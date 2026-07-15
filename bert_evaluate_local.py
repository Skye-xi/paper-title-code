# -*- coding: utf-8 -*-
"""
BERT vs Qwen3 基线对比实验 — 本地评估脚本
==========================================
针对审稿意见 3：LLM vs 微调模型基线对比

功能：
  1. 加载 15% 抽样数据，复现训练时的 train/test split (random_state=42, 8:2, stratified)
  2. 加载已训练的 BERT-base-Chinese 文化类型分类模型，在测试集上评估
     → Macro-F1 / Weighted-F1 / Accuracy / Cohen's Kappa / 分类报告 / 混淆矩阵
  3. 通过 vLLM API 运行 Qwen3 零样本推理（3个任务：文化类型/评价方面/情感极性）
  4. 生成论文表格

依赖环境：
  conda activate openAI
  (已安装: torch 1.12.1, transformers 4.24.0, pandas, numpy, aiohttp, scikit-learn, safetensors, joblib)

使用方式：
  python bert_evaluate_local.py

输出：
  ./baseline_results/ 目录下：
  - bert_culture_evaluation.json    BERT 文化分类详细结果
  - qwen_zeroshot_all_tasks.json    Qwen 零样本三个任务结果
  - baseline_comparison_table.csv   论文用对比表
  - bert_confusion_matrix.png       BERT 混淆矩阵图
  - classification_report.txt       详细分类报告
"""

import os
import sys
import json
import time
import logging
import asyncio
import warnings
import numpy as np
import pandas as pd
import torch
import aiohttp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings('ignore')

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(OUTPUT_DIR, 'evaluation.log'),
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════════════════════

# 数据文件
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_SCRIPT_DIR, '首都文化_情感得分_抽样15%.csv')

# 已训练的 BERT 模型路径
BERT_MODEL_PATH = os.path.join(_SCRIPT_DIR, 'bert_culture_classifier')

# === 请替换为你的实际 API 端点 ===
VLLM_BASE_URL = '<YOUR_VLLM_API_ENDPOINT>'
VLLM_API_KEY = '<YOUR_VLLM_API_KEY>'

# 输出目录（已在上方创建）
# OUTPUT_DIR 已在 logging 设置前定义

# 实验参数
SEED = 42
TEST_SIZE = 0.2
MAX_LENGTH = 128
BERT_BATCH_SIZE = 32

# Qwen 零样本并发数
QWEN_CONCURRENCY = 20

# 标签定义
CULTURE_TYPES = ['古都文化', '红色文化', '京味文化', '创新文化']
ASPECTS_17 = [
    '交通便利', '人文景观', '人流量', '体力消耗', '公共设施', '历史认知',
    '商业环境', '天气气候', '建筑美学', '情感共鸣', '文化体验', '文化内涵',
    '文化氛围', '文化遗产', '游客服务', '自然景观', '饮食体验',
]
POLARITIES = ['积极', '中立', '消极']


# ════════════════════════════════════════════════════════════════════════════
# 1. 数据加载与分割
# ════════════════════════════════════════════════════════════════════════════

def load_and_split_data():
    """加载 15% 抽样数据，复现训练时的 split。"""
    logger.info(f"加载数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    logger.info(f"原始数据: {len(df):,} 条")

    # 文化类型分类任务
    text_col = 'cleaned_content'
    label_col = '文化类型'

    df_clean = df.dropna(subset=[text_col, label_col]).copy()
    df_clean[text_col] = df_clean[text_col].astype(str)
    logger.info(f"删除空值后: {len(df_clean):,} 条")

    # LabelEncoder（与训练代码完全一致）
    le = LabelEncoder()
    df_clean['label'] = le.fit_transform(df_clean[label_col])
    logger.info(f"标签编码: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # 分割（与训练代码完全一致）
    X = df_clean[text_col].values
    y = df_clean['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    logger.info(f"训练集: {len(X_train):,} 条 | 测试集: {len(X_test):,} 条")

    # 同时准备 aspect 和 polarity 的测试集
    # 对于 aspect 和 polarity，使用同样的 test indices
    df_test = df_clean.iloc[
        train_test_split(df_clean.index, test_size=TEST_SIZE,
                         random_state=SEED, stratify=y)[1]
    ].copy()

    aspect_test = df_test.dropna(subset=['评价方面'])
    polarity_test = df_test.dropna(subset=['情感'])

    logger.info(f"方面识别测试集: {len(aspect_test):,} 条")
    logger.info(f"情感极性测试集: {len(polarity_test):,} 条")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'label_encoder': le,
        'df_test': df_test,
        'aspect_texts': aspect_test['cleaned_content'].astype(str).tolist(),
        'aspect_labels': aspect_test['评价方面'].tolist(),
        'polarity_texts': polarity_test['cleaned_content'].astype(str).tolist(),
        'polarity_labels': polarity_test['情感'].tolist(),
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. BERT 模型评估
# ════════════════════════════════════════════════════════════════════════════

def evaluate_bert_model(data):
    """加载已训练的 BERT 模型，在测试集上评估。"""
    logger.info("=" * 60)
    logger.info("任务 1: BERT-base-Chinese 文化类型分类 (微调)")
    logger.info("=" * 60)

    # 加载模型
    logger.info(f"加载模型: {BERT_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"设备: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    le = data['label_encoder']
    X_test = data['X_test']
    y_test = data['y_test']

    # 批量推理
    logger.info(f"开始推理 ({len(X_test):,} 条)...")
    all_preds = []
    all_probs = []

    start_time = time.time()
    with torch.no_grad():
        for i in range(0, len(X_test), BERT_BATCH_SIZE):
            batch_texts = [str(t) for t in X_test[i:i + BERT_BATCH_SIZE]]
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=MAX_LENGTH,
                return_tensors='pt'
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

            if (i // BERT_BATCH_SIZE) % 50 == 0:
                elapsed = time.time() - start_time
                progress = min(i + BERT_BATCH_SIZE, len(X_test))
                speed = progress / max(elapsed, 0.001)
                eta = (len(X_test) - progress) / max(speed, 0.001)
                logger.info(f"  {progress}/{len(X_test)} ({progress/len(X_test)*100:.1f}%) "
                           f"速度: {speed:.1f} 条/s ETA: {eta:.0f}s")

    infer_time = time.time() - start_time
    logger.info(f"推理完成，耗时: {infer_time:.1f}s ({len(X_test)/infer_time:.1f} 条/s)")

    # 计算指标
    y_true = y_test
    y_pred = np.array(all_preds)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # 分类报告
    target_names = le.classes_
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    logger.info(f"\n{'='*60}")
    logger.info("BERT 文化类型分类结果")
    logger.info(f"{'='*60}")
    logger.info(f"  Macro-F1:    {macro_f1:.4f}")
    logger.info(f"  Weighted-F1: {weighted_f1:.4f}")
    logger.info(f"  Accuracy:    {accuracy:.4f}")
    logger.info(f"  Cohen's Kappa: {kappa:.4f}")
    logger.info(f"\n分类报告:\n{report}")
    logger.info(f"\n混淆矩阵:\n{cm}")

    # 保存预测结果
    pred_labels = le.inverse_transform(y_pred)
    true_labels = le.inverse_transform(y_true)
    pred_df = pd.DataFrame({
        'text': X_test,
        'true_label': true_labels,
        'pred_label': pred_labels,
        'correct': [t == p for t, p in zip(true_labels, pred_labels)],
    })

    result = {
        'model': 'BERT-base-Chinese (fine-tuned)',
        'task': 'culture_classification',
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'accuracy': float(accuracy),
        'kappa': float(kappa),
        'infer_time_sec': float(infer_time),
        'n_test': len(X_test),
        'n_train': len(data['X_train']),
        'params': '110M',
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'label_names': list(target_names),
    }

    return result, pred_df


# ════════════════════════════════════════════════════════════════════════════
# 3. Qwen3 零样本推理
# ════════════════════════════════════════════════════════════════════════════

async def _get_model_name(session):
    """获取 vLLM 上可用的模型名（只调用一次）。"""
    try:
        async with session.get(f'{VLLM_BASE_URL}/models', timeout=aiohttp.ClientTimeout(total=5)) as resp:
            data = await resp.json()
            models = data.get('data', [])
            if models:
                return models[0]['id']
    except:
        pass
    return 'Qwen3-32B'


async def _qwen_batch_predict(texts, system_prompt, label_list, task_name, model_name):
    """异步批量调用 Qwen3 零样本推理。"""
    semaphore = asyncio.Semaphore(QWEN_CONCURRENCY)
    results = [None] * len(texts)
    completed = 0
    start_time = time.time()

    async def fetch_one(session, idx, text):
        nonlocal completed
        async with semaphore:
            payload = {
                'model': model_name,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f'文本：{text}\n请只输出分类标签，不要输出其他内容。'},
                ],
                'temperature': 0.0,
                'max_tokens': 20,
            }
            headers = {'Authorization': f'Bearer {VLLM_API_KEY}'}
            url = f'{VLLM_BASE_URL}/chat/completions'
            try:
                async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    result = await resp.json()
                    pred_text = result['choices'][0]['message']['content'].strip()
                    # 匹配标签
                    matched = None
                    for label in label_list:
                        if label in pred_text:
                            matched = label
                            break
                    if matched is None:
                        matched = label_list[0]
                    results[idx] = matched
            except Exception as e:
                logger.error(f"  推理失败 idx={idx}: {e}")
                results[idx] = label_list[0]

            completed += 1
            if completed % 200 == 0:
                elapsed = time.time() - start_time
                speed = completed / max(elapsed, 0.001)
                eta = (len(texts) - completed) / max(speed, 0.001)
                logger.info(f"  [{task_name}] {completed}/{len(texts)} "
                           f"速度: {speed:.1f} 条/s ETA: {eta:.0f}s")

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, i, t) for i, t in enumerate(texts)]
        await asyncio.gather(*tasks)

    infer_time = time.time() - start_time
    logger.info(f"  [{task_name}] 完成，耗时: {infer_time:.1f}s")
    return results, infer_time


def _compute_metrics(true_labels, pred_labels, label_list):
    """计算分类指标。"""
    label2id = {l: i for i, l in enumerate(label_list)}
    y_true = [label2id.get(l, 0) for l in true_labels]
    y_pred = [label2id.get(l, 0) for l in pred_labels]

    return {
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
    }


async def evaluate_qwen_zeroshot(data):
    """Qwen3 零样本评估三个任务。"""
    logger.info("=" * 60)
    logger.info("任务 2: Qwen3 零样本推理 (3个分类任务)")
    logger.info("=" * 60)

    # 检查 API 可达性并获取模型名
    try:
        import requests
        r = requests.get(f'{VLLM_BASE_URL}/models', timeout=5)
        models = r.json().get('data', [])
        model_name = models[0]['id'] if models else 'unknown'
        logger.info(f"Qwen API 可达，当前模型: {model_name}")
    except Exception as e:
        logger.error(f"Qwen API 不可达: {e}")
        return None, 'unknown'

    results = []

    # ── 任务 1: 文化类型判别 ──────────────────────────────
    logger.info("\n--- 文化类型判别 ---")
    prompt_culture = """你是文化文本分类助手。请判断微博文本属于以下哪一类首都文化：
古都文化、红色文化、京味文化、创新文化。
仅输出4个标签之一，不要输出其他内容。"""
    texts = data['X_test'].tolist()
    true_labels = data['label_encoder'].inverse_transform(data['y_test']).tolist()

    preds, infer_time = await _qwen_batch_predict(
        texts, prompt_culture, CULTURE_TYPES, 'culture', model_name
    )
    metrics = _compute_metrics(true_labels, preds, CULTURE_TYPES)
    result = {
        'model': f'Qwen3 (zero-shot, {model_name})',
        'task': 'culture_classification',
        **metrics,
        'infer_time_sec': float(infer_time),
        'n_test': len(texts),
        'n_train': 0,
        'params': '27B' if '27b' in model_name.lower() else '32B',
    }
    logger.info(f"  Macro-F1: {metrics['macro_f1']:.4f} | "
               f"Accuracy: {metrics['accuracy']:.4f} | "
               f"Kappa: {metrics['kappa']:.4f}")
    results.append(result)

    # ── 任务 2: 评价方面识别 ──────────────────────────────
    logger.info("\n--- 评价方面识别 ---")
    prompt_aspect = """你是ABSA分析助手。从微博中识别用户对物质文化载体的评价方面，
共17类：交通便利、人文景观、人流量、体力消耗、公共设施、历史认知、商业环境、
天气气候、建筑美学、情感共鸣、文化体验、文化内涵、文化氛围、文化遗产、游客服务、
自然景观、饮食体验。仅输出最相关的一个方面标签，不要输出其他内容。"""
    texts = data['aspect_texts']
    true_labels = data['aspect_labels']

    preds, infer_time = await _qwen_batch_predict(
        texts, prompt_aspect, ASPECTS_17, 'aspect', model_name
    )
    metrics = _compute_metrics(true_labels, preds, ASPECTS_17)
    result = {
        'model': f'Qwen3 (zero-shot, {model_name})',
        'task': 'aspect_classification',
        **metrics,
        'infer_time_sec': float(infer_time),
        'n_test': len(texts),
        'n_train': 0,
        'params': '27B' if '27b' in model_name.lower() else '32B',
    }
    logger.info(f"  Macro-F1: {metrics['macro_f1']:.4f} | "
               f"Accuracy: {metrics['accuracy']:.4f} | "
               f"Kappa: {metrics['kappa']:.4f}")
    results.append(result)

    # ── 任务 3: 情感极性判断 ──────────────────────────────
    logger.info("\n--- 情感极性判断 ---")
    prompt_polarity = """你是情感分类助手。判断微博文本的情感极性，仅输出以下三类之一：
积极、中立、消极。不要输出其他内容。"""
    texts = data['polarity_texts']
    true_labels = data['polarity_labels']

    preds, infer_time = await _qwen_batch_predict(
        texts, prompt_polarity, POLARITIES, 'polarity', model_name
    )
    metrics = _compute_metrics(true_labels, preds, POLARITIES)
    result = {
        'model': f'Qwen3 (zero-shot, {model_name})',
        'task': 'polarity_classification',
        **metrics,
        'infer_time_sec': float(infer_time),
        'n_test': len(texts),
        'n_train': 0,
        'params': '27B' if '27b' in model_name.lower() else '32B',
    }
    logger.info(f"  Macro-F1: {metrics['macro_f1']:.4f} | "
               f"Accuracy: {metrics['accuracy']:.4f} | "
               f"Kappa: {metrics['kappa']:.4f}")
    results.append(result)

    return results, model_name


# ════════════════════════════════════════════════════════════════════════════
# 4. 生成论文表格
# ════════════════════════════════════════════════════════════════════════════

def generate_paper_table(bert_result, qwen_results, qwen_model_name):
    """生成论文用对比表格。"""
    rows = []

    # BERT 文化分类
    if bert_result:
        rows.append({
            '任务': '文化类型判别',
            '模型': 'BERT-base-Chinese (微调)',
            '参数量': '110M',
            '训练数据': f"{bert_result['n_train']} 条",
            'Macro-F1': f"{bert_result['macro_f1']:.4f}",
            'Weighted-F1': f"{bert_result['weighted_f1']:.4f}",
            'Accuracy': f"{bert_result['accuracy']:.4f}",
            "Cohen's Kappa": f"{bert_result['kappa']:.4f}",
        })

    # Qwen 结果
    if qwen_results:
        task_map = {
            'culture_classification': '文化类型判别',
            'aspect_classification': '评价方面识别',
            'polarity_classification': '情感极性判断',
        }
        for r in qwen_results:
            rows.append({
                '任务': task_map.get(r['task'], r['task']),
                '模型': r['model'],
                '参数量': r['params'],
                '训练数据': '0 条 (零样本)',
                'Macro-F1': f"{r['macro_f1']:.4f}",
                'Weighted-F1': f"{r['weighted_f1']:.4f}",
                'Accuracy': f"{r['accuracy']:.4f}",
                "Cohen's Kappa": f"{r['kappa']:.4f}",
            })

    # 添加占位行（待训练的 BERT 任务）
    if bert_result and qwen_results:
        # 标注哪些任务还缺 BERT 基线
        for task_name, task_cn in [('aspect', '评价方面识别'), ('polarity', '情感极性判断')]:
            rows.append({
                '任务': task_cn,
                '模型': 'BERT-base-Chinese (微调)',
                '参数量': '110M',
                '训练数据': '待训练',
                'Macro-F1': '—',
                'Weighted-F1': '—',
                'Accuracy': '—',
                "Cohen's Kappa": '—',
            })

    df = pd.DataFrame(rows)
    return df


def save_confusion_matrix_plot(cm, label_names, save_path):
    """保存混淆矩阵为文本（不依赖 matplotlib）。"""
    lines = []
    lines.append("混淆矩阵 (行=真实, 列=预测)")
    header = "\t" + "\t".join(label_names)
    lines.append(header)
    for i, row in enumerate(cm):
        lines.append(f"{label_names[i]}\t" + "\t".join(str(v) for v in row))
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


# ════════════════════════════════════════════════════════════════════════════
# 5. 主函数
# ════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BERT vs Qwen3 基线对比实验")
    logger.info("=" * 60)

    # 1. 加载数据
    data = load_and_split_data()

    # 2. BERT 评估
    bert_result, bert_pred_df = evaluate_bert_model(data)

    # 保存 BERT 预测结果
    bert_pred_path = os.path.join(OUTPUT_DIR, 'bert_culture_predictions.csv')
    bert_pred_df.to_csv(bert_pred_path, index=False, encoding='utf-8-sig')
    logger.info(f"BERT 预测结果已保存: {bert_pred_path}")

    # 保存 BERT 详细结果
    bert_json_path = os.path.join(OUTPUT_DIR, 'bert_culture_evaluation.json')
    with open(bert_json_path, 'w', encoding='utf-8') as f:
        json.dump(bert_result, f, ensure_ascii=False, indent=2)
    logger.info(f"BERT 评估结果已保存: {bert_json_path}")

    # 保存分类报告
    report_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("BERT-base-Chinese 文化类型分类 — 详细报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Macro-F1:     {bert_result['macro_f1']:.4f}\n")
        f.write(f"Weighted-F1:  {bert_result['weighted_f1']:.4f}\n")
        f.write(f"Accuracy:     {bert_result['accuracy']:.4f}\n")
        f.write(f"Cohen's Kappa: {bert_result['kappa']:.4f}\n")
        f.write(f"测试集大小:    {bert_result['n_test']}\n")
        f.write(f"训练集大小:    {bert_result['n_train']}\n")
        f.write(f"推理耗时:      {bert_result['infer_time_sec']:.1f}s\n\n")
        f.write("分类报告:\n")
        f.write(bert_result['classification_report'])
        f.write("\n\n混淆矩阵:\n")
        cm = bert_result['confusion_matrix']
        names = bert_result['label_names']
        f.write("\t" + "\t".join(names) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{names[i]}\t" + "\t".join(str(v) for v in row) + "\n")
    logger.info(f"分类报告已保存: {report_path}")

    # 3. Qwen3 零样本评估
    qwen_results, qwen_model_name = asyncio.run(evaluate_qwen_zeroshot(data))

    # 保存 Qwen 结果
    if qwen_results:
        qwen_json_path = os.path.join(OUTPUT_DIR, 'qwen_zeroshot_all_tasks.json')
        with open(qwen_json_path, 'w', encoding='utf-8') as f:
            json.dump(qwen_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Qwen 零样本结果已保存: {qwen_json_path}")

    # 4. 生成论文表格
    table_df = generate_paper_table(bert_result, qwen_results, qwen_model_name)
    table_path = os.path.join(OUTPUT_DIR, 'baseline_comparison_table.csv')
    table_df.to_csv(table_path, index=False, encoding='utf-8-sig')
    logger.info(f"论文对比表格已保存: {table_path}")

    # 打印表格
    print("\n" + "=" * 80)
    print("基线对比实验结果（可直接用于论文）")
    print("=" * 80)
    print(table_df.to_string(index=False))
    print(f"\n模型: {qwen_model_name if qwen_results else 'Qwen API 不可达'}")
    print(f"\n结果保存目录: {OUTPUT_DIR}")
    print(f"  - bert_culture_evaluation.json")
    print(f"  - qwen_zeroshot_all_tasks.json")
    print(f"  - baseline_comparison_table.csv")
    print(f"  - classification_report.txt")
    print(f"  - bert_culture_predictions.csv")

    # 5. 总结
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)
    if bert_result:
        print(f"\n[文化类型判别]")
        print(f"  BERT (微调):    Macro-F1={bert_result['macro_f1']:.4f}, "
              f"Acc={bert_result['accuracy']:.4f}, Kappa={bert_result['kappa']:.4f}")
    if qwen_results:
        for r in qwen_results:
            task_cn = {
                'culture_classification': '文化类型判别',
                'aspect_classification': '评价方面识别',
                'polarity_classification': '情感极性判断',
            }.get(r['task'], r['task'])
            print(f"\n[{task_cn}]")
            print(f"  Qwen3 (零样本):  Macro-F1={r['macro_f1']:.4f}, "
                  f"Acc={r['accuracy']:.4f}, Kappa={r['kappa']:.4f}")

    print("\n⚠️ 待完成:")
    print("  - BERT 评价方面识别 (需在 Colab 训练)")
    print("  - BERT 情感极性判断 (需在 Colab 训练)")
    print("  - RoBERTa-wwm-ext 三个任务 (需在 Colab 训练)")
    print("  → 使用 bert_colab_complete.py 在 Colab 上完成")


if __name__ == '__main__':
    main()
