# -*- coding: utf-8 -*-
"""
BERT vs Qwen3 基线对比实验 — 优化版（CPU 友好）
=================================================
针对 BERT 在 CPU 上推理慢的问题优化：
  1. 先跑 Qwen3 零样本（API 调用，不占本地算力）
  2. BERT 推理用 2000 条子样本（随机抽样，保证可复现）
  3. 限制 torch 线程数避免争抢

使用方式：
  conda activate openAI
  python bert_evaluate_fast.py
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'baseline_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'evaluation.log'), encoding='utf-8')
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# 限制 torch 线程数（避免 CPU 过载）
torch.set_num_threads(4)

# ════════════════════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════════════════════

CSV_PATH = os.path.join(_SCRIPT_DIR, '首都文化_情感得分_抽样15%.csv')
BERT_MODEL_PATH = os.path.join(_SCRIPT_DIR, 'bert_culture_classifier')

# === 请替换为你的实际 API 端点 ===
VLLM_BASE_URL = '<YOUR_VLLM_API_ENDPOINT>'
VLLM_API_KEY = '<YOUR_VLLM_API_KEY>'

SEED = 42
TEST_SIZE = 0.2
MAX_LENGTH = 128
BERT_BATCH_SIZE = 16
BERT_SUBSAMPLE = 2000  # BERT 评估用 2000 条子样本
QWEN_CONCURRENCY = 20

CULTURE_TYPES = ['古都文化', '红色文化', '京味文化', '创新文化']
ASPECTS_17 = [
    '交通便利', '人文景观', '人流量', '体力消耗', '公共设施', '历史认知',
    '商业环境', '天气气候', '建筑美学', '情感共鸣', '文化体验', '文化内涵',
    '文化氛围', '文化遗产', '游客服务', '自然景观', '饮食体验',
]
POLARITIES = ['积极', '中立', '消极']


# ════════════════════════════════════════════════════════════════════════════
# 1. 数据加载
# ════════════════════════════════════════════════════════════════════════════

def load_data():
    logger.info(f"加载数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    df = df.dropna(subset=['cleaned_content', '文化类型']).copy()
    df['cleaned_content'] = df['cleaned_content'].astype(str)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['文化类型'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_content'].values, df['label'].values,
        test_size=TEST_SIZE, random_state=SEED, stratify=df['label']
    )
    logger.info(f"训练集: {len(X_train):,} | 测试集: {len(y_test):,}")

    # 同一 split 的 aspect 和 polarity 标签
    test_indices = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=SEED, stratify=df['label']
    )[1]
    df_test = df.loc[test_indices]

    return {
        'X_test': X_test, 'y_test': y_test, 'label_encoder': le,
        'df_test': df_test,
        'aspect_texts': df_test.dropna(subset=['评价方面'])['cleaned_content'].astype(str).tolist(),
        'aspect_labels': df_test.dropna(subset=['评价方面'])['评价方面'].tolist(),
        'polarity_texts': df_test.dropna(subset=['情感'])['cleaned_content'].astype(str).tolist(),
        'polarity_labels': df_test.dropna(subset=['情感'])['情感'].tolist(),
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. BERT 评估（子样本）
# ════════════════════════════════════════════════════════════════════════════

def evaluate_bert(data):
    logger.info("=" * 60)
    logger.info("BERT-base-Chinese 文化类型分类 (微调, 2000条子样本)")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    model.eval()

    le = data['label_encoder']
    X_test = data['X_test']
    y_test = data['y_test']

    # 子抽样 2000 条（分层抽样）
    rng = np.random.RandomState(SEED)
    sub_indices = []
    for label in range(len(le.classes_)):
        label_indices = np.where(y_test == label)[0]
        n_sample = min(BERT_SUBSAMPLE // len(le.classes_), len(label_indices))
        sub_indices.extend(rng.choice(label_indices, n_sample, replace=False).tolist())
    sub_indices = sorted(sub_indices)
    X_sub = X_test[sub_indices]
    y_sub = y_test[sub_indices]
    logger.info(f"子样本: {len(X_sub)} 条（分层抽样）")

    all_preds = []
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(X_sub), BERT_BATCH_SIZE):
            batch = [str(t) for t in X_sub[i:i + BERT_BATCH_SIZE]]
            inputs = tokenizer(batch, truncation=True, padding='max_length',
                             max_length=MAX_LENGTH, return_tensors='pt')
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.numpy().tolist())

            if i % (BERT_BATCH_SIZE * 10) == 0:
                elapsed = time.time() - start
                progress = min(i + BERT_BATCH_SIZE, len(X_sub))
                logger.info(f"  BERT {progress}/{len(X_sub)} ({progress/len(X_sub)*100:.0f}%) "
                           f"{elapsed:.0f}s")

    infer_time = time.time() - start
    y_pred = np.array(all_preds)

    macro_f1 = f1_score(y_sub, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_sub, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_sub, y_pred)
    kappa = cohen_kappa_score(y_sub, y_pred)
    report = classification_report(y_sub, y_pred, target_names=le.classes_, digits=4)
    cm = confusion_matrix(y_sub, y_pred)

    logger.info(f"\nBERT 结果:")
    logger.info(f"  Macro-F1: {macro_f1:.4f}")
    logger.info(f"  Weighted-F1: {weighted_f1:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Kappa: {kappa:.4f}")
    logger.info(f"\n{report}")

    return {
        'model': 'BERT-base-Chinese (fine-tuned)',
        'task': 'culture_classification',
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'accuracy': float(accuracy),
        'kappa': float(kappa),
        'infer_time_sec': float(infer_time),
        'n_test': len(X_sub),
        'n_train': 33222,
        'params': '110M',
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'label_names': list(le.classes_),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Qwen3 零样本
# ════════════════════════════════════════════════════════════════════════════

async def qwen_predict(texts, system_prompt, label_list, task_name, model_name):
    sem = asyncio.Semaphore(QWEN_CONCURRENCY)
    results = [None] * len(texts)
    done = 0
    t0 = time.time()

    async def fetch(session, idx, text):
        nonlocal done
        async with sem:
            payload = {
                'model': model_name,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f'文本：{text}\n请只输出分类标签。'},
                ],
                'temperature': 0.0,
                'max_tokens': 20,
            }
            headers = {'Authorization': f'Bearer {VLLM_API_KEY}'}
            try:
                async with session.post(f'{VLLM_BASE_URL}/chat/completions',
                                        json=payload, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    r = await resp.json()
                    pred = r['choices'][0]['message']['content'].strip()
                    matched = next((l for l in label_list if l in pred), label_list[0])
                    results[idx] = matched
            except Exception as e:
                results[idx] = label_list[0]

            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                logger.info(f"  [{task_name}] {done}/{len(texts)} {done/elapsed:.1f}/s")

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[fetch(session, i, t) for i, t in enumerate(texts)])

    elapsed = time.time() - t0
    logger.info(f"  [{task_name}] 完成 {elapsed:.0f}s")
    return results, elapsed


def metrics(y_true, y_pred, labels):
    l2i = {l: i for i, l in enumerate(labels)}
    yt = [l2i.get(l, 0) for l in y_true]
    yp = [l2i.get(l, 0) for l in y_pred]
    return {
        'macro_f1': float(f1_score(yt, yp, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(yt, yp, average='weighted', zero_division=0)),
        'accuracy': float(accuracy_score(yt, yp)),
        'kappa': float(cohen_kappa_score(yt, yp)),
    }


async def evaluate_qwen(data):
    logger.info("=" * 60)
    logger.info("Qwen3 零样本推理 (3个任务)")
    logger.info("=" * 60)

    import requests
    try:
        r = requests.get(f'{VLLM_BASE_URL}/models', timeout=5)
        model_name = r.json()['data'][0]['id']
        logger.info(f"模型: {model_name}")
    except Exception as e:
        logger.error(f"API 不可达: {e}")
        return None, 'unknown'

    results = []
    params = '27B' if '27b' in model_name.lower() else '32B'

    # 任务 1: 文化类型
    logger.info("\n--- 文化类型判别 ---")
    prompt_c = "判断微博属于哪类首都文化(古都文化/红色文化/京味文化/创新文化)，只输出标签。"
    texts = data['X_test'].tolist()
    trues = data['label_encoder'].inverse_transform(data['y_test']).tolist()
    preds, t = await qwen_predict(texts, prompt_c, CULTURE_TYPES, 'culture', model_name)
    m = metrics(trues, preds, CULTURE_TYPES)
    logger.info(f"  F1={m['macro_f1']:.4f} Acc={m['accuracy']:.4f} Kappa={m['kappa']:.4f}")
    results.append({'model': f'Qwen3 ({model_name}, zero-shot)', 'task': 'culture_classification',
                    **m, 'infer_time_sec': float(t), 'n_test': len(texts), 'n_train': 0, 'params': params})

    # 任务 2: 评价方面
    logger.info("\n--- 评价方面识别 ---")
    prompt_a = "识别微博评价方面(17类:交通便利/人文景观/人流量/体力消耗/公共设施/历史认知/商业环境/天气气候/建筑美学/情感共鸣/文化体验/文化内涵/文化氛围/文化遗产/游客服务/自然景观/饮食体验)，只输出一个标签。"
    texts = data['aspect_texts']
    trues = data['aspect_labels']
    preds, t = await qwen_predict(texts, prompt_a, ASPECTS_17, 'aspect', model_name)
    m = metrics(trues, preds, ASPECTS_17)
    logger.info(f"  F1={m['macro_f1']:.4f} Acc={m['accuracy']:.4f} Kappa={m['kappa']:.4f}")
    results.append({'model': f'Qwen3 ({model_name}, zero-shot)', 'task': 'aspect_classification',
                    **m, 'infer_time_sec': float(t), 'n_test': len(texts), 'n_train': 0, 'params': params})

    # 任务 3: 情感极性
    logger.info("\n--- 情感极性判断 ---")
    prompt_p = "判断微博情感极性(积极/中立/消极)，只输出标签。"
    texts = data['polarity_texts']
    trues = data['polarity_labels']
    preds, t = await qwen_predict(texts, prompt_p, POLARITIES, 'polarity', model_name)
    m = metrics(trues, preds, POLARITIES)
    logger.info(f"  F1={m['macro_f1']:.4f} Acc={m['accuracy']:.4f} Kappa={m['kappa']:.4f}")
    results.append({'model': f'Qwen3 ({model_name}, zero-shot)', 'task': 'polarity_classification',
                    **m, 'infer_time_sec': float(t), 'n_test': len(texts), 'n_train': 0, 'params': params})

    return results, model_name


# ════════════════════════════════════════════════════════════════════════════
# 4. 主函数
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("BERT vs Qwen3 基线对比实验 (优化版)")
    logger.info("=" * 60)

    data = load_data()

    # 先跑 Qwen3（不占本地算力）
    qwen_results, model_name = asyncio.run(evaluate_qwen(data))
    if qwen_results:
        with open(os.path.join(OUTPUT_DIR, 'qwen_zeroshot_all_tasks.json'), 'w', encoding='utf-8') as f:
            json.dump(qwen_results, f, ensure_ascii=False, indent=2)
        logger.info("Qwen 结果已保存")

    # 再跑 BERT（CPU 推理）
    bert_result = evaluate_bert(data)
    with open(os.path.join(OUTPUT_DIR, 'bert_culture_evaluation.json'), 'w', encoding='utf-8') as f:
        json.dump(bert_result, f, ensure_ascii=False, indent=2)
    logger.info("BERT 结果已保存")

    # 保存分类报告
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write("BERT-base-Chinese 文化类型分类\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Macro-F1:      {bert_result['macro_f1']:.4f}\n")
        f.write(f"Weighted-F1:   {bert_result['weighted_f1']:.4f}\n")
        f.write(f"Accuracy:      {bert_result['accuracy']:.4f}\n")
        f.write(f"Cohen's Kappa: {bert_result['kappa']:.4f}\n")
        f.write(f"测试样本:       {bert_result['n_test']} (子样本)\n\n")
        f.write(bert_result['classification_report'])

    # 生成对比表
    rows = []
    rows.append({
        '任务': '文化类型判别', '模型': 'BERT-base-Chinese (微调)',
        '参数量': '110M', '训练数据': '33,222 条',
        'Macro-F1': f"{bert_result['macro_f1']:.4f}",
        'Accuracy': f"{bert_result['accuracy']:.4f}",
        "Kappa": f"{bert_result['kappa']:.4f}",
    })
    task_cn = {'culture_classification': '文化类型判别',
               'aspect_classification': '评价方面识别',
               'polarity_classification': '情感极性判断'}
    for r in qwen_results or []:
        rows.append({
            '任务': task_cn[r['task']], '模型': r['model'],
            '参数量': r['params'], '训练数据': '0 (零样本)',
            'Macro-F1': f"{r['macro_f1']:.4f}",
            'Accuracy': f"{r['accuracy']:.4f}",
            "Kappa": f"{r['kappa']:.4f}",
        })
    # 待训练占位
    for t in ['评价方面识别', '情感极性判断']:
        rows.append({'任务': t, '模型': 'BERT-base-Chinese (微调)', '参数量': '110M',
                     '训练数据': '待训练', 'Macro-F1': '—', 'Accuracy': '—', 'Kappa': '—'})

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUTPUT_DIR, 'baseline_comparison_table.csv'),
                 index=False, encoding='utf-8-sig')

    print("\n" + "=" * 80)
    print("基线对比实验结果")
    print("=" * 80)
    print(table.to_string(index=False))
    print(f"\n结果目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
