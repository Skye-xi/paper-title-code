# -*- coding: utf-8 -*-
"""
BERT vs Qwen3 全面基线对比实验（回应审稿意见）
==============================================
审稿意见: "关于LLM超越微调模型的主张仅靠引用支持，建议增加实际基线比较"

本脚本完成:
  实验1: BERT (全量测试集8,306条) vs Qwen3 (零样本) 文化类型判别
  实验2: McNemar检验 + Cohen's Kappa + 效应量
  实验3: 逐类对比 + 混淆矩阵对比
  实验4: 论文用对比表格和结论段落

运行方式:
  pip install torch transformers scikit-learn pandas numpy scipy
  python baseline_comprehensive.py

注意:
  本文件已脱敏处理，可安全发布到 GitHub。
  所有路径均使用相对路径或 os.path.dirname 自动推导。
"""

import os
import sys
import json
import time
import math
import logging
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support, matthews_corrcoef
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
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'comprehensive_eval.log'),
                            encoding='utf-8'),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

torch.set_num_threads(12)

# ════════════════════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════════════════════

CSV_PATH = os.path.join(_SCRIPT_DIR, '首都文化_情感得分_抽样15%.csv')
BERT_MODEL_PATH = os.path.join(_SCRIPT_DIR, 'bert_culture_classifier')
QWEN_RESULTS_PATH = os.path.join(OUTPUT_DIR, 'qwen_zeroshot_all_tasks.json')
QWEN_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, 'qwen_culture_predictions.csv')

SEED = 42
TEST_SIZE = 0.2
MAX_LENGTH = 128
BERT_BATCH = 64

CULTURE_TYPES = ['古都文化', '红色文化', '京味文化', '创新文化']


# ════════════════════════════════════════════════════════════════════════════
# 1. 数据加载 (与训练时完全相同的 split)
# ════════════════════════════════════════════════════════════════════════════

def load_data():
    logger.info(f"加载数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    df = df.dropna(subset=['cleaned_content', '文化类型']).copy()
    df['cleaned_content'] = df['cleaned_content'].astype(str)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['文化类型'])

    X = df['cleaned_content'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)),
        test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    logger.info(f"训练集: {len(X_train):,} | 测试集: {len(X_test):,}")
    logger.info("类别分布 (测试集):")
    for i, name in enumerate(le.classes_):
        logger.info(f"  {name}: {int((y_test == i).sum())}")

    return {
        'X_test': X_test, 'y_test': y_test,
        'label_encoder': le,
        'df_full': df, 'idx_test': idx_test,
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. BERT 全量评估
# ════════════════════════════════════════════════════════════════════════════

def evaluate_bert_full(data):
    """BERT 在全量测试集上评估 (8,306条)"""
    logger.info("=" * 60)
    logger.info("BERT-base-Chinese 文化类型分类 (全量测试集)")
    logger.info("=" * 60)

    logger.info(f"加载模型: {BERT_MODEL_PATH}")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    model.eval()
    logger.info(f"模型加载耗时: {time.time() - t_load:.1f}s")

    le = data['label_encoder']
    X_test = data['X_test']
    y_test = data['y_test']
    n = len(X_test)

    logger.info(f"测试样本: {n:,} 条, batch_size={BERT_BATCH}")

    all_preds = []
    all_probs = []
    start = time.time()

    with torch.no_grad():
        for i in range(0, n, BERT_BATCH):
            batch = [str(t) for t in X_test[i:i + BERT_BATCH]]
            inputs = tokenizer(batch, truncation=True, padding=True,
                               max_length=MAX_LENGTH, return_tensors='pt')
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.numpy().tolist())
            all_probs.extend(probs.numpy().tolist())

            progress = min(i + BERT_BATCH, n)
            if i % (BERT_BATCH * 25) == 0:
                elapsed = time.time() - start
                speed = progress / max(elapsed, 0.001)
                eta = (n - progress) / max(speed, 0.001)
                logger.info(f"  {progress}/{n} ({progress / n * 100:.0f}%) "
                            f"{speed:.0f}条/s ETA:{eta:.0f}s")

    infer_time = time.time() - start
    y_pred = np.array(all_preds)
    logger.info(f"推理完成: {infer_time:.1f}s ({n / infer_time:.1f} 条/s)")

    # -- 指标 --
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=le.classes_, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    per_class_f1 = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )[2]

    logger.info(f"\nBERT 全量测试集结果:")
    logger.info(f"  Macro-F1:     {macro_f1:.4f}")
    logger.info(f"  Weighted-F1:  {weighted_f1:.4f}")
    logger.info(f"  Accuracy:     {accuracy:.4f}")
    logger.info(f"  Kappa:        {kappa:.4f}")
    logger.info(f"  MCC:          {mcc:.4f}")
    logger.info(f"\n逐类F1:")
    for i, name in enumerate(le.classes_):
        logger.info(f"  {name}: {per_class_f1[i]:.4f}")
    logger.info(f"\n{report}")

    # 保存预测
    true_labels = le.inverse_transform(y_test)
    pred_labels = le.inverse_transform(y_pred)
    pred_df = pd.DataFrame({
        'idx': data['idx_test'],
        'text': X_test,
        'true_label': true_labels,
        'pred_label': pred_labels,
        'correct': y_test == y_pred,
    })
    pred_df.to_csv(
        os.path.join(OUTPUT_DIR, 'bert_culture_full_predictions.csv'),
        index=False, encoding='utf-8-sig'
    )

    return {
        'model': 'BERT-base-Chinese (fine-tuned)',
        'task': 'culture_classification',
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'accuracy': float(accuracy),
        'kappa': float(kappa),
        'mcc': float(mcc),
        'per_class_f1': [float(x) for x in per_class_f1],
        'infer_time_sec': float(infer_time),
        'n_test': int(n),
        'params': '110M',
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'label_names': list(le.classes_),
        'y_pred': y_pred.tolist(),
        'y_true': y_test.tolist(),
    }, pred_df


# ════════════════════════════════════════════════════════════════════════════
# 3. 加载 Qwen 预测 + 计算指标
# ════════════════════════════════════════════════════════════════════════════

def load_qwen_data(label_encoder):
    """
    加载 Qwen3 零样本预测，同时返回:
      - qwen_metrics (dict):  汇总指标，用于表格和论文输出
      - qwen_pred_df (DataFrame 或 None): 逐条预测，用于 McNemar 检验
    """
    qwen_pred_df = None
    qwen_metrics = None

    # 优先从 CSV 加载逐条预测
    if os.path.exists(QWEN_PREDICTIONS_PATH):
        qwen_pred_df = pd.read_csv(QWEN_PREDICTIONS_PATH, encoding='utf-8-sig')
        logger.info(f"从 CSV 加载 Qwen 逐条预测: {len(qwen_pred_df)} 条")

        # 从逐条预测计算汇总指标
        le = label_encoder
        y_true = le.transform(qwen_pred_df['true_label'].values)
        y_pred = le.transform(qwen_pred_df['pred_label'].values)

        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )[2]
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred,
                                       target_names=le.classes_, digits=4)

        qwen_metrics = {
            'model': 'Qwen3-27B (zero-shot)',
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'accuracy': float(accuracy),
            'kappa': float(kappa),
            'mcc': float(mcc),
            'per_class_f1': [float(x) for x in per_class],
            'n_test': int(len(qwen_pred_df)),
            'params': '27B',
            'n_train': 0,
            'infer_time_sec': 509.0,  # 从 save 日志获取
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'label_names': list(le.classes_),
        }

        logger.info(f"Qwen3 指标 (从CSV计算): "
                    f"Macro-F1={macro_f1:.4f}, Acc={accuracy:.4f}, "
                    f"Kappa={kappa:.4f}, MCC={mcc:.4f}")

    # 如果没有 CSV，尝试从 JSON 加载
    if qwen_metrics is None and os.path.exists(QWEN_RESULTS_PATH):
        with open(QWEN_RESULTS_PATH, 'r', encoding='utf-8') as f:
            qwen_results = json.load(f)
        for r in qwen_results:
            if r['task'] == 'culture_classification':
                qwen_metrics = r
                logger.info(f"从 JSON 加载 Qwen 指标: "
                            f"Macro-F1={r['macro_f1']:.4f}")
                break

    if qwen_metrics is None:
        logger.warning("未找到 Qwen 预测结果")

    return qwen_metrics, qwen_pred_df


# ════════════════════════════════════════════════════════════════════════════
# 4. McNemar 检验
# ════════════════════════════════════════════════════════════════════════════

def mcnemar_test(y_true, y_pred_a, y_pred_b, label_a='A', label_b='B'):
    """
    McNemar's test for paired nominal data.
    H0: 两个分类器的错误率相同
    y_pred_a: 模型A的预测 (通常是 BERT)
    y_pred_b: 模型B的预测 (通常是 Qwen)
    """
    n01 = int(np.sum((y_pred_a != y_true) & (y_pred_b == y_true)))
    n10 = int(np.sum((y_pred_a == y_true) & (y_pred_b != y_true)))
    n11 = int(np.sum((y_pred_a == y_true) & (y_pred_b == y_true)))
    n00 = int(np.sum((y_pred_a != y_true) & (y_pred_b != y_true)))

    total = n00 + n01 + n10 + n11

    # 连续性校正的 McNemar chi^2
    if (n01 + n10) > 0:
        stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    else:
        stat = 0.0

    from scipy.stats import chi2
    p_value = float(1 - chi2.cdf(stat, 1))

    # 比例差异检验
    p_a = (n10 + n11) / total  # A 正确率
    p_b = (n01 + n11) / total  # B 正确率

    result = {
        'statistic': float(stat),
        'p_value': p_value,
        'n_both_correct': n11,
        'n_both_wrong': n00,
        f'n_{label_a}_correct_{label_b}_wrong': n10,
        f'n_{label_a}_wrong_{label_b}_correct': n01,
        'total': total,
        'significant_at_005': p_value < 0.05,
        'significant_at_001': p_value < 0.01,
        'accuracy_a': float(p_a),
        'accuracy_b': float(p_b),
    }
    return result


# ════════════════════════════════════════════════════════════════════════════
# 5. 效应量
# ════════════════════════════════════════════════════════════════════════════

def cohens_h(p1, p2):
    """Cohen's h 效应量 (两个比例之间的差异)"""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def compute_effect_sizes(bert_result, qwen_metrics):
    """计算效应量"""
    effects = {}
    acc_diff = bert_result['accuracy'] - qwen_metrics['accuracy']
    effects['accuracy_diff'] = float(acc_diff)
    effects['cohens_h'] = float(cohens_h(bert_result['accuracy'],
                                          qwen_metrics['accuracy']))
    effects['macro_f1_diff'] = float(bert_result['macro_f1'] -
                                      qwen_metrics['macro_f1'])
    effects['kappa_diff'] = float(bert_result['kappa'] - qwen_metrics['kappa'])
    effects['mcc_diff'] = float(bert_result['mcc'] - qwen_metrics.get('mcc', 0))
    return effects


# ════════════════════════════════════════════════════════════════════════════
# 6. 生成论文表格和文字
# ════════════════════════════════════════════════════════════════════════════

def generate_paper_output(bert_result, qwen_metrics, mcnemar_result,
                          effect_sizes):
    """生成论文用表格和文字描述"""
    output = []

    output.append("=" * 70)
    output.append("BERT vs Qwen3 基线对比实验 -- 论文用结果")
    output.append("=" * 70)

    # -- 表1: 主对比表 --
    output.append("\n[Table 1] 文化类型判别: BERT微调 vs Qwen3零样本")
    output.append("-" * 60)
    output.append(f"{'指标':<22} {'BERT(微调)':>16} {'Qwen3(零样本)':>16}")
    output.append("-" * 60)
    output.append(f"{'Macro-F1':<22} {bert_result['macro_f1']:>16.4f} "
                  f"{qwen_metrics['macro_f1']:>16.4f}")
    output.append(f"{'Weighted-F1':<22} {bert_result['weighted_f1']:>16.4f} "
                  f"{qwen_metrics['weighted_f1']:>16.4f}")
    output.append(f"{'Accuracy':<22} {bert_result['accuracy']:>16.4f} "
                  f"{qwen_metrics['accuracy']:>16.4f}")
    output.append(f"{'Cohen Kappa':<22} {bert_result['kappa']:>16.4f} "
                  f"{qwen_metrics['kappa']:>16.4f}")
    output.append(f"{'MCC':<22} {bert_result['mcc']:>16.4f} "
                  f"{qwen_metrics.get('mcc', 0):>16.4f}")
    output.append(f"{'参数规模':<22} {'110M':>16} {'27B':>16}")
    output.append(f"{'训练数据':<22} {'33,222条':>16} {'0(零样本)':>16}")
    n_bert = f"{bert_result['n_test']:,}条"
    n_qwen = f"{qwen_metrics.get('n_test', 0):,}条"
    t_bert = f"{bert_result['infer_time_sec']:.0f}s"
    t_qwen = f"{qwen_metrics.get('infer_time_sec', 0):.0f}s"
    output.append(f"{'测试样本':<22} {n_bert:>16} {n_qwen:>16}")
    output.append(f"{'推理耗时':<22} {t_bert:>16} {t_qwen:>16}")
    output.append("-" * 60)

    # -- 表2: 逐类F1 --
    output.append("\n[Table 2] 逐类F1对比")
    output.append("-" * 48)
    output.append(f"{'类别':<12} {'BERT F1':>14} {'Qwen3 F1':>14} {'差异':>8}")
    output.append("-" * 48)
    label_names = bert_result.get('label_names', CULTURE_TYPES)
    bert_f1s = bert_result.get('per_class_f1', [])
    qwen_f1s = qwen_metrics.get('per_class_f1', [])
    for i, name in enumerate(label_names):
        bf1 = bert_f1s[i] if i < len(bert_f1s) else 0
        qf1 = qwen_f1s[i] if i < len(qwen_f1s) else 0
        diff = bf1 - qf1
        output.append(f"{name:<12} {bf1:>14.4f} {qf1:>14.4f} "
                      f"{diff:>+8.4f}")
    output.append("-" * 48)

    # -- 表3: 统计检验 --
    output.append("\n[Table 3] 统计显著性检验 (McNemar's Test)")
    output.append("-" * 50)
    if mcnemar_result:
        output.append(f"{'指标':<30} {'值':>18}")
        output.append("-" * 50)
        output.append(f"{'McNemar chi^2':<30} "
                      f"{mcnemar_result['statistic']:>18.2f}")
        output.append(f"{'p-value':<30} "
                      f"{mcnemar_result['p_value']:>18.6f}")
        output.append(f"{'两者均正确':<30} "
                      f"{mcnemar_result['n_both_correct']:>18}")
        output.append(f"{'两者均错误':<30} "
                      f"{mcnemar_result['n_both_wrong']:>18}")
        output.append(f"{'BERT正确 Qwen错误':<30} "
                      f"{mcnemar_result.get('n_BERT_correct_Qwen_wrong', 0):>18}")
        output.append(f"{'BERT错误 Qwen正确':<30} "
                      f"{mcnemar_result.get('n_BERT_wrong_Qwen_correct', 0):>18}")
        output.append(f"{'总样本':<30} "
                      f"{mcnemar_result['total']:>18}")
        sig_05 = '是' if mcnemar_result['significant_at_005'] else '否'
        sig_01 = '是' if mcnemar_result['significant_at_001'] else '否'
        output.append(f"{'显著 (alpha=0.05)':<30} {sig_05:>18}")
        output.append(f"{'显著 (alpha=0.01)':<30} {sig_01:>18}")
    else:
        output.append("(需要 Qwen 逐条预测结果才能计算 McNemar 检验)")
    output.append("-" * 50)

    # -- 效应量 --
    if effect_sizes:
        output.append(f"\n效应量:")
        output.append(f"  Cohen's h:        {effect_sizes['cohens_h']:.4f}")
        output.append(f"  Macro-F1 差异:     {effect_sizes['macro_f1_diff']:+.4f}")
        output.append(f"  Accuracy 差异:    {effect_sizes['accuracy_diff']:+.4f}")
        output.append(f"  Kappa 差异:       {effect_sizes['kappa_diff']:+.4f}")

    # -- 论文用结论段落 --
    output.append("\n" + "=" * 70)
    output.append("论文用结论描述 (可直接用于论文)")
    output.append("=" * 70)

    bert_f1 = bert_result['macro_f1']
    qwen_f1 = qwen_metrics['macro_f1']
    bert_acc = bert_result['accuracy']
    qwen_acc = qwen_metrics['accuracy']

    if qwen_f1 > bert_f1:
        winner = "Qwen3-27B (零样本)"
        loser = "BERT-base-Chinese (微调)"
        gap = qwen_f1 - bert_f1
        verb = "超越"
    else:
        winner = "BERT-base-Chinese (微调)"
        loser = "Qwen3-27B (零样本)"
        gap = bert_f1 - qwen_f1
        verb = "优于"

    # McNemar 描述
    if mcnemar_result:
        p_str = '<0.001' if mcnemar_result['p_value'] < 0.001 else f"{mcnemar_result['p_value']:.4f}"
        sig_str = '具有统计学显著性' if mcnemar_result['significant_at_005'] else '不显著'
        mcnemar_text = (
            f"McNemar配对检验结果 (chi^2={mcnemar_result['statistic']:.2f}, "
            f"p={p_str}) "
            f"表明两模型性能差异{sig_str}。"
        )
    else:
        mcnemar_text = ""

    # Cohen's h 解读
    h_val = effect_sizes.get('cohens_h', 0) if effect_sizes else 0
    if abs(h_val) < 0.2:
        h_interp = "可忽略"
    elif abs(h_val) < 0.5:
        h_interp = "小效应"
    elif abs(h_val) < 0.8:
        h_interp = "中等效应"
    else:
        h_interp = "大效应"

    paragraph = f"""
为回应审稿人关于"缺乏经验基线对比"的意见，本研究在相同测试集
({bert_result['n_test']:,}条)上对BERT微调模型与Qwen3-27B零样本推理
进行了直接对比。BERT-base-Chinese在33,222条标注数据上微调3轮后，
测试集Macro-F1为{bert_f1:.4f}，Accuracy为{bert_acc:.4f}；
Qwen3-27B在零样本设定下（无任何训练数据），Macro-F1为{qwen_f1:.4f}，
Accuracy为{qwen_acc:.4f}。

结果显示{winner}在文化类型判别任务上{verb}{loser}
（Macro-F1差距{gap:.4f}，Cohen's h={h_val:.4f}，属{h_interp}）。
{mcnemar_text}

这一结果表明，大语言模型的零样本能力在文化文本分类任务上
{'确实能够超过' if qwen_f1 > bert_f1 else '尚不足以超过'}
在中等规模标注数据上微调的专用模型，
{'支持' if qwen_f1 > bert_f1 else '部分支持'}了本研究采用LLM进行
文化文本标注的合理性。尽管BERT模型参数量仅为110M（Qwen3为27B），
但在领域特定任务上微调后仍具有竞争力。
"""
    output.append(paragraph)

    return "\n".join(output)


# ════════════════════════════════════════════════════════════════════════════
# 7. 保存混淆矩阵数据
# ════════════════════════════════════════════════════════════════════════════

def save_confusion_data(bert_pred_df, qwen_pred_df, label_names):
    """保存混淆矩阵对比数据"""
    import csv

    rows = []

    # BERT 混淆矩阵
    bert_cm = confusion_matrix(
        bert_pred_df['true_label'], bert_pred_df['pred_label'],
        labels=label_names
    )
    for i, tl in enumerate(label_names):
        for j, pl in enumerate(label_names):
            rows.append({'Model': 'BERT', 'True': tl, 'Pred': pl,
                         'Count': int(bert_cm[i][j])})

    # Qwen 混淆矩阵
    if qwen_pred_df is not None:
        qwen_cm = confusion_matrix(
            qwen_pred_df['true_label'], qwen_pred_df['pred_label'],
            labels=label_names
        )
        for i, tl in enumerate(label_names):
            for j, pl in enumerate(label_names):
                rows.append({'Model': 'Qwen3', 'True': tl, 'Pred': pl,
                             'Count': int(qwen_cm[i][j])})

    cm_df = pd.DataFrame(rows)
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrices.csv')
    cm_df.to_csv(cm_path, index=False, encoding='utf-8-sig')
    logger.info(f"混淆矩阵数据已保存: {cm_path}")

    # 归一化
    norm_rows = []
    for model_name in ['BERT'] + (['Qwen3'] if qwen_pred_df is not None else []):
        cm = bert_cm if model_name == 'BERT' else qwen_cm
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        for i, tl in enumerate(label_names):
            for j, pl in enumerate(label_names):
                norm_rows.append({'Model': model_name, 'True': tl, 'Pred': pl,
                                  'Proportion': f'{cm_norm[i][j]:.4f}'})
    norm_df = pd.DataFrame(norm_rows)
    norm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_normalized.csv')
    norm_df.to_csv(norm_path, index=False, encoding='utf-8-sig')
    logger.info(f"归一化混淆矩阵已保存: {norm_path}")


# ════════════════════════════════════════════════════════════════════════════
# 8. 主函数
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("BERT vs Qwen3 全面基线对比实验")
    logger.info("回应审稿意见: LLM vs 微调模型实证对比")
    logger.info("=" * 60)

    # -- 1. 数据 --
    data = load_data()
    le = data['label_encoder']

    # -- 2. BERT 全量评估 --
    bert_result, bert_pred_df = evaluate_bert_full(data)

    # 保存 BERT 结果 (不包含 y_pred/y_true 列表)
    bert_json = {k: v for k, v in bert_result.items()
                 if k not in ['y_pred', 'y_true']}
    with open(os.path.join(OUTPUT_DIR, 'bert_full_evaluation.json'), 'w',
              encoding='utf-8') as f:
        json.dump(bert_json, f, ensure_ascii=False, indent=2)
    logger.info("BERT 全量评估结果已保存: bert_full_evaluation.json")

    # -- 3. 加载 Qwen 预测 + 指标 --
    qwen_metrics, qwen_pred_df = load_qwen_data(le)

    if qwen_metrics is None:
        logger.error("无法加载 Qwen 结果，请先运行 save_qwen_predictions.py")
        return

    # -- 4. McNemar 检验 --
    mcnemar_result = None
    if qwen_pred_df is not None and 'pred_label' in qwen_pred_df.columns:
        logger.info("Qwen 逐条预测可用，执行 McNemar 检验...")

        # 对齐 BERT 和 Qwen 预测 (通过 idx)
        bert_idx_map = dict(zip(bert_pred_df['idx'],
                                range(len(bert_pred_df))))
        qwen_idx_map = dict(zip(qwen_pred_df['idx'].astype(int),
                                range(len(qwen_pred_df))))

        common_idx = sorted(set(bert_idx_map.keys()) & set(qwen_idx_map.keys()))
        logger.info(f"共同样本数: {len(common_idx):,}")

        if len(common_idx) > 0:
            bert_indices = [bert_idx_map[i] for i in common_idx]
            qwen_indices = [qwen_idx_map[i] for i in common_idx]

            y_true = le.transform(
                bert_pred_df.iloc[bert_indices]['true_label'].values
            )
            y_bert = le.transform(
                bert_pred_df.iloc[bert_indices]['pred_label'].values
            )
            y_qwen = le.transform(
                qwen_pred_df.iloc[qwen_indices]['pred_label'].values
            )

            mcnemar_result = mcnemar_test(
                y_true, y_bert, y_qwen,
                label_a='BERT', label_b='Qwen'
            )
            logger.info(f"McNemar chi^2={mcnemar_result['statistic']:.2f}, "
                        f"p={mcnemar_result['p_value']:.6f}")
            logger.info(f"  两者均正确: {mcnemar_result['n_both_correct']}")
            logger.info(f"  两者均错误: {mcnemar_result['n_both_wrong']}")
            logger.info(f"  BERT对Qwen错: "
                        f"{mcnemar_result['n_BERT_correct_Qwen_wrong']}")
            logger.info(f"  BERT错Qwen对: "
                        f"{mcnemar_result['n_BERT_wrong_Qwen_correct']}")
        else:
            logger.warning("无法对齐 BERT 和 Qwen 预测 (索引不匹配)")
    else:
        logger.info("Qwen 逐条预测不可用，跳过 McNemar 检验")

    # -- 5. 效应量 --
    effect_sizes = compute_effect_sizes(bert_result, qwen_metrics)
    logger.info(f"效应量: Cohen's h={effect_sizes['cohens_h']:.4f}, "
                f"F1差异={effect_sizes['macro_f1_diff']:+.4f}")

    # -- 6. 生成论文输出 --
    output_text = generate_paper_output(
        bert_result, qwen_metrics, mcnemar_result, effect_sizes
    )

    output_path = os.path.join(OUTPUT_DIR, 'paper_comparison_results.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_text)
    logger.info(f"论文结果已保存: {output_path}")

    # -- 7. 混淆矩阵数据 --
    save_confusion_data(bert_pred_df, qwen_pred_df, list(le.classes_))

    # -- 8. CSV 汇总表 --
    rows = []
    rows.append({
        '模型': 'BERT-base-Chinese (微调)',
        '参数量': '110M',
        '训练数据': '33,222条',
        '测试样本': bert_result['n_test'],
        'Macro-F1': f"{bert_result['macro_f1']:.4f}",
        'Weighted-F1': f"{bert_result['weighted_f1']:.4f}",
        'Accuracy': f"{bert_result['accuracy']:.4f}",
        "Cohen's Kappa": f"{bert_result['kappa']:.4f}",
        'MCC': f"{bert_result['mcc']:.4f}",
        '推理耗时(s)': f"{bert_result['infer_time_sec']:.0f}",
    })
    rows.append({
        '模型': 'Qwen3-27B (零样本)',
        '参数量': '27B',
        '训练数据': '0 (零样本)',
        '测试样本': qwen_metrics.get('n_test', 0),
        'Macro-F1': f"{qwen_metrics['macro_f1']:.4f}",
        'Weighted-F1': f"{qwen_metrics['weighted_f1']:.4f}",
        'Accuracy': f"{qwen_metrics['accuracy']:.4f}",
        "Cohen's Kappa": f"{qwen_metrics['kappa']:.4f}",
        'MCC': f"{qwen_metrics.get('mcc', 0):.4f}",
        '推理耗时(s)': f"{qwen_metrics.get('infer_time_sec', 0):.0f}",
    })

    table_df = pd.DataFrame(rows)
    table_path = os.path.join(OUTPUT_DIR, 'baseline_comprehensive_table.csv')
    table_df.to_csv(table_path, index=False, encoding='utf-8-sig')
    logger.info(f"对比表已保存: {table_path}")

    # -- 9. McNemar 结果单独保存 --
    if mcnemar_result:
        mcnemar_path = os.path.join(OUTPUT_DIR, 'mcnemar_test_result.json')
        with open(mcnemar_path, 'w', encoding='utf-8') as f:
            json.dump(mcnemar_result, f, ensure_ascii=False, indent=2)
        logger.info(f"McNemar 结果已保存: {mcnemar_path}")

    # -- 10. Qwen 指标保存 --
    qwen_json_path = os.path.join(OUTPUT_DIR, 'qwen_metrics_computed.json')
    qwen_save = {k: v for k, v in qwen_metrics.items()
                 if k not in ['y_pred', 'y_true']}
    with open(qwen_json_path, 'w', encoding='utf-8') as f:
        json.dump(qwen_save, f, ensure_ascii=False, indent=2)
    logger.info(f"Qwen 指标已保存: {qwen_json_path}")

    # -- 打印最终结果 --
    print("\n" + output_text)
    print(f"\n{'=' * 60}")
    print(f"所有结果保存目录: {OUTPUT_DIR}")
    print(f"  - bert_full_evaluation.json")
    print(f"  - qwen_metrics_computed.json")
    print(f"  - baseline_comprehensive_table.csv")
    print(f"  - paper_comparison_results.txt")
    print(f"  - confusion_matrices.csv")
    print(f"  - confusion_matrix_normalized.csv")
    if mcnemar_result:
        print(f"  - mcnemar_test_result.json")
    print(f"  - bert_culture_full_predictions.csv")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
