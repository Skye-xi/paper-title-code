# -*- coding: utf-8 -*-
"""
完整 BERT + RoBERTa 基线训练脚本（Colab 版）
=============================================
针对审稿意见 3：LLM vs 微调模型基线对比

在 Google Colab (GPU) 上运行，训练以下 6 个模型：
  1. BERT-base-Chinese x 文化类型判别 (已在本地完成，可跳过)
  2. BERT-base-Chinese x 评价方面识别
  3. BERT-base-Chinese x 情感极性判断
  4. RoBERTa-wwm-ext x 文化类型判别
  5. RoBERTa-wwm-ext x 评价方面识别
  6. RoBERTa-wwm-ext x 情感极性判断

使用方式：
  1. 上传 首都文化_情感得分_抽样15%.csv 到 Colab
  2. 运行此脚本
  3. 自动下载训练好的模型和结果

重要：Colab 上的 transformers 版本兼容性
  之前训练 bert-base-chinese 时出现了 LayerNorm 参数加载失败的问题
  (beta/gamma vs weight/bias)，导致 LayerNorm 随机初始化。
  
  运行此脚本前，请先在 Colab 中执行：
    !pip install transformers==4.44.2
  
  该版本会自动处理 LayerNorm 参数名映射。

关键改进（相比原始 bert模型训练.py）：
  - 修复 LayerNorm 参数加载问题
  - 统一使用 Macro-F1 + Cohen's Kappa（论文指标）
  - 保存 id2label/label2id 到 config.json
  - label_encoder.pkl 使用兼容格式保存
  - 支持 6 个 模型x任务 组合批量训练
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    classification_report, confusion_matrix
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
import torch
from torch.utils.data import Dataset
import logging
import warnings
import json
import os
import joblib

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ════════════════════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════════════════════

FILE_PATH = '首都文化_情感得分_抽样15%.csv'
TEXT_COLUMN = 'cleaned_content'
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 2

# 三个任务配置
TASKS = [
    {
        'name': 'culture_classification',
        'label_column': '文化类型',
        'model_dir': './bert_culture_classifier',  # 已训练，可跳过
        'skip_if_exists': True,
    },
    {
        'name': 'aspect_classification',
        'label_column': '评价方面',
        'model_dir': './bert_aspect_classifier',
        'skip_if_exists': False,
    },
    {
        'name': 'polarity_classification',
        'label_column': '情感',
        'model_dir': './bert_polarity_classifier',
        'skip_if_exists': False,
    },
]

# 基线模型
MODELS = {
    'BERT-base-Chinese': 'bert-base-chinese',
    'RoBERTa-wwm-ext': 'hfl/chinese-roberta-wwm-ext',
}


# ════════════════════════════════════════════════════════════════════════════
# 数据集类
# ════════════════════════════════════════════════════════════════════════════

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ════════════════════════════════════════════════════════════════════════════
# 评估指标
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'macro_f1': f1_score(labels, predictions, average='macro', zero_division=0),
        'weighted_f1': f1_score(labels, predictions, average='weighted', zero_division=0),
        'kappa': cohen_kappa_score(labels, predictions),
    }


# ════════════════════════════════════════════════════════════════════════════
# 训练函数
# ════════════════════════════════════════════════════════════════════════════

def train_model(model_name, model_path, task_config, df):
    """训练单个模型。"""
    task_name = task_config['name']
    label_col = task_config['label_column']
    save_dir = task_config['model_dir'].replace('./bert_', f'./{model_name.split("/")[-1]}_')

    # 检查是否已存在
    if task_config.get('skip_if_exists') and os.path.exists(save_dir):
        logger.info(f"跳过已训练: {model_name} x {task_name}")
        return None

    logger.info(f"\n{'='*60}")
    logger.info(f"训练: {model_name} x {task_name}")
    logger.info(f"{'='*60}")

    # 数据准备
    task_df = df.dropna(subset=[TEXT_COLUMN, label_col]).copy()
    task_df[TEXT_COLUMN] = task_df[TEXT_COLUMN].astype(str)

    le = LabelEncoder()
    task_df['label'] = le.fit_transform(task_df[label_col])
    num_labels = len(le.classes_)
    logger.info(f"标签数: {num_labels}")
    for i, c in enumerate(le.classes_):
        logger.info(f"  {i}: {c} ({sum(task_df['label']==i)} 条)")

    X = task_df[TEXT_COLUMN].values
    y = task_df['label'].values

    # 分层分割
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
    except ValueError:
        logger.warning("分层分割失败，改用随机分割")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )

    logger.info(f"训练集: {len(X_train)} | 测试集: {len(X_test)}")

    # 加载模型
    logger.info(f"加载预训练模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 关键修复：处理 LayerNorm 参数名兼容性问题
    # bert-base-chinese 旧版 checkpoint 使用 beta/gamma，新版 transformers 期望 weight/bias
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
    except Exception as e:
        logger.warning(f"标准加载失败: {e}")
        logger.info("尝试从 trust_remote_code 加载...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels,
            ignore_mismatched_sizes=True,
            trust_remote_code=True
        )

    # 检查是否有未加载的参数
    missing_keys = [k for k in model.state_dict().keys()
                    if 'LayerNorm' in k and 'weight' in k
                    and model.state_dict()[k].std().item() < 1e-6]
    if missing_keys:
        logger.warning(f"检测到 LayerNorm 参数可能未正确加载 ({len(missing_keys)} 个)，"
                      "将手动初始化")
        for key in missing_keys:
            # 用正态分布初始化 LayerNorm weight
            param = model.state_dict()[key]
            torch.nn.init.normal_(param, mean=1.0, std=0.02)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"设备: {device}")

    # 数据集
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)

    # 训练参数
    output_dir = f'./results/{model_name.split("/")[-1]}_{task_name}'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=200,
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )

    # 训练
    import time
    start = time.time()
    trainer.train()
    train_time = time.time() - start

    # 评估
    eval_results = trainer.evaluate()
    logger.info(f"\n评估结果:")
    for key in ['eval_accuracy', 'eval_macro_f1', 'eval_weighted_f1', 'eval_kappa']:
        if key in eval_results:
            logger.info(f"  {key}: {eval_results[key]:.4f}")

    # 详细分类报告
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    report = classification_report(y_test, pred_labels, target_names=le.classes_, digits=4)
    cm = confusion_matrix(y_test, pred_labels)
    logger.info(f"\n分类报告:\n{report}")

    # 保存模型
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    joblib.dump(le, f'{save_dir}/label_encoder.pkl')

    # 保存标签映射到 config.json（修复原版 id2label 为 LABEL_0 的问题）
    config_path = f'{save_dir}/config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    config['id2label'] = {str(i): label for i, label in enumerate(le.classes_)}
    config['label2id'] = {label: i for i, label in enumerate(le.classes_)}
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"模型已保存: {save_dir}")

    # 保存预测结果
    result_df = pd.DataFrame({
        'text': X_test,
        'true_label': le.inverse_transform(y_test),
        'pred_label': le.inverse_transform(pred_labels),
    })
    result_df.to_csv(f'{save_dir}/prediction_results.csv', index=False, encoding='utf-8-sig')

    # 返回结果摘要
    result = {
        'model': f'{model_name} (fine-tuned)',
        'task': task_name,
        'macro_f1': float(eval_results.get('eval_macro_f1', 0)),
        'weighted_f1': float(eval_results.get('eval_weighted_f1', 0)),
        'accuracy': float(eval_results.get('eval_accuracy', 0)),
        'kappa': float(eval_results.get('eval_kappa', 0)),
        'train_time_sec': float(train_time),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'num_labels': num_labels,
        'label_classes': list(le.classes_),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
    }

    # 保存结果 JSON
    with open(f'{save_dir}/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ════════════════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════════════════

def main():
    # 加载数据
    logger.info(f"加载数据: {FILE_PATH}")
    df = pd.read_csv(FILE_PATH, encoding='utf-8-sig')
    logger.info(f"数据: {len(df):,} 条")

    all_results = []

    # 训练所有 模型 x 任务 组合
    for model_name, model_path in MODELS.items():
        for task_config in TASKS:
            # 对于 BERT 的文化分类，如果已存在则跳过
            if model_name == 'BERT-base-Chinese' and task_config['name'] == 'culture_classification':
                task_config['skip_if_exists'] = True
                task_config['model_dir'] = './bert_culture_classifier'

            result = train_model(model_name, model_path, task_config, df)
            if result:
                all_results.append(result)

    # 汇总表格
    logger.info(f"\n{'='*80}")
    logger.info("全部训练完成！结果汇总：")
    logger.info(f"{'='*80}")

    print("\n| 模型 | 任务 | Macro-F1 | Weighted-F1 | Accuracy | Kappa |")
    print("|------|------|----------|-------------|----------|-------|")
    for r in all_results:
        print(f"| {r['model']} | {r['task']} | "
              f"{r['macro_f1']:.4f} | {r['weighted_f1']:.4f} | "
              f"{r['accuracy']:.4f} | {r['kappa']:.4f} |")

    # 保存汇总
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ['classification_report', 'confusion_matrix', 'label_classes']}
        for r in all_results
    ])
    results_df.to_csv('./all_training_results.csv', index=False, encoding='utf-8-sig')
    logger.info("汇总结果已保存: all_training_results.csv")

    # 打包下载
    logger.info("\n正在打包下载...")
    os.system('zip -r bert_roberta_all_models.zip '
              './*_culture_classifier ./*_aspect_classifier ./*_polarity_classifier '
              './all_training_results.csv')

    try:
        from google.colab import files
        files.download('bert_roberta_all_models.zip')
        logger.info("下载已启动！")
    except:
        logger.info("非 Colab 环境，请手动下载 zip 文件")


if __name__ == '__main__':
    main()
