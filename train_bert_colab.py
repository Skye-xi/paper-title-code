# -*- coding: utf-8 -*-
"""
BERT-base-Chinese 训练脚本（Google Colab 版）
===============================================
用于训练文化类型四分类模型。

使用方式:
  1. 上传 首都文化_情感得分_抽样15%.csv 到 Colab
  2. 运行此脚本
  3. 自动下载 bert_culture_classifier 模型

注意: 本脚本设计在 Google Colab GPU 环境运行。
"""

from google.colab import files
uploaded = files.upload()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch
from torch.utils.data import Dataset
import logging
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 第一步：加载抽样数据
# ============================================================
file_path = r'首都文化_情感得分_抽样15%.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')
logger.info(f"数据加载完成：{len(df)} 条记录")

# ============================================================
# 第二步：选择任务 - 文化类型分类
# ============================================================
TEXT_COLUMN = 'cleaned_content'
LABEL_COLUMN = '文化类型'

# 检查列是否存在
if TEXT_COLUMN not in df.columns:
    raise ValueError(f"列 '{TEXT_COLUMN}' 不存在")
if LABEL_COLUMN not in df.columns:
    raise ValueError(f"列 '{LABEL_COLUMN}' 不存在")

# 删除空值
df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
logger.info(f"删除空值后：{len(df)} 条记录")

# 查看类别分布
logger.info("\n文化类型分布:")
culture_counts = df[LABEL_COLUMN].value_counts()
logger.info(culture_counts)

# ============================================================
# 第三步：编码标签
# ============================================================
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df[LABEL_COLUMN])
num_labels = len(label_encoder.classes_)
logger.info(f"\n类别数量: {num_labels}")
for i, label in enumerate(label_encoder.classes_):
    logger.info(f"  {i}: {label}")

# ============================================================
# 第四步：划分训练集和测试集
# ============================================================
X = df[TEXT_COLUMN].astype(str).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

logger.info(f"\n训练集: {len(X_train)} 条")
logger.info(f"测试集: {len(X_test)} 条")

# ============================================================
# 第五步：创建 PyTorch Dataset
# ============================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ============================================================
# 第六步：加载 BERT 模型
# ============================================================
model_name = 'bert-base-chinese'
logger.info(f"\n加载模型: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# 移动到设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
logger.info(f"使用设备: {device}")
if device.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# 准备数据集
train_dataset = TextDataset(X_train, y_train, tokenizer)
test_dataset = TextDataset(X_test, y_test, tokenizer)

# ============================================================
# 第七步：定义评估指标
# ============================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# ============================================================
# 第八步：配置训练参数（修复了 report_to 问题）
# ============================================================
training_args = TrainingArguments(
    output_dir='./bert_results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
)

# ============================================================
# 第九步：创建 Trainer 并开始训练
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

logger.info("\n" + "=" * 60)
logger.info("开始训练 BERT 文化类型分类模型...")
logger.info("=" * 60)

trainer.train()

# ============================================================
# 第十步：评估模型
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("评估模型...")
logger.info("=" * 60)

eval_results = trainer.evaluate()
logger.info(f"\n评估结果:")
for key, value in eval_results.items():
    logger.info(f"  {key}: {value:.4f}")

# 详细分类报告
predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
report = classification_report(y_test, pred_labels, target_names=label_encoder.classes_)
logger.info("\n分类报告:\n" + report)

# ============================================================
# 第十一步：保存模型
# ============================================================
model_save_path = './bert_culture_classifier'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

import joblib
joblib.dump(label_encoder, f'{model_save_path}/label_encoder.pkl')

logger.info(f"\n模型已保存到: {model_save_path}")

# ============================================================
# 第十二步：保存预测结果
# ============================================================
result_df = pd.DataFrame({
    'text': X_test,
    'true_label': label_encoder.inverse_transform(y_test),
    'pred_label': label_encoder.inverse_transform(pred_labels)
})
result_df.to_csv('./bert_prediction_results.csv', index=False, encoding='utf-8-sig')
logger.info("预测结果已保存到: bert_prediction_results.csv")

logger.info("\n" + "=" * 60)
logger.info("训练完成！")
logger.info("=" * 60)

# ============================================================
# 第十三步：自动下载结果到本地
# ============================================================
logger.info("\n正在打包并下载结果...")

# 打包所有结果
!zip -r bert_full_results.zip ./bert_culture_classifier ./bert_prediction_results.csv ./bert_results ./logs

# 下载到本地
from google.colab import files
files.download('bert_full_results.zip')

logger.info("结果已下载到本地！请解压 bert_full_results.zip 查看所有文件。")
