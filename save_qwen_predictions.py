# -*- coding: utf-8 -*-
"""
Qwen3 零样本逐条预测保存（用于 McNemar 检验）
=============================================
只跑文化类型判别任务，保存每个样本的预测标签，
以便与 BERT 预测做逐样本配对统计检验。

使用方式:
  conda activate openAI
  python save_qwen_predictions.py

前置条件:
  1. 需要运行 vLLM 服务，并将 VLLM_BASE_URL 指向正确的 API 端点
  2. 数据文件 首都文化_情感得分_抽样15%.csv 放在脚本同目录下
"""

import os
import sys
import time
import json
import logging
import asyncio
import warnings
import numpy as np
import pandas as pd
import aiohttp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'baseline_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'qwen_pred_save.log'), encoding='utf-8'),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# ============================================================
# 配置 (请根据实际环境修改以下两项)
# ============================================================

# 数据文件路径
CSV_PATH = os.path.join(_SCRIPT_DIR, '首都文化_情感得分_抽样15%.csv')

# vLLM API 端点 — 请替换为你的实际地址
# 示例: http://localhost:8000/v1 (本地 vLLM)
# 示例: http://10.0.0.1:8000/v1  (内网 vLLM)
VLLM_BASE_URL = '<YOUR_VLLM_API_ENDPOINT>'

# API Key — 如果使用 vLLM 默认不需要认证，可设置为 'EMPTY'
# 如果使用其他 LLM 服务 (如 OpenAI, DeepSeek)，请替换为你的实际 API Key
VLLM_API_KEY = '<YOUR_VLLM_API_KEY>'

# ============================================================

SEED = 42
TEST_SIZE = 0.2
QWEN_CONCURRENCY = 30
CULTURE_TYPES = ['古都文化', '红色文化', '京味文化', '创新文化']

PROMPT = """你是文化文本分类助手。请判断微博文本属于以下哪一类首都文化：
古都文化、红色文化、京味文化、创新文化。
仅输出4个标签之一，不要输出其他内容。"""


def load_data():
    logger.info(f"加载: {CSV_PATH}")
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

    logger.info(f"测试集: {len(X_test):,} 条")
    return {
        'X_test': X_test, 'y_test': y_test,
        'label_encoder': le, 'idx_test': idx_test,
    }


async def predict_all(texts, model_name):
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
                    {'role': 'system', 'content': PROMPT},
                    {'role': 'user', 'content': f'文本：{text}\n请只输出分类标签。'},
                ],
                'temperature': 0.0,
                'max_tokens': 20,
            }
            headers = {'Authorization': f'Bearer {VLLM_API_KEY}'}
            try:
                async with session.post(
                    f'{VLLM_BASE_URL}/chat/completions',
                    json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    r = await resp.json()
                    pred = r['choices'][0]['message']['content'].strip()
                    matched = next((l for l in CULTURE_TYPES if l in pred), CULTURE_TYPES[0])
                    results[idx] = matched
            except Exception as e:
                logger.error(f"idx={idx}: {e}")
                results[idx] = CULTURE_TYPES[0]

            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                logger.info(f"  {done}/{len(texts)} ({done/elapsed:.1f}/s)")

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[fetch(session, i, t) for i, t in enumerate(texts)])

    elapsed = time.time() - t0
    logger.info(f"完成: {elapsed:.0f}s")
    return results, elapsed


async def main():
    # 获取模型名
    import requests
    try:
        r = requests.get(f'{VLLM_BASE_URL}/models', timeout=5)
        model_name = r.json()['data'][0]['id']
        logger.info(f"模型: {model_name}")
    except Exception as e:
        logger.error(f"API不可达: {e}")
        return

    data = load_data()

    # 推理
    le = data['label_encoder']
    texts = data['X_test'].tolist()
    true_labels = le.inverse_transform(data['y_test']).tolist()

    logger.info(f"开始推理 {len(texts):,} 条...")
    preds, infer_time = await predict_all(texts, model_name)

    # 保存逐条预测
    pred_df = pd.DataFrame({
        'idx': data['idx_test'],
        'text': texts,
        'true_label': true_labels,
        'pred_label': preds,
        'correct': [t == p for t, p in zip(true_labels, preds)],
    })
    pred_path = os.path.join(OUTPUT_DIR, 'qwen_culture_predictions.csv')
    pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
    logger.info(f"预测已保存: {pred_path} ({len(pred_df)} 条)")
    logger.info(f"Accuracy: {(pred_df['correct'].sum()/len(pred_df)):.4f}")


if __name__ == '__main__':
    asyncio.run(main())
