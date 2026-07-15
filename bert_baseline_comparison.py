# -*- coding: utf-8 -*-
"""
LLM vs Fine-tuned Model Baseline Comparison Experiment (for Reviewer Comment 3) - Sampling Version
===================================================================================================

Features:
  - Load data from database, sample proportionally (default 20%)
  - Stratified sampling to ensure class balance
  - Fine-tune BERT-base-Chinese and RoBERTa-wwm-ext on 8:2 train/test split
  - Three tasks: culture type classification / evaluation aspect identification / sentiment polarity judgment
  - Compute Macro-F1 / Accuracy / Cohen's Kappa
  - Compare with Qwen3-32B zero-shot results, generate paper tables

Dependencies:
  pip install torch transformers scikit-learn pandas psycopg2-binary tqdm
"""

import os
import json
import logging
import random
import time
import numpy as np
import pandas as pd
import psycopg2
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm
import aiohttp
import asyncio

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Fixed random seed (ensure reproducibility)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ════════════════════════════════════════════════════════════════════════════
# 1. Configuration
# ════════════════════════════════════════════════════════════════════════════

DB_CONFIG = {
    "host": "<YOUR_DB_HOST>",
    "port": 5432,
    "user": "<YOUR_DB_USER>",
    "password": "<YOUR_DB_PASSWORD>",
    "database": "<YOUR_DB_NAME>",
}

# Final results table (including culture type + carrier + aspect + polarity + spatial classification)
DB_TABLE = "shihao.bj2019_culture_1_10_with_response_首都文化分类_全"

# Qwen3-32B configuration (vLLM API, for zero-shot baseline comparison)
LLM_CONFIG = {
    "base_url": "<YOUR_VLLM_SERVER_IP>:8000/v1",
    "api_key": "<YOUR_VLLM_API_KEY>",
    "model": "Qwen3-32B",
    "temperature": 0.0,
    "max_tokens": 10,
}

# Fallback CSV path (if database is unreachable)
CSV_FALLBACK = "<YOUR_DATA_DIR>/verification_samples_2000.csv"

# ==================== [Sampling Configuration] ====================
SAMPLE_CONFIG = {
    "enable_sampling": True,  # True=enable sampling, False=use full data
    "sample_ratio": 0.2,  # Sampling ratio (0.2 = 20%)
    "min_samples_per_class": 50,  # Minimum samples per class after sampling
    "max_samples_per_class": 2000,  # Maximum samples per class (prevent oversized classes)
}
# ============================================================

# Experiment configuration
EXPERIMENT_CONFIG = {
    "n_samples_per_culture": 500,  # Per culture type sample count (effective when enable_sampling=False)
    "train_ratio": 0.8,  # Training set ratio
    "max_length": 256,  # Maximum text length
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "weight_decay": 0.01,
    "early_stopping_patience": 2,
    "output_dir": "./baseline_results",
}

# Label configuration
CULTURE_TYPES = ["古都文化", "红色文化", "京味文化", "创新文化"]
ASPECTS_17 = [
    "交通便利", "人文景观", "人流量", "体力消耗", "公共设施", "历史认知",
    "商业环境", "天气气候", "建筑美学", "情感共鸣", "文化体验", "文化内涵",
    "文化氛围", "文化遗产", "游客服务", "自然景观", "饮食体验",
]
POLARITIES = ["积极", "中立", "消极"]

# Baseline models
BASELINE_MODELS = {
    "BERT-base-Chinese": "bert-base-chinese",
    "RoBERTa-wwm-ext": "hfl/chinese-roberta-wwm-ext",
}


# ════════════════════════════════════════════════════════════════════════════
# 2. Data Preparation (with Sampling)
# ════════════════════════════════════════════════════════════════════════════

def _try_db_load() -> pd.DataFrame:
    """
    Load data from database.
    Actual table structure includes a single column `文化类型`, evaluation aspect, polarity.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    query = f"""
        SELECT cleaned_content,
               "文化类型",
               "评价方面",
               "情感"
        FROM {DB_TABLE}
        WHERE "情感" IS NOT NULL
          AND "评价方面" IS NOT NULL
          AND "文化类型" IS NOT NULL
          AND "文化类型" IN ('古都文化', '京味文化', '创新文化', '红色文化')
    """
    df = pd.read_sql(query, conn)
    conn.close()
    logger.info(f"Database loading completed: {len(df):,} valid records")

    df = df.rename(columns={
        "cleaned_content": "text",
        "文化类型": "culture",
        "评价方面": "aspect",
        "情感": "polarity",
    })
    return df[["text", "culture", "aspect", "polarity"]]


def _try_csv_load() -> pd.DataFrame:
    """Load data from CSV fallback file."""
    logger.info(f"Loading from CSV: {CSV_FALLBACK}")
    df = pd.read_csv(CSV_FALLBACK, encoding="utf-8-sig")

    # Auto-detect column name mapping
    text_col = None
    culture_col = None
    aspect_col = None
    polarity_col = None

    for c in df.columns:
        if "cleaned_content" in c or "text" in c.lower() or "内容" in c:
            text_col = c
        elif "culture" in c.lower() or "文化类型" in c or "文化" in c:
            culture_col = c
        elif "aspect" in c or "评价方面" in c or "方面" in c:
            aspect_col = c
        elif "polarity" in c or "情感" in c:
            polarity_col = c

    if not all([text_col, culture_col, aspect_col, polarity_col]):
        logger.error(f"CSV column names do not match. Available columns: {list(df.columns)}")
        raise ValueError("Cannot auto-identify CSV column names, please check CSV format")

    df = df.rename(columns={
        text_col: "text",
        culture_col: "culture",
        aspect_col: "aspect",
        polarity_col: "polarity",
    })
    df["text"] = df["text"].astype(str)
    df = df.dropna(subset=["text", "culture", "aspect", "polarity"])
    logger.info(f"CSV loading completed: {len(df):,} valid records")
    return df[["text", "culture", "aspect", "polarity"]]


def apply_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sampling to data (stratified sampling).
    Supports two modes:
    1. Proportional sampling (sample_ratio)
    2. Fixed samples per class (n_samples_per_culture)
    """
    if not SAMPLE_CONFIG["enable_sampling"]:
        logger.info("Sampling disabled, using full data")
        return df

    logger.info(f"\n{'=' * 60}")
    logger.info("Starting data sampling...")
    logger.info(f"Original data volume: {len(df):,} records")
    logger.info(f"Sampling ratio: {SAMPLE_CONFIG['sample_ratio'] * 100:.1f}%")

    # Count original volume per class
    culture_counts = df["culture"].value_counts()
    logger.info(f"Original count per class:\n{culture_counts}")

    # Calculate per-class sample count (proportional)
    sampled_dfs = []
    for culture, group in df.groupby("culture"):
        n_original = len(group)
        n_sample = max(
            int(n_original * SAMPLE_CONFIG["sample_ratio"]),
            SAMPLE_CONFIG["min_samples_per_class"]  # Ensure minimum sample count
        )
        # Do not exceed maximum sample count
        n_sample = min(n_sample, SAMPLE_CONFIG["max_samples_per_class"])
        # Cannot exceed actual count
        n_sample = min(n_sample, n_original)

        sampled_group = group.sample(n=n_sample, random_state=SEED)
        sampled_dfs.append(sampled_group)
        logger.info(f"  {culture}: {n_original} -> {n_sample} records ({n_sample / n_original * 100:.1f}%)")

    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    logger.info(f"\nTotal data after sampling: {len(sampled_df):,} records")
    logger.info(f"Per-class count after sampling:\n{sampled_df['culture'].value_counts()}")
    logger.info("=" * 60)

    return sampled_df


def load_and_sample_data() -> pd.DataFrame:
    """
    Load data (prefer database, fallback to CSV on failure),
    then apply sampling according to SAMPLE_CONFIG.
    """
    logger.info("Loading data ...")

    # Try database loading
    try:
        df = _try_db_load()
    except Exception as e:
        logger.warning(f"Database loading failed: {e}")
        logger.info("Falling back to CSV backup data source ...")
        df = _try_csv_load()

    # Apply sampling
    df = apply_sampling(df)

    # Filter out classes with too few samples after sampling
    vc = df["culture"].value_counts()
    min_samples = SAMPLE_CONFIG["min_samples_per_class"]
    valid_cultures = vc[vc >= min_samples].index.tolist()
    df = df[df["culture"].isin(valid_cultures)]

    logger.info(f"Final valid data: {len(df):,} records, classes: {valid_cultures}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 3. PyTorch Dataset
# ════════════════════════════════════════════════════════════════════════════

class TextClassificationDataset(Dataset):
    """General text classification dataset."""

    def __init__(self, texts, labels, tokenizer, max_length, label2id):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.label2id[self.labels[idx]]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ════════════════════════════════════════════════════════════════════════════
# 4. Fine-tune BERT / RoBERTa
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(eval_pred):
    """Evaluation function for Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "kappa": cohen_kappa_score(labels, preds),
    }


def finetune_baseline(model_name: str, model_path: str,
                      train_texts, train_labels,
                      test_texts, test_labels,
                      label_list, task_name: str) -> dict:
    """
    Fine-tune a single baseline model.
    Returns dict: {macro_f1, accuracy, kappa, train_time_sec, n_train, n_test}
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Task: {task_name} | Model: {model_name}")
    logger.info(f"Training samples: {len(train_texts)} | Test samples: {len(test_texts)}")
    logger.info(f"{'=' * 60}")

    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer,
        EXPERIMENT_CONFIG["max_length"], label2id
    )
    test_dataset = TextClassificationDataset(
        test_texts, test_labels, tokenizer,
        EXPERIMENT_CONFIG["max_length"], label2id
    )

    output_dir = os.path.join(
        EXPERIMENT_CONFIG["output_dir"], task_name, model_name.replace("/", "_")
    )
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EXPERIMENT_CONFIG["num_epochs"],
        per_device_train_batch_size=EXPERIMENT_CONFIG["batch_size"],
        per_device_eval_batch_size=32,
        learning_rate=EXPERIMENT_CONFIG["learning_rate"],
        weight_decay=EXPERIMENT_CONFIG["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=EXPERIMENT_CONFIG["early_stopping_patience"]
        )],
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    # Evaluate on test set
    eval_results = trainer.evaluate(test_dataset)

    return {
        "model": model_name,
        "task": task_name,
        "macro_f1": eval_results["eval_macro_f1"],
        "accuracy": eval_results["eval_accuracy"],
        "kappa": eval_results["eval_kappa"],
        "train_time_sec": train_time,
        "n_train": len(train_dataset),
        "n_test": len(test_dataset),
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. Qwen3-32B Zero-Shot Inference
# ════════════════════════════════════════════════════════════════════════════

async def qwen_zero_shot_predict(texts: list, prompt_template: str,
                                 label_list: list) -> list:
    """Qwen3-32B zero-shot inference, returns predicted label list."""
    semaphore = asyncio.Semaphore(20)
    predictions = []

    async def fetch_one(session, text, idx):
        async with semaphore:
            payload = {
                "model": LLM_CONFIG["model"],
                "messages": [
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": f"Text: {text}\nPlease output classification result."},
                ],
                "temperature": 0.0,
                "max_tokens": 10,
            }
            headers = {"Authorization": f"Bearer {LLM_CONFIG['api_key']}"}
            url = f"{LLM_CONFIG['base_url']}/chat/completions"
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    result = await resp.json()
                    pred_text = result["choices"][0]["message"]["content"].strip()
                    # Match nearest label
                    matched = None
                    for label in label_list:
                        if label in pred_text:
                            matched = label
                            break
                    if matched is None:
                        matched = label_list[0]  # Default
                    predictions.append((idx, matched))
            except Exception as e:
                logger.error(f"Inference failed idx={idx}: {e}")
                predictions.append((idx, label_list[0]))

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, t, i) for i, t in enumerate(texts)]
        await asyncio.gather(*tasks)

    # Sort by idx
    predictions.sort(key=lambda x: x[0])
    return [p[1] for p in predictions]


def evaluate_qwen_zero_shot(texts, true_labels, label_list, task_name: str,
                            prompt_template: str) -> dict:
    """Evaluate Qwen3-32B zero-shot performance."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Task: {task_name} | Model: Qwen3-32B (zero-shot)")
    logger.info(f"Test samples: {len(texts)}")
    logger.info(f"{'=' * 60}")

    label2id = {l: i for i, l in enumerate(label_list)}
    start = time.time()
    pred_labels = asyncio.run(
        qwen_zero_shot_predict(texts, prompt_template, label_list)
    )
    infer_time = time.time() - start

    y_true = [label2id[l] for l in true_labels]
    y_pred = [label2id[l] for l in pred_labels]

    return {
        "model": "Qwen3-32B (zero-shot)",
        "task": task_name,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "infer_time_sec": infer_time,
        "n_train": 0,
        "n_test": len(texts),
    }


# ════════════════════════════════════════════════════════════════════════════
# 6. Prompt Templates
# ════════════════════════════════════════════════════════════════════════════
# NOTE: The prompt templates below are simplified versions for baseline
# comparison only. The full prompts for production use are provided in
# paper Appendix A. These simplified prompts are sufficient for the
# zero-shot baseline evaluation purpose.

PROMPT_CULTURE = """You are a cultural text classification assistant. Please determine which of the
following four capital culture categories the Weibo text belongs to:
古都文化, 红色文化, 京味文化, 创新文化.
Judgment is based on the core definitions of each culture type. Output only one of the 4 labels."""

PROMPT_ASPECT = """You are an ABSA analysis assistant. Identify the user's evaluation aspect regarding
the material culture carrier from the Weibo text. There are 17 categories:
交通便利, 人文景观, 人流量, 体力消耗, 公共设施, 历史认知, 商业环境,
天气气候, 建筑美学, 情感共鸣, 文化体验, 文化内涵, 文化氛围, 文化遗产, 游客服务,
自然景观, 饮食体验. Output only the most relevant aspect label."""

PROMPT_POLARITY = """You are a sentiment classification assistant. Determine the sentiment polarity of the
Weibo text, output only one of the following three categories:
积极, 中立, 消极."""


# ════════════════════════════════════════════════════════════════════════════
# 7. Main Function
# ════════════════════════════════════════════════════════════════════════════

def _qwen_available() -> bool:
    """Check if Qwen API is reachable."""
    try:
        import urllib.request
        url = f"{LLM_CONFIG['base_url']}/models"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {LLM_CONFIG['api_key']}")
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False


def _dynamic_labels(data_labels: list, full_labels: list) -> list:
    """Extract actually appearing labels from data, ordered by full_labels."""
    seen = set(data_labels)
    return [l for l in full_labels if l in seen]


def main():
    os.makedirs(EXPERIMENT_CONFIG["output_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cpu":
        logger.warning("No GPU detected, BERT fine-tuning will be very slow. Recommend running on a GPU server.")

    # Print sampling configuration
    logger.info(f"\n{'=' * 60}")
    logger.info("Sampling configuration:")
    logger.info(f"  Sampling enabled: {SAMPLE_CONFIG['enable_sampling']}")
    if SAMPLE_CONFIG['enable_sampling']:
        logger.info(f"  Sampling ratio: {SAMPLE_CONFIG['sample_ratio'] * 100:.1f}%")
        logger.info(f"  Min samples per class: {SAMPLE_CONFIG['min_samples_per_class']}")
        logger.info(f"  Max samples per class: {SAMPLE_CONFIG['max_samples_per_class']}")
    else:
        logger.info("  Using full data (500 records per class)")
    logger.info("=" * 60)

    qwen_ok = _qwen_available()
    if not qwen_ok:
        logger.warning("Qwen API is unreachable, will skip Qwen zero-shot evaluation")

    # Load data
    df = load_and_sample_data()
    texts = df["text"].tolist()

    # Dynamically get actually appearing labels from data
    culture_labels = _dynamic_labels(df["culture"].tolist(), CULTURE_TYPES)
    aspect_labels = _dynamic_labels(df["aspect"].tolist(), ASPECTS_17)
    polarity_labels = _dynamic_labels(df["polarity"].tolist(), POLARITIES)
    logger.info(f"Actual labels in data: culture={culture_labels}, "
                f"aspect={len(aspect_labels)}/17 types, polarity={polarity_labels}")

    all_results = []

    # ── Task 1: Culture Type Classification ─────────────────────────────
    task_name = "culture_classification"
    train_t, test_t, train_y, test_y = train_test_split(
        texts, df["culture"].tolist(),
        test_size=1 - EXPERIMENT_CONFIG["train_ratio"],
        random_state=SEED, stratify=df["culture"]
    )

    for model_name, model_path in BASELINE_MODELS.items():
        r = finetune_baseline(
            model_name, model_path,
            train_t, train_y, test_t, test_y,
            culture_labels, task_name
        )
        all_results.append(r)

    if qwen_ok:
        r = evaluate_qwen_zero_shot(
            test_t, test_y, culture_labels, task_name, PROMPT_CULTURE
        )
        all_results.append(r)

    # ── Task 2: Evaluation Aspect Identification ─────────────────────────────
    task_name = "aspect_classification"
    # Stratification may fail when some classes have <2 records with small samples; fallback to non-stratified split
    try:
        train_t, test_t, train_y, test_y = train_test_split(
            texts, df["aspect"].tolist(),
            test_size=1 - EXPERIMENT_CONFIG["train_ratio"],
            random_state=SEED, stratify=df["aspect"]
        )
    except ValueError:
        logger.warning("Aspect class distribution is uneven, using non-stratified split")
        train_t, test_t, train_y, test_y = train_test_split(
            texts, df["aspect"].tolist(),
            test_size=1 - EXPERIMENT_CONFIG["train_ratio"],
            random_state=SEED
        )

    for model_name, model_path in BASELINE_MODELS.items():
        r = finetune_baseline(
            model_name, model_path,
            train_t, train_y, test_t, test_y,
            aspect_labels, task_name
        )
        all_results.append(r)

    if qwen_ok:
        r = evaluate_qwen_zero_shot(
            test_t, test_y, aspect_labels, task_name, PROMPT_ASPECT
        )
        all_results.append(r)

    # ── Task 3: Sentiment Polarity Judgment ─────────────────────────────
    task_name = "polarity_classification"
    try:
        train_t, test_t, train_y, test_y = train_test_split(
            texts, df["polarity"].tolist(),
            test_size=1 - EXPERIMENT_CONFIG["train_ratio"],
            random_state=SEED, stratify=df["polarity"]
        )
    except ValueError:
        train_t, test_t, train_y, test_y = train_test_split(
            texts, df["polarity"].tolist(),
            test_size=1 - EXPERIMENT_CONFIG["train_ratio"],
            random_state=SEED
        )

    for model_name, model_path in BASELINE_MODELS.items():
        r = finetune_baseline(
            model_name, model_path,
            train_t, train_y, test_t, test_y,
            polarity_labels, task_name
        )
        all_results.append(r)

    if qwen_ok:
        r = evaluate_qwen_zero_shot(
            test_t, test_y, polarity_labels, task_name, PROMPT_POLARITY
        )
        all_results.append(r)

    # ── Summary Output ──────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(EXPERIMENT_CONFIG["output_dir"], "baseline_comparison.csv")
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Results saved to: {csv_path}")

    # Generate paper table
    print("\n" + "=" * 70)
    print("Baseline Comparison Experiment Results" +
          (f" (Sampling {SAMPLE_CONFIG['sample_ratio'] * 100:.1f}%)" if SAMPLE_CONFIG[
              'enable_sampling'] else " (Full data)"))
    print("=" * 70)

    for task in ["culture_classification", "aspect_classification",
                 "polarity_classification"]:
        task_df = results_df[results_df["task"] == task]
        if task_df.empty:
            continue
        print(f"\n## {task}")
        print("| Model | Parameters | Training Data | Macro-F1 | Accuracy | Kappa |")
        print("|------|-------|---------|----------|----------|-------|")
        for _, row in task_df.iterrows():
            model_lower = row["model"].lower()
            if "bert-base" in model_lower:
                params = "110M"
            elif "roberta" in model_lower:
                params = "102M"
            else:
                params = "32B"
            n_train = row["n_train"]
            train_str = f"{n_train} rec" if n_train > 0 else "0 rec (zero-shot)"
            print(f"| {row['model']} | {params} | {train_str} | "
                  f"{row['macro_f1']:.4f} | {row['accuracy']:.4f} | "
                  f"{row['kappa']:.4f} |")

    # Save sampling info to results
    info_path = os.path.join(EXPERIMENT_CONFIG["output_dir"], "sampling_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump({
            "sampling_config": SAMPLE_CONFIG,
            "total_samples": len(df),
            "culture_distribution": df["culture"].value_counts().to_dict(),
            "aspect_distribution": df["aspect"].value_counts().to_dict(),
            "polarity_distribution": df["polarity"].value_counts().to_dict(),
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Sampling info saved to: {info_path}")


if __name__ == "__main__":
    main()
