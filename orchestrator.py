# -*- coding: utf-8 -*-
"""
================================================================================
  Four-Agent Closed-Loop Pipeline Orchestrator
                    -- Pseudocode for Paper --
================================================================================

  IMPORTANT NOTICE
  ────────────────────────────────────────────────────────────────────────────
  This file is a pseudocode / illustrative skeleton for the paper's methodology
  section, not runnable production code or a complete reproducible experiment
  environment. Its purposes are:
    (1) To present the four-agent workflow's overall structure, call order,
        data flow, and closed-loop feedback mechanism to reviewers and readers
        in code form within paper Section 2.2 / Algorithm 1;
    (2) To provide conceptual descriptions of each Agent's input/output schema,
        key fields, prompt and LLM invocation method, corroborating the
        methodology description in the main text;
    (3) To direct readers to the real experiment code files in this directory
        (1.xxx.py, 2.xxx.py, 3.xxx.py, 4_xxx.py) for the full runnable
        implementation.

  Known Gaps vs. Real Implementation
  ────────────────────────────────────────────────────────────────────────────
  For brevity in paper presentation, this orchestrator makes the following
  simplifications/abstractions compared to the real experiment:
    1. Table/field names are illustrative: the real experiment splits data in
       PostgreSQL by "culture type x batch" (e.g., bj2019_culture_part5_with_
       response_01红色已对齐) into multiple tables; this orchestrator
       represents them as a single logical table name, and actual SQL JOINs
       are far more complex;
    2. Agent 3.3 / 3.4 / 3.5 function bodies are pseudocode skeletons
       (showing only logical structure); full implementations are in
       shiyan/code/ directory: 05地点包含关系.py, 07判断是否文保单位.py,
       载体分类汇总.py;
    3. Agent 2.2 (sentiment score computation) provides key formulas for
       regex parsing and 17-dimension weighted summation in comments; full
       implementation in 情感得分计算(实验).py and 统一元组内容.py;
    4. Agent 4's five-check details are in 4_自反思修正智能体.py in this
       directory;
    5. Some agents in real code use Qwen1.5-72B (e.g., Agent 3.1 early
       version), others use Qwen3-32B; this orchestrator uniformly uses
       Qwen3-32B;
    6. Real code has interactive input() branches for scenarios like column
       already exists; this orchestrator omits them;
    7. Database connections, LLM service addresses, API keys, and other
       sensitive information are represented by <YOUR_*> placeholders;
       actual values must be substituted for real execution.

  Non-Executable
  ────────────────────────────────────────────────────────────────────────────
  This file cannot be run directly with `python orchestrator.py`, because:
    - Some function bodies are logger.info placeholders, without real SQL/LLM
      invocations;
    - Database table/field names do not fully match the real environment;
    - Prompt file paths need user configuration;
    - Agent 3.3/3.4/3.5 real implementations are missing.
  To reproduce the experiment, follow the README.md in this directory or
  paper Appendix A to sequentially run each Agent's independent script
  (1.xxx.py -> 2.xxx.py -> 3.xxx.py -> 4_xxx.py).

================================================================================

[Pseudocode Structure Overview]

  ┌─────────────────────────────────────────────────────────────┐
  │ Source Weibo table: bj2019_cleaned_part9_with_response      │
  │   Fields: id, tid, cleaned_content, 广告移除                 │
  └──────────────────────┬──────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────┐
  │ Agent 1: Capital Culture Classification (4 types x 1 round) │
  │ LLM: max_tokens=1, Prompt: 首都文化分类{type}_3.txt          │
  │ Output: ★ New columns 古都文化/红色文化/京味文化/创新文化      │
  └──────────────────────┬──────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────┐
  │ Agent 3.0: Material Culture Carrier Extraction               │
  │ LLM: Qwen3-32B, Prompt: 红色文化分类&载体提取.txt             │
  │ Input: tid + cleaned_content + culture type label='1'        │
  │ Output: ★ New column {culture_type}_物质文化载体 (TEXT)      │
  └──────────────────────┬──────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────┐
  │ Agent 2.1: Aspect-level Sentiment Analysis (ABSA) x 4 types │
  │ LLM: Qwen3-32B, Prompt: 物质文化载体ABSA3.txt                │
  │ Input: cleaned_content + evaluation carrier                  │
  │ Output: Quadruple ("载体","方面","评价","情感")               │
  └──────────────────────┬──────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────┐
  │ Agent 2.2-Postprocessing: Tuple Standardization + Sentiment │
  │   Score Calculation                                          │
  │  2.2a: Regex extraction of quadruples (统一元组内容.py)       │
  │  2.2b: 17-dimension weighted sentiment score (情感得分计算.py)│
  │ Formula: Final sentiment score = Σ(score_aspect_i × weight_i)│
  └──────────────────────┬──────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────┐
  │ Agent 3.1: Place vs. Non-place Classification                │
  │ LLM: Qwen3-32B, Prompt: 00判别地点（实验）.txt               │
  │ Input: carrier name -> Output: "1"(place)/"0"(non-place)     │
  │ Output column: 区分 (TEXT)                                   │
  └──────────────────────┬──────────────────────────────────────┘
                         │ WHERE 区分='1'
  ┌──────────────────────▼──────────────────────────────────────┐
  │ Agent 3.2: Toponym Normalization & Alignment (hybrid)        │
  │  Rule layer: High-frequency mapping table (≥2 occurrences)   │
  │              + Standard mapping (紫禁城->故宫)                │
  │  LLM layer: Qwen3-32B fallback, Prompt: 00对齐（实验）.txt   │
  │ Output column: 对齐 (TEXT)                                   │
  └──────────────────────┬──────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────┐
  │ Agent 3.3: Place Containment [Pseudocode, see 05地点包含]    │
  │ LLM: Determine if carrier is a sub-place within a larger     │
  │      place                                                   │
  │ Output column: 是否小地点 (TEXT)                              │
  └──────────────────────┬──────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────┐
  │ Agent 3.4: Heritage Site Determination [Pseudocode, 07判断] │
  │ LLM: Determine if carrier is a heritage site / fine-grained  │
  │      protection level classification                         │
  │ Output column: 是否文保单位 (TEXT)                            │
  └──────────────────────┬──────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────┐
  │ Agent 3.5: Spatial Type 5-Classification (hybrid: rule+LLM) │
  │  Rules: Keyword matching (故宫->royal history, 胡同->trad.)  │
  │  LLM: Qwen3-32B fallback                                    │
  │  5 types: ①Royal historical heritage ②Traditional residential│
  │           ③Public culture & performing arts                  │
  │           ④Political symbolic red culture                    │
  │           ⑤Modern commercial & urban leisure                 │
  │ Output column: 载体分类 = INTEGER (1-5)                      │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                         │ ★ At this point all records converge into
                         │   a "wide table":
                         │   bj2019_culture_with_absa_aligned
                         │   Fields: id, tid, cleaned_content,
                         │         culture_type, carrier, carrier_list(JSON),
                         │         absa_results(JSON), 情感得分,
                         │         spatial_type (1-5), 是否小地点, 是否文保单位,
                         │         all_culture_types(JSON)
                         │
  ┌──────────────────────▼──────────────────────────────────────┐
  │ Agent 4: Self-Reflection & Correction (closed-loop core)    │
  │   [See 4_自反思修正智能体.py]                                 │
  │   Five checks: A(LLM) B(rule table) C(string dedup)         │
  │                D(LLM) E(rule table)                          │
  │   Check failure -> Auto re-invoke upstream Agent -> Recheck │
  │   max_retry=2, HITL sampling of 500 records every 5 batches│
  │   Output: bj2019_culture_agent4_verified                    │
  │   HITL: bj2019_agent4_hitl_samples                          │
  └─────────────────────────────────────────────────────────────┘
================================================================================
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        GLOBAL CONFIGURATION                                  ║
# ║       (All constants below are placeholder/illustrative values              ║
# ║        for paper presentation, not actual configuration)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# This orchestrator is pseudocode for paper presentation.
# For real execution, use each Agent's independent script in the same directory.
# The imports and configuration below are only for illustrating code structure
# and are not guaranteed to be directly runnable.

import asyncio
import aiohttp
import asyncpg
import json
import logging
import time
import os
from typing import Any, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orchestrator")

# ── Database Connection ──────────────────────────────────────────────────────
DB = {
    "host": "<YOUR_DB_HOST>",
    "port": 5432,
    "user": "<YOUR_DB_USER>",
    "password": "<YOUR_DB_PASSWORD>",
    "database": "<YOUR_DB_NAME>",
}

# ── LLM Server ──────────────────────────────────────────────────────────────
LLM = {
    "base_url": "<YOUR_LLM_BASE_URL>",
    "api_key": "<YOUR_API_KEY>",
    "model": "<YOUR_MODEL_NAME>",
    "temperature": 0.0,
}

# ── Culture Types ────────────────────────────────────────────────────────────
CULTURE_TYPES = ["古都文化", "红色文化", "京味文化", "创新文化"]

# ── Prompt File Paths (modify according to actual location) ──────────────────
PROMPT_DIR = r"<YOUR_PROMPT_DIR>"
PROMPTS = {
    "agent1": {
        "古都文化": os.path.join(PROMPT_DIR, "首都文化分类", "古都文化_3.txt"),
        "红色文化": os.path.join(PROMPT_DIR, "首都文化分类", "红色文化_3.txt"),
        "京味文化": os.path.join(PROMPT_DIR, "首都文化分类", "京味文化_3.txt"),
        "创新文化": os.path.join(PROMPT_DIR, "首都文化分类", "创新文化_3.txt"),
    },
    "agent3_0_carrier":  os.path.join(PROMPT_DIR, "首都文化分类", "红色文化分类&载体提取.txt"),
    "agent2_1_absa":     os.path.join(PROMPT_DIR, "物质文化载体ABSA3.txt"),
    "agent3_1_place":    os.path.join(PROMPT_DIR, "00判别地点（实验）.txt"),
    "agent3_2_align":    os.path.join(PROMPT_DIR, "00对齐（实验）.txt"),
    "agent3_3_contain":  os.path.join(PROMPT_DIR, "05地点与地点内包含关系.txt"),
    "agent3_4_heritage": os.path.join(PROMPT_DIR, "07判断载体文保单位.txt"),
}

# ── Table Names ──────────────────────────────────────────────────────────────
TABLE = {
    "source":       "bj2019_cleaned_part9_with_response",           # Source Weibo table
    "agent1_out":   "bj2019_culture_agent1_classified",            # Agent 1 output
    "agent3_0_out": "bj2019_culture_agent3_0_carriers",           # Agent 3.0 output
    "agent2_1_out": "bj2019_culture_agent2_1_absa",               # Agent 2.1 output
    "agent2_2_out": "bj2019_culture_agent2_2_sentiment",           # Agent 2.2 output
    "merged_wide":  "bj2019_culture_with_absa_aligned",            # Wide table (Agent 4 input)
    "agent4_out":   "bj2019_culture_agent4_verified",              # Agent 4 output
    "agent4_hitl":  "bj2019_agent4_hitl_samples",                  # HITL sampling table
    "corrections":  "bj2019_culture_agent4_corrections",           # Records pending correction
}

# ── 17-Dimension Sentiment Weights (embedded, same as 情感得分计算.py) ────────
ASPECT_WEIGHTS = {
    "交通便利": 0.0641, "人文景观": 0.0662, "人流量": 0.0528,
    "体力消耗": 0.0549, "公共设施": 0.0592, "历史认知": 0.0603,
    "商业环境": 0.0557, "天气气候": 0.0514, "建筑美学": 0.0625,
    "情感共鸣": 0.0586, "文化体验": 0.0573, "文化内涵": 0.0618,
    "文化氛围": 0.0597, "文化遗产": 0.0634, "游客服务": 0.0539,
    "自然景观": 0.0546, "饮食体验": 0.0585,
}
EMOTION_SCORE = {"积极": 1, "中立": 0, "消极": -1}

# ── Spatial Types (same as 载体分类汇总.py) ─────────────────────────────────
SPATIAL_TYPES = [
    "皇室历史文化遗产空间",
    "传统居住与历史街区空间",
    "公共文化展示与演艺空间",
    "政治象征与红色文化空间",
    "现代商业与都市休闲文化空间",
]

# ── Spatial Type Keyword Rules (same as 载体分类汇总.py) ────────────────────
SPATIAL_KEYWORD_RULES = {
    1: ["故宫", "长城", "天坛", "颐和园", "十三陵", "皇陵", "王府", "庙"],  # Royal historical
    2: ["胡同", "四合院", "南锣鼓巷", "什刹海", "大栅栏"],                # Traditional residential
    3: ["博物馆", "美术馆", "剧院", "书店", "图书馆", "音乐厅", "展览"],    # Public culture & arts
    4: ["纪念馆", "天安门", "广场", "纪念碑", "遗址", "烈士", "革命"],     # Political symbolic red
    5: ["商场", "三里屯", "咖啡", "酒吧", "商圈", "餐饮", "夜市"],        # Modern commercial leisure
}

# ── Manual Toponym Standard Mapping ──────────────────────────────────────────
STANDARD_NAME_MAP = {
    "紫禁城": "故宫", "故宫博物院": "故宫", "故宫博物馆": "故宫",
    "故官": "故宫", "紫荊城": "故宫",
    "八达岭长城": "长城", "慕田峪长城": "长城", "居庸关": "长城",
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          UTILITY FUNCTIONS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def read_prompt(path: str) -> str:
    """Read prompt file content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


async def call_llm(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1,
    timeout: int = 60,
    max_retries: int = 3,
) -> str:
    """
    Standard interface for calling Qwen3-32B.
    - max_tokens=1: for binary judgment (Agent 1, Agent 4 check A/D)
    - max_tokens=-1: no length limit (Agent 2.1 ABSA, Agent 3.0 carrier extraction, etc.)
    """
    async with semaphore:
        payload = {
            "model": LLM["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": LLM["temperature"],
        }
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {LLM['api_key']}",
            "Content-Type": "application/json",
        }

        for attempt in range(max_retries):
            try:
                async with session.post(
                    f"{LLM['base_url']}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"LLM call failed: {e}")
                    return "ERROR"
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    return "ERROR"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   AGENT 1: Capital Culture Classification                    ║
# ║ Based on real logic from 数据库_首都文化分类.py                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def agent1_classify_culture(
    pool: asyncpg.Pool,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    culture_type: str,
) -> None:
    """
    Perform binary classification ("1"/"0") for each record in the source
    Weibo table against a single culture type. Must be called 4 times for
    the 4 culture types.

    Real code logic:
    - Input table: bj2019_cleaned_part9_with_response
    - Read fields: id, tid, cleaned_content
    - LLM: max_tokens=1, temperature=0
    - New column: {culture_type} = TEXT ("1"/"0")
    - batch_size=90, concurrency=30 (Semaphore-controlled)
    """
    prompt = read_prompt(PROMPTS["agent1"][culture_type])
    table_source = TABLE["source"]
    table_output = TABLE["agent1_out"]
    column_name = culture_type

    logger.info(f"[Agent 1] Starting processing for {culture_type} ...")

    # 1. Create/prepare output table
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM pg_tables WHERE tablename=$1)", table_output
        )
        if not exists:
            await conn.execute(f"CREATE TABLE {table_output} AS TABLE {table_source} WITH NO DATA")
            await conn.execute(f"INSERT INTO {table_output} SELECT * FROM {table_source}")

        col_exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.columns "
            "WHERE table_name=$1 AND column_name=$2)", table_output, column_name
        )
        if not col_exists:
            await conn.execute(f"ALTER TABLE {table_output} ADD COLUMN \"{column_name}\" TEXT")

    # 2. Paginated query + concurrent LLM + batch UPDATE
    async with pool.acquire() as conn:
        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM {table_source} WHERE \"广告移除\"='1'"
        )

    BATCH_SIZE = 90
    offset = 0

    while offset < total:
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT id, tid, cleaned_content FROM {table_source}
                WHERE "广告移除"='1' ORDER BY tid
                LIMIT {BATCH_SIZE} OFFSET {offset}
            """)

        if not rows:
            break

        # Concurrent LLM requests (corresponds to get_openai_responses logic)
        tasks = [
            call_llm(session, semaphore, prompt, row["cleaned_content"], max_tokens=1)
            for row in rows if row["cleaned_content"]
        ]
        results = await asyncio.gather(*tasks)

        # Batch update (corresponds to executemany logic)
        update_data = [
            (result, row["tid"])
            for row, result in zip(rows, results)
            if result != "ERROR" and row["cleaned_content"]
        ]
        if update_data:
            async with pool.acquire() as conn:
                await conn.executemany(
                    f'UPDATE {table_output} SET "{column_name}"=$1 WHERE tid=$2',
                    update_data,
                )

        offset += BATCH_SIZE
        logger.info(f"[Agent 1] {culture_type}: {min(offset, total)}/{total}")

    logger.info(f"[Agent 1] {culture_type} completed")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   AGENT 3.0: Material Culture Carrier Extraction             ║
# ║ Based on real logic from 数据库_首都文化物质文化载体提取.py                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def agent3_0_extract_carriers(
    pool: asyncpg.Pool,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    culture_type: str,
) -> None:
    """
    For Weibo posts labeled with this culture type (column value='1'),
    extract material culture carrier names.

    Real code logic:
    - Input: Agent 1 output table, WHERE "{culture_type}"='1'
    - Read: id, tid, cleaned_content
    - LLM: No max_tokens limit (full carrier name list)
    - Prompt: 红色文化分类&载体提取.txt (actually switched per culture type)
    - batch_size=50
    """
    prompt = read_prompt(PROMPTS["agent3_0_carrier"])
    table_in = TABLE["agent1_out"]
    table_out = TABLE["agent3_0_out"]
    col_name = f"{culture_type}_物质文化载体"

    logger.info(f"[Agent 3.0] {culture_type} carrier extraction ...")

    # Create table
    async with pool.acquire() as conn:
        exists = await conn.fetchval("SELECT EXISTS (SELECT FROM pg_tables WHERE tablename=$1)", table_out)
        if not exists:
            await conn.execute(f"CREATE TABLE {table_out} (LIKE {table_in} INCLUDING ALL)")
            # Only insert data labeled with this culture type
            await conn.execute(f"""
                INSERT INTO {table_out} SELECT * FROM {table_in}
                WHERE \"{culture_type}\"='1'
            """)
        col = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.columns "
            "WHERE table_name=$1 AND column_name=$2)", table_out, col_name
        )
        if not col:
            await conn.execute(f'ALTER TABLE {table_out} ADD COLUMN "{col_name}" TEXT')

    async with pool.acquire() as conn:
        total = await conn.fetchval(f"SELECT COUNT(*) FROM {table_in} WHERE \"{culture_type}\"='1'")

    BATCH_SIZE = 50
    offset = 0

    while offset < total:
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT id, tid, cleaned_content FROM {table_in}
                WHERE \"{culture_type}\"='1' ORDER BY tid
                LIMIT {BATCH_SIZE} OFFSET {offset}
            """)

        if not rows:
            break

        # Concurrent carrier extraction (corresponds to get_openai_responses,
        # no max_tokens limit)
        tasks = [
            call_llm(session, semaphore, prompt, row["cleaned_content"], max_tokens=-1)
            for row in rows if row["cleaned_content"]
        ]
        results = await asyncio.gather(*tasks)

        # Batch update
        update_data = [
            (result, row["tid"])
            for row, result in zip(rows, results)
            if result != "ERROR"
        ]
        if update_data:
            async with pool.acquire() as conn:
                await conn.executemany(
                    f'UPDATE {table_out} SET "{col_name}"=$1 WHERE tid=$2',
                    update_data,
                )

        offset += BATCH_SIZE
        logger.info(f"[Agent 3.0] {culture_type}: {min(offset, total)}/{total}")

    logger.info(f"[Agent 3.0] {culture_type} completed")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   AGENT 2.1: Material Culture Carrier ABSA                   ║
# ║ Based on real logic from 数据库_首都文化分类_物质文化载体ABSA.py               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def agent2_1_absa(
    pool: asyncpg.Pool,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    culture_type: str,
) -> None:
    """
    Perform aspect-level sentiment analysis on Weibo posts with extracted carriers:

    Real code logic:
    - Input: cleaned_content + evaluation carrier
    - User content: f'Social media text: "{row[2]}"\nExtracted material culture carrier: "{row[3]}"'
    - LLM: No max_tokens limit (full ABSA quadruple)
    - Batch_size=1 (ABSA output is long, process one by one)
    - Concurrency=100 (TCPConnector limit)
    """
    prompt = read_prompt(PROMPTS["agent2_1_absa"])
    table_in = TABLE["agent3_0_out"]        # Agent 3.0 output
    table_out = TABLE["agent2_1_out"]
    carrier_col = f"{culture_type}_物质文化载体"
    absa_col = f"{culture_type}_物质文化载体方面级评价"

    logger.info(f"[Agent 2.1] {culture_type} ABSA ...")

    async with pool.acquire() as conn:
        exists = await conn.fetchval("SELECT EXISTS (SELECT FROM pg_tables WHERE tablename=$1)", table_out)
        if not exists:
            await conn.execute(f"CREATE TABLE {table_out} (LIKE {table_in} INCLUDING ALL)")
            await conn.execute(f"""
                INSERT INTO {table_out} SELECT * FROM {table_in}
                WHERE \"{carrier_col}\" IS NOT NULL AND \"{carrier_col}\" != ''
            """)
        col = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.columns "
            "WHERE table_name=$1 AND column_name=$2)", table_out, absa_col
        )
        if not col:
            await conn.execute(f'ALTER TABLE {table_out} ADD COLUMN "{absa_col}" TEXT')

    async with pool.acquire() as conn:
        total = await conn.fetchval(f"""
            SELECT COUNT(*) FROM {table_in}
            WHERE \"{carrier_col}\" IS NOT NULL AND \"{carrier_col}\" != ''
        """)

    BATCH_SIZE = 1  # One-to-one: ABSA output is long
    offset = 0

    while offset < total:
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT id, tid, cleaned_content, "{carrier_col}"
                FROM {table_in}
                WHERE \"{carrier_col}\" IS NOT NULL AND \"{carrier_col}\" != ''
                ORDER BY tid LIMIT {BATCH_SIZE} OFFSET {offset}
            """)

        if not rows:
            break

        # Process one by one: real user content format = social media text + extracted carrier
        tasks = [
            call_llm(
                session, semaphore, prompt,
                f'Social media text: "{row["cleaned_content"]}"\nExtracted material culture carrier: "{row[carrier_col]}"',
                max_tokens=-1,
            )
            for row in rows
        ]
        results = await asyncio.gather(*tasks)

        update_data = [
            (result, row["tid"]) for row, result in zip(rows, results) if result != "ERROR"
        ]
        if update_data:
            async with pool.acquire() as conn:
                await conn.executemany(
                    f'UPDATE {table_out} SET "{absa_col}"=$1 WHERE tid=$2',
                    update_data,
                )

        offset += BATCH_SIZE
        if offset % 100 == 0:
            logger.info(f"[Agent 2.1] {culture_type}: {min(offset, total)}/{total}")

    logger.info(f"[Agent 2.1] {culture_type} completed")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              AGENT 2.2: Tuple Standardization + Sentiment Score             ║
# ║ Based on real logic from 统一元组内容.py + 情感得分计算(实验).py              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def agent2_2_postprocess_and_sentiment(pool: asyncpg.Pool) -> None:
    """
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Pseudocode Skeleton                                                │
    │  Real implementation: 情感得分计算(实验).py + 统一元组内容.py         │
    └──────────────────────────────────────────────────────────────────────┘

    2.2a: Regex parsing of LLM-returned quadruple format
           ("载体","方面","评价","情感")
    2.2b: 17-dimension weighted sentiment score computation

    Real code logic (情感得分计算.py):
    - Pure pandas computation, no LLM
    - Formula: Final sentiment score = Σ(score_aspect × weight)
    - Weights: 17 pre-defined dimensions (computed via CRITIC method)
    - batch_size=1000 writes

    ─── Pseudocode ───────────────────────────────────────────────────

    # Step 1: Read ABSA raw results column from Agent 2.1 output table
    df = pd.read_sql(f"SELECT tid, \"{absa_col}\" FROM {TABLE['agent2_1_out']}", conn)

    # Step 2: Regex parse quadruples (from 统一元组内容.py)
    pattern = r'\\(\\s*"([^"]+)"\\s*,\\s*"([^"]+)"\\s*,\\s*"([^"]+)"\\s*,\\s*"([^"]+)"\\s*\\)'
    parsed = df[absa_col].str.extractall(pattern)
    parsed.columns = ["载体", "方面", "评价", "情感"]

    # Step 3: Sentiment polarity mapping
    parsed["情感得分"] = parsed["情感"].map({"积极": 1, "中立": 0, "消极": -1})

    # Step 4: 17-dimension weighted summation
    for aspect, weight in ASPECT_WEIGHTS.items():
        parsed[f"score_{aspect}"] = np.where(
            parsed["方面"] == aspect, parsed["情感得分"] * weight, 0.0
        )
    parsed["最终情感得分"] = parsed[[f"score_{a}" for a in ASPECT_WEIGHTS]].sum(axis=1)

    # Step 5: Aggregate by tid, write back to Agent 2.2 output table
    final = parsed.groupby("tid")["最终情感得分"].mean().reset_index()
    final.to_sql(TABLE['agent2_2_out'], conn, if_exists="replace", index=False)
    """
    logger.info("[Agent 2.2] Tuple standardization + sentiment score calculation (pseudocode skeleton, see real scripts) ...")
    # This is a pseudocode placeholder, does not contain real SQL/regex implementation.
    # Real runnable version: 情感得分计算(实验).py + 统一元组内容.py
    logger.info("[Agent 2.2] Sentiment score calculation completed (pseudocode)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   AGENT 3.1: Place vs. Non-place Classification             ║
# ║ Based on real logic from 04地点与非地点区分.py + 投稿文件 3.1地点与非地点区分 │
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def agent3_1_place_classification(
    pool: asyncpg.Pool,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Determine whether extracted carriers are geographic places (output "1"/"0").

    Real code logic:
    - Input: carrier column
    - User content: row[3] (carrier name)
    - LLM: Qwen3-32B, temperature=0
    - Prompt: 00判别地点（实验）.txt
    - batch_size=50
    """
    prompt = read_prompt(PROMPTS["agent3_1_place"])
    table_name = TABLE["agent2_2_out"]  # Operate on Agent 2.2 output table

    logger.info("[Agent 3.1] Place vs. non-place classification ...")

    async with pool.acquire() as conn:
        await conn.execute(f'ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS "区分" TEXT')

    async with pool.acquire() as conn:
        total = await conn.fetchval(
            f'SELECT COUNT(*) FROM {table_name} WHERE "载体" IS NOT NULL'
        )

    BATCH_SIZE = 50
    offset = 0

    while offset < total:
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT id, tid, "载体" FROM {table_name}
                WHERE "载体" IS NOT NULL
                ORDER BY tid LIMIT {BATCH_SIZE} OFFSET {offset}
            """)

        if not rows:
            break

        # field index depends on table structure
        tasks = [
            call_llm(session, semaphore, prompt, row[2], max_tokens=1)
            for row in rows if row[2]
        ]
        results = await asyncio.gather(*tasks)

        update_data = [(result, row[1]) for row, result in zip(rows, results) if result != "ERROR"]
        if update_data:
            async with pool.acquire() as conn:
                await conn.executemany(
                    f'UPDATE {table_name} SET "区分"=$1 WHERE tid=$2', update_data
                )

        offset += BATCH_SIZE
        if offset % 500 == 0:
            logger.info(f"[Agent 3.1] {min(offset, total)}/{total}")

    logger.info("[Agent 3.1] completed")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   AGENT 3.2: Toponym Normalization & Alignment              ║
# ║ Based on real logic from 对齐代码（实验）.py (most complete version)         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def agent3_2_name_alignment(
    pool: asyncpg.Pool,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Hybrid-mode toponym alignment:
    Step 1: Build high-frequency mapping table from 4 aligned training tables
            (carriers appearing >=2 times)
    Step 2: Query mapping table -> hit then return directly;
            miss -> call LLM fallback

    Real code logic:
    - Training tables: bj2019_culture_part5_with_response_01{创新/古都/红色/京味}已对齐
    - high_freq_threshold: 2
    - Standard mapping: {'紫禁城': '故宫', ...}
    - LLM: 00对齐（实验）.txt
    - WHERE: 区分='1' (only process carriers classified as places)
    - batch_size=30, max_retries=3
    """
    prompt = read_prompt(PROMPTS["agent3_2_align"])
    table_name = TABLE["agent2_2_out"]

    logger.info("[Agent 3.2] Toponym normalization & alignment (Hybrid) ...")

    # 1. Build high-frequency mapping table
    mapping = build_high_freq_mapping(pool)

    # 2. Iterate all records where 区分='1', first query mapping table
    async with pool.acquire() as conn:
        await conn.execute(f'ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS "对齐" TEXT')
        total = await conn.fetchval(f'SELECT COUNT(*) FROM {table_name} WHERE "区分"=\'1\'')

    BATCH_SIZE = 30
    offset = 0

    while offset < total:
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT tid, "载体" FROM {table_name}
                WHERE "区分"='1' ORDER BY tid
                LIMIT {BATCH_SIZE} OFFSET {offset}
            """)

        if not rows:
            break

        results = []
        llm_tasks = []
        llm_rows = []

        for row in rows:
            carrier = row["载体"].strip() if row["载体"] else ""
            tid = row["tid"]

            # Priority: check mapping table (corresponds to align_carrier logic)
            if carrier in mapping:
                results.append((tid, mapping[carrier]))
            elif carrier in STANDARD_NAME_MAP:
                results.append((tid, STANDARD_NAME_MAP[carrier]))
            else:
                # Miss -> LLM fallback
                llm_tasks.append(call_llm(session, semaphore, prompt, carrier, max_tokens=-1))
                llm_rows.append(row)

        if llm_tasks:
            llm_results = await asyncio.gather(*llm_tasks)
            for row, res in zip(llm_rows, llm_results):
                results.append((row["tid"], res if res != "ERROR" else ""))

        update_data = [r for r in results if r[1]]
        if update_data:
            async with pool.acquire() as conn:
                await conn.executemany(
                    f'UPDATE {table_name} SET "对齐"=$1 WHERE tid=$2', update_data
                )

        offset += BATCH_SIZE

    logger.info("[Agent 3.2] completed")


def build_high_freq_mapping(pool: asyncpg.Pool) -> dict:
    """
    Build high-frequency carrier standardization mapping from 4 training tables.
    Real code logic (build_high_freq_mapping):
    - Iterate 4 {type}已对齐 tables
    - Count carrier occurrences
    - >= high_freq_threshold (2) -> record alignment result
    - Prefer standard_mapping, otherwise use alignment from training set
    """
    # Pseudocode skeleton
    mapping = {}
    return mapping


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   AGENT 3.3: Place Containment                              ║
# ║ Based on real logic from 05地点包含关系.py                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def agent3_3_containment(pool, session, semaphore):
    """
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Pseudocode Skeleton                                                │
    │  Real implementation: shiyan/代码/05地点包含关系.py                  │
    └──────────────────────────────────────────────────────────────────────┘

    Determine whether an aligned toponym is a sub-place within a larger
    place (e.g., "太和殿" inside "故宫").
    LLM: Qwen3-32B, Prompt: 05地点与地点内包含关系.txt

    ─── Pseudocode ───────────────────────────────────────────────────

    # Step 1: Read all carriers with non-empty "对齐" column
    rows = await conn.fetch(f'SELECT tid, "对齐" FROM {table} WHERE "对齐" IS NOT NULL')

    # Step 2: For each carrier, call LLM to determine if it is a sub-place
    for row in rows:
        prompt = read_prompt(PROMPTS["agent3_3_contain"])
        is_subplace = await call_llm(session, semaphore, prompt, row["对齐"], max_tokens=1)
        # "1" = sub-place, "0" = not a sub-place
        await conn.execute(f'UPDATE {table} SET "是否小地点"=$1 WHERE tid=$2', is_subplace, row["tid"])

    # Step 3: For sub-places, query their parent large place (based on geographic containment table)
    """
    logger.info("[Agent 3.3] Place containment (pseudocode skeleton, see 05地点包含关系.py) ...")
    # Pseudocode placeholder, does not contain real implementation.
    logger.info("[Agent 3.3] completed (pseudocode)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   AGENT 3.4: Heritage Site Determination                    ║
# ║ Based on real logic from 07判断是否文保单位.py                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def agent3_4_heritage(pool, session, semaphore):
    """
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Pseudocode Skeleton                                                │
    │  Real implementation: shiyan/代码/07判断是否文保单位.py              │
    └──────────────────────────────────────────────────────────────────────┘

    Determine whether a carrier is a cultural heritage protection site
    and its protection level.
    LLM: Qwen3-32B, Prompt: 07判断载体文保单位.txt

    ─── Pseudocode ───────────────────────────────────────────────────

    # Step 1: Read all aligned carriers
    rows = await conn.fetch(f'SELECT tid, "对齐" FROM {table} WHERE "对齐" IS NOT NULL')

    # Step 2: For each carrier, call LLM to determine heritage attributes
    # Output: "1"=national, "2"=provincial, "3"=city/district, "0"=non-heritage
    for row in rows:
        prompt = read_prompt(PROMPTS["agent3_4_heritage"])
        heritage_level = await call_llm(session, semaphore, prompt, row["对齐"], max_tokens=1)
        await conn.execute(f'UPDATE {table} SET "是否文保单位"=$1 WHERE tid=$2', heritage_level, row["tid"])
    """
    logger.info("[Agent 3.4] Heritage site determination (pseudocode skeleton, see 07判断是否文保单位.py) ...")
    # Pseudocode placeholder, does not contain real implementation.
    logger.info("[Agent 3.4] completed (pseudocode)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   AGENT 3.5: Spatial Type 5-Classification                  ║
# ║ Based on real logic from 载体分类汇总.py (Hybrid: keyword rules + LLM)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def agent3_5_spatial_type(
    pool: asyncpg.Pool,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Pseudocode Skeleton                                                │
    │  Real implementation: shiyan/代码/载体分类汇总.py                    │
    └──────────────────────────────────────────────────────────────────────┘

    Assign carriers to 5 spatial types (Hybrid mode).

    Real code logic (载体分类汇总.py):
    Step 1: Keyword rule matching -> direct output of category number
      - 故宫/长城/天坛 -> 1 (royal historical)
      - 胡同/四合院 -> 2 (traditional residential)
      - 博物馆/美术馆 -> 3 (public culture)
      - 纪念馆/天安门 -> 4 (political symbolic)
      - 商场/三里屯 -> 5 (modern commercial)
    Step 2: Rule miss -> LLM fallback
      - Prompt: choose from 5 types (strict: output only category name)
      - Validation: ensure output is within SPATIAL_TYPES

    ─── Pseudocode ───────────────────────────────────────────────────

    # Step 1: Read all aligned carriers
    rows = await conn.fetch(f'SELECT tid, "对齐" FROM {table} WHERE "对齐" IS NOT NULL')

    for row in rows:
        carrier = row["对齐"]

        # Step 2: Rule layer priority
        rule_class = None
        for class_id, keywords in SPATIAL_KEYWORD_RULES.items():
            if any(kw in carrier for kw in keywords):
                rule_class = class_id
                break

        # Step 3: Rule miss -> LLM fallback
        if rule_class is None:
            prompt = read_prompt(PROMPTS["agent3_5_classify"])
            llm_result = await call_llm(session, semaphore, prompt, carrier, max_tokens=-1)
            # Validation: ensure output is within 5 types
            rule_class = parse_and_validate(llm_result, SPATIAL_TYPES)

        # Step 4: Write back to "载体分类" column
        await conn.execute(f'UPDATE {table} SET "载体分类"=$1 WHERE tid=$2', rule_class, row["tid"])
    """
    logger.info("[Agent 3.5] Spatial type 5-classification (Hybrid, pseudocode skeleton, see 载体分类汇总.py) ...")
    # Pseudocode placeholder, does not contain real implementation.
    logger.info("[Agent 3.5] completed (pseudocode)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                AGENT 4: Self-Reflection & Correction (Closed-Loop Core)    ║
# ║ Based on real logic from 4_自反思修正智能体.py                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def agent4_self_reflection(
    pool: asyncpg.Pool,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    max_iterations: int = 2,   # Closed-loop maximum rounds
) -> dict:
    """
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Pseudocode Skeleton                                                │
    │  Real implementation: 4_自反思修正智能体.py in this directory       │
    └──────────────────────────────────────────────────────────────────────┘

    Execute five structural consistency checks on each record in the wide
    table, and send failed records back to upstream agents for re-inference.

    Checks:
      A: Carrier-culture type semantic consistency (LLM, max_tokens=1)
         -> Failure re-invokes Agent 1
      B: Sentiment-aspect-carrier plausibility (rule table)
         -> Failure re-invokes Agent 2
      C: Toponym uniqueness (string deduplication, no LLM)
         -> Silent correction
      D: Culture type exclusivity (LLM, max_tokens=1)
         -> Failure re-invokes Agent 1
      E: Carrier-geographic type consistency (rule table)
         -> Failure re-invokes Agent 3.5

    Closed-loop logic:
      for iteration in range(max_iterations):
        1. Read all records, execute 4 LLM checks
        2. Record failure info to corrections table
        3. For failed records, invoke corresponding upstream Agent for re-inference
        4. Re-verify -> pass or mark as final failure
        5. HITL sampling (500 records every 5 batches)

    Real code implementation: 4_自反思修正智能体.py
    (In same directory; here is the core orchestration logic skeleton)
    """
    logger.info(f"[Agent 4] Self-reflection & correction (max_iterations={max_iterations}, pseudocode skeleton) ...")

    stats = {"total": 0, "pass": 0, "fail": 0, "corrections": {}}

    # ─── Pseudocode ───────────────────────────────────────────────────
    #
    # for iteration in range(max_iterations):
    #     logger.info(f"[Agent 4] Round {iteration+1}/{max_iterations} ...")
    #
    #     # Read all records from wide table
    #     total = await conn.fetchval(f"SELECT COUNT(*) FROM {TABLE['merged_wide']}")
    #
    #     BATCH = 500
    #     for offset in range(0, total, BATCH):
    #         rows = await conn.fetch(f"""
    #             SELECT * FROM {TABLE['merged_wide']}
    #             ORDER BY id LIMIT {BATCH} OFFSET {offset}
    #         """)
    #
    #         for row in rows:
    #             # === Five Checks ===
    #             # Check A: LLM judges carrier-culture type compatibility (max_tokens=1)
    #             check_a = await call_llm(..., f"Does carrier {row['carrier']} belong to {row['culture_type']}?", max_tokens=1)
    #
    #             # Check B: Rule table checks aspect-spatial_type conflict
    #             check_b = rule_check_aspect_spatial(row['aspect'], row['spatial_type'])
    #
    #             # Check C: String deduplication (no LLM)
    #             check_c = deduplicate_carrier(row['carrier'])
    #
    #             # Check D: LLM judges culture type exclusivity (max_tokens=1)
    #             check_d = await call_llm(..., f"Are {row['all_culture_types']} mutually exclusive?", max_tokens=1)
    #
    #             # Check E: Rule table checks culture_type-spatial_type conflict
    #             check_e = rule_check_culture_spatial(row['culture_type'], row['spatial_type'])
    #
    #             # Write failed records to corrections table:
    #             if any([check_a, check_b, check_c, check_d, check_e]) < pass_threshold:
    #                 await conn.execute(
    #                     "INSERT INTO corrections (id, failed_check, target_agent, payload) VALUES ($1,$2,$3,$4)",
    #                     row['id'], failed_check_name, target_agent, json.dumps(row)
    #                 )
    #
    #         # HITL trigger: sample 500 records every 5 batches
    #         if (offset // BATCH + 1) % 5 == 0:
    #             logger.info(f"[Agent 4] HITL sampling (batch {offset//BATCH + 1})")
    #             await sample_for_hitl(pool, BATCH)
    #
    #     # Process corrections table: send back for re-inference
    #     corrections = await conn.fetch("SELECT * FROM corrections WHERE iteration=$1", iteration)
    #     for c in corrections:
    #         if c['target_agent'] == 'agent1':
    #             await agent1_classify_culture(pool, session, semaphore, c['payload']['culture_type'])
    #         elif c['target_agent'] == 'agent2':
    #             await agent2_1_absa(pool, session, semaphore, c['payload']['culture_type'])
    #         elif c['target_agent'] == 'agent3_5':
    #             await agent3_5_spatial_type(pool, session, semaphore)
    # ────────────────────────────────────────────────────────────────────────

    logger.info(f"[Agent 4] completed (pseudocode): {stats}")
    return stats


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           MAIN ORCHESTRATOR                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def main():
    """
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Main Orchestrator (Pseudocode)                                      │
    │  Sequentially chains all Agents to form the complete closed loop.    │
    │  This function is pseudocode for paper presentation, not runnable.  │
    └──────────────────────────────────────────────────────────────────────┘
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Four-Agent Closed-Loop Pipeline Orchestrator (Pseudocode for Paper)")
    logger.info("This file is pseudocode, not directly runnable; use each Agent's independent script for real execution")
    logger.info("=" * 60)

    # Initialize connection pool
    pool = await asyncpg.create_pool(**DB, min_size=2, max_size=10)

    # Initialize HTTP session and semaphore
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=100)
    semaphore = asyncio.Semaphore(30)

    async with aiohttp.ClientSession(connector=connector) as session:

        # ═══════════════════════════════════════════════════════════
        # Phase 1: Agent 1 -- Capital Culture Classification (4 types)
        # ═══════════════════════════════════════════════════════════
        logger.info("\n>>> Phase 1: Agent 1 -- Capital Culture Classification <<<")
        for ctype in CULTURE_TYPES:
            await agent1_classify_culture(pool, session, semaphore, ctype)

        # ═══════════════════════════════════════════════════════════
        # Phase 2: Agent 3.0 -- Carrier Extraction
        # ═══════════════════════════════════════════════════════════
        logger.info("\n>>> Phase 2: Agent 3.0 -- Carrier Extraction <<<")
        for ctype in CULTURE_TYPES:
            await agent3_0_extract_carriers(pool, session, semaphore, ctype)

        # ═══════════════════════════════════════════════════════════
        # Phase 3: Agent 2.1 -- ABSA
        # ═══════════════════════════════════════════════════════════
        logger.info("\n>>> Phase 3: Agent 2.1 -- ABSA <<<")
        for ctype in CULTURE_TYPES:
            await agent2_1_absa(pool, session, semaphore, ctype)

        # ═══════════════════════════════════════════════════════════
        # Phase 4: Agent 2.2 -- Sentiment Score Calculation
        # ═══════════════════════════════════════════════════════════
        logger.info("\n>>> Phase 4: Agent 2.2 -- Sentiment Score Calculation <<<")
        agent2_2_postprocess_and_sentiment(pool)

        # ═══════════════════════════════════════════════════════════
        # Phase 5: Agent 3.1-3.5 -- Spatial Carrier Analysis
        # ═══════════════════════════════════════════════════════════
        logger.info("\n>>> Phase 5: Agent 3.1~3.5 -- Spatial Analysis <<<")
        await agent3_1_place_classification(pool, session, semaphore)
        await agent3_2_name_alignment(pool, session, semaphore)
        await agent3_3_containment(pool, session, semaphore)
        await agent3_4_heritage(pool, session, semaphore)
        await agent3_5_spatial_type(pool, session, semaphore)

        # ═══════════════════════════════════════════════════════════
        # Phase 6: Merge Wide Table (all Agent 1-3 outputs -> one table)
        # ═══════════════════════════════════════════════════════════
        logger.info("\n>>> Phase 6: Merge Wide Table <<<")
        await build_wide_table(pool)

        # ═══════════════════════════════════════════════════════════
        # Phase 7: Agent 4 -- Self-Reflection & Correction (Closed Loop)
        # ═══════════════════════════════════════════════════════════
        logger.info("\n>>> Phase 7: Agent 4 -- Self-Reflection & Correction (Closed Loop) <<<")
        stats = await agent4_self_reflection(pool, session, semaphore, max_iterations=2)

    await pool.close()

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info(f"Workflow completed! Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"Final output table: {TABLE['agent4_out']}")
    logger.info(f"HITL table: {TABLE['agent4_hitl']}")
    logger.info(f"Stats: {stats}")
    logger.info("=" * 60)


async def build_wide_table(pool: asyncpg.Pool) -> None:
    """
    Merge all Agent 1/2/3 outputs into one wide table.
    Final fields:
      id, tid, cleaned_content, culture_type,
      carrier, carrier_list(JSON), absa_results(JSON),
      最终情感得分, spatial_type(1-5),
      是否小地点, 是否文保单位, all_culture_types(JSON)
    """
    logger.info("Building wide table bj2019_culture_with_absa_aligned ...")

    # SQL JOIN pseudocode:
    #
    # CREATE TABLE bj2019_culture_with_absa_aligned AS
    # SELECT
    #   a.id, a.tid, a.cleaned_content,
    #   CASE WHEN a.古都文化='1' THEN '古都文化'
    #        WHEN a.红色文化='1' THEN '红色文化'
    #        WHEN a.京味文化='1' THEN '京味文化'
    #        WHEN a.创新文化='1' THEN '创新文化'
    #   END AS culture_type,
    #   b.{culture_type}_物质文化载体 AS carrier,
    #   c.{culture_type}_物质文化载体方面级评价 AS absa_raw,
    #   d.最终情感得分,
    #   e.载体分类 AS spatial_type,
    #   f.是否小地点,
    #   g.是否文保单位,
    #   ARRAY['古都文化','红色文化','京味文化','创新文化'] AS all_culture_types
    # FROM agent1表 a
    # LEFT JOIN agent3_0表 b ON a.tid=b.tid
    # LEFT JOIN agent2_1表 c ON a.tid=c.tid
    # LEFT JOIN agent2_2表 d ON a.tid=d.tid
    # LEFT JOIN agent3_5表 e ON a.tid=e.tid
    # LEFT JOIN agent3_3表 f ON a.tid=f.tid
    # LEFT JOIN agent3_4表 g ON a.tid=g.tid;

    logger.info("Wide table construction completed")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            ENTRY POINT                                      ║
# ║          (Pseudocode for paper presentation, not recommended to execute)   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # This file is pseudocode for paper presentation; running it directly
    # only outputs logger messages, without actual LLM calls or database writes.
    # To reproduce the experiment, follow README.md in this directory to run
    # each Agent's independent script.
    print("=" * 70)
    print("  This file is pseudocode for paper presentation")
    print("  Not directly runnable; see each Agent's independent script:")
    print("    1.首都文化分类.py             (Agent 1)")
    print("    3.首都文化空间载体提取.py     (Agent 3.0)")
    print("    2.1物质文化载体ABSA.py        (Agent 2.1)")
    print("    2.2情感得分计算.py            (Agent 2.2)")
    print("    3.1地点与非地点区分.py        (Agent 3.1)")
    print("    3.2空间载体对齐.py            (Agent 3.2)")
    print("    4_自反思修正智能体.py         (Agent 4)")
    print("=" * 70)
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Only demonstrates orchestration structure, does not actually execute
    # asyncio.run(main())
