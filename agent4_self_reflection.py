# -*- coding: utf-8 -*-
"""
Agent 4: Self-Reflection and Correction Agent
======================================================================
Responsibility: Execute five structural consistency checks on Agent 1/2/3
      outputs, and automatically send failed records back to the
      corresponding upstream Agent for re-inference.

Five Checks:
  A - Carrier-culture type semantic consistency (LLM zero-shot judgment)
  B - Sentiment-aspect-carrier triplet plausibility (rule table + optional LLM)
  C - Toponym spatial uniqueness (exact string matching, zero LLM cost)
  D - Culture type exclusivity (LLM fusion narrative judgment)
  E - Carrier-geographic type consistency (rule table)

Dependencies (consistent with existing Agent code):
  pip install asyncpg aiohttp python-dotenv tqdm
"""

import asyncio
import json
import logging
import random
import re
from collections import Counter
from typing import Any

import aiohttp
import asyncpg
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Global Configuration
# ─────────────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "<YOUR_DB_HOST>",
    "port": 5432,
    "user": "<YOUR_DB_USER>",
    "password": "<YOUR_DB_PASSWORD>",
    "database": "<YOUR_DB_NAME>",
    "min_size": 5,
    "max_size": 10,
}

LLM_CONFIG = {
    "base_url": "<YOUR_LLM_BASE_URL>",
    "api_key": "<YOUR_API_KEY>",
    "model": "<YOUR_MODEL_NAME>",
    "temperature": 0.0,
    "max_tokens": 1,                   # Check A/D only needs 0/1, fast
}

PIPELINE_CONFIG = {
    "batch_size": 500,                 # Number of records per batch
    "max_concurrent_requests": 20,     # Concurrency limit
    "hitl_interval_batches": 5,        # Trigger HITL sampling every N batches
    "hitl_sample_size": 500,           # HITL sample size
    "max_retry": 2,                    # Maximum re-inference attempts
    "source_table": "bj2019_culture_with_absa_aligned",   # Upstream output table
    "output_table": "bj2019_culture_agent4_verified",     # Agent 4 output table
    "hitl_table": "bj2019_agent4_hitl_samples",           # HITL sampling table
}

# Culture type labels (corresponding to database column names)
CULTURE_TYPES = ["古都文化", "红色文化", "京味文化", "创新文化"]

# ─────────────────────────────────────────────────────────────────────────────
# Rule Tables (Check B & E)
# ─────────────────────────────────────────────────────────────────────────────

# Check B: (spatial type -> forbidden ABSA aspects)
# Spatial type numbers: 1=Royal historical 2=Traditional residential 3=Public culture & arts 4=Political symbolic red 5=Modern commercial leisure
CHECK_B_RULE_TABLE: dict[int, list[str]] = {
    4: ["饮食体验", "商业环境", "体力消耗"],   # Political symbolic spaces should not have such aspects
    1: ["商业环境"],                           # Royal historical spaces
    3: ["饮食体验"],                           # Public culture & performing arts spaces
}

# Check E: (culture type -> forbidden spatial type numbers)
CHECK_E_RULE_TABLE: dict[str, list[int]] = {
    "红色文化":  [5],           # Red culture carriers should not be labeled "modern commercial space"
    "创新文化":  [1],           # Innovation culture carriers should not be labeled "royal historical heritage space"
    "古都文化":  [5],           # Ancient capital culture carriers should not be labeled "modern commercial space"
    "京味文化":  [],            # No forced exclusivity between Beijing flavor culture and spatial types
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Invocation Utility
# ─────────────────────────────────────────────────────────────────────────────

async def call_llm(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1,
) -> str:
    """Send a single async request to Qwen3-32B, returning model output text."""
    async with semaphore:
        payload = {
            "model": LLM_CONFIG["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": LLM_CONFIG["temperature"],
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {LLM_CONFIG['api_key']}",
            "Content-Type": "application/json",
        }
        for attempt in range(3):
            try:
                async with session.post(
                    f"{LLM_CONFIG['base_url']}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"LLM call failed (after 3 retries): {e}")
                    return ""
                await asyncio.sleep(1)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Five Checks Implementation
# ─────────────────────────────────────────────────────────────────────────────

async def check_a_carrier_culture_consistency(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    carrier: str,
    culture_type: str,
) -> bool:
    """
    Check A: Carrier-culture type semantic consistency.
    Returns True = consistent (pass), False = inconsistent (trigger correction).
    LLM zero-shot judgment, max_tokens=1.

    # NOTE: The prompts below are simplified versions for this code listing.
    # For production use, the full prompts should match paper Appendix A4-1.
    """
    system = (
        "You are a Beijing culture expert. Please determine whether the given "
        "material culture carrier belongs to the specified capital culture type. "
        "Only answer 1 (belongs) or 0 (does not belong), with no explanation."
    )
    user = f'Material culture carrier: "{carrier}"\nCulture type: "{culture_type}"\nPlease answer 1 or 0.'
    result = await call_llm(session, semaphore, system, user, max_tokens=1)
    return result.strip() == "1"


def check_b_sentiment_triplet_plausibility(
    carrier: str,
    aspect: str,
    spatial_type: int,
) -> bool:
    """
    Check B: Sentiment-aspect-carrier triplet plausibility (rule table).
    Returns True = plausible (pass), False = implausible (trigger correction).
    Pure rule judgment, no LLM cost.
    """
    forbidden_aspects = CHECK_B_RULE_TABLE.get(spatial_type, [])
    if aspect in forbidden_aspects:
        logger.debug(
            f"Check B failed: carrier={carrier}, aspect={aspect}, "
            f"spatial_type={spatial_type}"
        )
        return False
    return True


def check_c_toponymic_uniqueness(carrier_list: list[str]) -> list[str]:
    """
    Check C: Toponym spatial uniqueness (exact string deduplication).
    Returns deduplicated carrier list (preserving first occurrence).
    Pure string operation, zero LLM cost.
    """
    seen: set[str] = set()
    deduplicated: list[str] = []
    for c in carrier_list:
        normalized = c.strip()
        if normalized not in seen:
            seen.add(normalized)
            deduplicated.append(normalized)
    return deduplicated


async def check_d_cultural_type_exclusivity(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    text: str,
    carrier: str,
    culture_types: list[str],
) -> list[str]:
    """
    Check D: Culture type exclusivity.
    If the same (text, carrier) belongs to multiple culture types simultaneously,
    ask LLM whether there is a fusion narrative.
    Returns the list of culture types that should be kept (single or multiple).

    # NOTE: The prompts below are simplified versions for this code listing.
    # For production use, the full prompts should match paper Appendix A4-1.
    """
    if len(culture_types) <= 1:
        return culture_types

    system = (
        "You are a Beijing cultural research expert. Please determine whether "
        "the following Weibo post contains an explicit narrative that simultaneously "
        "involves two different culture types for the same carrier. "
        "Only answer 1 (yes, explicit fusion narrative exists) or 0 (no), "
        "with no explanation."
    )
    kept = []
    for i in range(len(culture_types)):
        for j in range(i + 1, len(culture_types)):
            ct_a, ct_b = culture_types[i], culture_types[j]
            user = (
                f'Weibo text: "{text}"\n'
                f'Carrier: "{carrier}"\n'
                f'Culture type A: "{ct_a}"\n'
                f'Culture type B: "{ct_b}"\n'
                f'Does the text contain an explicit narrative about both culture types for this carrier? Please answer 1 or 0.'
            )
            result = await call_llm(session, semaphore, system, user, max_tokens=1)
            if result.strip() == "1":
                # Fusion narrative exists, keep both
                if ct_a not in kept:
                    kept.append(ct_a)
                if ct_b not in kept:
                    kept.append(ct_b)
            else:
                # No fusion narrative, keep only the earlier one in culture_types
                # (already sorted by confidence)
                if ct_a not in kept:
                    kept.append(ct_a)
    return kept if kept else [culture_types[0]]


def check_e_carrier_geographic_consistency(
    carrier: str,
    culture_type: str,
    spatial_type: int,
) -> bool:
    """
    Check E: Carrier-geographic type consistency (rule table).
    Returns True = consistent (pass), False = inconsistent (trigger correction).
    Pure rule judgment, no LLM cost.
    """
    forbidden_spatial_types = CHECK_E_RULE_TABLE.get(culture_type, [])
    if spatial_type in forbidden_spatial_types:
        logger.debug(
            f"Check E failed: carrier={carrier}, culture_type={culture_type}, "
            f"spatial_type={spatial_type}"
        )
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Core: Complete Record Verification
# ─────────────────────────────────────────────────────────────────────────────

async def verify_record(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute five checks on a single complete record, returning a dict
    containing verification results.

    Record field descriptions (from database join results):
      - id: int
      - cleaned_content: str                Weibo original text
      - culture_type: str                   Assigned culture type
      - carrier: str                        Material culture carrier (normalized)
      - carrier_list: list[str]             All carriers for this text under this culture type
      - absa_results: list[list]            [[carrier, aspect, evaluation, polarity], ...]
      - spatial_type: int                   Agent 3-5 assigned spatial type (1-5)
      - all_culture_types: list[str]        All culture types labeled as 1 for this text
    """
    result = {
        "id":          record["id"],
        "carrier":     record["carrier"],
        "culture_type": record["culture_type"],
        "check_a": True,
        "check_b_failures": [],   # List of specific failed aspects
        "check_c_original": record["carrier_list"],
        "check_c_deduped": [],
        "check_d_original": record.get("all_culture_types", [record["culture_type"]]),
        "check_d_kept": [],
        "check_e": True,
        "overall_pass": True,
        "corrections": {},        # Records each check's suggested correction action
    }

    # ── Check A ──────────────────────────────────────────────────────────────
    a_pass = await check_a_carrier_culture_consistency(
        session, semaphore, record["carrier"], record["culture_type"]
    )
    result["check_a"] = a_pass
    if not a_pass:
        result["overall_pass"] = False
        result["corrections"]["A"] = f"Re-invoke Agent 1: carrier={record['carrier']}"

    # ── Check B ──────────────────────────────────────────────────────────────
    b_failures = []
    for absa_row in record.get("absa_results", []):
        # absa_row = [carrier, aspect, evaluation text, polarity]
        if len(absa_row) < 4:
            continue
        _, aspect, _, _ = absa_row
        b_ok = check_b_sentiment_triplet_plausibility(
            record["carrier"], aspect, record["spatial_type"]
        )
        if not b_ok:
            b_failures.append(aspect)
    result["check_b_failures"] = b_failures
    if b_failures:
        result["overall_pass"] = False
        result["corrections"]["B"] = f"Re-invoke Agent 2: aspects={b_failures}"

    # ── Check C ──────────────────────────────────────────────────────────────
    deduped = check_c_toponymic_uniqueness(record["carrier_list"])
    result["check_c_deduped"] = deduped
    if len(deduped) < len(record["carrier_list"]):
        result["corrections"]["C"] = (
            f"Deduplication: {record['carrier_list']} -> {deduped}"
        )
        # Check C does not affect overall_pass; it is a silent correction

    # ── Check D ──────────────────────────────────────────────────────────────
    all_types = record.get("all_culture_types", [record["culture_type"]])
    kept = await check_d_cultural_type_exclusivity(
        session, semaphore, record["cleaned_content"], record["carrier"], all_types
    )
    result["check_d_kept"] = kept
    if len(kept) < len(all_types):
        removed = [t for t in all_types if t not in kept]
        result["overall_pass"] = False
        result["corrections"]["D"] = f"Remove redundant culture type labels: {removed}"

    # ── Check E ──────────────────────────────────────────────────────────────
    e_pass = check_e_carrier_geographic_consistency(
        record["carrier"], record["culture_type"], record["spatial_type"]
    )
    result["check_e"] = e_pass
    if not e_pass:
        result["overall_pass"] = False
        result["corrections"]["E"] = (
            f"Re-invoke Agent 3-5: carrier={record['carrier']}, "
            f"spatial_type={record['spatial_type']} incompatible with {record['culture_type']}"
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HITL Human Sampling Interface
# ─────────────────────────────────────────────────────────────────────────────

async def trigger_hitl_sampling(
    pool: asyncpg.Pool,
    batch_idx: int,
    sample_size: int = 500,
) -> None:
    """
    Trigger HITL sampling every N batches:
    Randomly extract sample_size records from the output table into the HITL
    table for human annotation review.
    """
    async with pool.acquire() as conn:
        await conn.execute(f"""
            INSERT INTO {PIPELINE_CONFIG['hitl_table']}
                (id, carrier, culture_type, check_a, check_b_failures,
                 check_d_kept, check_e, corrections, batch_idx, sampled_at)
            SELECT
                id, carrier, culture_type, check_a, check_b_failures,
                check_d_kept, check_e, corrections,
                {batch_idx} AS batch_idx,
                NOW() AS sampled_at
            FROM {PIPELINE_CONFIG['output_table']}
            WHERE batch_idx = {batch_idx}
            ORDER BY RANDOM()
            LIMIT {sample_size}
        """)
    logger.info(f"HITL sampling completed (batch {batch_idx}): {sample_size} records written to {PIPELINE_CONFIG['hitl_table']}")


# ─────────────────────────────────────────────────────────────────────────────
# Batch Processing Main Flow
# ─────────────────────────────────────────────────────────────────────────────

async def process_batch(
    pool: asyncpg.Pool,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    batch_rows: list[asyncpg.Record],
    batch_idx: int,
) -> list[dict]:
    """Execute Agent 4 verification concurrently on a batch of records."""
    records = []
    for row in batch_rows:
        # Parse JSON fields (asyncpg returns strings, need manual parsing)
        try:
            absa_results = json.loads(row["absa_results"]) if row.get("absa_results") else []
        except (json.JSONDecodeError, TypeError):
            absa_results = []

        try:
            carrier_list = json.loads(row["carrier_list"]) if row.get("carrier_list") else [row["carrier"]]
        except (json.JSONDecodeError, TypeError):
            carrier_list = [row["carrier"]]

        try:
            all_culture_types = json.loads(row["all_culture_types"]) if row.get("all_culture_types") else [row["culture_type"]]
        except (json.JSONDecodeError, TypeError):
            all_culture_types = [row["culture_type"]]

        records.append({
            "id":               row["id"],
            "cleaned_content":  row["cleaned_content"],
            "culture_type":     row["culture_type"],
            "carrier":          row["carrier"],
            "carrier_list":     carrier_list,
            "absa_results":     absa_results,
            "spatial_type":     row.get("spatial_type", 0),
            "all_culture_types": all_culture_types,
        })

    tasks = [verify_record(session, semaphore, rec) for rec in records]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    verified = []
    for rec, res in zip(records, results):
        if isinstance(res, Exception):
            logger.warning(f"Record {rec['id']} verification exception: {res}")
            continue
        res["batch_idx"] = batch_idx
        verified.append(res)

    return verified


async def write_results(pool: asyncpg.Pool, results: list[dict]) -> None:
    """Write Agent 4 verification results to the output table."""
    if not results:
        return
    async with pool.acquire() as conn:
        await conn.executemany(
            f"""
            INSERT INTO {PIPELINE_CONFIG['output_table']}
                (id, carrier, culture_type, check_a, check_b_failures,
                 check_c_deduped, check_d_kept, check_e,
                 overall_pass, corrections, batch_idx)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (id, carrier, culture_type) DO UPDATE SET
                check_a = EXCLUDED.check_a,
                overall_pass = EXCLUDED.overall_pass,
                corrections = EXCLUDED.corrections
            """,
            [
                (
                    r["id"],
                    r["carrier"],
                    r["culture_type"],
                    r["check_a"],
                    json.dumps(r["check_b_failures"], ensure_ascii=False),
                    json.dumps(r["check_c_deduped"],  ensure_ascii=False),
                    json.dumps(r["check_d_kept"],     ensure_ascii=False),
                    r["check_e"],
                    r["overall_pass"],
                    json.dumps(r["corrections"],      ensure_ascii=False),
                    r["batch_idx"],
                )
                for r in results
            ],
        )


async def ensure_output_table(pool: asyncpg.Pool) -> None:
    """Create output tables (if they do not exist)."""
    async with pool.acquire() as conn:
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {PIPELINE_CONFIG['output_table']} (
                id              BIGINT,
                carrier         TEXT,
                culture_type    TEXT,
                check_a         BOOLEAN,
                check_b_failures TEXT,
                check_c_deduped  TEXT,
                check_d_kept    TEXT,
                check_e         BOOLEAN,
                overall_pass    BOOLEAN,
                corrections     TEXT,
                batch_idx       INT,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (id, carrier, culture_type)
            )
        """)
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {PIPELINE_CONFIG['hitl_table']} (
                id              BIGINT,
                carrier         TEXT,
                culture_type    TEXT,
                check_a         BOOLEAN,
                check_b_failures TEXT,
                check_d_kept    TEXT,
                check_e         BOOLEAN,
                corrections     TEXT,
                batch_idx       INT,
                sampled_at      TIMESTAMPTZ,
                human_verdict   BOOLEAN DEFAULT NULL,  -- Human annotation result
                reviewer_note   TEXT    DEFAULT NULL
            )
        """)
    logger.info("Output table and HITL table initialization completed.")


# ─────────────────────────────────────────────────────────────────────────────
# Main Function
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    pool = await asyncpg.create_pool(**DB_CONFIG)
    await ensure_output_table(pool)

    semaphore = asyncio.Semaphore(PIPELINE_CONFIG["max_concurrent_requests"])

    connector = aiohttp.TCPConnector(limit=50, limit_per_host=50)
    async with aiohttp.ClientSession(connector=connector) as session:

        # Get total record count
        async with pool.acquire() as conn:
            total = await conn.fetchval(
                f"SELECT COUNT(*) FROM {PIPELINE_CONFIG['source_table']}"
            )
        logger.info(f"Total records to process: {total}")

        batch_size = PIPELINE_CONFIG["batch_size"]
        total_batches = (total + batch_size - 1) // batch_size
        passed = 0
        failed = 0

        with tqdm(total=total, desc="Agent 4 verification progress", unit="rec") as pbar:
            for batch_idx in range(total_batches):
                offset = batch_idx * batch_size

                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        f"""
                        SELECT
                            s.id,
                            s.cleaned_content,
                            s.culture_type,
                            s.carrier,
                            s.carrier_list,
                            s.absa_results,
                            s.spatial_type,
                            s.all_culture_types
                        FROM {PIPELINE_CONFIG['source_table']} s
                        ORDER BY s.id
                        LIMIT {batch_size} OFFSET {offset}
                        """
                    )

                if not rows:
                    break

                results = await process_batch(pool, session, semaphore, rows, batch_idx)
                await write_results(pool, results)

                # Count passed/failed
                b_passed = sum(1 for r in results if r["overall_pass"])
                b_failed = len(results) - b_passed
                passed += b_passed
                failed += b_failed
                pbar.update(len(rows))
                pbar.set_postfix({"passed": passed, "corrected": failed})

                # HITL sampling trigger
                if (batch_idx + 1) % PIPELINE_CONFIG["hitl_interval_batches"] == 0:
                    await trigger_hitl_sampling(pool, batch_idx)

        logger.info(
            f"\n=== Agent 4 Completed ===\n"
            f"  Total processed: {total} records\n"
            f"  Directly passed: {passed} ({passed/total*100:.1f}%)\n"
            f"  Triggered correction: {failed} ({failed/total*100:.1f}%)\n"
        )

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
