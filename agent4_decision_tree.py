# -*- coding: utf-8 -*-
"""
Agent 4 Self-Reflection Agent Decision Tree (for Reviewer Comment 2)
====================================================================

This file is a standalone, simplified version of the core check logic in
4_自反思修正智能体.py, used for: (1) as a reference implementation alongside
the algorithm pseudocode in the paper; (2) as a minimal reproducible version
for GitHub release.

Contains:
  - Decision tree implementation of the five checks
  - Rule tables (Check B/E)
  - LLM invocation (Check A/D)
  - Feedback-loop target routing

Full implementation (with async concurrency, HITL, database interaction)
is in 4_自反思修正智能体.py.

# NOTE: The LLM prompts in this file are simplified versions for code listing.
# For production use, the full prompts should match paper Appendix A.
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# 1. Data Structures and Enums
# ════════════════════════════════════════════════════════════════════════════

class CheckType(Enum):
    """Five check types."""
    A_CARRIER_CULTURE = "A"   # Carrier-culture type consistency
    B_SPATIAL_ASPECT = "B"    # Spatial type-evaluation aspect plausibility
    C_PLACE_UNIQUE = "C"      # Toponym spatial uniqueness
    D_CULTURE_MUTEX = "D"     # Culture type exclusivity
    E_CULTURE_SPATIAL = "E"   # Culture type-geographic type consistency


class TargetAgent(Enum):
    """Feedback-loop target agent."""
    AGENT_1 = "Agent_1"  # Culture type classification + carrier extraction
    AGENT_2 = "Agent_2"  # ABSA + sentiment polarity
    AGENT_3 = "Agent_3"  # Spatial carrier + GIS alignment
    NONE = "NONE"        # Silent correction, no feedback loop


@dataclass
class Record:
    """Record to be checked (merged wide table from Agent 1/2/3 outputs)."""
    record_id: int
    text: str
    carrier: str                       # Material culture carrier
    culture_type: str                  # 古都/红色/京味/创新
    aspect: str                        # One of 17 evaluation aspects
    polarity: str                      # 积极/中立/消极
    spatial_type: int                  # 1-5
    # ... other fields omitted


@dataclass
class CheckResult:
    """Check result."""
    passed: bool
    failed_check: Optional[CheckType]
    target_agent: TargetAgent
    reason: str = ""


# ════════════════════════════════════════════════════════════════════════════
# 2. Rule Tables (Check B & E)
# ════════════════════════════════════════════════════════════════════════════

# Check B: Spatial type -> forbidden ABSA aspects
CHECK_B_RULE_TABLE = {
    # Spatial type numbers: 1=Royal historical 2=Traditional residential 3=Public culture & arts 4=Political symbolic red 5=Modern commercial leisure
    4: ["饮食体验", "商业环境", "体力消耗"],  # Political symbolic spaces
    1: ["商业环境"],                          # Royal historical spaces
    3: ["饮食体验"],                          # Public culture & performing arts spaces
}

# Check E: Culture type -> forbidden spatial types
CHECK_E_RULE_TABLE = {
    "红色文化":  [5],     # Red culture carriers should not be modern commercial space
    "创新文化":  [1],     # Innovation culture carriers should not be royal historical heritage space
    "古都文化":  [5],     # Ancient capital culture carriers should not be modern commercial space
    "京味文化":  [],      # No forced exclusivity between Beijing flavor culture and spatial types
}


# ════════════════════════════════════════════════════════════════════════════
# 3. LLM Invocation Interface (for Check A/D)
# ════════════════════════════════════════════════════════════════════════════

async def llm_zero_shot_judge(system_prompt: str, user_prompt: str) -> int:
    """
    Call Qwen3-32B for 0/1 binary classification judgment (max_tokens=1).
    Returns 0 or 1.

    In production, this is implemented by the call_llm function in
    4_自反思修正智能体.py; here it is an interface definition.
    """
    # Actual implementation in 4_自反思修正智能体.py lines 98-140
    raise NotImplementedError(
        "Actual implementation in 4_自反思修正智能体.py's call_llm function"
    )


# ════════════════════════════════════════════════════════════════════════════
# 4. Five Checks (Decision Tree Core)
# ════════════════════════════════════════════════════════════════════════════

def check_a_carrier_culture(record: Record) -> CheckResult:
    """
    Check A: Carrier-culture type semantic consistency (LLM zero-shot)

    Judgment: LLM determines "Does carrier X belong to culture type Y".
    Threshold: max_tokens=1 output, 1=yes (pass), 0=no (fail)

    # NOTE: The prompts below are simplified; full prompts match paper Appendix A.
    """
    system_prompt = (
        "You are a cultural geography expert. Determine whether the given "
        "material culture carrier belongs to the given culture type. "
        "Only output 1 (yes) or 0 (no), nothing else."
    )
    user_prompt = (
        f"Carrier: '{record.carrier}'\n"
        f"Culture type: {record.culture_type}\n"
        f"Does it belong to this culture type?"
    )
    # result = await llm_zero_shot_judge(system_prompt, user_prompt)
    # Simplified as pseudocode placeholder
    result = 1  # Actually returned by LLM

    if result == 0:
        return CheckResult(
            passed=False,
            failed_check=CheckType.A_CARRIER_CULTURE,
            target_agent=TargetAgent.AGENT_1,
            reason=f"Carrier '{record.carrier}' does not belong to {record.culture_type}"
        )
    return CheckResult(passed=True, failed_check=None, target_agent=TargetAgent.NONE)


def check_b_spatial_aspect(record: Record) -> CheckResult:
    """
    Check B: Spatial type-evaluation aspect plausibility (rule table)

    Judgment: Blacklist of "forbidden aspects" for each spatial type.
    Threshold: record.aspect in CHECK_B_RULE_TABLE[record.spatial_type]
    """
    forbidden = CHECK_B_RULE_TABLE.get(record.spatial_type, [])
    if record.aspect in forbidden:
        return CheckResult(
            passed=False,
            failed_check=CheckType.B_SPATIAL_ASPECT,
            target_agent=TargetAgent.AGENT_2,
            reason=(f"Spatial type {record.spatial_type} forbids aspect '{record.aspect}'")
        )
    return CheckResult(passed=True, failed_check=None, target_agent=TargetAgent.NONE)


def check_c_place_unique(record: Record, seen_carriers: dict) -> CheckResult:
    """
    Check C: Toponym spatial uniqueness (string matching, no LLM)

    Judgment: Whether the same carrier name is assigned different spatial coordinates.
    Threshold: Normalized string exact match
    """
    # Normalization: remove parenthetical notes, whitespace, unify case
    def normalize(s):
        s = re.sub(r"[\(（].*?[\)）]", "", s)
        return s.strip().lower()

    norm = normalize(record.carrier)
    if norm in seen_carriers:
        # Silent correction: keep coordinates from first occurrence, no feedback loop
        return CheckResult(
            passed=True,
            failed_check=CheckType.C_PLACE_UNIQUE,
            target_agent=TargetAgent.NONE,
            reason=f"Carrier '{record.carrier}' already exists, silently merged"
        )
    seen_carriers[norm] = record.spatial_type
    return CheckResult(passed=True, failed_check=None, target_agent=TargetAgent.NONE)


def check_d_culture_mutex(record: Record, cross_cultural_records: list) -> CheckResult:
    """
    Check D: Culture type exclusivity (LLM zero-shot)

    Judgment: When the same (text, carrier) spans multiple culture types,
    LLM determines whether there is a "fusion narrative".
    Threshold: max_tokens=1 output, 1=fusion exists (pass), 0=no fusion (fail)

    # NOTE: The prompts below are simplified; full prompts match paper Appendix A.
    """
    if not cross_cultural_records:
        return CheckResult(passed=True, failed_check=None, target_agent=TargetAgent.NONE)

    other_types = [r.culture_type for r in cross_cultural_records
                   if r.culture_type != record.culture_type]
    if not other_types:
        return CheckResult(passed=True, failed_check=None, target_agent=TargetAgent.NONE)

    system_prompt = (
        "You are a cultural text analysis expert. Determine whether the text "
        "uses the same material carrier as a fusion narrative simultaneously "
        "bearing multiple culture types. Only output 1 (yes) or 0 (no)."
    )
    user_prompt = (
        f"Text: '{record.text}'\n"
        f"Carrier: '{record.carrier}'\n"
        f"Simultaneously classified as: {record.culture_type}, {', '.join(other_types)}\n"
        f"Does a fusion narrative exist?"
    )
    result = 1  # Actually returned by LLM

    if result == 0:
        return CheckResult(
            passed=False,
            failed_check=CheckType.D_CULTURE_MUTEX,
            target_agent=TargetAgent.AGENT_1,
            reason=(f"Carrier '{record.carrier}' spans culture types but no fusion narrative")
        )
    return CheckResult(passed=True, failed_check=None, target_agent=TargetAgent.NONE)


def check_e_culture_spatial(record: Record) -> CheckResult:
    """
    Check E: Culture type-geographic type consistency (rule table)

    Judgment: Blacklist of "forbidden spatial types" for each culture type.
    Threshold: record.spatial_type in CHECK_E_RULE_TABLE[record.culture_type]
    """
    forbidden_st = CHECK_E_RULE_TABLE.get(record.culture_type, [])
    if record.spatial_type in forbidden_st:
        return CheckResult(
            passed=False,
            failed_check=CheckType.E_CULTURE_SPATIAL,
            target_agent=TargetAgent.AGENT_3,
            reason=(f"{record.culture_type} forbids spatial type {record.spatial_type}")
        )
    return CheckResult(passed=True, failed_check=None, target_agent=TargetAgent.NONE)


# ════════════════════════════════════════════════════════════════════════════
# 5. Decision Tree Main Function
# ════════════════════════════════════════════════════════════════════════════

def agent4_decision_tree(
    record: Record,
    seen_carriers: dict,
    cross_cultural_records: list = None,
) -> CheckResult:
    """
    Agent 4 self-reflection decision tree main entry point.

    Algorithm flow (sequential checks, return on first failure):
      1. Check A: Carrier-culture consistency (LLM)
      2. Check B: Spatial-aspect plausibility (rule table)
      3. Check C: Toponym uniqueness (string matching)
      4. Check D: Culture exclusivity (LLM)
      5. Check E: Culture-spatial consistency (rule table)

    All pass -> PASS
    Any failure -> Return (FAIL, target_agent, reason)

    Args:
        record: Record to be checked
        seen_carriers: Dictionary of already-seen carriers {normalized_name: spatial_type}
        cross_cultural_records: Other culture-type records for the same (text, carrier)

    Returns:
        CheckResult
    """
    # 1. Check A
    r = check_a_carrier_culture(record)
    if not r.passed:
        logger.warning(f"[Check A FAIL] record_id={record.record_id}: {r.reason}")
        return r

    # 2. Check B
    r = check_b_spatial_aspect(record)
    if not r.passed:
        logger.warning(f"[Check B FAIL] record_id={record.record_id}: {r.reason}")
        return r

    # 3. Check C
    r = check_c_place_unique(record, seen_carriers)
    # Check C failure is silently merged, does not return failure

    # 4. Check D
    r = check_d_culture_mutex(record, cross_cultural_records or [])
    if not r.passed:
        logger.warning(f"[Check D FAIL] record_id={record.record_id}: {r.reason}")
        return r

    # 5. Check E
    r = check_e_culture_spatial(record)
    if not r.passed:
        logger.warning(f"[Check E FAIL] record_id={record.record_id}: {r.reason}")
        return r

    return CheckResult(passed=True, failed_check=None, target_agent=TargetAgent.NONE)


# ════════════════════════════════════════════════════════════════════════════
# 6. Demo Use Cases
# ════════════════════════════════════════════════════════════════════════════

def demo():
    """Demo decision tree execution."""
    print("=" * 70)
    print("Agent 4 Self-Reflection Decision Tree Demo")
    print("=" * 70)

    seen = {}

    # Case 1: Normal record (should pass all checks)
    print("\n[Case 1] Normal record (Ancient Capital culture + Palace Museum + Cultural experience + Royal historical space)")
    r1 = Record(
        record_id=1, text="Today visited the Palace Museum, so awe-inspiring",
        carrier="故宫", culture_type="古都文化",
        aspect="文化体验", polarity="积极", spatial_type=1
    )
    result = agent4_decision_tree(r1, seen)
    print(f"  Result: passed={result.passed}, target={result.target_agent.value}")

    # Case 2: Check B failure (Political symbolic space should not have food experience aspect)
    print("\n[Case 2] Check B failure (Tiananmen + food experience aspect)")
    r2 = Record(
        record_id=2, text="Eating roast duck at Tiananmen Square",
        carrier="天安门广场", culture_type="红色文化",
        aspect="饮食体验", polarity="积极", spatial_type=4
    )
    result = agent4_decision_tree(r2, seen)
    print(f"  Result: passed={result.passed}, failed={result.failed_check}")
    print(f"  Feedback target: {result.target_agent.value}")
    print(f"  Reason: {result.reason}")

    # Case 3: Check E failure (Red culture should not be modern commercial space)
    print("\n[Case 3] Check E failure (Red culture + modern commercial space)")
    r3 = Record(
        record_id=3, text="Feeling red culture at Sanlitun",
        carrier="三里屯", culture_type="红色文化",
        aspect="文化体验", polarity="积极", spatial_type=5
    )
    result = agent4_decision_tree(r3, seen)
    print(f"  Result: passed={result.passed}, failed={result.failed_check}")
    print(f"  Feedback target: {result.target_agent.value}")
    print(f"  Reason: {result.reason}")

    # Case 4: Normal multi-culture fusion (Check D should pass)
    print("\n[Case 4] Multi-culture fusion narrative (should pass Check D)")
    r4 = Record(
        record_id=4, text="Palace Museum creative shop combines ancient capital culture with innovation very well",
        carrier="故宫文创店", culture_type="古都文化",
        aspect="文化体验", polarity="积极", spatial_type=1
    )
    cross = [Record(
        record_id=5, text="Palace Museum creative shop combines ancient capital culture with innovation very well",
        carrier="故宫文创店", culture_type="创新文化",
        aspect="文化体验", polarity="积极", spatial_type=1
    )]
    result = agent4_decision_tree(r4, seen, cross_cultural_records=cross)
    print(f"  Result: passed={result.passed}, target={result.target_agent.value}")

    print("\n" + "=" * 70)
    print("Demo completed. Full async concurrent implementation in 4_自反思修正智能体.py")
    print("=" * 70)


if __name__ == "__main__":
    demo()
