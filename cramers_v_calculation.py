# -*- coding: utf-8 -*-
"""
Cramér's V Effect Size Calculation (for Section 3.1.2 and Section 3.2.1)
Database connection and table structure reference: 情感得分计算(实验).py
"""

import pandas as pd
import numpy as np
import psycopg2
from scipy.stats import chi2_contingency
import json
import warnings

warnings.filterwarnings('ignore')

# ===================== [Database Configuration - Reference: Sentiment Score Code] =====================
DB_CONFIG = {
    "database": "<YOUR_DB_NAME>",
    "user": "<YOUR_DB_USER>",
    "password": "<YOUR_DB_PASSWORD>",
    "host": "<YOUR_DB_HOST>",
    "port": "5432"
}

SCHEMA_NAME = "shihao"
TABLE_NAME = "bj2019_culture_1_10_with_response_首都文化分类_全"
FULL_TABLE = f'"{SCHEMA_NAME}"."{TABLE_NAME}"'  # Double quotes to preserve case

# ===================== [Column Names - Based on actual table structure] =====================
COL_CULTURE = "文化类型"  # Present in table
COL_ASPECT = "评价方面"  # Present in table
COL_POLARITY = "情感"  # Present in table
COL_CARRIER = "载体分类"  # Carrier classification


# ===================================================================

def cramers_v(confusion_matrix):
    """Calculate Cramér's V"""
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if min_dim == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def bootstrap_ci(data, col1, col2, n_bootstrap=1000, alpha=0.95):
    """Bootstrap resampling to compute 95% confidence interval"""
    values = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = data.sample(n=n, replace=True)
        crosstab = pd.crosstab(sample[col1], sample[col2])
        if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
            continue
        v = cramers_v(crosstab.values)
        values.append(v)
    if not values:
        return (np.nan, np.nan)
    lower = np.percentile(values, (1 - alpha) / 2 * 100)
    upper = np.percentile(values, (1 + alpha) / 2 * 100)
    return (lower, upper)


def compute_task(conn, query, col1, col2, task_name, use_bootstrap=True):
    """Execute query and compute Cramér's V"""
    print(f"\n--- Computing {task_name} ---")
    df = pd.read_sql_query(query, conn)
    if df.empty:
        print("  Query returned empty data, skipping")
        return None
    df = df.dropna(subset=[col1, col2])
    if df.empty:
        print("  No data after removing nulls, skipping")
        return None

    crosstab = pd.crosstab(df[col1], df[col2])
    print(f"  Contingency table dimensions: {crosstab.shape}")
    if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
        print("  Contingency table dimension < 2, cannot compute V")
        return None

    chi2, p, dof, _ = chi2_contingency(crosstab.values)
    v = cramers_v(crosstab.values)
    print(f"  Cramér's V = {v:.4f}")

    # Convert numpy types to Python native types for JSON serialization
    result = {
        'task': task_name,
        'cramers_v': float(round(v, 4)),
        'chi2': float(round(chi2, 4)),
        'p_value': float(p),
        'df': int(dof),
        'n': int(crosstab.sum().sum()),
        'table_shape': [int(crosstab.shape[0]), int(crosstab.shape[1])]
    }

    if use_bootstrap:
        ci_low, ci_high = bootstrap_ci(df, col1, col2)
        result['ci_95_lower'] = float(round(ci_low, 4)) if not np.isnan(ci_low) else None
        result['ci_95_upper'] = float(round(ci_high, 4)) if not np.isnan(ci_high) else None
        print(f"  Bootstrap 95% CI: [{result['ci_95_lower']}, {result['ci_95_upper']}]")

    return result


def main():
    print("=" * 70)
    print("Cramér's V Effect Size Calculation (for Section 3.1.2 and Section 3.2.1)")
    print("=" * 70)
    print(f"Connecting to database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    print(f"Data table: {FULL_TABLE}")

    conn = psycopg2.connect(**DB_CONFIG)

    # Task 1: Culture type x Evaluation aspect
    query1 = f"""
    SELECT {COL_CULTURE} AS culture, {COL_ASPECT} AS aspect
    FROM {FULL_TABLE}
    WHERE {COL_ASPECT} IS NOT NULL
    """

    # Task 2: Carrier classification x Evaluation aspect
    query2 = f"""
    SELECT {COL_CARRIER} AS carrier, {COL_ASPECT} AS aspect
    FROM {FULL_TABLE}
    WHERE {COL_ASPECT} IS NOT NULL AND {COL_CARRIER} IS NOT NULL
    """

    # Task 3: Evaluation aspect x Sentiment polarity
    query3 = f"""
    SELECT {COL_ASPECT} AS aspect, {COL_POLARITY} AS polarity
    FROM {FULL_TABLE}
    WHERE {COL_ASPECT} IS NOT NULL AND {COL_POLARITY} IS NOT NULL
    """

    # Task 4: Culture type x Carrier classification
    query4 = f"""
    SELECT {COL_CULTURE} AS culture, {COL_CARRIER} AS carrier
    FROM {FULL_TABLE}
    WHERE {COL_CULTURE} IS NOT NULL AND {COL_CARRIER} IS NOT NULL
    """

    results = {}

    # Compute Task 1: Culture type x Evaluation aspect
    r1 = compute_task(conn, query1, 'culture', 'aspect', 'Culture type x Evaluation aspect')
    if r1:
        results['culture_vs_aspect'] = r1

    # Compute Task 2: Carrier classification x Evaluation aspect
    r2 = compute_task(conn, query2, 'carrier', 'aspect', 'Carrier classification x Evaluation aspect')
    if r2:
        results['carrier_vs_aspect'] = r2

    # Compute Task 3: Evaluation aspect x Sentiment polarity
    r3 = compute_task(conn, query3, 'aspect', 'polarity', 'Evaluation aspect x Sentiment polarity')
    if r3:
        results['aspect_vs_polarity'] = r3

    # Compute Task 4: Culture type x Carrier classification
    r4 = compute_task(conn, query4, 'culture', 'carrier', 'Culture type x Carrier classification')
    if r4:
        results['culture_vs_carrier'] = r4

    conn.close()

    # Save as JSON (using ensure_ascii=False to preserve Chinese characters)
    output_file = 'cramers_v_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nAll results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary results:")
    for key, val in results.items():
        ci_low = val.get('ci_95_lower', 'N/A')
        ci_up = val.get('ci_95_upper', 'N/A')
        print(f"  {key}: V={val['cramers_v']:.4f}  (95% CI: [{ci_low}, {ci_up}])")
    print("=" * 70)
    print("\nNotes:")
    print("  1. Cohen (1988) benchmarks: V<0.1 negligible, 0.1-0.3 weak, 0.3-0.5 moderate, >=0.5 strong")
    print("  2. Bootstrap 1000 resamples provide 95% confidence intervals")
    print("  3. Replace the chi-square test p<0.001 statements in paper Section 3.1.2 and 3.2.1 with these results")


if __name__ == "__main__":
    main()
