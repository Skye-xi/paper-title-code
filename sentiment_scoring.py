import pandas as pd
import numpy as np
import psycopg2
from tqdm import tqdm

# ===================== Database Configuration =====================
DB_CONFIG = {
    "database": "postgres",
    "user": "postgres",
    "password": "<YOUR_DB_PASSWORD>",
    "host": "<YOUR_DB_HOST>",
    "port": "5432"
}

SCHEMA_NAME = "shihao"
ORIGIN_TABLE = "bj2019_culture_1_10_with_response_首都文化分类_全"
NEW_TABLE    = "bj2019_culture_1_10_with_response_情感得分_全"

# ===================== Hard-coded weights: completely resolved file-not-found error =====================
ASPECT_LIST = [
    "交通便利", "人文景观", "人流量", "体力消耗", "公共设施", "历史认知",
    "商业环境", "天气气候", "建筑美学", "情感共鸣", "文化体验", "文化内涵",
    "文化氛围", "文化遗产", "游客服务", "自然景观", "饮食体验"
]

# Precomputed weights for 17 dimensions (built-in)
weight_dict = {
    "交通便利": 0.0641,
    "人文景观": 0.0662,
    "人流量": 0.0528,
    "体力消耗": 0.0549,
    "公共设施": 0.0592,
    "历史认知": 0.0603,
    "商业环境": 0.0557,
    "天气气候": 0.0514,
    "建筑美学": 0.0625,
    "情感共鸣": 0.0586,
    "文化体验": 0.0573,
    "文化内涵": 0.0618,
    "文化氛围": 0.0597,
    "文化遗产": 0.0634,
    "游客服务": 0.0539,
    "自然景观": 0.0546,
    "饮食体验": 0.0585
}

# Sentiment score mapping
EMOTION_SCORE = {"积极": 1, "中立": 0, "消极": -1}

# ===================== 1. Load original table =====================
def load_original_table():
    print("[DB] Connecting to database and reading data...")
    conn = psycopg2.connect(**DB_CONFIG)
    query = f'SELECT * FROM "{SCHEMA_NAME}"."{ORIGIN_TABLE}"'
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"[OK] Original table loaded: {len(df):,} records")
    return df

# ===================== 2. Compute sentiment scores (red progress bar) =====================
def calc_final_score(df):
    print("\n[Calc] Computing sentiment scores...")
    df["情感得分"] = df["情感"].map(EMOTION_SCORE)

    # 17 dimension scores + red progress bar
    for asp in tqdm(ASPECT_LIST, desc="17 evaluation dimensions", colour="red", ncols=80):
        df[f"score_{asp}"] = np.where(df["评价方面"] == asp, df["情感得分"], 0.0)

    # Final weighted sentiment score
    df["最终情感得分"] = 0.0
    for asp in ASPECT_LIST:
        df["最终情感得分"] += df[f"score_{asp}"] * weight_dict[asp]

    print("[OK] Final weighted sentiment score computed")
    return df

# ===================== 3. Write new table (red progress bar) =====================
def write_new_table(df):
    print("\n[DB] Writing to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute(f'DROP TABLE IF EXISTS "{SCHEMA_NAME}"."{NEW_TABLE}"')
    conn.commit()

    cols = []
    for col, dt in zip(df.columns, df.dtypes):
        if "int" in str(dt):
            t = "integer"
        elif "float" in str(dt):
            t = "double precision"
        else:
            t = "text"
        cols.append(f'"{col}" {t}')

    create_sql = f'CREATE TABLE "{SCHEMA_NAME}"."{NEW_TABLE}" ({",".join(cols)})'
    cur.execute(create_sql)
    conn.commit()

    data = [tuple(x) for x in df.to_numpy()]
    batch_size = 1000
    total = len(data)

    # Write to database with red progress bar
    with tqdm(total=total, desc="Writing to database", colour="red", ncols=80) as pbar:
        for i in range(0, total, batch_size):
            batch = data[i:i+batch_size]
            placeholders = ",".join(["%s"] * len(df.columns))
            insert_sql = f'INSERT INTO "{SCHEMA_NAME}"."{NEW_TABLE}" VALUES ({placeholders})'
            cur.executemany(insert_sql, batch)
            conn.commit()
            pbar.update(len(batch))

    cur.close()
    conn.close()
    print(f"\n[Done] New table created: {SCHEMA_NAME}.{NEW_TABLE}")

# ===================== Main =====================
if __name__ == "__main__":
    df_origin = load_original_table()
    df_final = calc_final_score(df_origin)
    write_new_table(df_final)
    print("\n=== All done! Ready for spatial visualization in ArcGIS Pro! ===")
