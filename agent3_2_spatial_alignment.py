import openai
import psycopg2
import asyncio
import aiohttp
import time
import logging
import warnings  # Added: import warnings module
from openai import OpenAI
import re

# Added: the following line is the original warning filter code, now works properly
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


# ===================== Simple progress bar implementation =====================
class SimpleProgressBar:
    def __init__(self, total, bar_length=50, color="red"):
        self.total = total
        self.bar_length = bar_length
        self.current = 0
        self.color_codes = {
            "red": "\033[31m",
            "bright_red": "\033[91m",
            "reset": "\033[0m"
        }
        self.color = self.color_codes[color]
        self.bright_color = self.color_codes["bright_red"]
        self.reset = self.color_codes["reset"]

    def update(self, increment=1):
        self.current += increment
        self._draw()

    def _draw(self):
        percent = (self.current / self.total) * 100
        filled_length = int(self.bar_length * self.current // self.total)
        bar = self.bright_color + '█' * filled_length + self.color + '-' * (
                self.bar_length - filled_length) + self.reset
        print(f'\r{bar} {percent:.1f}%', end='', flush=True)

    def finish(self):
        self.current = self.total
        self._draw()
        print()


# ===================== Core configuration (fitting your business logic) =====================
# Business config: prioritize high-frequency carriers, reuse low-frequency ones directly
BUSINESS_CONFIG = {
    'high_freq_threshold': 2,  # High-frequency carrier threshold: standardize only carriers appearing >=2 times
    'standard_mapping': {  # Custom standard mapping (extendable as needed)
        # Forbidden City related
        '故宫': '故宫',
        '紫禁城': '故宫',
        '故宫博物院': '故宫',
        '故宫博物馆': '故宫',
        '故官': '故宫',  # Typo
        '紫荊城': '故宫',  # Traditional Chinese character
        # Additional location mappings can be added
        '长城': '长城',
        '八达岭长城': '长城',
        '慕田峪长城': '长城'
    }
}

# Training set / database configuration
TRAIN_CONFIG = {
    'train_tables': [
        "bj2019_culture_part5_with_response_01创新已对齐",
        "bj2019_culture_part5_with_response_01古都已对齐",
        "bj2019_culture_part5_with_response_01红色已对齐",
        "bj2019_culture_part5_with_response_01京味已对齐"
    ],
    'text_col': '载体',  # Carrier column name
    'label_col': '对齐',  # Alignment result column name
    'align_mode': 'hybrid'  # Alignment mode: llm (pure LLM) / hybrid (high-freq mapping + LLM)
}

# Core database / LLM configuration
CONFIG = {
    'server_ip': '<YOUR_VLLM_SERVER_IP>',
    'server_port': '8000',
    'openai_api_key': "<YOUR_VLLM_API_KEY>",
    'database': 'postgres',
    'user': 'postgres',
    'password': '<YOUR_DB_PASSWORD>',
    'host': '<YOUR_DB_HOST>',
    'port': '5432',
    'table_name': 'shihao.bj2019_culture_part6_10_with_response_首都文化分类_全1',
    'batch_size': 30,
    # Prompt template loaded from external .txt file. The prompt content should match paper Appendix A.
    'system_content_file': '<YOUR_PROMPT_DIR>/00对齐（实验）.txt',
    'additional_column': '对齐',
    'where_condition': '区分 = \'1\'',
    'target_column': '载体',
    'request_timeout': 60,
    'max_retries': 3,
    'retry_delay': 2
}


# ===================== Step 1: Build high-frequency carrier mapping table =====================
def build_high_freq_mapping():
    """
    Build from training set:
    1. Standardized mapping for high-frequency carriers (occurrence >= threshold)
    2. Low-frequency carriers directly return original content
    """
    # 1. Count occurrences of all carriers
    carrier_count = {}
    # 2. Training set carrier -> alignment result mapping
    train_carrier_align = {}

    conn = None
    cur = None

    try:
        conn = psycopg2.connect(
            database=CONFIG['database'],
            user=CONFIG['user'],
            password=CONFIG['password'],
            host=CONFIG['host'],
            port=CONFIG['port']
        )
        conn.autocommit = True
        cur = conn.cursor()

        # Iterate through training tables to count carriers + build mapping
        for table_name in TRAIN_CONFIG['train_tables']:
            try:
                cur.execute(f"""
                    SELECT "{TRAIN_CONFIG['text_col']}", "{TRAIN_CONFIG['label_col']}"
                    FROM "{table_name}"
                    WHERE "{TRAIN_CONFIG['text_col']}" IS NOT NULL 
                    AND "{TRAIN_CONFIG['label_col']}" IS NOT NULL;
                """)
                rows = cur.fetchall()

                for carrier, align_result in rows:
                    carrier = str(carrier).strip()
                    align_result = str(align_result).strip()

                    # Count carrier occurrences
                    carrier_count[carrier] = carrier_count.get(carrier, 0) + 1
                    # Build carrier -> alignment result mapping (dedup, keep first occurrence)
                    if carrier not in train_carrier_align:
                        train_carrier_align[carrier] = align_result

                logging.info(f"Successfully processed training table {table_name}: {len(rows)} carrier records")

            except Exception as e:
                logging.error(f"Failed processing training table {table_name}: {e}, skipped")
                continue

        # 3. Build final mapping table
        final_mapping = {}
        high_freq_carriers = []
        low_freq_carriers = []

        for carrier, count in carrier_count.items():
            if count >= BUSINESS_CONFIG['high_freq_threshold']:
                # High-frequency carrier: prioritize custom standard mapping, else use training set alignment
                standard_name = BUSINESS_CONFIG['standard_mapping'].get(carrier,
                                                                        train_carrier_align.get(carrier, carrier))
                final_mapping[carrier] = standard_name
                high_freq_carriers.append((carrier, count, standard_name))
            else:
                # Low-frequency carrier: directly return original carrier content
                final_mapping[carrier] = carrier
                low_freq_carriers.append(carrier)

        # Print statistics
        logging.info(f"\n=== Carrier Statistics ===")
        logging.info(f"Total carriers: {len(carrier_count)}")
        logging.info(f"High-frequency carriers (>={BUSINESS_CONFIG['high_freq_threshold']} occurrences): {len(high_freq_carriers)}")
        logging.info(f"Low-frequency carriers: {len(low_freq_carriers)}")
        logging.info(f"\nHigh-frequency carrier standardization examples:")
        for carrier, count, standard in high_freq_carriers[:5]:
            logging.info(f"  {carrier} ({count} occurrences) -> {standard}")

        return final_mapping

    except Exception as e:
        logging.error(f"Failed building mapping table: {e}")
        return None
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# ===================== Step 2: Core carrier alignment function =====================
def align_carrier(carrier, mapping_table):
    """
    Carrier alignment logic:
    1. First check mapping table (high-freq standardized, low-freq returned as-is)
    2. No mapping found -> return None (delegated to LLM)
    """
    try:
        carrier = str(carrier).strip()
        if not carrier:
            return None

        # Prioritize mapping table lookup
        if carrier in mapping_table:
            return mapping_table[carrier]
        else:
            # New carrier without mapping, return None (delegated to LLM)
            return None
    except Exception as e:
        logging.error(f"Carrier alignment failed: {carrier} -> {e}")
        return None


# ===================== Read system prompt =====================
def read_system_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


# ===================== Async LLM call with retry =====================
async def fetch_with_retry(session, tid, carrier_content, system_content, max_retries, retry_delay, timeout):
    for attempt in range(max_retries):
        try:
            data = {
                "model": "Qwen3-32B",
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": carrier_content}
                ],
                "temperature": 0.0,
            }
            async with session.post(
                    url=f"http://{CONFIG['server_ip']}:{CONFIG['server_port']}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {CONFIG['openai_api_key']}"},
                    json=data,
                    timeout=timeout
            ) as response:
                if response.status != 200:
                    raise Exception(f"Server returned error status code: {response.status}")
                result = await response.json()
                return (tid, result['choices'][0]['message']['content'].strip())
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Attempt {attempt + 1} for tid={tid} failed: {e}, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                logging.error(f"tid={tid} failed after {max_retries} retries: {e}")
                return (tid, None)


# ===================== Async fetch alignment results =====================
async def get_align_responses(session, rows, system_content, mapping_table=None):
    """
    Alignment logic:
    - hybrid mode: check mapping table first, call LLM if no result
    - llm mode: always call LLM
    """
    align_mode = TRAIN_CONFIG['align_mode']
    responses = []
    tasks = []

    if align_mode == 'hybrid' and mapping_table is not None:
        # Hybrid mode: mapping table prioritized
        for row in rows:
            tid, carrier_content = row[0], row[1]
            # First check mapping table
            align_result = align_carrier(carrier_content, mapping_table)
            if align_result is not None:
                responses.append((tid, align_result))
            else:
                # No mapping, call LLM
                tasks.append(
                    fetch_with_retry(
                        session,
                        tid,
                        carrier_content,
                        system_content,
                        CONFIG['max_retries'],
                        CONFIG['retry_delay'],
                        CONFIG['request_timeout']
                    )
                )
    else:
        # Pure LLM mode
        tasks = [
            fetch_with_retry(
                session,
                row[0],
                row[1],
                system_content,
                CONFIG['max_retries'],
                CONFIG['retry_delay'],
                CONFIG['request_timeout']
            ) for row in rows
        ]

    # Execute LLM tasks
    if tasks:
        llm_responses = await asyncio.gather(*tasks)
        responses.extend(llm_responses)

    return responses


# ===================== Prepare target table (fix: compatible with schema-qualified table names) =====================
def prepare_table(cur, table_name, additional_column):
    # Split schema and table name, compatible with schema-qualified names
    if '.' in table_name:
        schema_name, table_name_only = table_name.split('.', 1)
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s AND column_name = %s;
        """, (schema_name, table_name_only, additional_column))
    else:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = %s AND column_name = %s;
        """, (table_name, additional_column))

    if cur.fetchone():
        print(f"Column '{additional_column}' already exists.")
        print("A. Clear and rerun")
        print("B. Continue from last checkpoint")
        option = input("Please select (A/B): ").upper()

        if option == 'A':
            cur.execute(f"UPDATE {table_name} SET {additional_column} = NULL;")
            return 0
        elif option == 'B':
            while True:
                try:
                    offset = int(input("Please enter the offset from last session: "))
                    if offset >= 0:
                        return offset
                except ValueError:
                    print("Enter a valid number!")
        else:
            return None
    else:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {additional_column} TEXT;")
        return 0


# ===================== Main function (core fix: proper quoting of schema-qualified table names in SQL) =====================
async def main():
    start_time = time.time()
    system_content = read_system_content(CONFIG['system_content_file'])
    target_col = CONFIG['target_column']
    table_name = CONFIG['table_name']
    align_mode = TRAIN_CONFIG['align_mode']

    # Split schema and table name (key: handle quoting for schema-qualified table names)
    schema_name = None
    table_name_only = None
    if '.' in table_name:
        schema_name, table_name_only = table_name.split('.', 1)
        # Build quoted table name: schema."table_name"
        quoted_table_name = f"{schema_name}.\"{table_name_only}\""
    else:
        table_name_only = table_name
        quoted_table_name = f"\"{table_name_only}\""

    # Initialize mapping table
    mapping_table = None

    # Hybrid mode: build mapping table first
    if align_mode == 'hybrid':
        logging.info("Building high-frequency carrier standardization mapping table from training set...")
        mapping_table = build_high_freq_mapping()
        if mapping_table is None:
            logging.error("Mapping table build failed, automatically switching to pure LLM mode")
            align_mode = 'llm'

    try:
        # Connect to target table database
        conn = psycopg2.connect(
            database=CONFIG['database'],
            user=CONFIG['user'],
            password=CONFIG['password'],
            host=CONFIG['host'],
            port=CONFIG['port']
        )
        cur = conn.cursor()

        # Validate carrier column (fix: compatible with schema-qualified table names)
        if schema_name:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s AND column_name = %s;
            """, (schema_name, table_name_only, target_col))
        else:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = %s AND column_name = %s;
            """, (table_name_only, target_col))

        if not cur.fetchone():
            logging.error(f"Error: column '{target_col}' does not exist in table {table_name}!")
            conn.close()
            return

        # Prepare target table (create/clear alignment column)
        offset = prepare_table(cur, table_name, CONFIG['additional_column'])
        if offset is None:
            conn.close()
            return

        # Count total rows (core fix: use correctly quoted table name)
        count_sql = f"""
            SELECT COUNT(*) FROM {quoted_table_name} WHERE {CONFIG['where_condition']};
        """
        cur.execute(count_sql)
        total_rows = cur.fetchone()[0]
        print(f"\nStarting target table processing, total rows to process: {total_rows}")
        print(f"Alignment mode: {align_mode.upper()}")

        # Initialize progress bar
        progress_bar = SimpleProgressBar(total=total_rows - offset, bar_length=50, color="red")

        # Configure LLM session
        timeout = aiohttp.ClientTimeout(total=CONFIG['request_timeout'])
        async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=10)) as session:
            while True:
                try:
                    # Batch read target table data (core fix: use correctly quoted table name)
                    select_sql = f"""
                        SELECT tid, "{target_col}" FROM {quoted_table_name}
                        WHERE {CONFIG['where_condition']}
                        ORDER BY tid
                        LIMIT %s OFFSET %s;
                    """
                    cur.execute(select_sql, (CONFIG['batch_size'], offset))

                    rows = cur.fetchall()
                    if not rows:
                        break

                    # Get alignment results
                    responses = await get_align_responses(
                        session, rows, system_content, mapping_table
                    )

                    # Batch update database (core fix: use correctly quoted table name)
                    success_count = 0
                    update_sql = f"""
                        UPDATE {quoted_table_name}
                        SET {CONFIG['additional_column']} = %s
                        WHERE tid = %s;
                    """
                    for tid, resp in responses:
                        if resp is not None:
                            cur.execute(update_sql, (resp, tid))
                            success_count += 1

                    conn.commit()
                    # Update progress bar
                    progress_bar.update(increment=len(rows))
                    # Log progress
                    current_percent = (progress_bar.current / progress_bar.total) * 100
                    logging.info(f"Progress: {current_percent:.1f}% | Batch complete: {success_count}/{len(rows)} rows processed")

                    offset += CONFIG['batch_size']
                    await asyncio.sleep(0.5)

                except psycopg2.OperationalError:
                    logging.warning("Database disconnected, reconnecting...")
                    conn.close()
                    conn = psycopg2.connect(
                        database=CONFIG['database'], user=CONFIG['user'],
                        password=CONFIG['password'], host=CONFIG['host'], port=CONFIG['port']
                    )
                    cur = conn.cursor()
                except Exception as e:
                    logging.error(f"Batch processing failed: {e}")
                    conn.rollback()
                    await asyncio.sleep(5)
                    continue

        # Finish progress bar
        progress_bar.finish()
        print(f"\n[OK] Target table alignment complete! Total time: {time.time() - start_time:.2f}s")
        print(f"[Logic summary:]")
        print(f"   1. High-frequency carriers (>={BUSINESS_CONFIG['high_freq_threshold']} occurrences): standardized alignment (e.g., 紫禁城 -> 故宫)")
        print(f"   2. Low-frequency carriers (<{BUSINESS_CONFIG['high_freq_threshold']} occurrences): retained as-is")
        print(f"   3. New carriers without mapping: delegated to LLM")

    except Exception as e:
        logging.error(f"Fatal program error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()


if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
