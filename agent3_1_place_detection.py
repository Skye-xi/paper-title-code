import openai
import psycopg2
import asyncio
import aiohttp
import time
import logging
from openai import OpenAI
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    'server_ip': '<YOUR_VLLM_SERVER_IP>',
    'server_port': '8000',
    'openai_api_key': "<YOUR_VLLM_API_KEY>",
    'database': 'postgres',
    'user': 'postgres',
    'password': '<YOUR_DB_PASSWORD>',
    'host': '<YOUR_DB_HOST>',
    'port': '5432',
    'table_name': 'bj2019_culture_part3_with_response_01红色已对齐',
    'new_table_name': 'bj2019_culture_part3_with_response_04红色地点区分',
    'batch_size': 1,
    # Prompt template loaded from external .txt file. The prompt content should match paper Appendix A.
    'system_content_file': '<YOUR_PROMPT_DIR>/04区分地点和非地点.txt',
    'additional_column': '地点判断',  # New column name
    'where_condition': '广告移除 = \'1\''  # Query condition
}

# Read system content (prompt template)
def read_system_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

# Asynchronously fetch model responses
async def get_openai_responses(session, rows, system_content):
    url = f"http://{CONFIG['server_ip']}:{CONFIG['server_port']}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {CONFIG['openai_api_key']}"}

    async def fetch_response(row):
        try:
            data = {
                "model": "Qwen1.5-72B-Chat-AWQ",
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": row[3]}
                ],
                "temperature": 0.0,
            }
            async with session.post(url, headers=headers, json=data) as response:
                result = await response.json()
                return (row[1], result['choices'][0]['message']['content'])
        except Exception as e:
            logging.error(f"Error processing row {row[1]}: {e}")
            return (row[1], None)

    tasks = [fetch_response(row) for row in rows]
    return await asyncio.gather(*tasks)

# Handle database operations
def handle_database_operations(cur, new_table_name, table_name, additional_column):
    cur.execute("SELECT EXISTS (SELECT FROM pg_tables WHERE tablename = %s);", (new_table_name,))
    if not cur.fetchone()[0]:
        cur.execute(f"CREATE TABLE {new_table_name} AS TABLE {table_name};")

    column_exists_query = "SELECT column_name FROM information_schema.columns WHERE table_name=%s AND column_name=%s;"
    cur.execute(column_exists_query, (new_table_name, additional_column))

    if cur.fetchone():
        print(f"Column '{additional_column}' already exists. Please select an option:")
        print("A. Overwrite existing column")
        print("B. Continue writing to existing column")
        print("C. Exit and rename column")
        option = input("Please enter option (A/B/C): ").upper()

        if option == 'A':
            print("You chose to overwrite the column. Column content will be cleared and overwritten.")
            cur.execute(f"UPDATE {new_table_name} SET {additional_column} = NULL;")
            return 0
        elif option == 'B':
            print("You chose to continue writing to the existing column.")
            while True:
                try:
                    offset = int(input("Please enter the data offset to start writing (last completed row count): "))
                    if offset >= 0:
                        return offset
                    else:
                        print("Offset must be a non-negative integer.")
                except ValueError:
                    print("Please enter a valid number.")
        elif option == 'C':
            print("You chose to exit the program. Please modify the new column name.")
            return None
        else:
            print("Invalid input, program exits.")
            return None
    else:
        cur.execute(f"ALTER TABLE {new_table_name} ADD COLUMN {additional_column} TEXT;")
        return 0

# Main function
async def main():
    start_time = time.time()
    system_content = read_system_content(CONFIG['system_content_file'])

    try:
        with psycopg2.connect(database=CONFIG['database'], user=CONFIG['user'], password=CONFIG['password'], host=CONFIG['host'], port=CONFIG['port']) as conn:
            with conn.cursor() as cur:
                offset = handle_database_operations(cur, CONFIG['new_table_name'], CONFIG['table_name'], CONFIG['additional_column'])
                if offset is None:
                    return

                cur.execute(f'SELECT COUNT(*) FROM "{CONFIG["table_name"]}";')
                total_rows = cur.fetchone()[0]

                with Progress(
                    "[progress.description]{task.description}",
                    BarColumn(bar_width=None),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    "•",
                    TextColumn("[bold green]{task.completed} rows"),
                    "of",
                    TextColumn("[bold cyan]{task.total} total rows"),
                    "•",
                    TimeElapsedColumn(),
                    "•",
                    TimeRemainingColumn(),
                    transient=True
                ) as progress:
                    task = progress.add_task(f"Processing rows...", total=total_rows - offset)

                    async with aiohttp.ClientSession() as session:
                        while True:
                            try:
                                cur.execute(f'SELECT id,tid,载体,对齐 FROM "{CONFIG["table_name"]}" ORDER BY tid LIMIT %s OFFSET %s', (CONFIG['batch_size'], offset))
                                rows = cur.fetchall()
                                if not rows:
                                    break

                                responses = await get_openai_responses(session, rows, system_content)

                                for tid, response_content in responses:
                                    if response_content is not None:
                                        cur.execute(f"UPDATE {CONFIG['new_table_name']} SET {CONFIG['additional_column']} = %s WHERE tid = %s", (response_content, tid))
                                conn.commit()
                                progress.update(task, advance=len(rows))
                                offset += CONFIG['batch_size']
                                await asyncio.sleep(0.1)

                            except psycopg2.OperationalError as e:
                                logging.error(f"Database operational error: {e}")
                                # Reconnect to database
                                conn = psycopg2.connect(database=CONFIG['database'], user=CONFIG['user'], password=CONFIG['password'], host=CONFIG['host'], port=CONFIG['port'])
                                cur = conn.cursor()
                                logging.info("Reconnected to the database.")
                            except Exception as e:
                                logging.error(f"Unexpected error: {e}")
                                break

                        progress.update(task, completed=total_rows - offset)

    except psycopg2.DatabaseError as e:
        logging.error(f"Database error: {e}")
    except openai.OpenAIError as e:  # Fixed OpenAI exception handling
        logging.error(f"OpenAI API error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    end_time = time.time()
    logging.info(f"\nRunning time: {end_time - start_time:.2f} seconds.")

# Run main program
if __name__ == "__main__":
    asyncio.run(main())
