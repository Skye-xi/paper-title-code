import asyncio
import aiohttp
import asyncpg  # Use async PostgreSQL driver
import time
import logging
# openai import is removed as it's not directly used for the API call
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import json  # For handling potential JSON parsing errors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    'server_ip': '<YOUR_VLLM_SERVER_IP>',
    'server_port': '8000',
    'openai_api_key': "<YOUR_VLLM_API_KEY>",  # API Key still required even for local deployment
    'database': 'postgres',
    'user': 'postgres',
    'password': '<YOUR_DB_PASSWORD>',
    'host': '<YOUR_DB_HOST>',
    'port': '5432',
    'table_name': 'bj2019_cleaned_part9_with_response',
    'new_table_name': 'bj2019_culture_part9_with_response_首都文化分类',
    'batch_size': 90,  # Try increasing batch_size to improve throughput; requires testing
    'max_concurrent_requests': 30,  # Limit concurrent requests to avoid overwhelming LLM service or network
    'db_pool_size': 10,  # Database connection pool size
    # Prompt template loaded from external .txt file. The prompt content should match paper Appendix A.
    'system_content_file': '<YOUR_PROMPT_DIR>/首都文化分类/XX文化_3.txt',
    'additional_column': '创新文化',
    'where_condition': "广告移除 = '1'",  # No extra quotes needed here when using parameterized queries
    'request_timeout': 30  # aiohttp request timeout (seconds)
}

# Read system content (prompt template)
def read_system_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.error(f"System content file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading system content file {file_path}: {e}")
        raise

# Asynchronously fetch model responses (with timeout and detailed error handling)
async def get_openai_responses(session, rows, system_content, semaphore):
    url = f"http://{CONFIG['server_ip']}:{CONFIG['server_port']}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {CONFIG['openai_api_key']}"}
    timeout = aiohttp.ClientTimeout(total=CONFIG['request_timeout'])

    async def fetch_response(row):
        # Use semaphore to limit concurrency
        async with semaphore:
            # Row structure assumed to be (id, tid, cleaned_content) -> indices 0, 1, 2
            if len(row) < 3 or not row[2]:  # Validate data
                logging.warning(f"Skipping invalid row data: {row}")
                return (row[1] if len(row) > 1 else None, None)  # Return tid (if exists) and None

            tid = row[1]
            content = row[2]

            try:
                data = {
                    "model": "Qwen3-32B",
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": content}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1
                }
                async with session.post(url, headers=headers, json=data, timeout=timeout) as response:
                    response.raise_for_status()  # Check HTTP error status (e.g., 4xx, 5xx)
                    try:
                        result = await response.json()
                        # Verify response structure matches expectations
                        if 'choices' in result and len(result['choices']) > 0 and \
                           'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                            return (tid, result['choices'][0]['message']['content'])
                        else:
                            logging.error(f"Unexpected response structure for tid {tid}: {result}")
                            return (tid, None)
                    except json.JSONDecodeError as json_err:
                        # Response is not valid JSON
                        response_text = await response.text()
                        logging.error(f"JSON decode error for tid {tid}. Status: {response.status}. Response text: {response_text[:500]}... Error: {json_err}")
                        return (tid, None)
            except aiohttp.ClientConnectorError as e:
                logging.error(f"Connection error for tid {tid}: {e}")
                return (tid, None)
            except aiohttp.ClientResponseError as e:
                logging.error(f"HTTP error for tid {tid}: Status {e.status}, Message: {e.message}")
                return (tid, None)
            except asyncio.TimeoutError:
                logging.error(f"Request timed out for tid {tid} after {CONFIG['request_timeout']} seconds.")
                return (tid, None)
            except Exception as e:
                # Catch all other possible errors
                logging.exception(f"Unexpected error processing tid {tid}: {e}")  # Use logging.exception to include traceback
                return (tid, None)

    tasks = [fetch_response(row) for row in rows]
    return await asyncio.gather(*tasks)

# Async database operations (using asyncpg)
async def handle_database_operations_async(pool, new_table_name, table_name, additional_column):
    async with pool.acquire() as conn:  # Acquire connection from pool
        async with conn.transaction():  # Use transaction for atomicity
            # Check if new table exists
            table_exists = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = $1);",
                new_table_name
            )
            if not table_exists:
                logging.info(f"Creating table {new_table_name} based on {table_name}...")
                # Note: f-string used here. If table name comes from untrusted source, SQL injection risk exists.
                # But in this context, table name comes from config, so risk is low.
                await conn.execute(f"CREATE TABLE {new_table_name} AS TABLE {table_name} WITH NO DATA;")
                logging.info(f"Table {new_table_name} created. Copying data...")
                # Using INSERT INTO ... SELECT ... to copy data may be more flexible than CREATE TABLE AS TABLE
                # This step may be time-consuming if the source table is large
                await conn.execute(f"INSERT INTO {new_table_name} SELECT * FROM {table_name};")
                logging.info(f"Data copied to {new_table_name}.")

            # Check if new column exists
            column_exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = $1 AND column_name = $2
                );
                """,
                new_table_name, additional_column
            )

            if column_exists:
                print(f"Column '{additional_column}' already exists in table '{new_table_name}'. Please select an option:")
                print("A. Clear and overwrite this column data")
                print("B. Continue writing to this column (skip rows with data - requires code support, or specify start position here)")
                print("C. Exit program")
                option = input("Please enter option (A/B/C): ").upper()

                if option == 'A':
                    logging.info(f"You chose to clear column '{additional_column}'.")
                    # Note: f-string used here as well
                    await conn.execute(f"UPDATE {new_table_name} SET {additional_column} = NULL;")
                    logging.info(f"Column '{additional_column}' cleared.")
                    return 0  # Return offset 0
                elif option == 'B':
                    logging.info(f"You chose to continue writing to column '{additional_column}'.")
                    while True:
                        try:
                            # For precise checkpoint resume, query the last non-NULL tid or id
                            # Simplified here: user inputs offset
                            offset_input = input("Please enter the data offset to start processing (number of rows completed based on source table query order): ")
                            offset = int(offset_input)
                            if offset >= 0:
                                logging.info(f"Will start processing from offset {offset}.")
                                return offset
                            else:
                                print("Offset must be a non-negative integer.")
                        except ValueError:
                            print("Please enter a valid number.")
                elif option == 'C':
                    logging.info("You chose to exit the program.")
                    return None  # Return None indicates exit
                else:
                    logging.warning("Invalid input, program exits.")
                    return None  # Return None indicates exit
            else:
                logging.info(f"Adding new column '{additional_column}' to table '{new_table_name}'.")
                # Note: f-string used here as well
                await conn.execute(f"ALTER TABLE {new_table_name} ADD COLUMN {additional_column} TEXT;")
                logging.info(f"Column '{additional_column}' added.")
                return 0  # Return offset 0

# Main function (using asyncpg and semaphore)
async def main():
    start_time = time.time()
    try:
        system_content = read_system_content(CONFIG['system_content_file'])
    except Exception:
        return  # Exit if system content file cannot be read

    pool = None  # Initialize pool variable
    try:
        # Create database connection pool
        pool = await asyncpg.create_pool(
            database=CONFIG['database'],
            user=CONFIG['user'],
            password=CONFIG['password'],
            host=CONFIG['host'],
            port=CONFIG['port'],
            min_size=1,
            max_size=CONFIG['db_pool_size']
        )
        logging.info("Database connection pool created.")

        # Process database tables and columns, get starting offset
        offset = await handle_database_operations_async(
            pool, CONFIG['new_table_name'], CONFIG['table_name'], CONFIG['additional_column']
        )
        if offset is None:  # User chose to exit
            if pool: await pool.close()
            return

        # Get total row count (only matching condition)
        async with pool.acquire() as conn:
            # Using parameterized query is safer
            total_rows_query = f'SELECT COUNT(*) FROM "{CONFIG["table_name"]}" WHERE {CONFIG["where_condition"]};'
            # Note: If WHERE condition is dynamic, ensure its safety or use more complex parameterization
            total_rows = await conn.fetchval(total_rows_query)
            if total_rows is None: total_rows = 0

        logging.info(f"Total rows matching condition '{CONFIG['where_condition']}': {total_rows}")
        rows_to_process = total_rows - offset
        if rows_to_process <= 0:
            logging.info("No rows need processing based on the offset.")
            if pool: await pool.close()
            return

        # Create aiohttp Session and Semaphore
        async with aiohttp.ClientSession() as session:
            # Create Semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(CONFIG['max_concurrent_requests'])

            with Progress(
                "[progress.description]{task.description}",
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.0f}%", "•",
                TextColumn("[bold green]{task.completed} rows"), "of",
                TextColumn("[bold cyan]{task.total} total rows"), "•",
                TimeElapsedColumn(), "•", TimeRemainingColumn(),
                # transient=True  # Set transient=True to hide progress bar after completion; remove to view final result
            ) as progress:
                task_description = f"Processing rows (Batch Size: {CONFIG['batch_size']}, Max Concurrent: {CONFIG['max_concurrent_requests']})"
                task = progress.add_task(task_description, total=rows_to_process)

                processed_count = 0
                while processed_count < rows_to_process:
                    batch_limit = CONFIG['batch_size']
                    # Async fetch data
                    async with pool.acquire() as conn:
                        # Use parameterized query
                        # ORDER BY tid ensures pagination consistency
                        fetch_query = f'''
                            SELECT id, tid, cleaned_content
                            FROM "{CONFIG["table_name"]}"
                            WHERE {CONFIG["where_condition"]}
                            ORDER BY tid
                            LIMIT $1 OFFSET $2
                        '''
                        rows = await conn.fetch(fetch_query, batch_limit, offset + processed_count)

                    if not rows:
                        logging.info("No more rows found in the database matching the criteria.")
                        break  # No more data

                    # Async fetch LLM responses
                    responses = await get_openai_responses(session, rows, system_content, semaphore)

                    # Prepare update data
                    update_data = []
                    for tid, response_content in responses:
                        if tid is not None and response_content is not None:
                            update_data.append((response_content, tid))
                        elif tid is not None:
                            logging.warning(f"Received null response for tid {tid}, skipping update.")

                    # Async batch update database
                    if update_data:
                        try:
                            async with pool.acquire() as conn:
                                async with conn.transaction():  # Execute batch update in transaction
                                    # Use executemany for efficient batch updates
                                    update_query = f"UPDATE {CONFIG['new_table_name']} SET {CONFIG['additional_column']} = $1 WHERE tid = $2"
                                    await conn.executemany(update_query, update_data)
                            logging.debug(f"Updated {len(update_data)} rows in the database.")
                        except Exception as db_update_err:
                            logging.error(f"Database update failed for batch starting at offset {offset + processed_count}: {db_update_err}")
                            # Consider retrying or recording failed tids
                            # For simplicity, continue to next batch but log the error

                    actual_processed_in_batch = len(rows)
                    processed_count += actual_processed_in_batch
                    progress.update(task, advance=actual_processed_in_batch)
                    logging.debug(f"Processed batch of {actual_processed_in_batch}. Total processed: {processed_count}")

                    # No longer need asyncio.sleep(0.1) since DB operations are now non-blocking

                # Ensure progress bar shows completion
                progress.update(task, completed=rows_to_process)

    except asyncpg.exceptions.PostgresError as e:
        logging.error(f"Database error: {e}")
    except aiohttp.ClientError as e:
        logging.error(f"HTTP Client error during setup or infrequent operations: {e}")
    except Exception as e:
        logging.exception(f"An unexpected error occurred in main: {e}")  # Use exception to record traceback
    finally:
        if pool:
            await pool.close()
            logging.info("Database connection pool closed.")

    end_time = time.time()
    logging.info(f"Script finished. Total running time: {end_time - start_time:.2f} seconds.")

# Run main program
if __name__ == "__main__":
    # Install asyncpg: pip install asyncpg
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Script interrupted by user.")
