import openai
import psycopg2
import asyncio
import aiohttp
import time
from openai import OpenAI
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# Record start time
start_time = time.time()

# vLLM server IP and port
server_ip = '<YOUR_VLLM_SERVER_IP>'
server_port = '8000'

# Set OpenAI API key and API base to use vLLM API server
openai_api_key = "<YOUR_VLLM_API_KEY>"  # Provide API key if needed
openai_api_base = f"http://{server_ip}:{server_port}/v1"

# Create OpenAI client instance
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Database connection info
database = 'postgres'
user = 'postgres'
password = '<YOUR_DB_PASSWORD>'
host = '<YOUR_DB_HOST>'
port = '5432'
table_name = 'bj2019_culture_part7_with_response_首都文化分类'
new_table_name = 'bj2019_culture_part7_with_response_首都文化分类_文化载体'
batch_size = 50  # Adjust batch size as needed

# Read system content (prompt template) from txt file.
# Prompt template loaded from external .txt file. The prompt content should match paper Appendix A.
with open('<YOUR_PROMPT_DIR>/首都文化分类/红色文化分类&载体提取.txt', 'r', encoding='utf-8') as f:
    system_content = f.read().strip()

async def get_openai_responses(session, rows):
    """Asynchronously fetch model responses"""
    url = f"{openai_api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_api_key}"
    }

    async def fetch_response(row):
        try:
            data = {
                "model": "Qwen3-32B",
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": row[2]}
                ],
                "temperature": 0.0,
            }
            async with session.post(url, headers=headers, json=data) as response:
                result = await response.json()
                return (row[1], result['choices'][0]['message']['content'])
        except Exception as e:
            print(f"Error processing row {row[1]}: {e}")
            return (row[1], None)

    tasks = [fetch_response(row) for row in rows]
    return await asyncio.gather(*tasks)

async def main():
    offset = 0
    additional_column = "红色文化_物质文化载体"
    where_condition: '红色文化 = \'1\''  # Query condition
    # Use resource management to ensure proper closing of database connection and cursor
    try:
        with psycopg2.connect(database=database, user=user, password=password, host=host, port=port) as conn:
            with conn.cursor() as cur:
                # Check if new table exists, create and copy data if not
                cur.execute("SELECT EXISTS (SELECT FROM pg_tables WHERE tablename = %s);", (new_table_name,))
                if not cur.fetchone()[0]:
                    cur.execute(f"CREATE TABLE {new_table_name} AS TABLE {table_name};")

                # Check if column exists
                column_exists_query = "SELECT column_name FROM information_schema.columns WHERE table_name=%s AND column_name=%s;"
                cur.execute(column_exists_query, (new_table_name, additional_column))

                if cur.fetchone():
                    # Column exists, provide options to user
                    print(f"Column '{additional_column}' already exists. Please select an option:")
                    print("A. Overwrite existing column")
                    print("B. Continue writing to existing column")
                    print("C. Exit and rename column")
                    option = input("Please enter option (A/B/C): ").upper()

                    if option == 'A':
                        # Overwrite existing column
                        print("You chose to overwrite the column. Column content will be cleared and overwritten.")
                        cur.execute(f"UPDATE {new_table_name} SET {additional_column} = NULL;")
                        offset = 0
                    elif option == 'B':
                        # Continue writing
                        print("You chose to continue writing to the existing column.")
                        while True:
                            try:
                                offset = int(input("Please enter the data offset to start writing (last completed row count): "))
                                if offset >= 0:
                                    break
                                else:
                                    print("Offset must be a non-negative integer.")
                            except ValueError:
                                print("Please enter a valid number.")
                    elif option == 'C':
                        # Exit
                        print("You chose to exit the program. Please modify the new column name.")
                        return
                    else:
                        print("Invalid input, program exits.")
                        return
                else:
                    # Column does not exist, add it directly
                    cur.execute(f"ALTER TABLE {new_table_name} ADD COLUMN {additional_column} TEXT;")
                    offset = 0.0
                # Get total row count
                cur.execute(f'SELECT COUNT(*) FROM "{table_name}" WHERE "京味文化" = \'1\';')
                total_rows = cur.fetchone()[0]

                # Use rich progress bar
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
                            # Read data from database, ensure both tid and id are fetched
                            cur.execute(f'SELECT id, tid, cleaned_content FROM "{table_name}" WHERE "京味文化" = \'1\' ORDER BY tid LIMIT %s OFFSET %s', (batch_size, offset))
                            rows = cur.fetchall()
                            if not rows:
                                break  # No more data, exit loop

                            # Get model responses
                            responses = await get_openai_responses(session, rows)

                            # Add response results to table, using tid to locate records to update
                            for tid, response_content in responses:
                                if response_content is not None:
                                    cur.execute(f"UPDATE {new_table_name} SET {additional_column} = %s WHERE tid = %s", (response_content, tid))
                            conn.commit()
                            progress.update(task, advance=len(rows))  # Update progress bar

                            # Update offset
                            offset += batch_size

                            await asyncio.sleep(0.1)

                        # Ensure the progress bar shows 100% completion
                        progress.update(task, completed=total_rows - offset)

    except psycopg2.DatabaseError as e:
        print(f"Database error: {e}")
    except openai.Error as e:
        print(f"OpenAI API error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Run main program
asyncio.run(main())

# Record end time
end_time = time.time()

# Calculate and print running time
runtime = end_time - start_time
print(f"\nRunning time: {runtime:.2f} seconds.")
