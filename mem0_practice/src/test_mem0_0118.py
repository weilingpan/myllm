import os
import logging
from openai import OpenAI
from pymilvus import connections, db

os.environ["MEM0_TELEMETRY"] = "false"
from mem0 import Memory


def get_openai_client():
    try:
        from env import OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    except ImportError:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    return OpenAI(api_key=OPENAI_API_KEY)


def get_config(database_name="test_mem0_db", collection_name="test_mem0_collection"):
    return {
        # https://github.com/mem0ai/mem0/blob/dba7f0458aeb50aa7078d36eaefa2405afbee620/mem0/configs/vector_stores/milvus.py#L22
        "vector_store": {
            "provider": "milvus",
            "config": {
                "collection_name": collection_name,
                # "embedding_model_dims": ,  # 使用項目的向量維度
                "url": os.environ.get("MILVUS_URI", "http://milvus-standalone:19530"),
                "token": "",
                "db_name": database_name,
            },
        },
        "version": "v1.1",
    }


def init_milvus_db(config):
    milvus_config = config["vector_store"]["config"]
    milvus_url = milvus_config["url"]
    milvus_db_name = milvus_config.get("db_name")

    if milvus_db_name and not milvus_url.startswith("./"):  # Only for server-based Milvus
        print(f"Checking database '{milvus_db_name}' in Milvus at {milvus_url}...")
        try:
            connections.connect(uri=milvus_url)
            if milvus_db_name not in db.list_database():
                print(f"Creating database: {milvus_db_name}")
                db.create_database(milvus_db_name)
            else:
                print(f"Database '{milvus_db_name}' already exists.")
        except Exception as e:
            print(f"Warning: Failed to initialize Milvus database: {e}")

def clean_up_milvus_db(config):
    milvus_config = config["vector_store"]["config"]
    milvus_url = milvus_config["url"]
    milvus_db_name = milvus_config.get("db_name")

    if milvus_db_name and not milvus_url.startswith("./"):  # Only for server-based Milvus
        print(f"Cleaning up database '{milvus_db_name}' in Milvus at {milvus_url}...")
        try:
            connections.connect(uri=milvus_url)
            if milvus_db_name in db.list_database():
                # Switch to the target database
                db.using_database(milvus_db_name)
                from pymilvus import utility
                collections = utility.list_collections()
                for coll in collections:
                    print(f"Dropping collection: {coll}")
                    utility.drop_collection(coll)
                db.drop_database(milvus_db_name)
                print(f"Database '{milvus_db_name}' has been dropped.")
            else:
                print(f"Database '{milvus_db_name}' does not exist.")
        except Exception as e:
            print(f"Warning: Failed to clean up Milvus database: {e}")

def add_memory(m, user_id: str, content: str):
    res = m.add(
        messages=content,
        user_id=user_id,
        metadata={"category": "fact"},
    )
    return res


def get_memory(m, user_id: str):
    # get all memories for user
    memories = m.get_all(user_id=user_id)
    print(f"\n目前記憶內容:\n{memories}")
    return memories


def search_memory(m, user_id: str, query: str, top_k: int = 5):
    memories = m.search(query=query, user_id=user_id)
    print(f"\n搜尋到的記憶內容:\n{memories}")
    return memories


def chat_with_memory(client, m, user_id: str, text: str, model_name="gpt-4o"):
    memories = get_memory(m, user_id)

    existed_memory = memories.get("results", [])
    if existed_memory:
        relevant_memories = search_memory(m, user_id, text)

    add_memory(m, user_id, text)

    system_prompt = f"You are a helpful AI. Answer the question based on query."
    if relevant_memories["results"]:
        memories = "\n".join(
            f"- {entry['memory']}" for entry in relevant_memories["results"]
        )
        system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser memories:\n{memories}"
    print(f"\nSystem Prompt:\n{system_prompt}")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


def main():
    logging.basicConfig(level=logging.INFO)  # 或 logging.DEBUG
    # logger = logging.getLogger(__name__)

    database_name = "test_mem0_db"
    collection_name = "test_mem0_collection"

    memory_config = get_config(database_name, collection_name)
    client = get_openai_client()

    init_milvus_db(memory_config)
    m = Memory.from_config(memory_config)

    user_id = "regina_pan"
    print(f"歡迎 {user_id}！輸入 'exit' 結束對話。")

    while True:
        try:
            question = input(f"\n{user_id} 問: ").strip()
            if question.lower() == "exit":
                confirm = input("是否要清除 Milvus DB? (y/N): ").strip().lower()
                if confirm == "y":
                    clean_up_milvus_db(memory_config)
                print("Bye!")
                break

            if not question:
                continue

            response = chat_with_memory(client, m, user_id, question)
            print(f"\n答:\n{response}")
        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    main()

