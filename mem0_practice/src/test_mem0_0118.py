import os
import logging
from openai import OpenAI
from pymilvus import connections, db

os.environ["MEM0_TELEMETRY"] = "false"
from mem0 import Memory

import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info(f"Running file: {os.path.abspath(sys.argv[0])}")

def get_openai_client():
    try:
        from env import OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    except ImportError:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    return OpenAI(api_key=OPENAI_API_KEY)


def get_config(
        database_name: str, 
        collection_name: str,
        model_name: str,
        embedding_model: str):
    return {
        # https://github.com/mem0ai/mem0/blob/dba7f0458aeb50aa7078d36eaefa2405afbee620/mem0/configs/vector_stores/milvus.py#L22
        "vector_store": {
            "provider": "milvus",
            "config": {
                "collection_name": collection_name,
                "url": os.environ.get("MILVUS_URI", "http://milvus-standalone:19530"),
                "token": "",
                "db_name": database_name,
            },
        },
        # #https://github.com/mem0ai/mem0/blob/main/mem0/configs/llms
        "llm": {
            "provider": "openai",
            "config": {
                "model": model_name,
            }
        },
        #https://github.com/mem0ai/mem0/blob/dba7f0458aeb50aa7078d36eaefa2405afbee620/mem0/configs/embeddings/base.py#L10
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embedding_model,
            }
        },
        "version": "v1.1",
    }


def init_milvus_db(config):
    milvus_config = config["vector_store"]["config"]
    milvus_url = milvus_config["url"]
    milvus_db_name = milvus_config.get("db_name")

    if milvus_db_name and not milvus_url.startswith("./"):  # Only for server-based Milvus
        logger.info(f"Checking database '{milvus_db_name}' in Milvus at {milvus_url}...")
        try:
            connections.connect(uri=milvus_url)
            if milvus_db_name not in db.list_database():
                logger.info(f"Creating database: {milvus_db_name}")
                db.create_database(milvus_db_name)
            else:
                logger.info(f"Database '{milvus_db_name}' already exists.")
        except Exception as e:
            logger.warning(f"Failed to initialize Milvus database: {e}")

def clean_up_milvus_db(config):
    milvus_config = config["vector_store"]["config"]
    milvus_url = milvus_config["url"]
    milvus_db_name = milvus_config.get("db_name")

    if milvus_db_name and not milvus_url.startswith("./"):  # Only for server-based Milvus
        logger.info(f"Cleaning up database '{milvus_db_name}' in Milvus at {milvus_url}...")
        try:
            connections.connect(uri=milvus_url)
            if milvus_db_name in db.list_database():
                # Switch to the target database
                db.using_database(milvus_db_name)
                from pymilvus import utility
                collections = utility.list_collections()
                for coll in collections:
                    logger.info(f"Dropping collection: {coll}")
                    utility.drop_collection(coll)
                db.drop_database(milvus_db_name)
                logger.info(f"Database '{milvus_db_name}' has been dropped.")
            else:
                logger.info(f"Database '{milvus_db_name}' does not exist.")
        except Exception as e:
            logger.warning(f"Failed to clean up Milvus database: {e}")

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
    # print(f"\n目前記憶內容:\n{memories}")
    return memories


def search_memory(m, user_id: str, query: str, top_k: int = 5):
    memories = m.search(query=query, user_id=user_id)
    logger.debug(f"Relevant memories for the query '{query}':")
    for i, entry in enumerate(memories.get("results", []), start=1):
        logger.debug(f"- {entry['memory']} (Score: {entry['score']})")
    return memories


def chat_with_memory(
        client,
        m,
        user_id: str,
        text: str,
        model_name: str):
    system_prompt = f"You are a helpful AI. Answer the question based on query."

    memories = get_memory(m, user_id)

    existing_memory = memories.get("results", [])
    # logger.info(f"Existing memories count: {len(existing_memory)}")
    if existing_memory:
        relevant_memories = search_memory(m, user_id, text)
        if relevant_memories["results"]:
            memories = "\n".join(
                f"- {entry['memory']}" for entry in relevant_memories["results"]
            )
            system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser memories:\n{memories}"

    logger.debug(f"System Prompt:\n{system_prompt}")
    add_memory(m, user_id, text)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


def main():
    database_name = "test_mem0_db"
    collection_name = "test_mem0_collection"
    model_name = "gpt-5.2-2025-12-11"
    embedding_model = "text-embedding-3-large"

    memory_config = get_config(
        database_name=database_name, 
        collection_name=collection_name, 
        model_name=model_name, 
        embedding_model=embedding_model
    )
    client = get_openai_client()

    init_milvus_db(memory_config)

    memory = Memory.from_config(memory_config)
    logger.info("[Memory Instance Config]")
    if hasattr(memory, 'config'):
        logger.info(memory.config)
    else:
        logger.info(vars(memory))

    user_icon = "👤"
    bot_icon = "🤖"
    user_id = input(f"\n{user_icon} Please enter your name: ").strip() or "User"
    logger.info(f"{bot_icon} Welcome, {user_id}! Type 'exit' to end the conversation.")

    question_count = 1
    history = []  # 記錄 human/ai 對話
    while True:
        try:
            question = input(f"\n[{question_count}] Human Question:\n").strip()
            if question.lower() == "exit":
                confirm = input(f"{user_icon} Do you want to clean up the Milvus DB? (y/N): ").strip().lower()
                if confirm == "y":
                    clean_up_milvus_db(memory_config)
                logger.info(f"{user_icon} Bye!")
                break

            if not question:
                continue

            question = question.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
            response = chat_with_memory(client, memory, user_id, question, model_name)
            history.append({"human": question, "ai": response})
            logger.info(f"[{question_count}] AI Response:\n{response}")
            logger.info("\n" + "=" * 50)
            question_count += 1
        except KeyboardInterrupt:
            logger.info(f"\n{user_icon} Bye!")
            break


if __name__ == "__main__":
    main()

