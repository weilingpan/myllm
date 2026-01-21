import os
import time
import logging
from openai import OpenAI
from pymilvus import connections, db
import redis

os.environ["MEM0_TELEMETRY"] = "false"
from mem0 import Memory

import sys
from datetime import datetime

# 建立 logs 資料夾
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = datetime.now().strftime("%Y%m%d_%H%M%S.log")
log_path = os.path.join(log_dir, log_filename)
print(f"Logging to: {log_path}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    #handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.info(f"Running file: {os.path.abspath(sys.argv[0])}")

def get_openai_client(base_url: str = None):
    try:
        from env import OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    except ImportError:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    if base_url:
        return OpenAI(api_key=OPENAI_API_KEY, base_url=base_url)
    return OpenAI(api_key=OPENAI_API_KEY)


def get_config(
        database_name: str, 
        collection_name: str,
        model_name: str,
        embedding_model: str,
        vector_store: str
    ):
    
    # custom_extraction_prompt = None

    # custom_extraction_prompt = """
    # If it contains long-term useful information about the user's preferences, identity, or status, extract it.
    # Here are some few shot examples:

    # Input: Hi.
    # Output: {{"facts" : []}}

    # Input: The weather is nice today.
    # Output: {{"facts" : []}}

    # Input: My order #12345 hasn't arrived yet.
    # Output: {{"facts" : ["Order #12345 not received"]}}

    # Input: I'm John Doe, and I'd like to return the shoes I bought last week.
    # Output: {{"facts" : ["Customer name: John Doe", "Wants to return shoes", "Purchase made last week"]}}

    # Input: I ordered a red shirt, size medium, but received a blue one instead.
    # Output: {{"facts" : ["Ordered red shirt, size medium", "Received blue shirt instead"]}}

    # Return the facts and customer information in a json format as shown above.
    # """

    custom_extraction_prompt = """
    If it contains long-term useful information about the user's preferences, identity, or status, extract it.
    If not, output \"None\".
    Only output the extracted facts, do not explain

    Input: Hi.
    Output: {{"facts" : []}}

    Input: The weather is nice today.
    Output: {{"facts" : []}}

    Input: My order #12345 hasn't arrived yet.
    Output: {{"facts" : ["Order #12345 not received"]}}

    Input: I'm John Doe, and I'd like to return the shoes I bought last week.
    Output: {{"facts" : ["Customer name: John Doe", "Wants to return shoes", "Purchase made last week"]}}

    Input: I ordered a red shirt, size medium, but received a blue one instead.
    Output: {{"facts" : ["Ordered red shirt, size medium", "Received blue shirt instead"]}}

    Return the facts and customer information in a json format as shown above.
    """

    # custom_extraction_prompt = """
    # Extract key facts from the conversation focusing on:
    # 1. Personal preferences
    # 2. Technical skills
    # 3. Project requirements
    # 4. Important dates and deadlines

    # Conversation: {messages}
    # """


    custom_update_memory_prompt = None

    if vector_store == "milvus":
        # https://github.com/mem0ai/mem0/blob/dba7f0458aeb50aa7078d36eaefa2405afbee620/mem0/configs/vector_stores/milvus.py#L22
        vector_store_config = {
            "provider": "milvus",
            "config": {
                "collection_name": collection_name,
                "url": os.environ.get("MILVUS_URI", "http://milvus-standalone:19530"),
                "token": "",
                "db_name": database_name,
                # "enable_vision",
                "embedding_model_dims": 1024,
            },
        }
    elif vector_store == "redis":
        # Redis 向量搜尋預設回傳的是距離，距離越小越相關
        vector_store_config = {
            "provider": "redis",
            "config": {
                "collection_name": collection_name,
                "redis_url": os.environ.get("REDIS_URI", "redis://redis-stack:6379/0"),
                # "embedding_model_dims": 1536,
            }
        }

    return {
        "vector_store": vector_store_config,
        # #https://github.com/mem0ai/mem0/blob/main/mem0/configs/llms
        "llm": {
            "provider": "vllm",
            "config": {
                "model": model_name,
                "vllm_base_url": os.environ.get("LLAMA_BASE_URL"),
            }
        },
        #https://github.com/mem0ai/mem0/blob/dba7f0458aeb50aa7078d36eaefa2405afbee620/mem0/configs/embeddings/base.py#L10
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": embedding_model,
                "huggingface_base_url": os.environ.get("HUGGINGFACE_URL"),  # 指定 embedding gateway URL
            }
        },
        "custom_fact_extraction_prompt": custom_extraction_prompt,
        "custom_update_memory_prompt": custom_update_memory_prompt,
        # "reanker": {
        #     "provider": "",
        #     "config": {}
        # },
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



def clean_up_redis_db(config):
    redis_config = config["vector_store"]["config"]
    redis_url = redis_config["redis_url"]

    try:
        logger.info(f"Connecting to Redis at {redis_url} for cleanup...")
        r = redis.from_url(redis_url)

        # List all keys in Redis
        keys = r.keys('*')
        for key in keys:
            key_str = key.decode('utf-8')
            logger.info(f"Deleting key: {key_str}")
            r.delete(key_str)

        logger.info("Redis database cleanup complete.")
    except Exception as e:
        logger.warning(f"Failed to clean up Redis database: {e}")


def add_memory(m, user_id: str, content: str):
    logger.info(f"Try to add memory for user '{user_id}': {content}")
    res = m.add(
        messages=content,
        user_id=user_id,
        metadata={"category": "fact"},
        infer=True, # LLM is used to extract key facts from 'messages' and decide whether to add, update, or delete related memories. If False, 'messages' are added as raw memories directly.
    )
    # def add(
    #     self,
    #     messages,
    #     *,
    #     user_id: Optional[str] = None,
    #     agent_id: Optional[str] = None,
    #     run_id: Optional[str] = None,
    #     metadata: Optional[Dict[str, Any]] = None,
    #     infer: bool = True,
    #     memory_type: Optional[str] = None,
    #     prompt: Optional[str] = None,
    # )
    if res.get("results"):
        logger.info(f"Memories operation: {len(res.get('results'))}")
        for i, entry in enumerate(res.get("results")):
            logger.info(f"- event: {entry['event']}")
    else:
        logger.warning("No memory was added.")
    return res


def get_memory_milvus(m, user_id: str):
    # get all memories for user
    memories = m.get_all(user_id=user_id)
    # print(f"\n目前記憶內容:\n{memories}")
    return memories

def get_memory_redis(m, user_id: str, filters: dict):
    REDISHOST = "172.18.246.170"
    REDISPORT = 31379
    REDISDB = 0
    redis_client = redis.Redis(host=REDISHOST, port=REDISPORT, db=REDISDB)
    for key in redis_client.scan_iter(match='mem0:test_mem0_collection*'):
       hash_values = redis_client.hgetall(key)
       if hash_values and hash_values.get(b'user_id') == str(user_id).encode():
            print(key)

    # get all memories for user
    memories = m.get_all(user_id=user_id, filters=filters)
    print(f"\n目前記憶內容:\n{memories}")
    return memories

def search_memory(m, user_id: str, query: str, top_k: int = 5):
    # def search(
    #     self,
    #     query: str,
    #     *,
    #     user_id: Optional[str] = None,
    #     agent_id: Optional[str] = None,
    #     run_id: Optional[str] = None,
    #     limit: int = 100,
    #     filters: Optional[Dict[str, Any]] = None,
    #     threshold: Optional[float] = None,
    #     rerank: bool = True,
    # ):
    memories = m.search(query=query, user_id=user_id)
    logger.info(f"Relevant memories for the query '{query}':")
    for i, entry in enumerate(memories.get("results", []), start=1):
        logger.info(f"- {entry['memory']} (Score: {entry['score']})")
    return memories


def chat_with_memory(
        client,
        m,
        user_id: str,
        text: str,
        model_name: str,
        vector_store: str):
    system_prompt = f"You are a helpful AI."

    if vector_store == "milvus":
        memories = get_memory_milvus(m, user_id)
    elif vector_store == "redis":
        filters = {"user_id": user_id}
        memories = get_memory_redis(m, user_id=user_id, filters=filters)

    existing_memory = memories.get("results", [])
    logger.info(f"Existing memories count: {len(existing_memory)}")
    st = time.time()
    if len(existing_memory) > 0:
        logger.info("Searching for relevant memories...")
        relevant_memories = search_memory(m, user_id, text)
        if relevant_memories["results"]:
            memories = "\n".join(
                f"- {entry['memory']}" for entry in relevant_memories["results"]
            )
            system_prompt = (
                "You are a helpful AI. Answer the question based on query and memories.\n"
                f"User memories:\n{memories}\n"
            )
    et = time.time()
    logger.info(f"Memory search took {et - st:.2f} seconds")

    logger.info(f"System Prompt: {system_prompt}")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )

    return response.choices[0].message.content

def chat_with_stm(
        client, 
        user_id: str, 
        text: str, 
        model_name: str, 
        history: list,
        stm: int = 2
    ):
    system_prompt = f"You are a helpful AI."

    history_text = "\n".join([
        f"Human: {entry['human']}\nAI: {entry['ai']}" for entry in history[-stm:]
    ])
    if history_text:
        system_prompt = f"You are a helpful AI.\nRecent conversation:\n{history_text}"

    logger.info(f"System Prompt: {system_prompt}")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )

    return response.choices[0].message.content

## llm + mem0
def main_llm_mem0(vector_store: str = "milvus", async_mode: bool = False):
    logger.info(f"\n{'='*20} LLM + Mem0 {'='*20}\n")
    base_url = os.environ.get("LLAMA_BASE_URL")
    database_name = "test_mem0_db"
    collection_name = "test_mem0_collection"
    model_name = os.environ.get("MODEL_NAME")
    embedding_model = os.environ.get("EMBEDDING_MODEL")

    logger.info("Initializing Memory with Milvus backend...")
    logger.info(f"Database Name: {database_name}")
    logger.info(f"Collection Name: {collection_name}")
    logger.info(f"LLM Model: {model_name}")
    logger.info(f"Embedding Model: {embedding_model}\n")

    memory_config = get_config(
        database_name=database_name, 
        collection_name=collection_name, 
        model_name=model_name, 
        embedding_model=embedding_model,
        vector_store=vector_store
    )
    client = get_openai_client(base_url)
    
    if vector_store == "milvus":
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
    preset_questions = [
        "我最喜歡的水果是蘋果",          # 關鍵
        "我不太喜歡吃太甜的水果",        # 關鍵偏好
        "你覺得水果每天吃好嗎？",        # 無關
        "很多人早餐會吃水果，你怎麼看？", # 無關
        "最近天氣變熱了",                # 干擾
        "幫我推薦一種「適合我」的水果"
    ]

    for question in preset_questions:
        logger.info("\n" + "=" * 50)
        logger.info(f"[{question_count}] Human Question:\n{question}")
        question = question.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        response = chat_with_memory(client, memory, user_id, question, model_name, vector_store)
        history.append({"human": question, "ai": response})
        logger.info(f"[{question_count}] AI Response:\n{response}")
        add_memory(memory, user_id, question)
        question_count += 1
        time.sleep(3)

    logger.info("\n=== Conversation History ===")
    for i, entry in enumerate(history, 1):
        logger.info(f"Round {i} Human: {entry['human']}")
        logger.info(f"Round {i} AI: {entry['ai']}")

    if vector_store == "milvus":
        logger.info("Cleaning up Milvus DB...")
        clean_up_milvus_db(memory_config)
        logger.info("Milvus DB cleanup complete.")
    # elif vector_store == "redis":
    #     logger.info("Cleaning up Redis DB...")
    #     clean_up_redis_db(memory_config)
    #     logger.info("Redis DB cleanup complete.")


## llm+stm2
def main_llm_stm():
    logger.info(f"\n{'='*20} LLM + STM {'='*20}\n")
    base_url = os.environ.get("LLAMA_BASE_URL")
    database_name = "test_mem0_db"
    collection_name = "test_mem0_collection"
    model_name = os.environ.get("MODEL_NAME")
    embedding_model = os.environ.get("EMBEDDING_MODEL")

    logger.info("Initializing Memory with Milvus backend...")
    logger.info(f"Database Name: {database_name}")
    logger.info(f"Collection Name: {collection_name}")
    logger.info(f"Base URL: {base_url}")
    logger.info(f"LLM Model: {model_name}")
    logger.info(f"Embedding Model: {embedding_model}\n")

    client = get_openai_client(base_url)

    user_icon = "👤"
    bot_icon = "🤖"
    user_id = input(f"{user_icon} Please enter your name: ").strip() or "User"
    logger.info(f"{bot_icon} Welcome, {user_id}! Type 'exit' to end the conversation.")

    question_count = 1
    history = []  # 記錄 human/ai 對話
    preset_questions = [
        "我最喜歡的水果是蘋果",          # 關鍵
        "我不太喜歡吃太甜的水果",        # 關鍵偏好
        # "你覺得水果每天吃好嗎？",        # 無關
        # "很多人早餐會吃水果，你怎麼看？", # 無關
        # "最近天氣變熱了",                # 干擾
        # "幫我推薦一種「適合我」的水果"
    ]

    for question in preset_questions:
        logger.info("\n" + "=" * 50)
        logger.info(f"[{question_count}] Human Question:\n{question}")
        question = question.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        response = chat_with_stm(client, user_id, question, model_name, history)
        history.append({"human": question, "ai": response})
        logger.info(f"[{question_count}] AI Response:\n{response}")
        question_count += 1

    logger.info("\n=== Conversation History ===")
    for i, entry in enumerate(history, 1):
        logger.info(f"Round {i} Human: {entry['human']}")
        logger.info(f"Round {i} AI: {entry['ai']}")

if __name__ == "__main__":
    main_llm_mem0(vector_store="redis")
    # main_llm_stm()



# reference: https://github.com/mem0ai/mem0/blob/dba7f0458aeb50aa7078d36eaefa2405afbee620/LLM.md
