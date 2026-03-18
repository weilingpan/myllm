import os
import sys
import time
import logging
from datetime import datetime
from openai import AsyncOpenAI
from pymilvus import connections, db

import redis

os.environ["MEM0_TELEMETRY"] = "false"
from mem0 import AsyncMemory

# 建立 logs 資料夾
log_dir = "logs-async"
os.makedirs(log_dir, exist_ok=True)
log_filename = datetime.now().strftime("%Y%m%d_%H%M%S.log")
log_path = os.path.join(log_dir, log_filename)
print(f"Logging to: {log_path}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    # handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.info(f"Running file: {os.path.abspath(sys.argv[0])}")


def get_openai_client(base_url: str = None) -> AsyncOpenAI:
    try:
        from env import OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    except ImportError:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    if base_url:
        return AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=base_url)
    return AsyncOpenAI(api_key=OPENAI_API_KEY)


def get_config(
        database_name: str, 
        collection_name: str,
        model_name: str,
        embedding_model: str,
        rerank_model: str = None,
    ):
    
    custom_extraction_prompt = None

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

    # custom_extraction_prompt = """
    # If it contains long-term useful information about the user's preferences, identity, or status, extract it.
    # If not, output \"None\".
    # Only output the extracted facts, do not explain

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

    # custom_extraction_prompt = """
    # Extract key facts from the conversation focusing on:
    # 1. Personal preferences
    # 2. Technical skills
    # 3. Project requirements
    # 4. Important dates and deadlines

    # Conversation: {messages}
    # """


    custom_update_memory_prompt = None

    return {
        # https://github.com/mem0ai/mem0/blob/dba7f0458aeb50aa7078d36eaefa2405afbee620/mem0/configs/vector_stores/milvus.py#L22
        # "vector_store": {
        #     "provider": "milvus",
        #     "config": {
        #         "collection_name": collection_name,
        #         "url": os.environ.get("MILVUS_URI", "http://milvus-standalone:19530"),
        #         "token": "",
        #         "db_name": database_name,
        #         # "embedding_model_dims": 1536,
        #     },
        # },
        # Redis 向量搜尋預設回傳的是距離，距離越小越相關
        "vector_store": {
            "provider": "redis",
            "config": {
                "collection_name": collection_name,
                "redis_url": os.environ.get("REDIS_URI", "redis://redis-stack:6379"),
                # "embedding_model_dims": 1536,
            }
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
        "custom_fact_extraction_prompt": custom_extraction_prompt,
        "custom_update_memory_prompt": custom_update_memory_prompt,
        "reanker": {
            "provider": "openai",
            "config": {
                "model": rerank_model,
            }
        },
        "version": "v1.1",
    }

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

async def get_memory(async_memory, user_id: str):
    # get all memories for user
    memories = await async_memory.get_all(user_id=user_id)
    # print(f"\n目前記憶內容:\n{memories}")
    return memories

async def search_memory(async_memory, user_id: str, query: str, limit: int = 100):
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
    memories = await async_memory.search(
        query=query, 
        user_id=user_id,
        limit=limit,
    )
    logger.info(f"Relevant memories for the query '{query}':")
    for i, entry in enumerate(memories.get("results", []), start=1):
        logger.info(f"- {entry['memory']} (Score: {entry['score']})")
    return memories

async def add_memory(async_memory, user_id: str, content: str|list, agent_id: str = None):
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
    logger.info(f"Try to add memory for user '{user_id}': {content}")
    if agent_id:
        res = await async_memory.add(
            messages=content,
            user_id=user_id,
            agent_id=agent_id,
            memory_type="procedural_memory",
            metadata={"category": "fact"},
        )
    else:
        res = await async_memory.add(
            messages=content,
            user_id=user_id,
            metadata={"category": "fact"},
        )
    """memory.add messages vs metadata 差異
    - messages:
        - 影響語意搜尋的結果
        - 會被向量化，影響相似度比對
        - 直接成為記憶的一部分，LLM 讀取時會看到
    - metadata:
        - 用於資料管理、分組或特定條件查詢
        - 預設不會影響語意比對，僅供過濾
        - LLM 通常看不到，除非你額外傳遞給它
    """
    if res.get("results"):
        logger.info(f"Memories operation: {len(res.get('results'))}")
        for i, entry in enumerate(res.get("results")):
            logger.info(entry)
    else:
        logger.warning("No memory was added.")
    return res

async def chat_with_memory(
        asyncclient,
        async_memory,
        user_id: str,
        text: str,
        model_name: str,
        top_k: int = 5,):
    system_prompt = f"You are a helpful AI."

    memories = await get_memory(async_memory, user_id)
    existing_memory = memories.get("results", [])
    logger.info(f"{user_id} has {len(existing_memory)} existing memories")

    if len(existing_memory) > 0:
        # logger.info(f"Searching Top 100 relevant memories, user query: {text}")
        # relevant_memories = await search_memory(
        #     async_memory, 
        #     user_id, 
        #     text
        # )

        logger.info(f"Searching Top {top_k} relevant memories, user query: {text}")
        relevant_memories = await search_memory(
            async_memory, 
            user_id, 
            text, 
            limit=top_k
        )
        if relevant_memories["results"]:
            memories = "\n".join(
                f"- {entry['memory']}" for entry in relevant_memories["results"]
            )
            system_prompt = (
                "You are a helpful AI. Answer the question based on query and memories.\n"
                f"User memories:\n{memories}\n"
            )

    logger.info(f"System Prompt:\n{system_prompt}")
    response = await asyncclient.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )

    return response.choices[0].message.content


async def main_llm_mem0(vector_store: str = "milvus", async_mode: bool = False):
    database_name = "test_mem0_db"
    collection_name = "test_mem0_collection_both3"
    model_name = "gpt-5.2-2025-12-11"
    embedding_model = "text-embedding-3-large"
    rerank_model = "GPT-4o-mini"

    logger.info(f"Initializing Memory with {vector_store} backend with async_mode={async_mode}...")
    logger.info(f"Database Name: {database_name}")
    logger.info(f"Collection Name: {collection_name}")
    logger.info(f"LLM Model: {model_name}")
    logger.info(f"Embedding Model: {embedding_model}")
    logger.info(f"Rerank Model: {rerank_model}\n")

    memory_config = get_config(
        database_name=database_name, 
        collection_name=collection_name, 
        model_name=model_name, 
        embedding_model=embedding_model,
        rerank_model=rerank_model
    )
    asyncclient = get_openai_client()

    async_memory = await AsyncMemory.from_config(memory_config)
    logger.info("[Memory Instance Config]")
    if hasattr(async_memory, 'config'):
        logger.info(async_memory.config)
    else:
        logger.info(vars(async_memory))
    
    user_icon = "👤"
    bot_icon = "🤖"
    user_id = input(f"\n{user_icon} Please enter your name: ").strip() or "User"
    logger.info(f"{bot_icon} Welcome, {user_id}! Type 'exit' to end the conversation.")

    question_count = 1
    history = []  # 記錄 human/ai 對話
    # preset_questions = [
    #     f"你好，我叫{user_id}，我目前在台北的一家科技公司擔任前端工程師。",
    #     "我最近開始學習 Python，因為我想把 AI 功能整合到我們的產品中。",
    #     "我對海鮮過敏，所以聚餐時我通常只吃素食或牛排。",
    #     "我有一隻叫「麻糬」的柴犬，牠每天早上 6 點就會吵著要出門散步。",
    #     "其實我最近換工作了，我現在轉職成了後端工程師，主要用 Go 語言。",
    #     "下週我要去日本東京出差，我想在那邊找幾間好吃的素食餐廳。",
    #     "最近「麻糬」變得很懶，現在都要到 8 點才肯起床，真拿牠沒辦法。",
    #     "我正在考慮把我的筆電換成 Mac，因為 Go 的開發環境好像比較方便。",
    #     "我發現我上次說錯了，我不是對海鮮過敏，我是對「蝦蟹類」過敏，魚肉是可以吃的", # add feedback
    #     "幫我規劃一下東京出差的晚餐。", # add feedback
    #     "其實「麻糬」上個月送給住在南部的親戚養了，我現在家裡沒有寵物。",
    #     "你還記得我養的是什麼狗，以及我現在主要用什麼程式語言工作嗎？"
    # ]

    preset_questions = [
        f"你好，我叫{user_id}, 我有一隻叫「麻糬」的柴犬",
        "我最喜歡的水果是香蕉",
        "我最近開始學習 Python，因為我想把 AI 功能整合到我們的產品中。",
        "我昨天跟lily去吃飯了",
        # "其實「麻糬」上個月送給住在南部的親戚養了，我現在家裡沒有寵物。"
        # "你還記得我養的是什麼狗，以及我現在主要用什麼程式語言工作嗎？"
    ]
    preset_questions = ["我也喜歡apple"]

    user_feedbacks = {}
    # user_feedbacks = {
    #     9: "非常重要且正確",
    #     10: "不對，你提到的餐廳仍有過敏源"
    # }

    for question in preset_questions:
        logger.info("=" * 50)
        logger.info(f"[Round {question_count}] Human Question:\n{question}")
        question = question.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        response = await chat_with_memory(asyncclient, async_memory, user_id, question, model_name)
        history.append({"human": question, "ai": response})
        logger.info(f"[Round {question_count}] AI Response:\n{response}")
        # await add_memory(async_memory, user_id, question)
        if question_count in user_feedbacks.keys():
            feedback = user_feedbacks[question_count]
            context = f"User Message: {question}\nLLM Response: {response}\nUser Feedback: {feedback}"
            logger.info(f"[Round {question_count}] User Feedback:\n{feedback}")
            await add_memory(async_memory, user_id, context)
        question_count += 1
        await asyncio.sleep(3)

    logger.info("\n=== Conversation History ===")
    for i, entry in enumerate(history, 1):
        logger.info(f"Round {i} Human: {entry['human']}")
        logger.info(f"Round {i} AI: {entry['ai']}")

    logger.info("Adding conversation history to memory ...")
    for i, entry in enumerate(history, 1):
        conversations = [
            {"role": "user", "content": entry['human']},
            {"role": "assistant", "content": entry['ai']},
        ]
        await add_memory(
            async_memory, 
            user_id=user_id, 
            content=conversations, 
            agent_id="test_agent"
        )


    # if vector_store == "redis":
    #     logger.info("Cleaning up Redis DB...")
    #     clean_up_redis_db(memory_config)
    #     logger.info("Redis DB cleanup complete.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main_llm_mem0(vector_store="redis", async_mode=True))
