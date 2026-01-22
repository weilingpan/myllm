import os
import redis
import orjson
import asyncio

os.environ["MEM0_TELEMETRY"] = "false"
from mem0 import AsyncMemory

RQ_MEM0_QUEUE_NAME = "save_memory"
REDISHOST = ""
REDISPORT = 31384
REDISDB = 4

base_url = os.environ.get("LLAMA_BASE_URL")
database_name = "test_mem0_db"
collection_name = "test_mem0_collection"
model_name = os.environ.get("MODEL_NAME")
embedding_model = os.environ.get("EMBEDDING_MODEL")
rerank_model = os.environ.get("RERANK_MODEL")

def get_redis_client(REDISHOST, REDISPORT, REDISDB):
    redis_client = redis.Redis(host=REDISHOST, port=REDISPORT, db=REDISDB)
    return redis_client


class Mem0Worker:
    def __init__(
            self,
            database_name,
            collection_name,
            model_name,
            embedding_model,
            rerank_model,
            base_url,
            queue_name, 
            redishost, 
            redisport, 
            redisdb, 
            exception_handlers=None, 
            max_retries=3, 
            fail_queue_suffix="_fail"
        ):
        self.base_url = base_url
        self.queue_name = queue_name
        self.redis_client = get_redis_client(redishost, redisport, redisdb)
        self.exception_handlers = exception_handlers or []
        self.max_retries = max_retries
        self.fail_queue = f"{queue_name}{fail_queue_suffix}"
        self.collection_name = collection_name

        self.memory_config = self.get_config(
            database_name=database_name, 
            collection_name=collection_name, 
            model_name=model_name, 
            embedding_model=embedding_model,
            rerank_model=rerank_model
        )
        self.async_memory = None

    def get_config(
            self,
            database_name: str, 
            collection_name: str,
            model_name: str,
            embedding_model: str,
            rerank_model: str = None,
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

        return {
            # Redis 向量搜尋預設回傳的是距離，距離越小越相關
            "vector_store": {
                "provider": "redis",
                "config": {
                    "collection_name": collection_name,
                    "redis_url": os.environ.get("REDIS_URI", "redis://redis-stack:6379"),
                    "embedding_model_dims": 1024,
                }
            },
            "llm": {
                "provider": "vllm",
                "config": {
                    "model": model_name,
                    "vllm_base_url": os.environ.get("LLAMA_BASE_URL"),
                }
            },
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
            #     "provider": "openai",
            #     "config": {
            #         "model": rerank_model,
            #     }
            # },
            "version": "v1.1",
        }

    async def add_memory(
            self,
            user_name: str,
            human_message: str,
            job_data: dict
        ):
        print(f"[Mem0Worker] Adding memory for user {user_name} with message: {human_message}")
        res = await self.async_memory.add(
            messages=human_message,
            user_id=user_name,
            metadata={
                "category": "fact",
                "chat_id": job_data.get("chat_id"),
                "pair_chat_id": job_data.get("pair_chat_id")
            }
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
            print(f"Memories operation: {len(res.get('results'))}")
            for i, entry in enumerate(res.get("results")):
                print(entry)
        else:
            print("No memory was added.")
        return res

    async def save_memory_job(
            self,
            job_data: bytes
        ):
        job_data = orjson.loads(job_data)
        # print(job_data)
        self.memory_config["vector_store"]["config"]["collection_name"] = job_data.get("user_name", self.collection_name)
        self.async_memory = await AsyncMemory.from_config(self.memory_config)

        print("[Memory Instance Config]")
        if hasattr(self.async_memory, 'config'):
            print(self.async_memory.config)
        else:
            print(vars(self.async_memory))

        # print(f"[Mem0Worker] Processing job data: {job_data}")
        human_message = job_data.get("human_message").encode("utf-8", "ignore").decode("utf-8", "ignore")
        user_name = job_data.get("user_name")
        await self.add_memory(user_name, human_message, job_data)


    async def process_with_retry(self, job_data):
        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                await self.save_memory_job(job_data)
                print(f"[Mem0Worker] finished")
                return True
            except Exception as e:
                last_exception = e
                print(f"[Mem0Worker] Job failed (attempt {attempt}/{self.max_retries}): {e}")
                if attempt == self.max_retries:
                    fail_record = orjson.dumps({
                        "job_data": orjson.loads(job_data),
                        "error": str(last_exception)
                    })
                    self.redis_client.rpush(self.fail_queue, fail_record)
                    print(f"[Mem0Worker] Job pushed to fail queue: {self.fail_queue}")
                else:
                    continue
        return False

    async def start(self):
        print(f"[Mem0Worker] Listening for jobs on queue: {self.queue_name}")
        loop = asyncio.get_event_loop()
        while True:
            try:
                job = self.redis_client.blpop(self.queue_name)
                if job:
                    _, job_data = job
                    await self.process_with_retry(job_data)
            except Exception as e:
                print(f"[Mem0Worker] Exception: {e}")
                for handler in self.exception_handlers:
                    handler(e)

    def clean_up_redis_db(self):
        redis_config = self.memory_config["vector_store"]["config"]
        redis_url = redis_config["redis_url"]

        try:
            print(f"Connecting to Redis at {redis_url} for cleanup...")
            r = redis.from_url(redis_url)

            # List all keys in Redis
            keys = r.keys('*')
            for key in keys:
                key_str = key.decode('utf-8')
                #print(f"Deleting key: {key_str}")
                r.delete(key_str)

            print("Redis database cleanup complete.\n")
        except Exception as e:
            print(f"Failed to clean up Redis database: {e}")

def print_exception(e):
    print(f"[Exception Handler] {e}")

async def main():
    vector_store = "redis"
    async_mode = True
    print(f"Initializing Memory with {vector_store} backend with async_mode={async_mode}...\n")
    print(f"Database Name: {database_name}")
    print(f"Collection Name: {collection_name}")
    print(f"LLM Model: {model_name}")
    print(f"Embedding Model: {embedding_model}")
    print(f"Rerank Model: {rerank_model}\n")

    worker = Mem0Worker(
        database_name,
        collection_name,
        model_name,
        embedding_model,
        rerank_model,
        base_url,
        RQ_MEM0_QUEUE_NAME,
        REDISHOST,
        REDISPORT,
        REDISDB,
        exception_handlers=[print_exception],
    )
    # worker.clean_up_redis_db()
    await worker.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down worker.")
