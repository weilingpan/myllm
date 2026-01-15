import os
from openai import OpenAI
import logging

os.environ["MEM0_TELEMETRY"] = "false"
from mem0 import Memory

from env import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)  # 或 logging.DEBUG
logger = logging.getLogger(__name__)


collection_name = "test_mem0_collection"
model_name = "gpt-4o"
embedding_model = "text-embedding-3-large" 

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

config = {
    #https://github.com/mem0ai/mem0/blob/dba7f0458aeb50aa7078d36eaefa2405afbee620/mem0/configs/vector_stores/milvus.py#L22
    "vector_store": {
        "provider": "milvus",
        "config": {
            "collection_name": collection_name,
            # "embedding_model_dims": ,  # 使用項目的向量維度
            "url": "./milvus.db",
            "token": "",
        },
    },
    # #https://github.com/mem0ai/mem0/blob/main/mem0/configs/llms/vllm.py
    # "llm": {
    #     "provider": "vllm",
    #     "config": {
    #         "model": model_name,
    #         "vllm_base_url": llama_url,
    #     }
    # },
    # #https://github.com/mem0ai/mem0/blob/dba7f0458aeb50aa7078d36eaefa2405afbee620/mem0/configs/embeddings/base.py#L10
    # "embedder": {
    #     "provider": "huggingface",
    #     "config": {
    #         "model": VectorDBConfig.EMBEDDING_MODEL_NAME,
    #         "huggingface_base_url": VectorDBConfig.EMBEDDING_URL,  # 指定 embedding gateway URL
    #     }
    # },
    "version": "v1.1",
}


m = Memory.from_config(config)

def add_memory(user_id: str, content: str):
    res = m.add(
        messages=content,
        user_id=user_id,
        metadata={"category": "fact"},
    )
    return res

def get_memory(user_id: str):
    # get all memories for user
    memories = m.get_all(user_id=user_id)
    print(f"\n目前記憶內容:\n{memories}")
    return memories

def search_memory(user_id: str, query: str, top_k: int = 5):
    memories = m.search(query=query, user_id=user_id)
    print(f"\n搜尋到的記憶內容:\n{memories}")
    return memories

def extrace_memory(text: str):
    prompt = f"""
    Analyze the user's statement: "{text}"
    If it contains long-term useful information about the user's preferences, identity, or status, extract it.
    If not, output "None".
    Only output the extracted facts, do not explain.
    """

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    fact = response.choices[0].message.content.strip()
    print(f"\n提取結果:\n{fact}\n")
    return fact

def chat_with_memory(user_id: str, text: str):
    get_memory(user_id)
    relevant_memories = search_memory(user_id, text)
    # fact = extrace_memory(text)

    # if fact != "None":
    #     add_memory(user_id, fact)
    add_memory(user_id, text)

    system_prompt = f"You are a helpful AI. Answer the question based on query."
    if relevant_memories["results"]:
        memories = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
        system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser memories:\n{memories}"
    print(f"\nSystem Prompt:\n{system_prompt}")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content


user_id = "regina_pan"
question = "最近在學習Python"
# question = "請推薦我一些書籍"
print(f"{user_id} 問: {question}")
response = chat_with_memory(user_id, question)
print(f"\n答:\n{response}")
