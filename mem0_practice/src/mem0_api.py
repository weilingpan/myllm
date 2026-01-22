import os
import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from mem0 import AsyncMemory

class SearchRequest(BaseModel):
    user_name: str
    query: str
    limit: int= 5

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[lifespan] FastAPI app startup")
    yield
    print("[lifespan] FastAPI app shutdown")


router = APIRouter(tags=["Memory"])
app = FastAPI(lifespan=lifespan)


@router.get("/", include_in_schema=False)
async def redirect_root_docs():
    return RedirectResponse(url="/docs")

@router.post("/search", name="search_memory")
async def search(payload: SearchRequest):
    """
    Query semantic memory based on user_name (as collection) and query string.
    """
    try:
        query = payload.query
        user_id = payload.user_name
        limit = payload.limit

        # init async memory instance
        vector_store_config = {
            "provider": "redis",
            "config": {
                "collection_name": user_id,
                "redis_url": os.environ.get("REDIS_URI", "redis://redis-stack:6379"),
                "embedding_model_dims": os.environ.get("EMBEDDING_DIMENSION"),
            }
        }
        llm_config = {
            "provider": "vllm",
            "config": {
                "model": os.environ.get("MODEL_NAME"),
                "vllm_base_url": os.environ.get("LLAMA_BASE_URL"),
            }
        }
        embedder_config = {
            "provider": "huggingface",
            "config": {
                "model": os.environ.get("EMBEDDING_MODEL"),
                "huggingface_base_url": os.environ.get("HUGGINGFACE_URL"),
            }
        }
        reranker_config = {}
        custom_extraction_prompt = None
        custom_update_memory_prompt = None
        memory_config = {
            "vector_store": vector_store_config,
            "llm": llm_config,
            "embedder": embedder_config,
            "custom_fact_extraction_prompt": custom_extraction_prompt,
            "custom_update_memory_prompt": custom_update_memory_prompt,
            "reanker": reranker_config,
            "version": "v1.1",
        }
        async_memory = await AsyncMemory.from_config(memory_config)

        memories = await async_memory.search(
            query=query, 
            user_id=user_id,
            limit=limit,
        )
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
        print(f"{len(memories.get('results'))} Memories found")
        return memories
    
    except Exception as e:
        return {"error": str(e)}
    



app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        "mem0_api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
