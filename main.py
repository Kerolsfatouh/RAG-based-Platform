import logging
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from clustering.models import CommentInput
from clustering.pipeline import ClusteringPipeline
from core.vram_manager import clear_vram
from core.chunk_formatter import format_clusters_to_chunks
from core.smart_cache import check_cache, save_to_cache, clear_smart_cache
from rag.qwen_engine import QwenEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

global_gpu_lock = asyncio.Lock()
qwen_engine = None 

# QWEN_LOCAL_PATH = "Qwen/Qwen3.5-4B"
QWEN_LOCAL_PATH = "qwen/qwen2.5-vl-72b-instruct"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qwen_engine
    clear_vram()
    yield
    if qwen_engine:
        del qwen_engine
    clear_vram()

app = FastAPI(title="Agentic RAG System (BGE + Qwen)", lifespan=lifespan)

@app.get("/health", tags=["Monitoring"])
def health_check():
    return {
        "status": "ready",
        "gpu_locked": global_gpu_lock.locked(),
        "active_model": "Qwen" if qwen_engine else "None"
    }

@app.post("/api/v1/update-knowledge", tags=["Knowledge Base"])
async def update_knowledge(payload: CommentInput):
    global qwen_engine
    
    async with global_gpu_lock:
        
        # 1. Clear the cache completely for the new data
        clear_smart_cache()
        
        if qwen_engine is not None:
            del qwen_engine
            qwen_engine = None
        clear_vram()
        
        pipeline = None
        try:
            pipeline = ClusteringPipeline()
            result = await asyncio.wait_for(
                asyncio.to_thread(pipeline.process, payload.posts),
                timeout=180.0
            )
            
            # 2. Build per-post metadata (page_name/post_content per post_id) and
            #    pass it to the formatter. payload now holds a *list* of posts,
            #    so there's no single page_name/post_content on payload itself.
            posts_metadata = {
                post.post_id: {
                    "page_name": post.page_name,
                    "post_content": post.post_content
                }
                for post in payload.posts
            }
            
            llm_ready_chunks = format_clusters_to_chunks(result["optimized_data"], posts_metadata)
            
            os.makedirs("data", exist_ok=True)
            with open("data/daily_clusters.txt", "w", encoding="utf-8") as f:
                f.write(llm_ready_chunks)
                
            return {
                "status": "success", 
                "message": "Knowledge base updated and cache cleared.",
                "clusters_found": result["num_clusters"],
                "posts_processed": len(payload.posts),
                "failed_posts": result.get("failed_posts", [])
            }
            
        except Exception as e:
            logger.error(f"Error during update: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Update failed.")
            
        finally:
            if pipeline:
                pipeline.free_vram()
            clear_vram()

class QueryInput(BaseModel):
    question: str

@app.post("/api/v1/ask", tags=["RAG AI Engine"])
async def ask_qwen(payload: QueryInput):
    global qwen_engine
    
    cached_response = await asyncio.to_thread(check_cache, payload.question)
    if cached_response:
        return {
            "source": "Smart Cache",
            "answer": cached_response["answer"],
            "clusters_used": cached_response["clusters_used"]
        }

    async with global_gpu_lock:
        cached_response_second_check = await asyncio.to_thread(check_cache, payload.question)
        if cached_response_second_check:
            return {
                "source": "Smart Cache",
                "answer": cached_response_second_check["answer"],
                "clusters_used": cached_response_second_check["clusters_used"]
            }

        if qwen_engine is None:
            try:
                qwen_engine = await asyncio.to_thread(QwenEngine, model_name=QWEN_LOCAL_PATH)
            except Exception as e:
                clear_vram()
                raise HTTPException(status_code=500, detail="Failed to load LLM.")

        try:
            result = await asyncio.to_thread(qwen_engine.ask, payload.question)
            
            if "I don't have" not in result["answer"] and "I couldn't find" not in result["answer"]:
                await asyncio.to_thread(save_to_cache, payload.question, result["answer"], result.get("clusters_used", []))
                
            return {
                "source": "Qwen Model",
                "answer": result["answer"],
                "clusters_used": result.get("clusters_used", [])
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error generating answer.")