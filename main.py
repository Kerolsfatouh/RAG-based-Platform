import logging
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from clustering.models import CommentInput
from clustering.pipeline import ClusteringPipeline
from core.vram_manager import clear_vram
from core.database import VectorDB
from core.smart_cache import check_cache, save_to_cache, clear_smart_cache
from rag.qwen_engine import QwenEngine
from scraper.orchestrator import run as run_scraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

global_gpu_lock = asyncio.Lock()
qwen_engine = None 

# Toggle between local in-VRAM inference and the OpenRouter API.
# Set QWEN_USE_LOCAL=true in the environment to run Qwen locally instead.
QWEN_USE_LOCAL = os.getenv("QWEN_USE_LOCAL", "false").strip().lower() in ("1", "true", "yes")
QWEN_LOCAL_MODEL = os.getenv("QWEN_LOCAL_MODEL", "Qwen/Qwen3.5-4B")
QWEN_API_MODEL = os.getenv("QWEN_API_MODEL", "qwen/qwen2.5-vl-72b-instruct")
QWEN_MODEL_PATH = QWEN_LOCAL_MODEL if QWEN_USE_LOCAL else QWEN_API_MODEL

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qwen_engine
    clear_vram()
    yield
    if qwen_engine:
        qwen_engine.free_vram()
        del qwen_engine
    clear_vram()

app = FastAPI(title="Agentic RAG System (BGE + Qwen)", lifespan=lifespan)

@app.get("/health", tags=["Monitoring"])
def health_check():
    return {
        "status": "ready",
        "gpu_locked": global_gpu_lock.locked(),
        "active_model": "Qwen" if qwen_engine else "None",
        "qwen_backend": "local" if QWEN_USE_LOCAL else "api",
        "qwen_model": QWEN_MODEL_PATH
    }

@app.post("/api/v1/update-knowledge", tags=["Knowledge Base"])
async def update_knowledge(payload: CommentInput):
    global qwen_engine
    
    async with global_gpu_lock:
        
        # 1. Clear the cache completely for the new data
        clear_smart_cache()
        
        if qwen_engine is not None:
            qwen_engine.free_vram()
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
            # Save to ChromaDB Vector Database
            db = VectorDB()
            db.add_clusters(result["optimized_data"], posts_metadata)
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
                qwen_engine = await asyncio.to_thread(
                    QwenEngine, model_name=QWEN_MODEL_PATH, use_local=QWEN_USE_LOCAL
                )
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

@app.post("/api/v1/scrape", tags=["Scraping Pipeline"])
async def trigger_scraper(background_tasks: BackgroundTasks):
    """
    Triggers the scraping orchestrator. It runs in the background, 
    scrapes the configured pages using Playwright, and automatically 
    feeds the newly scraped data directly into the RAG VectorDB.
    """
    background_tasks.add_task(run_scraper)
    return {
        "status": "success",
        "message": "Scraper pipeline started in the background! It will finish silently and update the Knowledge Base."
    }

@app.get("/api/v1/knowledge", tags=["Knowledge Base"], response_class=HTMLResponse)
async def view_knowledge_base():
    """Returns a simple HTML dashboard to visualize what is inside ChromaDB"""
    db = VectorDB()
    try:
        results = db.collection.get()
    except Exception:
        return "<h1>Database is empty or not initialized.</h1>"
        
    if not results or not results.get('ids'):
        return "<h1>No data scraped yet!</h1>"
        
    html = "<!DOCTYPE html><html><head><title>Knowledge Dashboard</title></head><body style='font-family: Arial; padding: 20px; background-color: #f4f6f9;'>"
    html += "<div style='max-width: 800px; margin: 0 auto;'>"
    html += "<h2 style='color: #333;'>🧠 Live Scraper Knowledge Dashboard</h2>"
    html += f"<p style='color: #666;'>Total clustered comments in database: <b>{len(results['ids'])}</b></p><hr style='border: 1px solid #ddd;'>"
    
    for i in range(len(results['ids'])):
        meta = results['metadatas'][i]
        content = results['documents'][i].replace('\n', '<br>')
        html += f"<div style='border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 8px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>"
        html += f"<span style='background: #007bff; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold;'>{meta.get('intent', 'General')}</span> "
        html += f"<span style='background: #28a745; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold;'>{meta.get('page_name', 'Unknown Page')}</span>"
        html += f"<p style='margin-top: 15px; color: #333; line-height: 1.5;'>{content}</p>"
        html += "</div>"
        
    html += "</div></body></html>"
    return html