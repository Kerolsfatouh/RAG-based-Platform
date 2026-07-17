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
from core.smart_cache import check_cache, save_to_cache, clear_smart_cache, invalidate_by_post_ids
from core.debug_dump import save_debug_clusters
from rag.qwen_engine import QwenEngine
from scraper.orchestrator import run as run_scraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

global_gpu_lock = asyncio.Lock()

# Both the embedding pipeline and Qwen are lazy singletons -- loaded once and kept
# warm across requests, instead of being torn down/rebuilt on every single call.
qwen_engine = None
pipeline = None

# Single shared VectorDB instance, opened once at import time -- reused by every
# endpoint instead of each one opening its own Chroma client.
vector_db = VectorDB()

# Toggle between local in-VRAM inference and the OpenRouter API.
# Set QWEN_USE_LOCAL=true in the environment to run Qwen locally instead.
QWEN_USE_LOCAL = os.getenv("QWEN_USE_LOCAL", "false").strip().lower() in ("1", "true", "yes")
QWEN_LOCAL_MODEL = os.getenv("QWEN_LOCAL_MODEL", "Qwen/Qwen3.5-4B")
QWEN_API_MODEL = os.getenv("QWEN_API_MODEL", "qwen/qwen2.5-vl-72b-instruct")
QWEN_MODEL_PATH = QWEN_LOCAL_MODEL if QWEN_USE_LOCAL else QWEN_API_MODEL

# In API mode (QWEN_USE_LOCAL=false), the embedding pipeline (BGE-M3) is never
# freed automatically during runtime -- the swap logic in _ensure_qwen_loaded()
# only frees it when local Qwen needs the VRAM, which never happens in API mode.
# Set this to true to explicitly free it right after every /update-knowledge call
# instead of keeping it warm indefinitely. Trades a reload cost on the next
# update-knowledge call for a smaller idle VRAM footprint -- useful on a
# GPU-constrained box, or while testing.
AUTO_FREE_EMBEDDING_AFTER_UPDATE = os.getenv("AUTO_FREE_EMBEDDING_AFTER_UPDATE", "false").strip().lower() in ("1", "true", "yes")


def _ensure_pipeline_loaded() -> ClusteringPipeline:
    """
    Lazily loads (or reuses) the clustering pipeline. In local-Qwen mode, the
    embedding model and the LLM can't both fit in VRAM at once, so if Qwen is
    currently loaded we free it first. In OpenRouter mode Qwen never touches local
    VRAM, so the pipeline can just stay warm indefinitely -- this only matters when
    QWEN_USE_LOCAL is true. Must be called while holding global_gpu_lock.
    """
    global pipeline, qwen_engine

    if pipeline is None:
        if QWEN_USE_LOCAL and qwen_engine is not None:
            qwen_engine.free_vram()
            qwen_engine = None
        pipeline = ClusteringPipeline()

    return pipeline


def _ensure_qwen_loaded() -> QwenEngine:
    """
    Lazily loads (or reuses) QwenEngine. Mirrors _ensure_pipeline_loaded(): in
    local mode, frees the embedding pipeline's VRAM first if it's currently
    holding it. Must be called while holding global_gpu_lock.
    """
    global pipeline, qwen_engine

    if qwen_engine is None:
        if QWEN_USE_LOCAL and pipeline is not None:
            pipeline.free_vram()
            pipeline = None
        qwen_engine = QwenEngine(model_name=QWEN_MODEL_PATH, use_local=QWEN_USE_LOCAL, db=vector_db)

    return qwen_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qwen_engine, pipeline
    clear_vram()
    yield
    if qwen_engine:
        qwen_engine.free_vram()
        qwen_engine = None
    if pipeline:
        pipeline.free_vram()
        pipeline = None
    clear_vram()

app = FastAPI(title="Agentic RAG System (BGE + Qwen)", lifespan=lifespan)


@app.get("/health", tags=["Monitoring"])
def health_check():
    return {
        "status": "ready",
        "gpu_locked": global_gpu_lock.locked(),
        "active_model": "Qwen" if qwen_engine else ("Embedding Pipeline" if pipeline else "None"),
        "qwen_backend": "local" if QWEN_USE_LOCAL else "api",
        "qwen_model": QWEN_MODEL_PATH,
        "auto_free_embedding_after_update": AUTO_FREE_EMBEDDING_AFTER_UPDATE
    }


@app.post("/api/v1/free-vram", tags=["Monitoring"])
async def free_vram_endpoint():
    """
    Manually releases whichever model(s) are currently loaded in VRAM.
    Mainly useful in API mode (QWEN_USE_LOCAL=false), where the embedding
    pipeline (BGE-M3) is otherwise never freed during runtime -- the swap
    logic in _ensure_qwen_loaded()/_ensure_pipeline_loaded() only frees one
    model to make room for the other, which never triggers when Qwen is
    running through the OpenRouter API instead of locally.
    """
    global pipeline, qwen_engine
    async with global_gpu_lock:
        freed = []
        if pipeline is not None:
            await asyncio.to_thread(pipeline.free_vram)
            pipeline = None
            freed.append("embedding_pipeline")
        if qwen_engine is not None:
            await asyncio.to_thread(qwen_engine.free_vram)
            qwen_engine = None
            freed.append("qwen")
        await asyncio.to_thread(clear_vram)

    return {"status": "success", "freed": freed}


@app.post("/api/v1/update-knowledge", tags=["Knowledge Base"])
async def update_knowledge(payload: CommentInput):
    async with global_gpu_lock:
        try:
            pipeline_instance = await asyncio.to_thread(_ensure_pipeline_loaded)

            result = await asyncio.wait_for(
                asyncio.to_thread(pipeline_instance.process, payload.posts),
                timeout=180.0
            )

            # Build per-post metadata (page_name/post_content per post_id) and pass
            # it to the VectorDB. payload holds a *list* of posts, so there's no
            # single page_name/post_content on payload itself.
            posts_metadata = {
                post.post_id: {
                    "page_name": post.page_name,
                    "post_content": post.post_content
                }
                for post in payload.posts
            }

            # Debug: dump the raw clustering output (pre-storage) to disk, so it can
            # be inspected exactly as the pipeline produced it. Only writes when
            # SAVE_DEBUG_CLUSTERS=true is set in the environment.
            debug_path = await asyncio.to_thread(save_debug_clusters, result, posts_metadata)
            if debug_path:
                logger.info(f"Saved debug cluster dump to {debug_path}")

            # IMPORTANT: only rebuild posts that actually produced new clusters this
            # run. `pipeline.process()` silently drops posts that threw during
            # clustering into failed_posts and excludes them from optimized_data --
            # if we deleted those posts' old clusters unconditionally, a previously
            # successful post would lose all its data the moment a later re-process
            # attempt failed (e.g. bad encoding, degenerate text), with nothing to
            # replace it. So we scope both the delete and the cache invalidation to
            # only the posts that actually succeeded this run.
            failed_post_ids = set(result.get("failed_posts", []))
            requested_post_ids = {post.post_id for post in payload.posts}
            succeeded_post_ids = requested_post_ids - failed_post_ids

            if failed_post_ids:
                logger.warning(
                    f"Skipping delete/invalidate for {len(failed_post_ids)} post(s) "
                    f"that failed clustering this run (existing data left untouched): "
                    f"{sorted(failed_post_ids)}"
                )

            # Clustering isn't deterministic run-to-run, so we delete each *successful*
            # post's old clusters first -- this avoids both extremes: never wiping the
            # whole DB (which would lose everything the scraper built up over time),
            # and never leaving orphaned stale clusters behind under different ids.
            for post_id in succeeded_post_ids:
                vector_db.delete_post(post_id)

            vector_db.add_clusters(result["optimized_data"], posts_metadata)

            # Only invalidate cached answers that actually depended on posts we
            # actually changed -- not the entire Smart Cache, and not posts whose
            # clustering failed (their old data, and any cache built on it, is still
            # valid and untouched) -- so answers about unrelated topics survive a
            # continuous, single-post-at-a-time scrape.
            await asyncio.to_thread(invalidate_by_post_ids, succeeded_post_ids)

            return {
                "status": "success",
                "message": "Knowledge base updated.",
                "clusters_found": result["num_clusters"],
                "posts_requested": len(payload.posts),
                "posts_succeeded": len(succeeded_post_ids),
                "failed_posts": result.get("failed_posts", []),
                "debug_dump": debug_path
            }

        except Exception as e:
            logger.error(f"Error during update: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Update failed.")

        finally:
            # If enabled, explicitly free the embedding pipeline right after this
            # request instead of leaving it warm -- see AUTO_FREE_EMBEDDING_AFTER_UPDATE
            # above. Off by default: normally the pipeline stays warm for the next
            # update instead of being torn down every call.
            if AUTO_FREE_EMBEDDING_AFTER_UPDATE and pipeline is not None:
                await asyncio.to_thread(pipeline.free_vram)
                pipeline = None
            clear_vram()


class QueryInput(BaseModel):
    question: str


@app.post("/api/v1/ask", tags=["RAG AI Engine"])
async def ask_qwen(payload: QueryInput):
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

        try:
            engine = await asyncio.to_thread(_ensure_qwen_loaded)
        except Exception:
            clear_vram()
            raise HTTPException(status_code=500, detail="Failed to load LLM.")

        try:
            result = await asyncio.to_thread(engine.ask, payload.question)

            if "I don't have" not in result["answer"] and "I couldn't find" not in result["answer"]:
                await asyncio.to_thread(save_to_cache, payload.question, result["answer"], result.get("clusters_used", []))

            return {
                "source": "Qwen Model",
                "answer": result["answer"],
                "clusters_used": result.get("clusters_used", [])
            }
        except Exception:
            raise HTTPException(status_code=500, detail="Error generating answer.")


@app.post("/api/v1/reset-knowledge", tags=["Knowledge Base"])
async def reset_knowledge():
    """
    Manually wipes the entire knowledge base (VectorDB) and the Q&A cache.
    Use this deliberately (e.g. before a clean re-scrape) -- /update-knowledge
    itself only rebuilds the specific posts it's given, since the scraper feeds
    it incrementally, one post at a time.
    """
    async with global_gpu_lock:
        vector_db.clear()
        await asyncio.to_thread(clear_smart_cache)
    return {"status": "success", "message": "Knowledge base and cache wiped."}


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
    try:
        results = vector_db.collection.get()
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
