import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from models import CommentInput, PipelineResponse
from pipeline import ClusteringPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

pipeline = None
# allowing only 2 requests to be processed concurrently to prevent memory overload
concurrency_limiter = asyncio.Semaphore(2) 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Starting up... Loading BAAI/bge-m3 into Memory/VRAM.")
    pipeline = ClusteringPipeline()
    yield
    logger.info("Shutting down... Releasing resources.")
    pipeline = None

app = FastAPI(
    title="RAG Context Optimization API",
    description="API for Semantic Deduplication and Topic Clustering",
    lifespan=lifespan
)

# Readiness / Health Endpoint for Orchestrators (like Kubernetes)
@app.get("/health", tags=["Monitoring"])
def health_check():
    if pipeline is not None:
        return {"status": "ready"}
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is still loading.")

@app.post("/api/v1/cluster", response_model=PipelineResponse)
async def optimize_context(payload: CommentInput):
    async with concurrency_limiter:
        try:
            # let 120 seconds be the extreme safety net for processing, which is highly unlikely to be reached in practice.
            # also, note that in case of a timeout, the Python thread will continue to run in the background.
            # and since max_length=2000 is very close to the limit, it is unlikely to be reached in practice.
            
            result = await asyncio.wait_for(
                asyncio.to_thread(pipeline.process, payload.comments),
                timeout=120.0
            )
            return result
            
        except asyncio.TimeoutError:
            logger.critical("CRITICAL: Request timed out. Background thread may still hold the inference lock!")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT, 
                detail="Request timed out while processing on the ML pipeline."
            )
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Internal server error during clustering."
            )