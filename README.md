# RAG-based Platform for contant Communities

An **Agentic RAG** system built to analyze comments, turn them into a queryable knowledge base, and let users ask natural-language questions (up to 100 language) that get answered based on what fans actually said.

The core idea: instead of someone scrolling through thousands of comments under a post, the system **clusters** semantically similar comments, summarizes them, and exposes them to a local LLM (Qwen) through a RAG pipeline with a smart router that decides which cluster(s) likely hold the answer.

---

## Architecture Overview

```
(Scraper) → /update-knowledge → Clustering Pipeline → daily_clusters.txt
                                                                              │
User Question → /ask → Smart Cache (hit?) → Qwen Router → Qwen QA → Answer
```

### Knowledge Update Flow
1. Comments arrive via `POST /api/v1/update-knowledge` (Pydantic validation + text sanitization).
2. `ClusteringPipeline`:
   - Embeds every comment using **BAAI/bge-m3**.
   - If there are fewer than 15 comments → everything goes into a single cluster.
   - Otherwise → **UMAP** for dimensionality reduction, then **HDBSCAN** clustering orchestrated through **BERTopic**.
   - Within each topic, `community_detection` (Sentence-Transformers) groups near-duplicate comments together and picks a representative text + frequency count.
3. `chunk_formatter` converts the clustering output into a structured text format (`<CLUSTER_START> ... <CLUSTER_END>`) with metadata (page name, post, post content), written to `data/daily_clusters.txt`.
4. The old cache is fully wiped (`clear_smart_cache`) since the underlying data has changed.
5. VRAM is cleared after every heavy step (`vram_manager`), since the embedding model and the LLM can't both fit in memory at once.

### Ask Flow
1. `POST /api/v1/ask` receives the user's question.
2. It first checks the **Smart Cache** (`smart_cache.json`) — if this question was asked before, the cached answer is returned immediately, skipping model inference entirely.
3. If there's no cache hit, **Qwen** is lazy-loaded if it isn't already in memory — either locally (8-bit quantized, in VRAM) or through the OpenRouter API, depending on the configured backend (see [Qwen Backend: Local vs. OpenRouter](#qwen-backend-local-vs-openrouter)).
4. `QwenEngine.ask()` runs a 3-stage process:
   - **Rewriter**: turns the user's raw (often slang) question into a clean, well-formed search query.
   - **Router**: shows the model a summary of every cluster (keywords + preview) and asks it to pick the cluster ID(s) likely to contain the answer (JSON list).
   - **QA**: feeds the full content of the selected clusters as context and generates the final answer, **in the same language as the user's question**.
5. The answer is cached (unless it's a "don't know" response) and returned to the user.

### GPU Lock
A `global_gpu_lock` (`asyncio.Lock`) prevents two models from running on the GPU at the same time (the embedding model and the LLM), since both together wouldn't fit in the available VRAM.

### Qwen Backend: Local vs. OpenRouter
`QwenEngine` doesn't talk to a model directly — it delegates the actual generation call to one of two interchangeable "callers":

- **`LocalQwenCaller`** (`rag/local_call.py`) — loads Qwen locally, 8-bit quantized via `bitsandbytes`, and runs inference in VRAM.
- **`OpenRouterQwenCaller`** (`rag/openrouter_call.py`) — sends the same prompts to Qwen through the [OpenRouter](https://openrouter.ai/) API instead.

Both expose the same tiny interface (`generate(...)` / `free_vram()`), so `QwenEngine`'s Rewrite → Route → QA logic never needs to know which one is active.

Which one runs is controlled by environment variables, no code changes required:

| Variable | Default | Purpose |
|---|---|---|
| `QWEN_USE_LOCAL` | `false` | Set to `true`/`1` to run Qwen locally instead of via OpenRouter |
| `QWEN_LOCAL_MODEL` | `Qwen/Qwen3.5-4B` | HF model id/path used when `QWEN_USE_LOCAL=true` |
| `QWEN_API_MODEL` | `qwen/qwen2.5-vl-72b-instruct` | OpenRouter model slug used when `QWEN_USE_LOCAL=false` |
| `OPENROUTER_API_KEY` | — | Required when running in OpenRouter mode |

`GET /health` reports which backend and model are currently configured (`qwen_backend`, `qwen_model`).

---

## Project Structure

```
RAG-based-Platform-for-Sports-Fan-Communities/
│
├── main.py                  # FastAPI app + endpoints + lifespan + GPU lock
├── requirements.txt
│
├── clustering/
│   ├── models.py             # Pydantic schemas (input/output validation)
│   └── pipeline.py           # BGE-M3 embeddings + UMAP/HDBSCAN/BERTopic clustering
│
├── core/
│   ├── chunk_formatter.py    # Converts clustering results into LLM-ready text chunks
│   ├── smart_cache.py        # Question/answer cache (JSON file-based)
│   ├── string_cleaner.py     # Strips control characters and safely escapes text
│   └── vram_manager.py       # Frees RAM/VRAM after heavy operations
│
├── data/                      # (auto-generated at runtime)
│   ├── daily_clusters.txt    # Latest extracted clusters (overwritten on every update)
│   └── smart_cache.json      # Cached questions and answers
│
├── rag/
│   ├── prompt_templates.py   # System/user prompts for the Rewriter, Router, and QA stages
│   ├── local_call.py         # LocalQwenCaller — runs Qwen locally in VRAM (8-bit quantized)
│   ├── openrouter_call.py    # OpenRouterQwenCaller — calls Qwen through the OpenRouter API
│   └── qwen_engine.py        # QwenEngine — drives the 3-stage pipeline (Rewrite→Route→Answer), delegates the actual model call to whichever caller is active
│
└── scraper/                   # 🚧 Work in progress — pulls comments from Facebook pages
```

> **Note:** The files under `data/` in this repository are just **sample data** (from a Real Madrid vs Bayern Munich match), not permanent data. These files are regenerated/overwritten automatically on every `update-knowledge` call.

---

## Requirements

- Python 3.10
- A CUDA 12.1-capable GPU (optional but strongly recommended — models will run on CPU otherwise, but very slowly)

Core dependencies (`requirements.txt`):
- `fastapi`, `uvicorn` — API layer
- `pydantic` — validation
- `sentence-transformers` — BGE-M3 embeddings
- `bertopic`, `umap-learn`, `hdbscan`, `scikit-learn`, `nltk` — clustering
- `transformers`, `accelerate`, `bitsandbytes` — running Qwen locally (8-bit quantized)
- `openai` — OpenRouter client (OpenRouter exposes an OpenAI-compatible API)

---

## Getting Started

> **Note:** Docker packaging is still in progress and not part of this repo yet.

### Locally
```bash
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Default: Qwen runs via the OpenRouter API
export OPENROUTER_API_KEY=your_key_here

# Optional: run Qwen locally instead (see "Qwen Backend: Local vs. OpenRouter" below)
# export QWEN_USE_LOCAL=true

uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will be available at: `http://localhost:8000`

---

## API Endpoints

### `GET /health`
Returns server status: whether the GPU lock is currently held, and which model (if any) is loaded.

### `POST /api/v1/update-knowledge`
Takes a post and its comments, runs clustering, and refreshes the knowledge base.

**Request body (example):**
```json
{
  "page_name": "Global Football Live",
  "post_id": "FB_POST_99281_2026",
  "post_user_id": "page_owner_id",
  "post_content": "Full time! Real Madrid turn the game around and beat Bayern Munich 3-2...",
  "comments": [
    { "comment_id": "fb_c_001", "user_id": "user_ahmed_cairo", "text": "..." }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Knowledge base updated and cache cleared.",
  "clusters_found": 2
}
```

### `POST /api/v1/ask`
Takes a question in any language/dialect, and returns an answer grounded in the latest knowledge update.

**Request body:**
```json
{ "question": "What do people think about Ancelotti's substitutions?" }
```

**Response:**
```json
{
  "source": "Qwen Model",
  "answer": "...",
  "clusters_used": [1]
}
```

---

## Key Design Notes

- **The cache is fully wiped** on every knowledge update, to guarantee no stale answers are served from data that no longer exists.
- **GPU lock (`asyncio.Lock`)** ensures the embedding model and the LLM never run on the GPU at the same time.
- **Double cache-check** before and after acquiring the lock in `/ask`, to avoid race conditions when multiple concurrent requests hit the same question.
- **Text sanitization** (`string_cleaner`) strips control characters and prevents issues that could break JSON or cause unintended injection.
- Answers are **always returned in the same language as the user's question**, regardless of the original comments' language (translated internally when needed).
- **Pluggable Qwen backend** — `QwenEngine` delegates the raw model call to either `LocalQwenCaller` or `OpenRouterQwenCaller`, switchable via env vars with no code changes.

---

## Work in Progress 🚧

- **`scraper/`**: module for automatically pulling comments from Facebook pages (still in development) — will be the actual data source feeding `/update-knowledge` instead of manual input.
- **Docker packaging**: containerized build/run setup, still being finalized for deployment.
- Incremental/partial knowledge updates instead of a full cache wipe every time.
- Persisting clusters in a database instead of a flat text file.

---
