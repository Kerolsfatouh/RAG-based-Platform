# RAG-based Platform for Fan Communities

An **Agentic RAG** system built to analyze comments, turn them into a queryable knowledge base, and let users ask natural-language questions (up to 100 languages) that get answered based on what fans actually said.

The core idea: instead of someone scrolling through thousands of comments under a post, the system **clusters** semantically similar comments, summarizes them, and exposes them to an LLM (Qwen) through a RAG pipeline with a smart router that decides which cluster(s) likely hold the answer.

---

## Architecture Overview

```
Scraper (Facebook GraphQL) ──► POST /api/v1/update-knowledge ──► Clustering Pipeline ──► ChromaDB
        │                              (batch of posts,                (per-post,               (VectorDB)
        │                               1+ per request)                 BGE-M3 + BERTopic)              │
        │                                                                                                │
        ▼                                                                                                ▼
retry_queue.jsonl                                                                              Smart Cache invalidation
(failed sends, retried                                                                        (only for posts that
 next run)                                                                                      actually changed)

User Question ──► POST /api/v1/ask ──► Smart Cache (hit?) ──► Qwen Rewrite ──► VectorDB search
                                                                                     │
                                                                                     ▼
                                                                          Qwen Router (picks clusters + depth)
                                                                                     │
                                                                                     ▼
                                                                          Qwen QA (answers from selected context)
```

### Knowledge Update Flow
1. A **batch** of one or more posts (each with its own comments) arrives via `POST /api/v1/update-knowledge` (Pydantic validation + text sanitization, `clustering/models.py`).
2. `ClusteringPipeline.process()` runs **independently per post** (topics from different posts are never mixed):
   - Embeds every comment using **BAAI/bge-m3**.
   - If there are fewer than 15 comments in that post → skips BERTopic, everything goes into a single cluster. Keywords for this cluster are still derived from real word-frequency counting over the post's own comments (not a static placeholder).
   - Otherwise → **UMAP** for dimensionality reduction, then **HDBSCAN** clustering orchestrated through **BERTopic**.
   - If BERTopic itself fails (e.g. empty vocabulary from a stopword-heavy topic), falls back to the same single-cluster path instead of crashing the whole request.
   - Within each topic, `community_detection` (Sentence-Transformers) groups near-duplicate comments together and picks a representative text + frequency count.
   - A post that throws during clustering (bad encoding, degenerate text, etc.) is skipped and reported in `failed_posts` — it does **not** take down the rest of the batch.
3. `VectorDB` (`core/database.py`, Chroma-backed) stores each cluster directly with its metadata (page name, post id, main text, similar/sub-cluster text). Only posts that **actually produced new clusters this run** have their old clusters deleted first — a post that failed clustering keeps its previously stored data untouched instead of being wiped with nothing to replace it.
4. The Smart Cache is **selectively invalidated** — only cached answers that depended on clusters from the posts just updated are removed (matched by `post_id` prefix on `clusters_used`), not the whole cache. This matters because the scraper feeds posts in continuously, one at a time.
5. Optionally, the raw pre-storage clustering output can be dumped to `data/debug_clusters/` for inspection (see [Debugging](#debugging)).
6. VRAM is cleared after every heavy step (`vram_manager`), since the embedding model and the LLM can't both fit in memory at once in local mode.

> `core/chunk_formatter.py` exists in the codebase but is **not currently used** by the live flow — cluster storage goes through `VectorDB.add_clusters()` directly, and context formatting at query time is handled by `QwenEngine._format_chunk()` instead.

### Ask Flow
1. `POST /api/v1/ask` receives the user's question.
2. It first checks the **Smart Cache** (`data/smart_cache.json`) — if this exact question was asked before, the cached answer is returned immediately, skipping model inference entirely. A second cache check happens right after the GPU lock is acquired, to avoid two concurrent identical requests both triggering a redundant LLM call.
3. If there's no cache hit, **Qwen** is lazy-loaded if it isn't already in memory — either locally (8-bit quantized, in VRAM) or through the OpenRouter API, depending on the configured backend (see [Qwen Backend](#qwen-backend-local-vs-openrouter)).
4. If the knowledge base is completely empty, the request short-circuits with a "no data yet" response before spending any LLM call.
5. `QwenEngine.ask()` then runs a 3-stage process:
   - **Rewriter**: turns the user's raw (often slang) question into a clean, well-formed search query.
   - **Retrieval**: pulls a pool of the top semantically relevant clusters from ChromaDB.
   - **Router**: shows the model a summary of each retrieved cluster (keywords + preview) and asks it to pick the cluster ID(s) likely to contain the answer, plus a retrieval **depth** (`1` = representative comment only, `2` = also include similar/sub-cluster comments). If the router's output isn't valid JSON, a regex fallback extracts candidate cluster numbers instead; if nothing valid is selected, the full retrieved pool is used as a fallback.
   - **QA**: feeds the content of the selected clusters (at the requested depth) as context and generates the final answer, **in the same language as the user's raw question**.
6. The answer is cached (unless it's a "don't know" response) and returned to the user.

---

## GPU Lock & VRAM Management

A `global_gpu_lock` (`asyncio.Lock`) prevents two models from running on the GPU at the same time (the embedding model and the LLM), since both together wouldn't fit in the available VRAM in local mode.

- In **local Qwen mode**, loading one model automatically frees the other first if it's currently resident (`_ensure_pipeline_loaded` / `_ensure_qwen_loaded` in `main.py`).
- In **OpenRouter/API mode**, Qwen never touches local VRAM at all — so the embedding pipeline (BGE-M3) is **not** automatically freed during normal operation; it stays warm indefinitely for fast repeat clustering, and is only released when the server shuts down.
- To manually release whatever is currently loaded (useful on a GPU-constrained box, or mid-testing): `POST /api/v1/free-vram`.
- To automatically free the embedding pipeline after every single `/update-knowledge` call instead of keeping it warm, set `AUTO_FREE_EMBEDDING_AFTER_UPDATE=true`. This trades a reload cost on the next update for a smaller idle VRAM footprint.

---

## Qwen Backend: Local vs. OpenRouter

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
| `AUTO_FREE_EMBEDDING_AFTER_UPDATE` | `false` | Free BGE-M3 from VRAM after every `/update-knowledge` call instead of keeping it warm |
| `SAVE_DEBUG_CLUSTERS` | `false` | Dump raw pre-storage clustering output to `data/debug_clusters/` on every update |

`GET /health` reports which backend and model are currently configured (`qwen_backend`, `qwen_model`).

> **Important:** `main.py` reads these purely from process environment variables (`os.getenv`) — it does **not** call `load_dotenv()`. A `.env` file only works if something loads it first (the scraper's own modules do call `load_dotenv()` internally, so `.env` works for scraper-related vars regardless). For vars consumed directly by `main.py` (`QWEN_USE_LOCAL`, `OPENROUTER_API_KEY`, etc.), either export them in your shell before starting `uvicorn`, or add an explicit `load_dotenv()` call at the top of `main.py`.

---

## Scraper Pipeline

The scraper pulls posts and their full comment threads directly from Facebook's internal GraphQL API (no browser automation), and streams them to `/update-knowledge` one post at a time.

**Flow:** `POST /api/v1/scrape` → runs `scraper/orchestrator.py` as a background task →

1. Drains any posts that failed to send in a previous run (`scraper/retry_queue.py`, backed by `data/failed_posts.jsonl`) and retries them first.
2. For each page configured in `scraper/config.json`:
   - `FacebookAdapter` resolves the page URL to a numeric page ID, then paginates posts via `post_scraper.py` up to `scrape_limit`.
   - For each post, paginates comments via `comment_scraper.py` up to `max_comments_per_post`, then fetches replies for each collected comment (currently uncapped).
   - Comments shorter than 15 characters are dropped.
   - Each post is sent to the backend **immediately, one at a time** — so a mid-run crash doesn't lose already-sent posts, and a single page failing to scrape doesn't take down the rest of the run.
3. A local JSON backup of each scraped post+comments is also saved under `data/page_post/<page_name>/<post_id>/`.
4. Proxy rotation (`scraper/graphql_api/proxy_utils.py`) automatically kicks in on proxy errors, IP blocks, or rate limiting (HTTP 407/403/429/503, or a checkpoint/login-wall response body).

**`scraper/config.json`:**
```json
{
  "facebook_pages": ["https://www.facebook.com/realmadrid"],
  "reddit_subs": ["soccer"],
  "api_endpoint": "http://localhost:8000/api/v1/update-knowledge",
  "scrape_limit": 5,
  "max_comments_per_post": 30
}
```
| Key | Default if omitted | Purpose |
|---|---|---|
| `scrape_limit` | `50` | Max posts to pull per configured page, per scrape run |
| `max_comments_per_post` | `100` | Max top-level comments to pull per post before stopping pagination |

> **Reddit support is not implemented.** `RedditAdapter.scrape_latest_posts()` is a stub that always returns an empty list, and it's commented out entirely in the orchestrator.

> **Facebook auth:** the scraper currently expects session cookies / an `fb_dtsg` token to be set (`COOKIES`, `FB_DTSG` globals in `comment_scraper.py` / `post_scraper.py`, currently empty by default) for reliable comment access — logged-out requests to public pages may return partial or no data.

---

## Debugging

Set `SAVE_DEBUG_CLUSTERS=true` to have every `/update-knowledge` call write the raw clustering output (before it's written to ChromaDB) to `data/debug_clusters/clusters_<timestamp>.json` — includes `num_clusters`, `noise_count`, `failed_posts`, and the full `optimized_data` list. Useful for inspecting cluster quality independently of what the `/api/v1/knowledge` dashboard shows (which only reflects what actually made it into storage).

---

## Project Structure

```
project_root/
│
├── main.py                    # FastAPI app + endpoints + lifespan + GPU lock
├── requirements.txt
├── .env                        # Environment variables (see tables above)
│
├── clustering/
│   ├── models.py               # Pydantic schemas (CommentInput / PostInput / FBComment, response models)
│   └── pipeline.py             # BGE-M3 embeddings + UMAP/HDBSCAN/BERTopic clustering, per-post
│
├── core/
│   ├── database.py             # VectorDB — ChromaDB wrapper (add/search/delete/clear)
│   ├── chunk_formatter.py      # Cluster→text formatter (currently unused by the live flow)
│   ├── smart_cache.py          # Question/answer cache (JSON file-based, selective invalidation)
│   ├── string_cleaner.py       # Strips control characters and safely escapes text
│   ├── vram_manager.py         # Frees RAM/VRAM after heavy operations
│   └── debug_dump.py           # Optional raw cluster-output dump for debugging
│
├── data/                        # (auto-generated at runtime)
│   ├── chroma_db/              # Persistent Vector Database (Chroma)
│   ├── page_post/               # Local JSON backups of raw scraped threads
│   ├── debug_clusters/          # Raw clustering dumps (if SAVE_DEBUG_CLUSTERS=true)
│   ├── failed_posts.jsonl       # Posts that failed to reach the backend, queued for retry
│   └── smart_cache.json         # Cached questions and answers
│
├── rag/
│   ├── prompt_templates.py     # System/user prompts for the Rewriter, Router, and QA stages
│   ├── local_call.py           # LocalQwenCaller — runs Qwen locally in VRAM (8-bit quantized)
│   ├── openrouter_call.py      # OpenRouterQwenCaller — calls Qwen through the OpenRouter API
│   └── qwen_engine.py          # QwenEngine — drives Rewrite→Retrieve→Route→QA, delegates the model call to whichever caller is active
│
└── scraper/
    ├── config.json              # Scraper configuration (pages, limits, backend endpoint)
    ├── orchestrator.py          # Coordinates the full scrape → send → retry flow
    ├── retry_queue.py           # Persists posts that failed to send, for retry on the next run
    ├── utils/
    │   ├── schema_mapper.py     # Maps raw scraped data to the backend's PostInput schema
    │   └── stealth.py           # Random user-agent rotation helper
    ├── graphql_api/
    │   ├── post_scraper.py      # Paginates a page's posts via Facebook's GraphQL API
    │   ├── comment_scraper.py   # Paginates a post's comments + replies
    │   ├── helpers.py           # Page-ID extraction, comment-fetch orchestration
    │   └── proxy_utils.py       # Proxy selection/rotation, block/error detection
    └── adapters/
        ├── base_adapter.py      # Abstract interface all platform adapters implement
        ├── facebook_adapter.py  # Facebook implementation
        └── reddit_adapter.py    # Stub — not implemented
```

---

## Requirements

- Python 3.10
- A CUDA 12.1-capable GPU (optional but strongly recommended for local Qwen/embedding inference — models will run on CPU otherwise, but very slowly)

Core dependencies (`requirements.txt`):
- `fastapi`, `uvicorn` — API layer
- `pydantic` — validation
- `sentence-transformers` — BGE-M3 embeddings
- `bertopic`, `umap-learn`, `hdbscan`, `scikit-learn`, `nltk` — clustering
- `transformers`, `accelerate`, `bitsandbytes` — running Qwen locally (8-bit quantized)
- `openai` — OpenRouter client (OpenRouter exposes an OpenAI-compatible API)
- `requests`, `python-dotenv` — scraper HTTP calls + env loading
- `chromadb` — persistent vector storage

---

## Getting Started

> **Note:** Docker packaging is still in progress and not part of this repo yet.

### Locally
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA version
pip install -r requirements.txt

# Default: Qwen runs via the OpenRouter API
export OPENROUTER_API_KEY=your_key_here

# Optional: run Qwen locally instead (see "Qwen Backend" above)
# export QWEN_USE_LOCAL=true
# export QWEN_LOCAL_MODEL=Qwen/Qwen3.5-4B

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at: `http://localhost:8000`

To scrape and ingest data, configure `scraper/config.json` (page URLs, `scrape_limit`, `max_comments_per_post`) and then:
```bash
curl -X POST http://localhost:8000/api/v1/scrape
```

---

## API Endpoints

### `GET /health`
Returns server status: whether the GPU lock is currently held, which model (if any) is loaded, the configured Qwen backend/model, and the current `AUTO_FREE_EMBEDDING_AFTER_UPDATE` setting.

### `POST /api/v1/update-knowledge`
Takes a **batch** of one or more posts (each with its own comments), runs clustering per post, and refreshes the knowledge base.

**Request body:**
```json
{
  "posts": [
    {
      "page_name": "Global Football Live",
      "post_id": "FB_POST_99281_2026",
      "post_user_id": "page_owner_id",
      "post_content": "Full time! Real Madrid turn the game around and beat Bayern Munich 3-2...",
      "comments": [
        { "comment_id": "fb_c_001", "user_id": "user_ahmed_cairo", "text": "..." }
      ]
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Knowledge base updated.",
  "clusters_found": 2,
  "posts_requested": 1,
  "posts_succeeded": 1,
  "failed_posts": [],
  "debug_dump": null
}
```
`failed_posts` lists any `post_id`s that threw during clustering — their previous data (if any) is left untouched rather than deleted. `debug_dump` is the path to the raw cluster dump if `SAVE_DEBUG_CLUSTERS=true`, otherwise `null`.

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
  "clusters_used": ["FB_POST_99281_2026_0_1"]
}
```
`source` is `"Smart Cache"` instead of `"Qwen Model"` when the answer came from cache.

### `POST /api/v1/reset-knowledge`
Manually wipes the entire ChromaDB knowledge base **and** the Smart Cache. Intended for a deliberate full reset (e.g. before a clean re-scrape) — `/update-knowledge` itself never wipes everything, only the specific posts it's given.

### `POST /api/v1/scrape`
Triggers the scraper orchestrator as a background task. Returns immediately; scraping progress is visible in the server logs.

### `GET /api/v1/knowledge`
Returns a simple HTML dashboard showing what's currently stored in ChromaDB (page, intent/keywords, main comment text per cluster).

### `POST /api/v1/free-vram`
Manually releases whichever model(s) are currently loaded in VRAM (embedding pipeline and/or Qwen), regardless of backend mode. Mainly useful in OpenRouter/API mode, where the embedding pipeline otherwise stays resident until server shutdown.

---

## Key Design Notes

- **Selective cache invalidation** — only cached answers tied to posts that were *actually* updated this run are removed, not the whole cache. This matters because the scraper feeds posts in continuously, one at a time; wiping everything on every single post would defeat the cache's purpose.
- **Failed posts don't destroy existing data** — a post that fails clustering on a re-process attempt keeps its previously stored clusters untouched, instead of being deleted with nothing to replace it.
- **GPU lock (`asyncio.Lock`)** ensures the embedding model and the LLM never run on the GPU at the same time in local mode.
- **Double cache-check** before and after acquiring the lock in `/ask`, to avoid race conditions when multiple concurrent requests hit the same question.
- **Text sanitization** (`string_cleaner`) strips control characters and prevents issues that could break JSON or cause unintended injection.
- Answers are **always returned in the same language as the user's question**, regardless of the original comments' language (translated internally when needed).
- **Pluggable Qwen backend** — `QwenEngine` delegates the raw model call to either `LocalQwenCaller` or `OpenRouterQwenCaller`, switchable via env vars with no code changes.
- **Per-page scrape isolation** — a single Facebook page failing to scrape (blocked, layout change, network issue) doesn't stop the rest of the configured pages from being scraped.
- **Bounded scraping** — both the number of posts per page and the number of comments per post are capped via `scraper/config.json`, instead of paginating until Facebook's API runs out of pages.

---

## Known Limitations

- **Clustering needs volume to be meaningful.** With fewer than ~15 comments per post, real topic modeling (BERTopic) is skipped entirely in favor of a single cluster. Even above that threshold, short, multi-lingual, informal comments (typical of Facebook threads) need a fairly large sample (generally 50+) before BERTopic/HDBSCAN can reliably separate distinct topics — with small samples, most or all comments may collapse into one or two clusters.
- **Reply count per comment is uncapped** — while top-level comments per post are now bounded by `max_comments_per_post`, replies to each collected comment are still fetched in full.
- **Reddit is not implemented** — config and code paths exist but are stubbed/commented out.
- **Facebook auth is not yet configurable end-to-end** — session cookies and `fb_dtsg` token are hardcoded empty globals in the scraper modules rather than being sourced from config/env.

---

## Completed Milestones 

- **High-Volume Facebook Scraper**: Direct, ultra-fast GraphQL API integration (no browser automation). Extracts full post comment threads natively and saves local JSON backups.
- **Streaming Pipeline**: The scraper streams posts to `/update-knowledge` one at a time in real-time, avoiding memory bottlenecks and ensuring a mid-run crash doesn't lose already-sent data.
- **Persistent Vector Storage (ChromaDB)**: Clusters are embedded and persisted into **ChromaDB** instead of relying on ephemeral memory or flat text files.
- **Hybrid LLM Infrastructure (OpenRouter)**: `.env`-driven toggle between running Qwen locally via HuggingFace or offloading generation to OpenRouter (e.g. `qwen/qwen2.5-vl-72b-instruct`).
- **Resilient batch clustering**: A single post's clustering failure no longer takes down a whole `/update-knowledge` batch, and no longer silently deletes that post's previously good data.
- **Bounded, configurable scraping**: post and comment limits are explicit, config-driven, and enforced during pagination rather than relying on the source API to run out of pages.
- **VRAM visibility & control**: manual `/api/v1/free-vram` endpoint and optional auto-free-after-update flag, for GPU-constrained environments.

## Work in Progress 

- **Docker packaging**: containerized build/run setup, still being finalized for deployment.
- **Reply-count limiting**: capping replies per comment, mirroring the existing top-level comment cap.
- **Configurable Facebook session auth**: sourcing cookies/`fb_dtsg` from config or environment instead of hardcoded empty globals.
- **Reddit adapter implementation**.

---
