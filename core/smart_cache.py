import json
import os
import threading
import logging

logger = logging.getLogger(__name__)

CACHE_FILE = "data/smart_cache.json"
cache_lock = threading.Lock()

def init_cache():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)

def check_cache(question: str):
    with cache_lock:
        init_cache()
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        return cache.get(question.strip())

def save_to_cache(question: str, answer: str, clusters_used: list):
    with cache_lock:
        init_cache()
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        
        cache[question.strip()] = {
            "answer": answer,
            "clusters_used": clusters_used
        }
        
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)

def clear_smart_cache():
    # Flushes the cache completely -- used for deliberate full resets
    # (see /api/v1/reset-knowledge), not the normal per-post update path.
    with cache_lock:
        os.makedirs("data", exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)

def invalidate_by_post_ids(post_ids: set):
    """
    Selectively removes cached answers that relied on clusters belonging to any of
    the given post_ids, instead of wiping the entire Smart Cache. VectorDB cluster
    ids are formatted as f"{post_id}_{topic_id}_{i}", so a cached answer is stale
    if any of its stored clusters_used starts with one of these post_id prefixes.
    This keeps cached answers for unrelated posts/topics valid even while the
    scraper is continuously feeding in single posts one at a time.
    """
    if not post_ids:
        return

    prefixes = tuple(f"{pid}_" for pid in post_ids)

    with cache_lock:
        init_cache()
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

        stale_questions = [
            question for question, entry in cache.items()
            if any(str(cid).startswith(prefixes) for cid in entry.get("clusters_used", []))
        ]

        if not stale_questions:
            return

        for question in stale_questions:
            del cache[question]

        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)

        logger.info(f"Invalidated {len(stale_questions)} cached answer(s) tied to updated post(s): {sorted(post_ids)}")