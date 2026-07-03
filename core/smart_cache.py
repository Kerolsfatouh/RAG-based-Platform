import json
import os
import threading

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
    # Flushes the cache completely during the daily update
    with cache_lock:
        os.makedirs("data", exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)