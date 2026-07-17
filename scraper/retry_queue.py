import json
import os
import threading
import logging

logger = logging.getLogger(__name__)

QUEUE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "failed_posts.jsonl")
_lock = threading.Lock()


def _ensure_data_dir():
    os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)


def enqueue_failed(post: dict):
    """Appends a post that failed to reach the backend after all retries, so it
    isn't lost -- it will be retried automatically on the next orchestrator run."""
    _ensure_data_dir()
    with _lock:
        with open(QUEUE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(post, ensure_ascii=False) + "\n")


def drain_failed() -> list:
    """Reads and clears every queued failed post, returning them for retry.
    Any post that fails again should be re-queued by the caller via enqueue_failed()
    rather than assumed to be flushed for good."""
    _ensure_data_dir()
    with _lock:
        if not os.path.exists(QUEUE_FILE):
            return []

        with open(QUEUE_FILE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Clear the file now; callers re-enqueue anything that fails again so we
        # never hold the lock across a slow network retry.
        open(QUEUE_FILE, "w", encoding="utf-8").close()

    posts = []
    for line in lines:
        try:
            posts.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"Skipping corrupted queued post entry: {line[:80]}...")
    return posts
