import json
import os
import time
import logging

logger = logging.getLogger(__name__)

DEBUG_DIR = "data/debug_clusters"

# Off by default so production runs don't write a file on every /update-knowledge
# call. Set SAVE_DEBUG_CLUSTERS=true in the environment to enable while testing.
SAVE_DEBUG_CLUSTERS = os.getenv("SAVE_DEBUG_CLUSTERS", "false").strip().lower() in ("1", "true", "yes")


def save_debug_clusters(result: dict, posts_metadata: dict) -> str | None:
    """
    Dumps the raw clustering output (pipeline.process() result) to disk for
    inspection, independent of what actually ends up written into Chroma.
    Filename is timestamped so repeated test runs don't overwrite each other.

    Returns the filepath written, or None if SAVE_DEBUG_CLUSTERS is not enabled.
    """
    if not SAVE_DEBUG_CLUSTERS:
        return None

    os.makedirs(DEBUG_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(DEBUG_DIR, f"clusters_{timestamp}.json")

    debug_payload = {
        "num_clusters": result.get("num_clusters"),
        "noise_count": result.get("noise_count"),
        "failed_posts": result.get("failed_posts", []),
        "posts_metadata": posts_metadata,
        "optimized_data": result.get("optimized_data", []),
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to write debug cluster dump: {e}")
        return None

    return filepath
