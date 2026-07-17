import sys
import json
import os
import time
import requests
import logging

# Add the root project directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.adapters.facebook_adapter import FacebookAdapter
# from scraper.adapters.reddit_adapter import RedditAdapter
from scraper.retry_queue import enqueue_failed, drain_failed

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

DEFAULT_SCRAPE_LIMIT = 50   # a full day's worth of posts, used if config.json doesn't override it
DEFAULT_MAX_COMMENTS = 100  # cap per post if config.json doesn't override it; previously unbounded
MAX_SEND_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5   # doubles after each failed attempt


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Scraper config not found at {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not config.get("api_endpoint"):
        raise ValueError("config.json is missing the required 'api_endpoint' field.")

    return config


def send_to_backend(posts_data: list, api_url: str) -> bool:
    """
    Sends posts to the backend, retrying transient failures with exponential backoff.
    Returns True on success, False if every attempt failed -- the caller is then
    responsible for queuing the post(s) for a later retry instead of dropping them.
    """
    if not posts_data:
        logger.warning("No posts to send.")
        return True

    payload = {"posts": posts_data}
    delay = RETRY_BACKOFF_SECONDS

    for attempt in range(1, MAX_SEND_RETRIES + 1):
        try:
            logger.info(
                f"Sending {len(posts_data)} post(s) to backend API "
                f"(attempt {attempt}/{MAX_SEND_RETRIES}): {api_url}"
            )
            # The backend clustering can take a while if there are many comments,
            # so we set a long timeout.
            response = requests.post(api_url, json=payload, timeout=300)
            response.raise_for_status()
            logger.info(f"Backend response: {response.json()}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Attempt {attempt}/{MAX_SEND_RETRIES} failed: {e}")
            if attempt < MAX_SEND_RETRIES:
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2

    logger.error(f"Giving up on {len(posts_data)} post(s) after {MAX_SEND_RETRIES} attempts.")
    return False


def retry_previously_failed_posts(api_url: str):
    """
    Before scraping anything new, flush any posts that failed to send in a previous
    run. Anything that fails again is re-queued rather than dropped, so nothing is
    silently lost across restarts.
    """
    queued_posts = drain_failed()
    if not queued_posts:
        return

    logger.info(f"Found {len(queued_posts)} previously failed post(s) queued. Retrying...")
    for post in queued_posts:
        if not send_to_backend([post], api_url):
            enqueue_failed(post)


def run():
    logger.info("Starting Scraper Orchestrator...")

    try:
        config = load_config()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"Cannot start scraper -- invalid config: {e}")
        return

    api_endpoint = config["api_endpoint"]
    scrape_limit = config.get("scrape_limit", DEFAULT_SCRAPE_LIMIT)
    max_comments = config.get("max_comments_per_post", DEFAULT_MAX_COMMENTS)

    logger.info(f"Config: scrape_limit={scrape_limit}, max_comments_per_post={max_comments}")

    # 0. Flush anything left over from a previous run before scraping new data.
    retry_previously_failed_posts(api_endpoint)

    # 1. Scrape Facebook
    fb_pages = config.get("facebook_pages", [])
    if not fb_pages:
        logger.warning("No facebook_pages configured in config.json -- nothing to scrape.")

    fb_adapter = FacebookAdapter()
    for fb_page in fb_pages:
        try:
            posts = fb_adapter.scrape_latest_posts(fb_page, limit=scrape_limit, max_comments=max_comments)

            for post in posts:
                # Send each post instantly to the backend! If the script itself stops,
                # everything sent so far is already safe. Anything that fails to send
                # (backend down, network blip) is queued and retried on the next run
                # instead of being silently lost.
                if not send_to_backend([post], api_endpoint):
                    enqueue_failed(post)

        except Exception as e:
            # A single page failing to scrape (blocked, layout change, network issue,
            # etc.) shouldn't take down the rest of the run. Note: scrape_latest_posts
            # is a generator, so this try/except must wrap the consuming loop too --
            # otherwise exceptions raised during iteration (e.g. post_scraper giving up
            # after its retries) would propagate past this block entirely.
            logger.error(f"Failed to scrape page '{fb_page}': {e}", exc_info=True)
            continue

    # 2. Scrape Reddit (when implemented)
    # reddit_adapter = RedditAdapter()
    # for sub in config.get("reddit_subs", []):
    #     for post in reddit_adapter.scrape_latest_posts(sub):
    #         if not send_to_backend([post], api_endpoint):
    #             enqueue_failed(post)

    logger.info("Scraper Orchestrator finished.")


if __name__ == "__main__":
    run()