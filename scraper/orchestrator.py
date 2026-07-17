import sys
import json
import os
import requests
import logging

# Add the root project directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scraper.adapters.facebook_adapter import FacebookAdapter
# from scraper.adapters.reddit_adapter import RedditAdapter

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def send_to_backend(posts_data: list, api_url: str):
    if not posts_data:
        logger.warning("No posts to send.")
        return

    payload = {"posts": posts_data}
    
    try:
        logger.info(f"Sending {len(posts_data)} posts to backend API: {api_url}")
        # The backend clustering can take a while if there are many comments, so we set a long timeout
        response = requests.post(api_url, json=payload, timeout=300)
        response.raise_for_status()
        logger.info(f"Backend response: {response.json()}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send data to backend: {e}")

def run():
    logger.info("Starting Scraper Orchestrator...")
    config = load_config()
    
    # 1. Scrape Facebook
    fb_adapter = FacebookAdapter()
    for fb_page in config.get("facebook_pages", []):
        # Increased limit to 50 to ensure we capture a full day's worth of posts
        for post in fb_adapter.scrape_latest_posts(fb_page, limit=2):
            # Send each post instantly to backend! If the script stops, the data is already safe.
            send_to_backend([post], config.get("api_endpoint"))
        
    # 2. Scrape Reddit (when implemented)
    # reddit_adapter = RedditAdapter()
    # for sub in config.get("reddit_subs", []):
    #     for post in reddit_adapter.scrape_latest_posts(sub):
    #         send_to_backend([post], config.get("api_endpoint"))

if __name__ == "__main__":
    run()
