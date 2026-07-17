import sys
import os
import logging
from typing import List, Dict

# Add the graphql_api folder to sys.path so we can import its powerful GraphQL modules
graphql_api_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graphql_api")
if graphql_api_dir not in sys.path:
    sys.path.append(graphql_api_dir)

from helpers import extract_user_id_from_url, fetch_comments_for_post
import post_scraper
from scraper.utils.schema_mapper import create_post_payload
from .base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

class FacebookAdapter(BaseAdapter):
    
    def scrape_latest_posts(self, target_url: str, limit: int = 1) -> List[Dict]:
        logger.info(f"Starting High-Volume GraphQL scrape for: {target_url}")
        
        # 1. Get Page ID
        page_id = extract_user_id_from_url(target_url)
        if not page_id:
            logger.error("Could not extract User ID from URL")
            return []
            
        # 2. Configure the post scraper
        post_scraper.USER_ID = page_id
        post_scraper.BASE_HEADERS["referer"] = f"https://www.facebook.com/profile.php?id={page_id}"
        

        
        # 3. Fetch Posts using the ultra-fast GraphQL backend
        logger.info(f"Fetching {limit} posts using GraphQL...")
        posts = post_scraper.fetch_posts(limit)
        
        all_posts_data = []
        
        for i, post in enumerate(posts):
            post_id = post.get("post_id")
            if not post_id:
                continue
                
            page_name = post.get("page_name", "Unknown Page")
            post_content = post.get("text", "No text found")
            
            logger.info(f"[{i+1}/{len(posts)}] Processing post {post_id} - {str(post_content)[:40]}...")
            
            # 4. Fetch Comments (GraphQL pulls hundreds of comments instantly!)
            try:
                comments_data_raw, _ = fetch_comments_for_post(post_id)
                
                # Filter and format comments
                valid_comments = []
                for c in comments_data_raw:
                    c_text = c.get("text", "").strip()
                    # Keep comments that have meaningful length
                    if len(c_text) > 15:
                        valid_comments.append({
                            "comment_id": "unknown",
                            "user_id": "anonymous_fan",
                            "text": c_text
                        })
                        
                logger.info(f"Extracted {len(valid_comments)} substantial comments for post {post_id}")
                
                if valid_comments:
                    post_payload = create_post_payload(
                        page_name=page_name,
                        post_id=post_id,
                        post_user_id=page_id,
                        post_content=post_content,
                        comments=valid_comments
                    )
                    
                    # Save a local backup JSON with the comments
                    try:
                        safe_page_name = "".join(c for c in page_name if c.isalnum() or c in (' ', '-', '_')).strip()
                        save_dir = os.path.join("page_post", safe_page_name, post_id)
                        os.makedirs(save_dir, exist_ok=True)
                        with open(os.path.join(save_dir, f"{post_id}.json"), "w", encoding="utf-8") as f:
                            import json
                            json.dump(post_payload, f, ensure_ascii=False, indent=2)
                        logger.info(f"Saved local JSON backup to {save_dir}/{post_id}.json")
                    except Exception as json_e:
                        logger.error(f"Could not save local JSON: {json_e}")
                        
                    yield post_payload
            except Exception as e:
                logger.error(f"Error fetching comments for {post_id}: {e}")
