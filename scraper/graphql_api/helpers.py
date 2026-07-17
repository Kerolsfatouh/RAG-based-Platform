import base64
import json
import os
import time
import requests
import re
from html import unescape
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import scraper modules
from comment_scraper import fetch_comments, fetch_replies, fb_json, GRAPHQL, PROXIES


def extract_user_id_from_url(url, cookies=None):
    """Extract Facebook User ID from a profile URL"""
    # First, try to extract ID directly from URL
    url_patterns = [
        r'profile\.php\?id=(\d+)',
        r'/profile/(\d+)',
        r'id=(\d+)'
    ]
    
    for pattern in url_patterns:
        match = re.search(pattern, url)
        if match:
            user_id = match.group(1)
            print(f"  ✅ Found User ID in URL: {user_id}")
            return user_id
    
    # If no ID in URL, fetch the page and search in HTML
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    try:
        print(f"  No ID in URL, fetching page: {url}")
        response = requests.get(url, headers=headers, cookies=cookies, proxies=PROXIES, timeout=20)
        html = response.text
        
        # Try multiple patterns to find user ID in HTML
        patterns = [
            r'fb://profile/(\d+)',           # BEST signal
            r'"profile_owner":"(\d+)"',
            r'"userID":"(\d+)"',
            r'owner_id=(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                user_id = match.group(1)
                print(f"  ✅ Found User ID: {user_id}")
                return user_id
        
        print("  ❌ User ID not found (profile may be private or login wall)")
        return None
    
    except Exception as e:
        print(f"  ❌ Error fetching URL: {e}")
        return None


def convert_post_id_to_feedback_id(post_id):
    """Convert post_id to feedback_id using base64 encoding"""
    feedback_id = base64.b64encode(f"feedback:{post_id}".encode()).decode()
    return feedback_id


def fetch_comments_for_post(post_id, cookies=None, max_comments=None):
    """
    Fetch all comments and replies for a given post_id.

    max_comments: if set, caps how many top-level comments are pulled for this
    post before stopping (see comment_scraper.fetch_comments). Replies for each
    collected comment are still fetched in full.
    """
    feedback_id = convert_post_id_to_feedback_id(post_id)
    print(f"  Fetching comments for post {post_id}...")
    print(f"  Using feedback_id: {feedback_id}")
    
    all_data = []
    comments, post_info = fetch_comments(feedback_id, cookies=cookies, max_comments=max_comments)
    
    for c in comments:
        print(f"    🗨️ {c.get('text', '')[:50]}...")
        c["replies"] = fetch_replies(c, cookies=cookies)
        
        for r in c["replies"]:
            print(f"       ↳ {r.get('text', '')[:50]}...")
        
        # Remove internal fields before appending
        c_clean = {k: v for k, v in c.items() if not k.startswith('_')}
        all_data.append(c_clean)
    
    print(f"  ✓ Found {len(all_data)} comments")
    return all_data, post_info