from abc import ABC, abstractmethod
from typing import List, Dict

class BaseAdapter(ABC):
    @abstractmethod
    def scrape_latest_posts(self, target_url: str, limit: int = 1) -> List[Dict]:
        """
        Should return a list of dictionaries, where each dict represents a post 
        and matches the `PostInput` schema expected by the API:
        {
            "page_name": "...",
            "post_id": "...",
            "post_user_id": "...",
            "post_content": "...",
            "comments": [
                {"comment_id": "...", "user_id": "...", "text": "..."},
                ...
            ]
        }
        """
        pass
