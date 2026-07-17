def create_post_payload(page_name: str, post_id: str, post_user_id: str, post_content: str, comments: list) -> dict:
    """
    Ensures the data perfectly matches the backend PostInput Pydantic model.
    """
    # Filter out empty comments
    valid_comments = [c for c in comments if c.get("text") and str(c.get("text")).strip()]
    
    return {
        "page_name": page_name,
        "post_id": post_id,
        "post_user_id": post_user_id,
        "post_content": post_content,
        "comments": valid_comments
    }
