from pydantic import BaseModel, Field, field_validator
from typing import List
from core.string_cleaner import clean_text_for_json

class FBComment(BaseModel):
    comment_id: str
    user_id: str
    text: str

    @field_validator("comment_id", "user_id", "text")
    @classmethod
    def clean_comment_fields(cls, v):
        return clean_text_for_json(v)

class PostInput(BaseModel):
    page_name: str
    post_id: str
    post_user_id: str
    post_content: str
    comments: List[FBComment] = Field(..., min_length=1, max_length=2000)

    @field_validator("page_name", "post_id", "post_user_id", "post_content")
    @classmethod
    def clean_metadata_fields(cls, v):
        return clean_text_for_json(v)

    @field_validator("comments")
    @classmethod
    def filter_empty_comments(cls, v):
        cleaned = [c for c in v if c.text.strip()]
        if not cleaned:
            raise ValueError("All provided comments contain only whitespace.")
        return cleaned


class CommentInput(BaseModel):
    posts: List[PostInput] = Field(..., min_length=1, max_length=50)

    @field_validator("posts")
    @classmethod
    def unique_post_ids(cls, v):
        ids = [p.post_id for p in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate post_id values found across posts.")
        return v

class OptimizedCluster(BaseModel):
    topic_id: int
    topic_keywords: str
    representative_text: str
    frequency: int
    similar_docs_count: int
    page_name: str
    post_id: str

class PipelineResponse(BaseModel):
    num_clusters: int
    noise_count: int
    optimized_data: List[OptimizedCluster]
    failed_posts: List[str] = Field(default_factory=list)