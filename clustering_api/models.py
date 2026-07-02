from pydantic import BaseModel, Field, field_validator
from typing import List

class CommentInput(BaseModel):
    # limit the maximum length to 2000 to ensure that the timeout is never broken under any load
    comments: List[str] = Field(..., min_length=1, max_length=2000)

    @field_validator("comments")
    @classmethod
    def strip_and_filter(cls, v):
        # strip whitespace and filter out empty comments
        cleaned = [c.strip() for c in v if c.strip()]
        
        # ensure there is valid data after cleaning
        if not cleaned:
            raise ValueError("All provided comments are empty or contain only whitespace.")
            
        return cleaned

class OptimizedCluster(BaseModel):
    topic_id: int
    representative_text: str
    frequency: int
    similar_docs_count: int

class PipelineResponse(BaseModel):
    num_clusters: int
    noise_count: int
    optimized_data: List[OptimizedCluster]