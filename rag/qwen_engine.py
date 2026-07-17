import os
import re
import json
import logging

from rag.local_call import LocalQwenCaller
from rag.openrouter_call import OpenRouterQwenCaller
from core.database import VectorDB
from rag.prompt_templates import (
    QA_SYSTEM_PROMPT, QA_USER_TEMPLATE,
    REWRITER_SYSTEM_PROMPT, REWRITER_USER_TEMPLATE
)

logger = logging.getLogger(__name__)


class QwenEngine:
    """
    Drives the Rewrite -> VectorDB Retrieval -> QA pipeline. The actual model call is delegated
    to a "caller" (LocalQwenCaller or OpenRouterQwenCaller) so this class doesn't need to
    know or care whether Qwen is running locally in VRAM or through the OpenRouter API.
    """

    def __init__(self, model_name: str, use_local: bool = None):
        """
        model_name:
            - use_local=False (API mode): OpenRouter model slug, e.g. "qwen/qwen2.5-vl-72b-instruct"
            - use_local=True (local mode): local/HF model id or path, e.g. "Qwen/Qwen3.5-4B"
        use_local:
            Picks which caller backs this engine. Defaults to the QWEN_USE_LOCAL env var
            ("true"/"1" -> local), falling back to False (OpenRouter API).
        """
        if use_local is None:
            use_local = os.getenv("QWEN_USE_LOCAL", "false").strip().lower() in ("1", "true", "yes")

        self.use_local = use_local
        self.model_name = model_name
        self.caller = LocalQwenCaller(model_name) if use_local else OpenRouterQwenCaller(model_name)

    def free_vram(self):
        """Releases any locally-held VRAM (no-op when running through the OpenRouter API)."""
        self.caller.free_vram()

    def _generate_response(self, system_prompt: str, user_prompt: str, max_new_tokens=512) -> str:
        return self.caller.generate(system_prompt, user_prompt, max_new_tokens=max_new_tokens)

    def ask(self, question: str) -> dict:
        # 1. Connect to Vector Database
        db = VectorDB()

        # 2. Rewrite the User's Query
        logger.info(f"Phase 0: Reformulating raw query: {question}") 
        rewriter_prompt = REWRITER_USER_TEMPLATE.format(raw_question=question) 
        optimized_query = self._generate_response(REWRITER_SYSTEM_PROMPT, rewriter_prompt, max_new_tokens=150) 
        logger.info(f"Optimized Query for RAG: {optimized_query}")

        # 3. Vector DB Similarity Search (Instantly finds best clusters, no "Router LLM" needed!)
        logger.info("Phase 1: Retrieving relevant data from Vector DB...")
        target_chunks_list = db.search(optimized_query, n_results=4)
        
        if not target_chunks_list:
            return {
                "answer": "I don't have any daily data available yet. Please run the scraper/update pipeline first.",
                "clusters_used": []
            }

        # 4. Format the retrieved chunks for the QA prompt
        target_chunks = ""
        for i, chunk in enumerate(target_chunks_list):
            target_chunks += f"<CLUSTER {i}>\n{chunk}\n</CLUSTER>\n\n"

        # 5. Generate Grounded Answer
        logger.info("Phase 2: Generating grounded answer based on Vector DB context...")
        qa_prompt = QA_USER_TEMPLATE.format(
            target_chunks=target_chunks, 
            search_intent=optimized_query,
            question=question
        ) 
        
        final_answer = self._generate_response(QA_SYSTEM_PROMPT, qa_prompt, max_new_tokens=1024) 

        return {
            "answer": final_answer,
            "clusters_used": list(range(len(target_chunks_list))) # We just use the index of retrieved chunks
        }
