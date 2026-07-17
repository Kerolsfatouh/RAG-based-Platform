import os
import re
import json
import logging

from rag.local_call import LocalQwenCaller
from rag.openrouter_call import OpenRouterQwenCaller
from core.database import VectorDB
from rag.prompt_templates import (
    ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE,
    QA_SYSTEM_PROMPT, QA_USER_TEMPLATE,
    REWRITER_SYSTEM_PROMPT, REWRITER_USER_TEMPLATE
)

logger = logging.getLogger(__name__)

# How many candidates to pull from the VectorDB before letting the Router LLM narrow
# them down. Kept bigger than the final answer context so the Router has a real pool
# to choose from (and to fall back to) instead of a single fixed top-4.
CANDIDATE_POOL_SIZE = 10


class QwenEngine:
    """
    Drives the Rewrite -> VectorDB Retrieve -> Route -> QA pipeline.

    - The actual model call is delegated to a "caller" (LocalQwenCaller or
      OpenRouterQwenCaller), so this class doesn't need to know whether Qwen is running
      locally in VRAM or through the OpenRouter API.
    - Cluster retrieval is delegated to a VectorDB instance (Chroma-backed) instead of
      parsing data/daily_clusters.txt, but the Router LLM still picks which retrieved
      candidates are actually relevant and at what depth, same as before.
    """

    def __init__(self, model_name: str, use_local: bool = None, db: VectorDB = None):
        """
        model_name / use_local: see LocalQwenCaller / OpenRouterQwenCaller.
        db: a shared VectorDB instance. Pass one in (created once at app startup, the
            same way the engine itself is lazy-loaded once) instead of letting the
            engine open a new Chroma client on every request.
        """
        if use_local is None:
            use_local = os.getenv("QWEN_USE_LOCAL", "false").strip().lower() in ("1", "true", "yes")

        self.use_local = use_local
        self.model_name = model_name
        self.caller = LocalQwenCaller(model_name) if use_local else OpenRouterQwenCaller(model_name)
        self.db = db or VectorDB()

    def free_vram(self):
        """Releases any locally-held VRAM (no-op when running through the OpenRouter API)."""
        self.caller.free_vram()

    def _generate_response(self, system_prompt: str, user_prompt: str, max_new_tokens=512) -> str:
        return self.caller.generate(system_prompt, user_prompt, max_new_tokens=max_new_tokens)

    @staticmethod
    def _format_chunk(candidate: dict, depth: int) -> str:
        """
        Rebuilds a cluster's text block from a VectorDB candidate. depth=1 (Standard)
        hides sub-cluster/similar comments; depth=2 (Deep Dive) includes them -- same
        contract the Router prompt already describes, just applied to VectorDB
        metadata instead of a regex-stripped file chunk.
        """
        chunk = f"[FACEBOOK_PAGE: {candidate.get('page_name', 'N/A')}]\n"
        chunk += f"[POST_ID: {candidate.get('post_id', 'N/A')}]\n"
        chunk += f"[TOP_INTENT: {candidate.get('intent', 'General/Mixed Content')}]\n"
        chunk += f"[CONTENT]:\n{candidate.get('main_text', '')}"

        if depth == 2 and candidate.get('similar_text'):
            chunk += f"\n  |_ [Sub-cluster Hidden]\n{candidate['similar_text']}"

        return chunk

    def ask(self, question: str) -> dict:
        no_data_response = {
            "answer": "I don't have any daily data available yet. Please run the scraper/update pipeline first.",
            "clusters_used": []
        }

        # Bail out before spending an LLM call if the knowledge base is empty --
        # restores the old "check clusters before Phase 0" short-circuit.
        if self.db.is_empty():
            return no_data_response

        logger.info(f"Phase 0: Reformulating raw query: {question}")
        rewriter_prompt = REWRITER_USER_TEMPLATE.format(raw_question=question)
        optimized_query = self._generate_response(REWRITER_SYSTEM_PROMPT, rewriter_prompt, max_new_tokens=150)
        logger.info(f"Optimized Query for RAG: {optimized_query}")

        # Phase 1: pull a pool of semantically relevant candidates from the VectorDB.
        # This replaces "parse every cluster from daily_clusters.txt" -- the Router LLM
        # below narrows down *this* pool instead of the whole knowledge base.
        candidates = self.db.search(optimized_query, n_results=CANDIDATE_POOL_SIZE)
        if not candidates:
            return no_data_response

        # The Router prompt expects small integer cluster ids, so we expose the pool as
        # local indices (0..N-1) here and map back to the real VectorDB ids afterward.
        cluster_summaries = ""
        for local_id, c in enumerate(candidates):
            preview = c['main_text'][:80].strip()
            cluster_summaries += f"[{local_id}] Keywords: {c['intent']} | Preview: {preview}...\n"

        logger.info("Pass 1: Asking Qwen for intuition (Routing)...")
        router_prompt = ROUTER_USER_TEMPLATE.format(question=optimized_query, cluster_summaries=cluster_summaries)
        router_response = self._generate_response(ROUTER_SYSTEM_PROMPT, router_prompt, max_new_tokens=50)
        logger.info(f"Qwen Router output: {router_response}")

        try:
            router_output = json.loads(router_response.strip())
            target_local_ids_raw = router_output.get("target_clusters", [])
            required_depth = router_output.get("depth", 1)
        except json.JSONDecodeError:
            logger.warning("Router output was not valid JSON. Falling back to full search.")
            target_local_ids_raw = [int(num) for num in re.findall(r'\d+', router_response)]
            required_depth = 1

        valid_local_ids = set(range(len(candidates)))
        target_local_ids = list({
            int(n) for n in target_local_ids_raw if int(n) in valid_local_ids
        })

        if not target_local_ids:
            logger.warning("Router failed to select specific clusters. Falling back to Full Search (candidate pool).")
            target_local_ids = list(valid_local_ids)
            required_depth = 1

        logger.info(f"Pass 2: Deep diving into clusters {target_local_ids} with depth {required_depth}...")

        target_chunks = ""
        target_real_ids = []
        for local_id in target_local_ids:
            candidate = candidates[local_id]
            target_real_ids.append(candidate['id'])
            chunk_content = self._format_chunk(candidate, required_depth)
            target_chunks += f"<CLUSTER {local_id}>\n{chunk_content}\n</CLUSTER>\n\n"

        qa_prompt = QA_USER_TEMPLATE.format(
            target_chunks=target_chunks,
            search_intent=optimized_query,
            question=question
        )

        final_answer = self._generate_response(QA_SYSTEM_PROMPT, qa_prompt, max_new_tokens=1024)

        return {
            "answer": final_answer,
            # Real VectorDB ids (post/topic-traceable), not positional indices.
            "clusters_used": target_real_ids
        }
