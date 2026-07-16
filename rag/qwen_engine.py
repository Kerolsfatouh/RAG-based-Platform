import os
import re
import json
import logging

from rag.local_call import LocalQwenCaller
from rag.openrouter_call import OpenRouterQwenCaller
from rag.prompt_templates import (
    ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE,
    QA_SYSTEM_PROMPT, QA_USER_TEMPLATE,
    REWRITER_SYSTEM_PROMPT, REWRITER_USER_TEMPLATE
)

logger = logging.getLogger(__name__)


class QwenEngine:
    """
    Drives the 3-stage Rewrite -> Route -> QA pipeline. The actual model call is delegated
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

    def _parse_clusters_file(self, filepath="data/daily_clusters.txt"):
        if not os.path.exists(filepath):
            return {}

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        clusters = {}
        blocks = re.findall(r"<CLUSTER_START>(.*?)<CLUSTER_END>", content, re.DOTALL)

        for block in blocks:
            id_match = re.search(r"\[CLUSTER_ID:\s*(\d+)\]", block)
            intent_match = re.search(r"\[TOP_INTENT:\s*(.*?)\]", block)
            content_match = re.search(r"\[CONTENT\]:\s*(.*)", block, re.DOTALL)

            if id_match and intent_match and content_match:
                c_id = int(id_match.group(1))
                clusters[c_id] = {
                    "intent": intent_match.group(1).strip(),
                    "content": content_match.group(1).strip(),
                    "full_chunk": block.strip()
                }
        return clusters

    def _generate_response(self, system_prompt: str, user_prompt: str, max_new_tokens=512) -> str:
        return self.caller.generate(system_prompt, user_prompt, max_new_tokens=max_new_tokens)

    def ask(self, question: str) -> dict:
        clusters = self._parse_clusters_file()
        if not clusters:
            return {"answer": "I don't have any daily data available yet. Please run the update pipeline first."}

        logger.info(f"Phase 0: Reformulating raw query: {question}")
        rewriter_prompt = REWRITER_USER_TEMPLATE.format(raw_question=question)
        optimized_query = self._generate_response(REWRITER_SYSTEM_PROMPT, rewriter_prompt, max_new_tokens=150)
        logger.info(f"Optimized Query for RAG: {optimized_query}")

        cluster_summaries = ""
        for c_id, data in clusters.items():
            preview = data['content'].split('\n')[0].replace('- ', '').strip()[:80]
            cluster_summaries += f"[{c_id}] Keywords: {data['intent']} | Preview: {preview}...\n"

        logger.info("Pass 1: Asking Qwen for intuition (Routing)...")
        router_prompt = ROUTER_USER_TEMPLATE.format(question=optimized_query, cluster_summaries=cluster_summaries)

        router_response = self._generate_response(ROUTER_SYSTEM_PROMPT, router_prompt, max_new_tokens=50)
        logger.info(f"Qwen Router output: {router_response}")

        try:
            router_output = json.loads(router_response.strip())
            target_ids_raw = router_output.get("target_clusters", [])
            required_depth = router_output.get("depth", 1)
        except json.JSONDecodeError:
            logger.warning("Router output was not valid JSON. Falling back to full search.")
            target_ids_raw = [int(num) for num in re.findall(r'\d+', router_response)]
            required_depth = 1

        target_ids = []
        valid_keys = list(clusters.keys())

        for num in target_ids_raw:
            if int(num) in valid_keys:
                target_ids.append(int(num))

        target_ids = list(set(target_ids))

        if not target_ids:
            logger.warning("Router failed to select specific clusters. Falling back to Full Search.")
            target_ids = valid_keys
            required_depth = 1
        logger.info(f"Pass 2: Deep diving into clusters {target_ids} with depth {required_depth}...")

        target_chunks = ""
        for t_id in target_ids:
            chunk_content = clusters[t_id]['full_chunk']

            if required_depth == 1:
                chunk_content = re.sub(r'\|_ \[Sub-cluster Hidden\].*?(?=- \[Main\]|<CLUSTER_END>)', '', chunk_content, flags=re.DOTALL)

            target_chunks += f"<CLUSTER {t_id}>\n{chunk_content}\n</CLUSTER>\n\n"

        qa_prompt = QA_USER_TEMPLATE.format(
            target_chunks=target_chunks,
            search_intent=optimized_query,
            question=question
        )

        final_answer = self._generate_response(QA_SYSTEM_PROMPT, qa_prompt, max_new_tokens=1024)

        return {
            "answer": final_answer,
            "clusters_used": target_ids
        }
