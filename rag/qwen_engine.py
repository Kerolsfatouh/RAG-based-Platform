import os
import re
import json
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from rag.prompt_templates import (
    ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE,
    QA_SYSTEM_PROMPT, QA_USER_TEMPLATE,
    REWRITER_SYSTEM_PROMPT, REWRITER_USER_TEMPLATE
)

logger = logging.getLogger(__name__)

class QwenEngine:
    def __init__(self, model_name):
        logger.info(f"Loading {model_name} into VRAM with INT8 Quantization...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        logger.info("Qwen is ready for inference!")

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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

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
        
        target_ids = []
        numbers = re.findall(r'\d+', router_response)
        valid_keys = list(clusters.keys())
        
        for num in numbers:
            if int(num) in valid_keys:
                target_ids.append(int(num))
                
        target_ids = list(set(target_ids))

        if not target_ids:
            logger.warning("Router failed to select specific clusters. Falling back to Full Search.")
            target_ids = valid_keys

        logger.info(f"Pass 2: Deep diving into clusters {target_ids}...")
        target_chunks = ""
        for t_id in target_ids:
            target_chunks += f"<CLUSTER {t_id}>\n{clusters[t_id]['full_chunk']}\n</CLUSTER>\n\n"

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