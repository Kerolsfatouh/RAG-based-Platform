import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenRouterQwenCaller:
    """Calls Qwen through the OpenRouter API instead of running it locally."""

    def __init__(self, model_name: str):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        logger.info(f"Connecting to OpenRouter API for {model_name}...")
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        logger.info("OpenRouter client is ready!")

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_new_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def free_vram(self):
        """No local VRAM is held by the API client; kept for interface parity with LocalQwenCaller."""
        pass
