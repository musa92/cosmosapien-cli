"""Hugging Face model implementation."""

from typing import List, Optional

import httpx

from ..core.models import BaseModel, ChatMessage, ModelResponse


class HuggingFace(BaseModel):
    """Hugging Face model implementation."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "meta-llama/Llama-2-7b-chat-hf",
        **kwargs,
    ):
        super().__init__(api_key, **kwargs)
        self.model = model
        self.base_url = base_url or "https://api-inference.huggingface.co"
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": "Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from Hugging Face."""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": kwargs.get("max_tokens", 512),
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "do_sample": True,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["max_tokens", "temperature", "top_p"]
                    },
                },
            }

            response = await self.client.post(
                "{self.base_url}/models/{self.model}", json=payload
            )
            response.raise_for_status()
            data = response.json()

            # Handle different response formats from Hugging Face
            if isinstance(data, list) and len(data) > 0:
                content = data[0].get("generated_text", "")
                # Remove the original prompt from the response
                if content.startswith(prompt):
                    content = content[len(prompt):].strip()
            elif isinstance(data, dict):
                content = data.get("generated_text", str(data))
            else:
                content = str(data)

            return ModelResponse(
                content=content,
                model=self.model,
                provider="huggingface",
                usage={
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(content.split()),
                    "total_tokens": len(prompt.split()) + len(content.split()),
                },
                metadata={"model_id": self.model, "response_type": "generated_text"},
            )
        except Exception:
            raise Exception("Hugging Face API error: {str(e)}")

    async def chat(self, messages: List[ChatMessage], **kwargs) -> ModelResponse:
        """Generate a chat response from Hugging Face."""
        try:
            # Convert chat messages to a single prompt
            prompt = self._format_chat_messages(messages)

            # Use the generate method for chat
            return await self.generate(prompt, **kwargs)
        except Exception:
            raise Exception("Hugging Face API error: {str(e)}")

    def _format_chat_messages(self, messages: List[ChatMessage]) -> str:
        """Format chat messages into a single prompt string."""
        formatted_messages = []

        for msg in messages:
            if msg.role == "system":
                formatted_messages.append("System: {msg.content}")
            elif msg.role == "user":
                formatted_messages.append("User: {msg.content}")
            elif msg.role == "assistant":
                formatted_messages.append("Assistant: {msg.content}")

        return "\n".join(formatted_messages) + "\nAssistant:"

    def get_available_models(self) -> List[str]:
        """Get list of available Hugging Face models."""
        return [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "microsoft/DialoGPT-small",
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B",
            "EleutherAI/gpt-j-6B",
            "bigscience/bloom-560m",
            "bigscience/bloom-1b1",
            "bigscience/bloom-3b",
            "bigscience/bloom-7b1",
            "tiiuae/falcon-7b",
            "tiiuae/falcon-40b",
            "tiiuae/falcon-180b",
            "mosaicml/mpt-7b",
            "mosaicml/mpt-30b",
            "mosaicml/mpt-7b-instruct",
            "mosaicml/mpt-30b-instruct",
            "stabilityai/stablelm-base-alpha-3b",
            "stabilityai/stablelm-base-alpha-7b",
            "stabilityai/stablelm-tuned-alpha-3b",
            "stabilityai/stablelm-tuned-alpha-7b",
            "THUDM/chatglm-6b",
            "THUDM/chatglm2-6b",
            "THUDM/chatglm3-6b",
            "baichuan-inc/Baichuan-7B",
            "baichuan-inc/Baichuan-13B",
            "baichuan-inc/Baichuan2-7B-Base",
            "baichuan-inc/Baichuan2-13B-Base",
            "Qwen/Qwen-7B",
            "Qwen/Qwen-14B",
            "Qwen/Qwen-72B",
            "Qwen/Qwen-7B-Chat",
            "Qwen/Qwen-14B-Chat",
            "Qwen/Qwen-72B-Chat",
        ]

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
