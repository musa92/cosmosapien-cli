"""Perplexity model implementation."""

from typing import List, Optional

import httpx

from ..core.models import BaseModel, ChatMessage, ModelResponse


class Perplexity(BaseModel):
    """Perplexity model implementation."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "llama-3.1-sonar-small-128k-online",
        **kwargs,
    ):
        super().__init__(api_key, **kwargs)
        self.model = model
        self.base_url = base_url or "https://api.perplexity.ai"
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": "Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from Perplexity."""
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                **kwargs,
            }

            response = await self.client.post(
                "{self.base_url}/chat/completions", json=payload
            )
            response.raise_for_status()
            data = response.json()

            return ModelResponse(
                content=data["choices"][0]["message"]["content"],
                model=self.model,
                provider="perplexity",
                usage=data.get("usage"),
                metadata={
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "index": data["choices"][0].get("index"),
                },
            )
        except Exception:
            raise Exception("Perplexity API error: {str(e)}")

    async def chat(self, messages: List[ChatMessage], **kwargs) -> ModelResponse:
        """Generate a chat response from Perplexity."""
        try:
            # Convert our ChatMessage format to Perplexity format
            perplexity_messages = []
            for msg in messages:
                perplexity_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )

            payload = {"model": self.model, "messages": perplexity_messages, **kwargs}

            response = await self.client.post(
                "{self.base_url}/chat/completions", json=payload
            )
            response.raise_for_status()
            data = response.json()

            return ModelResponse(
                content=data["choices"][0]["message"]["content"],
                model=self.model,
                provider="perplexity",
                usage=data.get("usage"),
                metadata={
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "index": data["choices"][0].get("index"),
                },
            )
        except Exception:
            raise Exception("Perplexity API error: {str(e)}")

    def get_available_models(self) -> List[str]:
        """Get list of available Perplexity models."""
        return [
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-small-128k",
            "llama-3.1-sonar-medium-128k-online",
            "llama-3.1-sonar-medium-128k",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-large-128k",
            "mixtral-8x7b-instruct",
            "mistral-7b-instruct",
            "codellama-70b-instruct",
            "pplx-7b-online",
            "pplx-70b-online",
            "pplx-7b-chat",
            "pplx-70b-chat",
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
