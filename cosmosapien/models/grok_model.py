"""Grok (xAI) model implementation."""

from typing import List, Optional

import httpx

from ..core.models import BaseModel, ChatMessage, ModelResponse


class Grok(BaseModel):
    """Grok (xAI) model implementation."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "grok-beta",
        **kwargs,
    ):
        super().__init__(api_key, **kwargs)
        self.model = model
        self.base_url = base_url or "https://api.x.ai/v1"
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": "Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from Grok."""
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
                provider="grok",
                usage=(
                    {
                        "prompt_tokens": data["usage"]["prompt_tokens"],
                        "completion_tokens": data["usage"]["completion_tokens"],
                        "total_tokens": data["usage"]["total_tokens"],
                    }
                    if "usage" in data
                    else None
                ),
                metadata={
                    "finish_reason": data["choices"][0]["finish_reason"],
                    "index": data["choices"][0]["index"],
                },
            )
        except Exception:
            raise Exception("Grok API error: {str(e)}")

    async def chat(self, messages: List[ChatMessage], **kwargs) -> ModelResponse:
        """Generate a chat response from Grok."""
        try:
            # Convert our ChatMessage format to Grok format
            grok_messages = []
            for msg in messages:
                grok_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                        **({"name": msg.name} if msg.name else {}),
                    }
                )

            payload = {"model": self.model, "messages": grok_messages, **kwargs}

            response = await self.client.post(
                "{self.base_url}/chat/completions", json=payload
            )
            response.raise_for_status()
            data = response.json()

            return ModelResponse(
                content=data["choices"][0]["message"]["content"],
                model=self.model,
                provider="grok",
                usage=(
                    {
                        "prompt_tokens": data["usage"]["prompt_tokens"],
                        "completion_tokens": data["usage"]["completion_tokens"],
                        "total_tokens": data["usage"]["total_tokens"],
                    }
                    if "usage" in data
                    else None
                ),
                metadata={
                    "finish_reason": data["choices"][0]["finish_reason"],
                    "index": data["choices"][0]["index"],
                },
            )
        except Exception:
            raise Exception("Grok API error: {str(e)}")

    def get_available_models(self) -> List[str]:
        """Get list of available Grok models."""
        return [
            "grok-beta",
            "grok-2",
            "grok-2-vision",
        ]

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
