"""OpenAI model implementation."""

import asyncio
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI
from ..core.models import BaseModel, ModelResponse, ChatMessage


class OpenAI(BaseModel):
    """OpenAI model implementation."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-4", **kwargs):
        super().__init__(api_key, **kwargs)
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            return ModelResponse(
                content=response.choices[0].message.content,
                model=self.model,
                provider="openai",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "index": response.choices[0].index,
                }
            )
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def chat(self, messages: List[ChatMessage], **kwargs) -> ModelResponse:
        """Generate a chat response from OpenAI."""
        try:
            # Convert our ChatMessage format to OpenAI format
            openai_messages = []
            for msg in messages:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    **({"name": msg.name} if msg.name else {})
                })
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                **kwargs
            )
            
            return ModelResponse(
                content=response.choices[0].message.content,
                model=self.model,
                provider="openai",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "index": response.choices[0].index,
                }
            )
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ] 