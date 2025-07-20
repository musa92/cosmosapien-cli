"""Claude (Anthropic) model implementation."""

import asyncio
from typing import List, Optional, Dict, Any
from anthropic import AsyncAnthropic
from ..core.models import BaseModel, ModelResponse, ChatMessage


class Claude(BaseModel):
    """Claude (Anthropic) model implementation."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "claude-3-sonnet-20240229", **kwargs):
        super().__init__(api_key, **kwargs)
        self.model = model
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url
        )
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from Claude."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 1000),
                messages=[{"role": "user", "content": prompt}],
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )
            
            return ModelResponse(
                content=response.content[0].text,
                model=self.model,
                provider="claude",
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                } if response.usage else None,
                metadata={
                    "finish_reason": response.stop_reason,
                    "stop_sequence": response.stop_sequence,
                }
            )
        except Exception as e:
            raise Exception(f"Claude API error: {str(e)}")
    
    async def chat(self, messages: List[ChatMessage], **kwargs) -> ModelResponse:
        """Generate a chat response from Claude."""
        try:
            # Convert our ChatMessage format to Claude format
            claude_messages = []
            for msg in messages:
                claude_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 1000),
                messages=claude_messages,
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )
            
            return ModelResponse(
                content=response.content[0].text,
                model=self.model,
                provider="claude",
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                } if response.usage else None,
                metadata={
                    "finish_reason": response.stop_reason,
                    "stop_sequence": response.stop_sequence,
                }
            )
        except Exception as e:
            raise Exception(f"Claude API error: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Claude models."""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ] 