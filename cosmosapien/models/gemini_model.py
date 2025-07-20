"""Google Gemini model implementation."""

import asyncio
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from ..core.models import BaseModel, ModelResponse, ChatMessage


class Gemini(BaseModel):
    """Google Gemini model implementation."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gemini-pro", **kwargs):
        super().__init__(api_key, **kwargs)
        self.model = model
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from Gemini."""
        try:
            # Run in executor since google.generativeai is not async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.generate_content(prompt, **kwargs)
            )
            
            return ModelResponse(
                content=response.text,
                model=self.model,
                provider="gemini",
                usage={
                    "prompt_tokens": getattr(response, 'usage_metadata', {}).get('prompt_token_count'),
                    "completion_tokens": getattr(response, 'usage_metadata', {}).get('candidates_token_count'),
                } if hasattr(response, 'usage_metadata') else None,
                metadata={
                    "finish_reason": getattr(response, 'candidates', [{}])[0].get('finish_reason'),
                }
            )
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    async def chat(self, messages: List[ChatMessage], **kwargs) -> ModelResponse:
        """Generate a chat response from Gemini."""
        try:
            # Convert messages to Gemini format
            chat = self.client.start_chat(history=[])
            
            # Add previous messages to chat history
            for msg in messages[:-1]:  # All but the last message
                if msg.role == "user":
                    chat.send_message(msg.content)
                elif msg.role == "assistant":
                    # Gemini doesn't have a direct way to add assistant messages
                    # We'll skip them for now
                    pass
            
            # Send the last message and get response
            last_message = messages[-1]
            if last_message.role == "user":
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: chat.send_message(last_message.content, **kwargs)
                )
                
                return ModelResponse(
                    content=response.text,
                    model=self.model,
                    provider="gemini",
                    usage={
                        "prompt_tokens": getattr(response, 'usage_metadata', {}).get('prompt_token_count'),
                        "completion_tokens": getattr(response, 'usage_metadata', {}).get('candidates_token_count'),
                    } if hasattr(response, 'usage_metadata') else None,
                    metadata={
                        "finish_reason": getattr(response, 'candidates', [{}])[0].get('finish_reason'),
                    }
                )
            else:
                raise ValueError("Last message must be from user")
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Gemini models."""
        return [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ] 