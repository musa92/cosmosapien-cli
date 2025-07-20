"""Base model classes and response types."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass


class ModelResponse(BaseModel):
    """Response from an LLM model."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    """A chat message."""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None


class BaseModel(ABC):
    """Base class for all LLM models."""
    
    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.kwargs = kwargs
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    async def chat(self, messages: List[ChatMessage], **kwargs) -> ModelResponse:
        """Generate a response in chat format."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider."""
        pass


class ModelRegistry:
    """Registry for all available models."""
    
    def __init__(self):
        self._models: Dict[str, type] = {}
    
    def register(self, name: str, model_class: type):
        """Register a model class."""
        self._models[name] = model_class
    
    def get(self, name: str) -> Optional[type]:
        """Get a model class by name."""
        return self._models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())


# Global model registry
model_registry = ModelRegistry() 