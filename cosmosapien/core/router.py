"""Router for delegating requests to appropriate models."""

from typing import Dict, Any, List, Optional
from .config import ConfigManager
from .model_library import ModelLibrary
from .models import BaseModel, ModelResponse, ChatMessage, model_registry


class Router:
    """Routes requests to appropriate model providers."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.model_library = ModelLibrary(config_manager)
    
    def get_model_instance(self, provider: str, model: str, **kwargs) -> BaseModel:
        """Get a model instance for the specified provider and model."""
        # Get provider configuration
        provider_config = self.config_manager.get_provider_config(provider)
        
        # Check if API key is required (local models don't need API keys)
        local_providers = ["llama"]  # Add other local providers here
        if provider not in local_providers:
            if not provider_config or not provider_config.api_key:
                raise ValueError(f"No API key configured for provider: {provider}")
        
        # Get model class from registry
        model_class = model_registry.get(provider)
        if not model_class:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Create model instance
        api_key = provider_config.api_key if provider_config else ""
        base_url = provider_config.base_url if provider_config else None
        
        return model_class(
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs
        )
    
    async def generate(
        self, 
        prompt: str, 
        provider: Optional[str] = None, 
        model: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate a response using the specified provider and model."""
        config = self.config_manager.load()
        
        # Use defaults if not specified
        provider = provider or config.default_provider
        model = model or config.default_model
        
        # Get model instance
        model_instance = self.get_model_instance(provider, model, **kwargs)
        
        # Generate response
        return await model_instance.generate(prompt, **kwargs)
    
    async def chat(
        self,
        messages: List[ChatMessage],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate a chat response using the specified provider and model."""
        config = self.config_manager.load()
        
        # Use defaults if not specified
        provider = provider or config.default_provider
        model = model or config.default_model
        
        # Get model instance
        model_instance = self.get_model_instance(provider, model, **kwargs)
        
        # Generate chat response
        return await model_instance.chat(messages, **kwargs)
    
    async def debate(
        self,
        prompt: str,
        models: List[Dict[str, str]],
        rounds: int = 3,
        **kwargs
    ) -> List[ModelResponse]:
        """Run a debate between multiple models."""
        responses = []
        current_prompt = prompt
        
        for round_num in range(rounds):
            round_responses = []
            
            for model_config in models:
                provider = model_config.get("provider")
                model = model_config.get("model")
                
                if not provider or not model:
                    continue
                
                # Generate response for this model
                response = await self.generate(
                    current_prompt,
                    provider=provider,
                    model=model,
                    **kwargs
                )
                round_responses.append(response)
                
                # Add model's response to the prompt for next round
                current_prompt += f"\n\n{model.upper()}: {response.content}"
            
            responses.extend(round_responses)
        
        return responses
    
    def list_available_providers(self) -> List[str]:
        """List all available providers."""
        return model_registry.list_models()
    
    def list_available_models(self, provider: str) -> List[str]:
        """List available models for a specific provider."""
        try:
            model_instance = self.get_model_instance(provider, "dummy")
            return model_instance.get_available_models()
        except Exception:
            return []
    
    def get_model_info(self, provider: str, model: str):
        """Get model information from the library."""
        model_id = f"{provider}:{model}"
        return self.model_library.get_model(model_id)
    
    def list_library_models(self, provider: Optional[str] = None, active_only: bool = True):
        """List models from the model library."""
        return self.model_library.list_models(provider=provider, active_only=active_only) 