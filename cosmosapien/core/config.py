"""Configuration management for Cosmosapien CLI."""

import os
import toml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for a specific provider."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: Dict[str, str] = Field(default_factory=dict)


class SmartRoutingConfig(BaseModel):
    """Configuration for smart routing quotas."""
    enabled: bool = True
    prefer_local: bool = True
    cost_threshold: float = 0.01  # Maximum cost per call before considering alternatives
    free_tier_limits: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    custom_costs: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class Config(BaseModel):
    """Main configuration class."""
    default_provider: str = "openai"
    default_model: str = "gpt-4"
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    memory_enabled: bool = True
    memory_path: str = "~/.cosmosapien/memory"
    plugins_path: str = "~/.cosmosapien/plugins"
    smart_routing: SmartRoutingConfig = Field(default_factory=SmartRoutingConfig)
    
    class Config:
        extra = "allow"


class ConfigManager:
    """Manages configuration file operations."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path.home() / ".cosmosrc"
    
    def load(self) -> Config:
        """Load configuration from file."""
        if not self.config_path.exists():
            return Config()
        
        try:
            with open(self.config_path, 'r') as f:
                data = toml.load(f)
            return Config(**data)
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return Config()
    
    def save(self, config: Config) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            toml.dump(config.dict(), f)
    
    def get_provider_config(self, provider: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider."""
        config = self.load()
        return config.providers.get(provider)
    
    def set_provider_config(self, provider: str, provider_config: ProviderConfig) -> None:
        """Set configuration for a specific provider."""
        config = self.load()
        config.providers[provider] = provider_config
        self.save(config)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        provider_config = self.get_provider_config(provider)
        if provider_config:
            return provider_config.api_key
        return None
    
    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for a provider."""
        provider_config = self.get_provider_config(provider) or ProviderConfig()
        provider_config.api_key = api_key
        self.set_provider_config(provider, provider_config)
    
    def get_smart_routing_config(self) -> SmartRoutingConfig:
        """Get smart routing configuration."""
        config = self.load()
        return config.smart_routing
    
    def set_smart_routing_config(self, smart_routing_config: SmartRoutingConfig) -> None:
        """Set smart routing configuration."""
        config = self.load()
        config.smart_routing = smart_routing_config
        self.save(config)
    
    def set_free_tier_limit(self, provider: str, model: str, limit: int) -> None:
        """Set free tier limit for a specific provider/model."""
        config = self.load()
        if provider not in config.smart_routing.free_tier_limits:
            config.smart_routing.free_tier_limits[provider] = {}
        config.smart_routing.free_tier_limits[provider][model] = limit
        self.save(config)
    
    def set_custom_cost(self, provider: str, model: str, cost: float) -> None:
        """Set custom cost for a specific provider/model."""
        config = self.load()
        if provider not in config.smart_routing.custom_costs:
            config.smart_routing.custom_costs[provider] = {}
        config.smart_routing.custom_costs[provider][model] = cost
        self.save(config) 