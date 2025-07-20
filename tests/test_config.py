"""Tests for configuration module."""

import pytest
from pathlib import Path
from cosmosapien.core.config import ConfigManager, Config, ProviderConfig


def test_config_manager_creation():
    """Test ConfigManager creation."""
    manager = ConfigManager()
    assert manager.config_path == Path.home() / ".cosmosrc"


def test_config_creation():
    """Test Config creation with defaults."""
    config = Config()
    assert config.default_provider == "openai"
    assert config.default_model == "gpt-4"
    assert config.memory_enabled is True


def test_provider_config():
    """Test ProviderConfig creation."""
    provider_config = ProviderConfig(
        api_key="test-key",
        base_url="https://api.test.com",
        models={"test": "test-model"}
    )
    assert provider_config.api_key == "test-key"
    assert provider_config.base_url == "https://api.test.com"
    assert provider_config.models["test"] == "test-model" 