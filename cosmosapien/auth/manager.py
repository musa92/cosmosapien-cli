"""Authentication manager for secure API key storage."""

import getpass
import webbrowser
from typing import Optional

import keyring

from ..core.config import ConfigManager
from ..core.provider_info import get_all_providers


class AuthManager:
    """Manages authentication for different providers."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.service_name = "cosmosapien-cli"

    def login(self, provider: str, api_key: Optional[str] = None) -> bool:
        """Login to a provider by storing the API key securely."""
        try:
            # Special handling for Hugging Face
            if provider == "huggingface":
                print("Opening Hugging Face login page...")
                print("1. Go to https://huggingface.co/settings/tokens")
                print("2. Create a new token with 'read' permissions")
                print("3. Copy the token and paste it below")
                webbrowser.open("https://huggingface.co/settings/tokens")

            # Get API key from user if not provided
            if not api_key:
                api_key = getpass.getpass(f"Enter API key for {provider}: ")

            if not api_key.strip():
                print("Error: API key cannot be empty")
                return False

            # Store API key securely using keyring
            keyring.set_password(self.service_name, provider, api_key)

            # Also store in config for easy access
            self.config_manager.set_api_key(provider, api_key)

            print(f"Successfully logged in to {provider}")
            return True

        except Exception as e:
            print(f"Error logging in to {provider}: {str(e)}")
            return False

    def logout(self, provider: str) -> bool:
        """Logout from a provider by removing the API key."""
        try:
            # Remove from keyring
            keyring.delete_password(self.service_name, provider)

            # Remove from config
            provider_config = self.config_manager.get_provider_config(provider)
            if provider_config:
                provider_config.api_key = None
                self.config_manager.set_provider_config(provider, provider_config)

            print(f"Successfully logged out from {provider}")
            return True

        except Exception as e:
            print(f"Error logging out from {provider}: {str(e)}")
            return False

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider from secure storage."""
        try:
            # Try keyring first
            api_key = keyring.get_password(self.service_name, provider)
            if api_key:
                return api_key

            # Fallback to config
            return self.config_manager.get_api_key(provider)

        except Exception:
            # Fallback to config if keyring fails
            return self.config_manager.get_api_key(provider)

    def is_logged_in(self, provider: str) -> bool:
        """Check if user is logged in to a provider."""
        # Local models don't need API keys
        if provider in ["llama"]:
            return True

        api_key = self.get_api_key(provider)
        return api_key is not None and api_key.strip() != ""

    def list_providers(self) -> list:
        """List all providers with login status."""
        providers = []
        for provider in get_all_providers():
            status = "✓" if self.is_logged_in(provider) else "✗"
            providers.append(
                {
                    "provider": provider,
                    "status": status,
                    "logged_in": self.is_logged_in(provider),
                }
            )
        return providers
