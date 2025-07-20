"""Provider information and metadata."""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ProviderInfo:
    """Information about a provider."""
    name: str
    display_name: str
    tier_type: str  # "individual", "bundled", "mixed"
    tier_icon: str  # Symbol to show tier type
    description: str
    website: str
    api_docs: str
    models_included: Optional[List[str]] = None
    subscription_required: bool = False
    free_tier_available: bool = False


# Provider information database
PROVIDER_INFO: Dict[str, ProviderInfo] = {
    "openai": ProviderInfo(
        name="openai",
        display_name="OpenAI",
        tier_type="individual",
        tier_icon="•",
        description="Individual model access with pay-per-use pricing",
        website="https://openai.com",
        api_docs="https://platform.openai.com/docs",
        subscription_required=False,
        free_tier_available=True,
    ),
    "gemini": ProviderInfo(
        name="gemini",
        display_name="Google Gemini",
        tier_type="individual",
        tier_icon="•",
        description="Individual model access with generous free tier",
        website="https://ai.google.dev",
        api_docs="https://ai.google.dev/docs",
        subscription_required=False,
        free_tier_available=True,
    ),
    "claude": ProviderInfo(
        name="claude",
        display_name="Claude (Anthropic)",
        tier_type="individual",
        tier_icon="•",
        description="Individual model access with pay-per-use pricing",
        website="https://anthropic.com",
        api_docs="https://docs.anthropic.com",
        subscription_required=False,
        free_tier_available=False,
    ),
    "perplexity": ProviderInfo(
        name="perplexity",
        display_name="Perplexity",
        tier_type="bundled",
        tier_icon="*",
        description="Bundled access to multiple models with subscription",
        website="https://perplexity.ai",
        api_docs="https://docs.perplexity.ai",
        models_included=["llama-3.1-sonar-small", "llama-3.1-sonar-medium", "mixtral", "codellama"],
        subscription_required=True,
        free_tier_available=True,
    ),
    "llama": ProviderInfo(
        name="llama",
        display_name="LLaMA (Ollama)",
        tier_type="local",
        tier_icon="○",
        description="Local models via Ollama - no API key required",
        website="https://ollama.ai",
        api_docs="https://github.com/ollama/ollama",
        subscription_required=False,
        free_tier_available=True,
    ),
    "grok": ProviderInfo(
        name="grok",
        display_name="Grok (xAI)",
        tier_type="individual",
        tier_icon="•",
        description="Individual model access with xAI subscription",
        website="https://x.ai",
        api_docs="https://docs.x.ai",
        subscription_required=True,
        free_tier_available=False,
    ),
    "huggingface": ProviderInfo(
        name="huggingface",
        display_name="Hugging Face",
        tier_type="bundled",
        tier_icon="*",
        description="Access to thousands of open-source models",
        website="https://huggingface.co",
        api_docs="https://huggingface.co/docs/api-inference",
        models_included=["llama-2", "gpt2", "falcon", "mpt", "bloom", "qwen", "chatglm"],
        subscription_required=False,
        free_tier_available=True,
    ),
}


def get_provider_info(provider: str) -> Optional[ProviderInfo]:
    """Get information about a provider."""
    return PROVIDER_INFO.get(provider.lower())


def get_provider_display_name(provider: str) -> str:
    """Get the display name for a provider with tier indicator."""
    info = get_provider_info(provider)
    if info:
        return f"{info.tier_icon} {info.display_name}"
    return provider.title()


def get_all_providers() -> List[str]:
    """Get list of all available providers."""
    return list(PROVIDER_INFO.keys())


def get_providers_by_tier(tier_type: str) -> List[str]:
    """Get providers by tier type."""
    return [name for name, info in PROVIDER_INFO.items() if info.tier_type == tier_type]


def get_bundled_providers() -> List[str]:
    """Get providers that offer bundled access to multiple models."""
    return get_providers_by_tier("bundled")


def get_individual_providers() -> List[str]:
    """Get providers that offer individual model access."""
    return get_providers_by_tier("individual")


def get_local_providers() -> List[str]:
    """Get providers that run locally."""
    return get_providers_by_tier("local") 