"""Smart router for cost-efficient model selection."""

import json
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ConfigManager
from .local_manager import LocalModelManager


class TaskComplexity(Enum):
    """Task complexity levels."""

    TRIVIAL = "trivial"  # Simple questions, greetings
    SIMPLE = "simple"  # Basic explanations, short answers
    MEDIUM = "medium"  # Detailed explanations, analysis
    COMPLEX = "complex"  # Creative writing, coding, reasoning
    EXPERT = "expert"  # Advanced tasks, research, complex coding


@dataclass
class ProviderQuota:
    """Provider usage quota information."""

    provider: str
    model: str
    free_tier_limit: int
    used_calls: int
    reset_date: str  # ISO date string
    cost_per_call: float = 0.0
    is_local: bool = False
    is_available: bool = True

    @property
    def remaining_calls(self) -> int:
        """Get remaining free tier calls."""
        if self.is_local:
            return float("inf")  # Local models have unlimited calls
        return max(0, self.free_tier_limit - self.used_calls)

    @property
    def is_exhausted(self) -> bool:
        """Check if free tier is exhausted."""
        return not self.is_local and self.remaining_calls <= 0


@dataclass
class RoutingDecision:
    """Result of smart routing decision."""

    selected_provider: str
    selected_model: str
    complexity: TaskComplexity
    reasoning: str
    alternatives: List[Tuple[str, str, str]]  # provider, model, reason
    estimated_cost: float = 0.0


class SmartRouter:
    """Smart router for cost-efficient model selection."""

    def __init__(self, config_manager: ConfigManager, local_manager: LocalModelManager):
        self.config_manager = config_manager
        self.local_manager = local_manager
        self.usage_file = Path.home() / ".cosmo" / "usage.json"
        self.usage_file.parent.mkdir(exist_ok=True)

        # Load or initialize usage data
        self.usage_data = self._load_usage_data()

        # Define provider tiers and quotas
        self.provider_quotas = self._initialize_provider_quotas()

        # Complexity detection patterns
        self.complexity_patterns = {
            TaskComplexity.TRIVIAL: [
                r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b",
                r"\b(how are you|what\'s up|how\'s it going)\b",
                r"\b(thanks|thank you|bye|goodbye)\b",
                r"^.{1,20}$",  # Very short prompts
            ],
            TaskComplexity.SIMPLE: [
                r"\b(what is|what are|define|explain|tell me about)\b",
                r"\b(how to|how do|can you help)\b",
                r"^.{21,100}$",  # Short to medium prompts
            ],
            TaskComplexity.MEDIUM: [
                r"\b(analyze|compare|discuss|describe|elaborate)\b",
                r"\b(write|create|generate|develop)\b",
                r"\b(example|sample|demonstrate)\b",
                r"^.{101,500}$",  # Medium to long prompts
            ],
            TaskComplexity.COMPLEX: [
                r"\b(code|program|script|function|algorithm)\b",
                r"\b(design|architecture|system|framework)\b",
                r"\b(research|investigate|study|examine)\b",
                r"\b(creative|story|poem|article|essay)\b",
                r"^.{501,1000}$",  # Long prompts
            ],
            TaskComplexity.EXPERT: [
                r"\b(advanced|complex|sophisticated|optimize)\b",
                r"\b(debug|fix|error|problem|issue)\b",
                r"\b(performance|efficiency|scalability)\b",
                r"\b(machine learning|ai|neural|deep learning)\b",
                r"^.{1001,}$",  # Very long prompts
            ],
        }

    def _load_usage_data(self) -> Dict[str, Any]:
        """Load usage data from file."""
        if self.usage_file.exists():
            try:
                with open(self.usage_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"providers": {}, "last_reset": date.today().isoformat()}

    def _save_usage_data(self):
        """Save usage data to file."""
        try:
            with open(self.usage_file, "w") as f:
                json.dump(self.usage_data, f, indent=2)
        except IOError:
            pass  # Silently fail if we can't save

    def _initialize_provider_quotas(self) -> Dict[str, ProviderQuota]:
        """Initialize provider quotas with default values."""
        quotas = {}

        # Get smart routing configuration
        smart_config = self.config_manager.get_smart_routing_config()

        # Local models (unlimited)
        if self.local_manager.detect_ollama():
            available_models = self.local_manager.get_available_models("ollama")
            for model in available_models[:3]:  # Top 3 models
                quotas["llama_{model}"] = ProviderQuota(
                    provider="llama",
                    model=model,
                    free_tier_limit=float("inf"),
                    used_calls=0,
                    reset_date=date.today().isoformat(),
                    cost_per_call=0.0,
                    is_local=True,
                )

        # Cloud providers with free tiers (default values)
        cloud_providers = {
            "openai": {
                "gpt-3.5-turbo": {"limit": 3, "cost": 0.002},
                "gpt-4": {"limit": 0, "cost": 0.03},
                "gpt-4-turbo-preview": {"limit": 0, "cost": 0.01},
            },
            "claude": {
                "claude-3-haiku-20240307": {"limit": 5, "cost": 0.00025},
                "claude-3-sonnet-20240229": {"limit": 0, "cost": 0.003},
                "claude-3-opus-20240229": {"limit": 0, "cost": 0.015},
            },
            "gemini": {
                "gemini-pro": {"limit": 15, "cost": 0.0},  # Free tier
                "gemini-1.5-flash": {"limit": 15, "cost": 0.0},
            },
            "perplexity": {
                "llama-3.1-sonar-small-128k-online": {"limit": 5, "cost": 0.0},
                "llama-3.1-sonar-medium-128k-online": {"limit": 0, "cost": 0.0},
            },
            "grok": {"grok-beta": {"limit": 0, "cost": 0.0}},  # No free tier info
            "huggingface": {"gpt2": {"limit": 100, "cost": 0.0}},  # Free tier
        }

        # Override with custom configuration
        for provider, models in cloud_providers.items():
            for model, info in models.items():
                key = "{provider}_{model}"

                # Use custom limits if configured
                custom_limit = smart_config.free_tier_limits.get(provider, {}).get(
                    model, info["limit"]
                )
                custom_cost = smart_config.custom_costs.get(provider, {}).get(
                    model, info["cost"]
                )

                quotas[key] = ProviderQuota(
                    provider=provider,
                    model=model,
                    free_tier_limit=custom_limit,
                    used_calls=self.usage_data.get("providers", {})
                    .get(key, {})
                    .get("used_calls", 0),
                    reset_date=self.usage_data.get("providers", {})
                    .get(key, {})
                    .get("reset_date", date.today().isoformat()),
                    cost_per_call=custom_cost,
                    is_local=False,
                )

        return quotas

    def _detect_complexity(self, prompt: str) -> TaskComplexity:
        """Detect task complexity based on prompt content and length."""
        prompt_lower = prompt.lower()

        # Check patterns from most complex to least
        for complexity in reversed(list(TaskComplexity)):
            patterns = self.complexity_patterns[complexity]
            for pattern in patterns:
                if re.search(pattern, prompt_lower, re.IGNORECASE):
                    return complexity

        # Default based on length
        if len(prompt) <= 20:
            return TaskComplexity.TRIVIAL
        elif len(prompt) <= 100:
            return TaskComplexity.SIMPLE
        elif len(prompt) <= 500:
            return TaskComplexity.MEDIUM
        elif len(prompt) <= 1000:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.EXPERT

    def _get_available_providers(self) -> List[ProviderQuota]:
        """Get list of available providers with remaining quota."""
        available = []

        for quota in self.provider_quotas.values():
            # Check if provider is logged in
            if quota.provider == "llama":
                # Local provider is always available if Ollama is running
                if self.local_manager.detect_ollama():
                    available.append(quota)
            else:
                # For cloud providers, check if logged in and not exhausted
                from ..auth.manager import AuthManager

                auth_manager = AuthManager(self.config_manager)
                if auth_manager.is_logged_in(quota.provider) and not quota.is_exhausted:
                    available.append(quota)

        return available

    def _select_model_for_complexity(
        self, complexity: TaskComplexity, available_providers: List[ProviderQuota]
    ) -> Tuple[str, str, str]:
        """Select the best model for a given complexity level."""

        # Sort providers by cost efficiency (local first, then by cost)
        sorted_providers = sorted(
            available_providers, key=lambda p: (not p.is_local, p.cost_per_call)
        )

        if complexity == TaskComplexity.TRIVIAL:
            # Prefer local models for trivial tasks
            for provider in sorted_providers:
                if provider.is_local:
                    return (
                        provider.provider,
                        provider.model,
                        "Local model for trivial task",
                    )

            # Fallback to cheapest cloud model
            for provider in sorted_providers:
                if provider.remaining_calls > 0:
                    return (
                        provider.provider,
                        provider.model,
                        "Cheapest available model ({provider.cost_per_call}$/call)",
                    )

        elif complexity == TaskComplexity.SIMPLE:
            # Prefer local or free tier models
            for provider in sorted_providers:
                if provider.is_local or provider.cost_per_call == 0.0:
                    if provider.remaining_calls > 0:
                        return (
                            provider.provider,
                            provider.model,
                            "Free/local model for simple task",
                        )

            # Fallback to cheapest paid model
            for provider in sorted_providers:
                if provider.remaining_calls > 0:
                    return (
                        provider.provider,
                        provider.model,
                        "Cheapest available model ({provider.cost_per_call}$/call)",
                    )

        elif complexity == TaskComplexity.MEDIUM:
            # Use free tier models or local models
            for provider in sorted_providers:
                if provider.remaining_calls > 0:
                    if provider.is_local:
                        return (
                            provider.provider,
                            provider.model,
                            "Local model for medium task",
                        )
                    elif provider.cost_per_call == 0.0:
                        return (
                            provider.provider,
                            provider.model,
                            "Free tier model for medium task",
                        )
                    else:
                        return (
                            provider.provider,
                            provider.model,
                            "Available model ({provider.cost_per_call}$/call)",
                        )

        elif complexity == TaskComplexity.COMPLEX:
            # Prefer more capable models for complex tasks
            # Look for models with good capabilities
            for provider in sorted_providers:
                if provider.remaining_calls > 0:
                    if "gpt-4" in provider.model or "claude-3-sonnet" in provider.model:
                        return (
                            provider.provider,
                            provider.model,
                            "High-capability model for complex task",
                        )
                    elif provider.is_local and "mixtral" in provider.model:
                        return (
                            provider.provider,
                            provider.model,
                            "Local high-capability model",
                        )
                    elif provider.cost_per_call <= 0.01:  # Cheap but capable
                        return (
                            provider.provider,
                            provider.model,
                            "Cost-effective capable model ({provider.cost_per_call}$/call)",
                        )

            # Fallback to any available
            for provider in sorted_providers:
                if provider.remaining_calls > 0:
                    return (
                        provider.provider,
                        provider.model,
                        "Available model for complex task ({provider.cost_per_call}$/call)",
                    )

        else:  # EXPERT
            # Use the most capable models for expert tasks
            for provider in sorted_providers:
                if provider.remaining_calls > 0:
                    if "gpt-4" in provider.model or "claude-3-opus" in provider.model:
                        return (
                            provider.provider,
                            provider.model,
                            "Most capable model for expert task",
                        )
                    elif "claude-3-sonnet" in provider.model:
                        return (
                            provider.provider,
                            provider.model,
                            "High-capability model for expert task",
                        )
                    elif provider.is_local and (
                        "mixtral" in provider.model or "llama3.2" in provider.model
                    ):
                        return (
                            provider.provider,
                            provider.model,
                            "Local high-capability model",
                        )

            # Fallback to any available
            for provider in sorted_providers:
                if provider.remaining_calls > 0:
                    return (
                        provider.provider,
                        provider.model,
                        "Available model for expert task ({provider.cost_per_call}$/call)",
                    )

        # If no provider available, return None
        return None, None, "No available providers"

    def smart_route(self, prompt: str, explain_only: bool = False) -> RoutingDecision:
        """Smart route a prompt to the most cost-efficient model."""

        # Detect complexity
        complexity = self._detect_complexity(prompt)

        # Get available providers
        available_providers = self._get_available_providers()

        if not available_providers:
            return RoutingDecision(
                selected_provider="none",
                selected_model="none",
                complexity=complexity,
                reasoning="No available providers. Please login to cloud providers or ensure Ollama is running.",
                alternatives=[],
                estimated_cost=0.0,
            )

        # Select best model for complexity
        selected_provider, selected_model, reasoning = (
            self._select_model_for_complexity(complexity, available_providers)
        )

        if not selected_provider:
            return RoutingDecision(
                selected_provider="none",
                selected_model="none",
                complexity=complexity,
                reasoning="No suitable provider found for this complexity level.",
                alternatives=[],
                estimated_cost=0.0,
            )

        # Get selected provider quota
        selected_quota = None
        for quota in available_providers:
            if quota.provider == selected_provider and quota.model == selected_model:
                selected_quota = quota
                break

        # Generate alternatives
        alternatives = []
        for provider in available_providers[:5]:  # Top 5 alternatives
            if (
                provider.provider != selected_provider
                or provider.model != selected_model
            ):
                alt_reason = "Alternative: {provider.provider}/{provider.model}"
                if provider.is_local:
                    alt_reason += " (local)"
                elif provider.cost_per_call == 0.0:
                    alt_reason += " (free)"
                else:
                    alt_reason += " (${provider.cost_per_call}/call)"
                alternatives.append((provider.provider, provider.model, alt_reason))

        # Calculate estimated cost
        estimated_cost = selected_quota.cost_per_call if selected_quota else 0.0

        decision = RoutingDecision(
            selected_provider=selected_provider,
            selected_model=selected_model,
            complexity=complexity,
            reasoning=reasoning,
            alternatives=alternatives,
            estimated_cost=estimated_cost,
        )

        # Update usage if not explain-only
        if not explain_only and selected_provider != "none":
            self._update_usage(selected_provider, selected_model)

        return decision

    def _update_usage(self, provider: str, model: str):
        """Update usage statistics."""
        key = "{provider}_{model}"

        if key not in self.usage_data.get("providers", {}):
            self.usage_data.setdefault("providers", {})[key] = {
                "used_calls": 0,
                "reset_date": date.today().isoformat(),
            }

        self.usage_data["providers"][key]["used_calls"] += 1

        # Update quota object
        if key in self.provider_quotas:
            self.provider_quotas[key].used_calls += 1

        # Save to file
        self._save_usage_data()

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary for all providers."""
        summary = {"total_calls": 0, "providers": {}, "estimated_cost": 0.0}

        for key, quota in self.provider_quotas.items():
            if quota.used_calls > 0:
                provider_cost = quota.used_calls * quota.cost_per_call
                summary["providers"][key] = {
                    "provider": quota.provider,
                    "model": quota.model,
                    "used_calls": quota.used_calls,
                    "remaining_calls": quota.remaining_calls,
                    "cost": provider_cost,
                    "is_local": quota.is_local,
                }
                summary["total_calls"] += quota.used_calls
                summary["estimated_cost"] += provider_cost

        return summary

    def reset_usage(self, provider: Optional[str] = None):
        """Reset usage for a specific provider or all providers."""
        if provider:
            # Reset specific provider
            for key in list(self.usage_data.get("providers", {}).keys()):
                if key.startswith("{provider}_"):
                    self.usage_data["providers"][key]["used_calls"] = 0
                    self.usage_data["providers"][key][
                        "reset_date"
                    ] = date.today().isoformat()
                    if key in self.provider_quotas:
                        self.provider_quotas[key].used_calls = 0
        else:
            # Reset all providers
            self.usage_data = {"providers": {}, "last_reset": date.today().isoformat()}
            for quota in self.provider_quotas.values():
                quota.used_calls = 0

        self._save_usage_data()
