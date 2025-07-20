"""Model library for storing and managing model configurations."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ConfigManager


class ModelType(Enum):
    """Model type classification."""

    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class ModelTier(Enum):
    """Model tier classification."""

    FREE = "free"
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class ModelCapability:
    """Model capability specification."""

    max_tokens: Optional[int] = None
    max_input_tokens: Optional[int] = None
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    supports_embeddings: bool = False
    context_window: Optional[int] = None
    training_data_cutoff: Optional[str] = None


@dataclass
class ModelPricing:
    """Model pricing information."""

    input_cost_per_1k_tokens: float = 0.0
    output_cost_per_1k_tokens: float = 0.0
    free_tier_limit: int = 0
    free_tier_reset_period: str = "monthly"  # daily, weekly, monthly
    currency: str = "USD"


@dataclass
class ModelConfig:
    """Complete model configuration."""

    name: str
    provider: str
    model_id: str
    display_name: str
    description: str
    model_type: ModelType
    tier: ModelTier
    capabilities: ModelCapability
    pricing: ModelPricing
    tags: List[str]
    is_active: bool = True
    is_local: bool = False
    created_at: str = None
    updated_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["model_type"] = self.model_type.value
        data["tier"] = self.tier.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        data = data.copy()
        data["model_type"] = ModelType(data["model_type"])
        data["tier"] = ModelTier(data["tier"])
        data["capabilities"] = ModelCapability(**data["capabilities"])
        data["pricing"] = ModelPricing(**data["pricing"])
        return cls(**data)


class ModelLibrary:
    """Model library for storing and managing model configurations."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.library_file = Path.home() / ".cosmo" / "model_library.json"
        self.library_file.parent.mkdir(exist_ok=True)
        self.models: Dict[str, ModelConfig] = {}
        self._load_library()
        self._initialize_default_models()

    def _load_library(self):
        """Load model library from file."""
        if self.library_file.exists():
            try:
                with open(self.library_file, "r") as f:
                    data = json.load(f)
                    for model_id, model_data in data.items():
                        self.models[model_id] = ModelConfig.from_dict(model_data)
            except (json.JSONDecodeError, IOError):
                print("Warning: Could not load model library: {e}")

    def _save_library(self):
        """Save model library to file."""
        try:
            data = {
                model_id: model.to_dict() for model_id, model in self.models.items()
            }
            with open(self.library_file, "w") as f:
                json.dump(data, f, indent=2)
        except IOError:
            print("Warning: Could not save model library: {e}")

    def _initialize_default_models(self):
        """Initialize with default model configurations."""
        if not self.models:
            self._add_default_models()

    def _add_default_models(self):
        """Add default model configurations."""
        default_models = [
            # OpenAI Models
            ModelConfig(
                name="gpt-4",
                provider="openai",
                model_id="gpt-4",
                display_name="GPT-4",
                description="Most capable GPT model for complex reasoning tasks",
                model_type=ModelType.CHAT,
                tier=ModelTier.PREMIUM,
                capabilities=ModelCapability(
                    max_tokens=8192,
                    max_input_tokens=8192,
                    supports_streaming=True,
                    supports_function_calling=True,
                    context_window=8192,
                    training_data_cutoff="2023-04",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.03,
                    output_cost_per_1k_tokens=0.06,
                    free_tier_limit=0,
                ),
                tags=["reasoning", "complex-tasks", "function-calling"],
            ),
            ModelConfig(
                name="gpt-4-turbo",
                provider="openai",
                model_id="gpt-4-turbo-preview",
                display_name="GPT-4 Turbo",
                description="Faster and more efficient GPT-4 model",
                model_type=ModelType.CHAT,
                tier=ModelTier.STANDARD,
                capabilities=ModelCapability(
                    max_tokens=4096,
                    max_input_tokens=128000,
                    supports_streaming=True,
                    supports_function_calling=True,
                    context_window=128000,
                    training_data_cutoff="2023-12",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.01,
                    output_cost_per_1k_tokens=0.03,
                    free_tier_limit=0,
                ),
                tags=["fast", "long-context", "function-calling"],
            ),
            ModelConfig(
                name="gpt-3.5-turbo",
                provider="openai",
                model_id="gpt-3.5-turbo",
                display_name="GPT-3.5 Turbo",
                description="Fast and efficient model for most tasks",
                model_type=ModelType.CHAT,
                tier=ModelTier.BASIC,
                capabilities=ModelCapability(
                    max_tokens=4096,
                    max_input_tokens=4096,
                    supports_streaming=True,
                    supports_function_calling=True,
                    context_window=4096,
                    training_data_cutoff="2021-09",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.0015,
                    output_cost_per_1k_tokens=0.002,
                    free_tier_limit=3,
                ),
                tags=["fast", "efficient", "general-purpose"],
            ),
            # Anthropic Models
            ModelConfig(
                name="claude-3-opus",
                provider="claude",
                model_id="claude-3-opus-20240229",
                display_name="Claude 3 Opus",
                description="Most capable Claude model for complex reasoning",
                model_type=ModelType.CHAT,
                tier=ModelTier.PREMIUM,
                capabilities=ModelCapability(
                    max_tokens=4096,
                    max_input_tokens=200000,
                    supports_streaming=True,
                    supports_function_calling=True,
                    context_window=200000,
                    training_data_cutoff="2023-08",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.015,
                    output_cost_per_1k_tokens=0.075,
                    free_tier_limit=0,
                ),
                tags=["reasoning", "analysis", "long-context"],
            ),
            ModelConfig(
                name="claude-3-sonnet",
                provider="claude",
                model_id="claude-3-sonnet-20240229",
                display_name="Claude 3 Sonnet",
                description="Balanced Claude model for most tasks",
                model_type=ModelType.CHAT,
                tier=ModelTier.STANDARD,
                capabilities=ModelCapability(
                    max_tokens=4096,
                    max_input_tokens=200000,
                    supports_streaming=True,
                    supports_function_calling=True,
                    context_window=200000,
                    training_data_cutoff="2023-08",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.003,
                    output_cost_per_1k_tokens=0.015,
                    free_tier_limit=0,
                ),
                tags=["balanced", "analysis", "long-context"],
            ),
            ModelConfig(
                name="claude-3-haiku",
                provider="claude",
                model_id="claude-3-haiku-20240307",
                display_name="Claude 3 Haiku",
                description="Fast and efficient Claude model",
                model_type=ModelType.CHAT,
                tier=ModelTier.BASIC,
                capabilities=ModelCapability(
                    max_tokens=4096,
                    max_input_tokens=200000,
                    supports_streaming=True,
                    supports_function_calling=True,
                    context_window=200000,
                    training_data_cutoff="2023-08",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.00025,
                    output_cost_per_1k_tokens=0.00125,
                    free_tier_limit=5,
                ),
                tags=["fast", "efficient", "long-context"],
            ),
            # Google Models
            ModelConfig(
                name="gemini-pro",
                provider="gemini",
                model_id="gemini-pro",
                display_name="Gemini Pro",
                description="Google's advanced language model",
                model_type=ModelType.CHAT,
                tier=ModelTier.STANDARD,
                capabilities=ModelCapability(
                    max_tokens=2048,
                    max_input_tokens=30720,
                    supports_streaming=True,
                    supports_function_calling=False,
                    context_window=30720,
                    training_data_cutoff="2023-02",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                    free_tier_limit=15,
                ),
                tags=["google", "free-tier", "general-purpose"],
            ),
            ModelConfig(
                name="gemini-flash",
                provider="gemini",
                model_id="gemini-1.5-flash",
                display_name="Gemini 1.5 Flash",
                description="Fast and efficient Gemini model",
                model_type=ModelType.CHAT,
                tier=ModelTier.BASIC,
                capabilities=ModelCapability(
                    max_tokens=8192,
                    max_input_tokens=1048576,
                    supports_streaming=True,
                    supports_function_calling=False,
                    context_window=1048576,
                    training_data_cutoff="2023-12",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                    free_tier_limit=15,
                ),
                tags=["fast", "long-context", "free-tier"],
            ),
            # Local Models (Ollama)
            ModelConfig(
                name="llama3.2-8b",
                provider="llama",
                model_id="llama3.2:8b",
                display_name="Llama 3.2 8B",
                description="Local Llama 3.2 model (8B parameters)",
                model_type=ModelType.CHAT,
                tier=ModelTier.FREE,
                capabilities=ModelCapability(
                    max_tokens=4096,
                    max_input_tokens=8192,
                    supports_streaming=True,
                    supports_function_calling=False,
                    context_window=8192,
                    training_data_cutoff="2023-12",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                    free_tier_limit=float("inf"),
                ),
                tags=["local", "free", "llama", "8b"],
                is_local=True,
            ),
            ModelConfig(
                name="mixtral-8x7b",
                provider="llama",
                model_id="mixtral:8x7b",
                display_name="Mixtral 8x7B",
                description="High-performance local model with mixture of experts",
                model_type=ModelType.CHAT,
                tier=ModelTier.FREE,
                capabilities=ModelCapability(
                    max_tokens=4096,
                    max_input_tokens=32768,
                    supports_streaming=True,
                    supports_function_calling=False,
                    context_window=32768,
                    training_data_cutoff="2023-12",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                    free_tier_limit=float("inf"),
                ),
                tags=["local", "free", "mixtral", "high-performance"],
                is_local=True,
            ),
            ModelConfig(
                name="codellama-13b",
                provider="llama",
                model_id="codellama:13b",
                display_name="Code Llama 13B",
                description="Specialized local model for code generation",
                model_type=ModelType.CHAT,
                tier=ModelTier.FREE,
                capabilities=ModelCapability(
                    max_tokens=4096,
                    max_input_tokens=16384,
                    supports_streaming=True,
                    supports_function_calling=False,
                    context_window=16384,
                    training_data_cutoff="2023-07",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                    free_tier_limit=float("inf"),
                ),
                tags=["local", "free", "code", "programming"],
                is_local=True,
            ),
            # Perplexity Models
            ModelConfig(
                name="perplexity-sonar-small",
                provider="perplexity",
                model_id="llama-3.1-sonar-small-128k-online",
                display_name="Perplexity Sonar Small",
                description="Web-connected AI with real-time information",
                model_type=ModelType.CHAT,
                tier=ModelTier.BASIC,
                capabilities=ModelCapability(
                    max_tokens=4096,
                    max_input_tokens=128000,
                    supports_streaming=True,
                    supports_function_calling=False,
                    context_window=128000,
                    training_data_cutoff="2024-01",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                    free_tier_limit=5,
                ),
                tags=["web-search", "real-time", "research"],
            ),
            ModelConfig(
                name="perplexity-sonar-medium",
                provider="perplexity",
                model_id="llama-3.1-sonar-medium-128k-online",
                display_name="Perplexity Sonar Medium",
                description="Advanced web-connected AI with enhanced capabilities",
                model_type=ModelType.CHAT,
                tier=ModelTier.STANDARD,
                capabilities=ModelCapability(
                    max_tokens=4096,
                    max_input_tokens=128000,
                    supports_streaming=True,
                    supports_function_calling=False,
                    context_window=128000,
                    training_data_cutoff="2024-01",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                    free_tier_limit=0,
                ),
                tags=["web-search", "advanced", "research"],
            ),
            # Grok Model
            ModelConfig(
                name="grok-beta",
                provider="grok",
                model_id="grok-beta",
                display_name="Grok Beta",
                description="xAI's creative and witty AI assistant",
                model_type=ModelType.CHAT,
                tier=ModelTier.STANDARD,
                capabilities=ModelCapability(
                    max_tokens=4096,
                    max_input_tokens=8192,
                    supports_streaming=True,
                    supports_function_calling=False,
                    context_window=8192,
                    training_data_cutoff="2023-12",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                    free_tier_limit=0,
                ),
                tags=["creative", "humor", "entertainment"],
            ),
            # Hugging Face Models
            ModelConfig(
                name="gpt2",
                provider="huggingface",
                model_id="gpt2",
                display_name="GPT-2",
                description="Open-source GPT-2 model from Hugging Face",
                model_type=ModelType.CHAT,
                tier=ModelTier.FREE,
                capabilities=ModelCapability(
                    max_tokens=1024,
                    max_input_tokens=1024,
                    supports_streaming=False,
                    supports_function_calling=False,
                    context_window=1024,
                    training_data_cutoff="2019-10",
                ),
                pricing=ModelPricing(
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                    free_tier_limit=100,
                ),
                tags=["open-source", "experimental", "text-generation"],
            ),
        ]

        for model in default_models:
            self.add_model(model)

    def add_model(self, model: ModelConfig) -> bool:
        """Add a model to the library."""
        model_id = "{model.provider}:{model.model_id}"
        if model_id in self.models:
            return False

        self.models[model_id] = model
        self._save_library()
        return True

    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the library."""
        if model_id in self.models:
            del self.models[model_id]
            self._save_library()
            return True
        return False

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get a model by ID."""
        return self.models.get(model_id)

    def get_models_by_provider(self, provider: str) -> List[ModelConfig]:
        """Get all models for a specific provider."""
        return [
            model
            for model_id, model in self.models.items()
            if model_id.startswith("{provider}:")
        ]

    def get_models_by_tier(self, tier: ModelTier) -> List[ModelConfig]:
        """Get all models for a specific tier."""
        return [model for model in self.models.values() if model.tier == tier]

    def get_models_by_type(self, model_type: ModelType) -> List[ModelConfig]:
        """Get all models for a specific type."""
        return [
            model for model in self.models.values() if model.model_type == model_type
        ]

    def get_models_by_tag(self, tag: str) -> List[ModelConfig]:
        """Get all models with a specific tag."""
        return [model for model in self.models.values() if tag in model.tags]

    def get_active_models(self) -> List[ModelConfig]:
        """Get all active models."""
        return [model for model in self.models.values() if model.is_active]

    def get_local_models(self) -> List[ModelConfig]:
        """Get all local models."""
        return [model for model in self.models.values() if model.is_local]

    def get_free_models(self) -> List[ModelConfig]:
        """Get all free models."""
        return [
            model
            for model in self.models.values()
            if model.pricing.free_tier_limit > 0 or model.is_local
        ]

    def search_models(self, query: str) -> List[ModelConfig]:
        """Search models by name, description, or tags."""
        query_lower = query.lower()
        results = []

        for model in self.models.values():
            if (
                query_lower in model.name.lower()
                or query_lower in model.display_name.lower()
                or query_lower in model.description.lower()
                or any(query_lower in tag.lower() for tag in model.tags)
            ):
                results.append(model)

        return results

    def update_model(self, model_id: str, **kwargs) -> bool:
        """Update a model's configuration."""
        if model_id not in self.models:
            return False

        model = self.models[model_id]
        for key, value in kwargs.items():
            if hasattr(model, key):
                setattr(model, key, value)

        model.updated_at = datetime.now().isoformat()
        self._save_library()
        return True

    def list_models(
        self,
        provider: Optional[str] = None,
        tier: Optional[ModelTier] = None,
        model_type: Optional[ModelType] = None,
        active_only: bool = True,
    ) -> List[ModelConfig]:
        """List models with optional filtering."""
        models = self.models.values()

        if active_only:
            models = [m for m in models if m.is_active]

        if provider:
            models = [m for m in models if m.provider == provider]

        if tier:
            models = [m for m in models if m.tier == tier]

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        return list(models)

    def export_library(self, file_path: str) -> bool:
        """Export the model library to a file."""
        try:
            data = {
                model_id: model.to_dict() for model_id, model in self.models.items()
            }
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except IOError:
            return False

    def import_library(self, file_path: str, overwrite: bool = False) -> bool:
        """Import models from a file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            imported_count = 0
            for model_id, model_data in data.items():
                if overwrite or model_id not in self.models:
                    self.models[model_id] = ModelConfig.from_dict(model_data)
                    imported_count += 1

            self._save_library()
            return imported_count > 0
        except (IOError, json.JSONDecodeError):
            return False

    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about the model library."""
        total_models = len(self.models)
        active_models = len(self.get_active_models())
        local_models = len(self.get_local_models())
        free_models = len(self.get_free_models())

        providers = {}
        tiers = {}
        types = {}

        for model in self.models.values():
            providers[model.provider] = providers.get(model.provider, 0) + 1
            tiers[model.tier.value] = tiers.get(model.tier.value, 0) + 1
            types[model.model_type.value] = types.get(model.model_type.value, 0) + 1

        return {
            "total_models": total_models,
            "active_models": active_models,
            "local_models": local_models,
            "free_models": free_models,
            "providers": providers,
            "tiers": tiers,
            "types": types,
        }
