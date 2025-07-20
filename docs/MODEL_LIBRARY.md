# Model Library System

The Cosmosapien CLI includes a comprehensive model library system that allows you to store, manage, and use AI models in a standardized way. This system provides industry-standard model management capabilities without relying on emojis or informal language.

## Overview

The model library provides:
- **Centralized Model Management**: Store all model configurations in a single location
- **Standardized Metadata**: Consistent model information including capabilities, pricing, and usage limits
- **Flexible Filtering**: Find models by provider, tier, type, tags, or custom criteria
- **Import/Export**: Share model configurations across environments
- **Integration**: Seamless integration with the CLI and smart routing system

## Architecture

### Core Components

1. **ModelConfig**: Complete model configuration with metadata
2. **ModelLibrary**: Main library management class
3. **ModelType**: Classification of model types (chat, completion, embedding, etc.)
4. **ModelTier**: Pricing tier classification (free, basic, standard, premium, enterprise)
5. **ModelCapability**: Technical capabilities specification
6. **ModelPricing**: Cost and usage limit information

### Data Structure

```python
@dataclass
class ModelConfig:
    name: str                    # Internal name
    provider: str               # Provider (openai, claude, etc.)
    model_id: str              # Model identifier
    display_name: str          # Human-readable name
    description: str           # Model description
    model_type: ModelType      # Model type classification
    tier: ModelTier           # Pricing tier
    capabilities: ModelCapability  # Technical capabilities
    pricing: ModelPricing     # Cost and limits
    tags: List[str]           # Searchable tags
    is_active: bool           # Whether model is available
    is_local: bool           # Whether model runs locally
```

## Usage

### CLI Commands

#### List Models
```bash
# List all models
cosmo models

# Filter by provider
cosmo models --provider openai

# Filter by tier
cosmo models --tier premium

# Filter by type
cosmo models --type chat

# Filter by tag
cosmo models --tag coding

# Export to different formats
cosmo models --format json
cosmo models --format csv
```

#### Model Information
```bash
# Get detailed information about a specific model
cosmo model-info openai:gpt-4
```

#### Search Models
```bash
# Search models by name, description, or tags
cosmo search-models "GPT"
cosmo search-models "coding"
```

#### Statistics
```bash
# View model library statistics
cosmo model-stats
```

#### Import/Export
```bash
# Export model library
cosmo export-models models.json

# Import model library
cosmo import-models models.json --overwrite
```

#### Model Registration
```bash
# Interactive registration wizard
cosmo register

# Quick registration with minimal input
cosmo register-quick myprovider mymodel --name "My Model" --tier premium

# Register from predefined template
cosmo register-template gpt4
cosmo register-template claude --provider myclaude --model-id claude-v2

# Register from JSON file
cosmo register-from-file model_config.json
cosmo register-from-file model_config.json --provider custom --model-id v2
```

#### Model Management
```bash
# Unregister a model
cosmo unregister openai:gpt-4
cosmo unregister custom:mymodel --confirm
```

### Programmatic Usage

#### Basic Operations

```python
from cosmosapien.core.config import ConfigManager
from cosmosapien.core.model_library import ModelLibrary, ModelType, ModelTier

# Initialize
config_manager = ConfigManager()
model_library = ModelLibrary(config_manager)

# List all models
models = model_library.list_models()

# Get specific model
model = model_library.get_model("openai:gpt-4")

# Search models
results = model_library.search_models("GPT")
```

#### Filtering Models

```python
# Get models by provider
openai_models = model_library.get_models_by_provider("openai")

# Get models by tier
premium_models = model_library.get_models_by_tier(ModelTier.PREMIUM)

# Get models by type
chat_models = model_library.get_models_by_type(ModelType.CHAT)

# Get models by tag
coding_models = model_library.get_models_by_tag("coding")

# Get free models
free_models = model_library.get_free_models()

# Get local models
local_models = model_library.get_local_models()
```

#### Advanced Filtering

```python
# Complex filtering
models = model_library.list_models(
    provider="openai",
    tier=ModelTier.STANDARD,
    model_type=ModelType.CHAT,
    active_only=True
)
```

#### Model Management

```python
# Add custom model
from cosmosapien.core.model_library import ModelConfig, ModelCapability, ModelPricing

custom_model = ModelConfig(
    name="my-model",
    provider="custom",
    model_id="v1",
    display_name="My Custom Model",
    description="A custom model for specific tasks",
    model_type=ModelType.CHAT,
    tier=ModelTier.STANDARD,
    capabilities=ModelCapability(
        max_tokens=2048,
        supports_streaming=True
    ),
    pricing=ModelPricing(
        input_cost_per_1k_tokens=0.005,
        free_tier_limit=100
    ),
    tags=["custom", "specialized"]
)

model_library.add_model(custom_model)

# Update model
model_library.update_model("custom:v1", is_active=False)

# Remove model
model_library.remove_model("custom:v1")
```

## Integration with Smart Routing

The model library integrates seamlessly with the smart routing system:

```python
# Smart router uses model library for decision making
from cosmosapien.core.smart_router import SmartRouter

smart_router = SmartRouter(config_manager, local_manager)
decision = smart_router.smart_route("Your prompt here")

# The decision includes model information from the library
print(f"Selected: {decision.selected_provider}:{decision.selected_model}")
```

## Model Metadata Standards

### Model Types
- **CHAT**: Conversational models (GPT, Claude, etc.)
- **COMPLETION**: Text completion models
- **EMBEDDING**: Text embedding models
- **VISION**: Image understanding models
- **AUDIO**: Audio processing models
- **MULTIMODAL**: Multi-modal models

### Model Tiers
- **FREE**: No cost models (local models, free tiers)
- **BASIC**: Low-cost models (GPT-3.5, Claude Haiku)
- **STANDARD**: Mid-tier models (GPT-4 Turbo, Claude Sonnet)
- **PREMIUM**: High-performance models (GPT-4, Claude Opus)
- **ENTERPRISE**: Enterprise-grade models

### Capabilities
- **max_tokens**: Maximum output tokens
- **max_input_tokens**: Maximum input tokens
- **context_window**: Total context window size
- **supports_streaming**: Whether streaming is supported
- **supports_function_calling**: Whether function calling is supported
- **supports_vision**: Whether vision capabilities are available
- **supports_audio**: Whether audio processing is supported
- **supports_embeddings**: Whether embeddings are supported

### Pricing
- **input_cost_per_1k_tokens**: Cost per 1K input tokens
- **output_cost_per_1k_tokens**: Cost per 1K output tokens
- **free_tier_limit**: Monthly free tier limit
- **free_tier_reset_period**: Reset period (daily, weekly, monthly)

## File Storage

The model library is stored in:
- **Location**: `~/.cosmo/model_library.json`
- **Format**: JSON with standardized structure
- **Backup**: Automatic backup on changes

## Best Practices

### Model Configuration
1. **Use Descriptive Names**: Choose clear, descriptive names for models
2. **Add Relevant Tags**: Include tags for easy searching and filtering
3. **Accurate Pricing**: Keep pricing information up to date
4. **Complete Capabilities**: Specify all relevant capabilities

### Library Management
1. **Regular Updates**: Keep model information current
2. **Version Control**: Use export/import for version control
3. **Documentation**: Document custom models and their use cases
4. **Testing**: Test model configurations before production use

### Integration
1. **Smart Routing**: Use the library with smart routing for cost optimization
2. **Fallbacks**: Always provide fallback models for reliability
3. **Monitoring**: Monitor model usage and performance
4. **Scaling**: Plan for scaling with multiple model providers

## Examples

See `examples/model_library_usage.py` for comprehensive examples of:
- Basic library operations
- Advanced filtering and search
- Custom model creation
- Integration patterns
- Best practices implementation

### Registration Examples

#### Interactive Registration
```bash
# Start the interactive wizard
cosmo register
```

#### Quick Registration
```bash
# Register a basic model
cosmo register-quick myprovider mymodel

# Register with custom parameters
cosmo register-quick myprovider mymodel \
  --name "My Advanced Model" \
  --desc "A powerful model for complex tasks" \
  --tier premium \
  --type chat \
  --max-tokens 8192 \
  --context 16384 \
  --input-cost 0.01 \
  --output-cost 0.02 \
  --free-limit 50 \
  --tags "advanced,complex,reasoning" \
  --local
```

#### Template Registration
```bash
# Register from predefined templates
cosmo register-template gpt4
cosmo register-template claude
cosmo register-template gemini
cosmo register-template local
cosmo register-template custom

# Override template parameters
cosmo register-template gpt4 --provider myopenai --model-id gpt4-custom
```

#### File-based Registration
```bash
# Register from JSON file
cosmo register-from-file examples/model_template.json

# Override file parameters
cosmo register-from-file examples/model_template.json \
  --provider mycompany \
  --model-id production-v1
```

### JSON Template Format

Create a JSON file with the following structure:

```json
{
  "name": "my-custom-model",
  "provider": "custom",
  "model_id": "my-model-v1",
  "display_name": "My Custom Model v1",
  "description": "A custom model for specific use cases",
  "model_type": "chat",
  "tier": "standard",
  "capabilities": {
    "max_tokens": 4096,
    "max_input_tokens": 8192,
    "supports_streaming": true,
    "supports_function_calling": true,
    "supports_vision": false,
    "supports_audio": false,
    "supports_embeddings": false,
    "context_window": 8192,
    "training_data_cutoff": "2024-01"
  },
  "pricing": {
    "input_cost_per_1k_tokens": 0.005,
    "output_cost_per_1k_tokens": 0.01,
    "free_tier_limit": 100,
    "free_tier_reset_period": "monthly",
    "currency": "USD"
  },
  "tags": ["custom", "specialized", "example"],
  "is_active": true,
  "is_local": false
}
```

See `examples/model_template.json` for a complete example.

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure the model ID format is `provider:model_id`
2. **Import Errors**: Check JSON format and required fields
3. **Permission Errors**: Ensure write access to `~/.cosmo/`
4. **Duplicate Models**: Use `--overwrite` flag for import conflicts

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check library status
stats = model_library.get_model_statistics()
print(f"Library contains {stats['total_models']} models")

# Verify model existence
model = model_library.get_model("provider:model")
if model:
    print(f"Model found: {model.display_name}")
else:
    print("Model not found")
``` 