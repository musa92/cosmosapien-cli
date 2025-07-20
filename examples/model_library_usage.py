#!/usr/bin/env python3
"""
Example: Using the Model Library in Code

This example demonstrates how to use the model library to:
1. List and search models
2. Get model information
3. Use models in your code
4. Manage model configurations
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cosmosapien.core.config import ConfigManager
from cosmosapien.core.model_library import ModelLibrary, ModelType, ModelTier, ModelConfig, ModelCapability, ModelPricing


async def main():
    """Main example function."""
    
    # Initialize the model library
    config_manager = ConfigManager()
    model_library = ModelLibrary(config_manager)
    
    print("=== Model Library Usage Example ===\n")
    
    # 1. List all models
    print("1. All Available Models:")
    models = model_library.list_models()
    for model in models[:5]:  # Show first 5
        print(f"  - {model.provider}:{model.model_id} ({model.display_name})")
    print(f"  ... and {len(models) - 5} more models\n")
    
    # 2. Search for models
    print("2. Searching for 'GPT' models:")
    gpt_models = model_library.search_models("GPT")
    for model in gpt_models:
        print(f"  - {model.provider}:{model.model_id} - {model.description}")
    print()
    
    # 3. Get models by provider
    print("3. OpenAI Models:")
    openai_models = model_library.get_models_by_provider("openai")
    for model in openai_models:
        print(f"  - {model.model_id} ({model.tier.value} tier)")
    print()
    
    # 4. Get free models
    print("4. Free Models:")
    free_models = model_library.get_free_models()
    for model in free_models:
        print(f"  - {model.provider}:{model.model_id} - {model.pricing.free_tier_limit} calls/month")
    print()
    
    # 5. Get model details
    print("5. Detailed Model Information:")
    model = model_library.get_model("openai:gpt-4")
    if model:
        print(f"  Model: {model.display_name}")
        print(f"  Description: {model.description}")
        print(f"  Tier: {model.tier.value}")
        print(f"  Type: {model.model_type.value}")
        print(f"  Context Window: {model.capabilities.context_window}")
        print(f"  Input Cost: ${model.pricing.input_cost_per_1k_tokens:.4f}/1K tokens")
        print(f"  Tags: {', '.join(model.tags)}")
    print()
    
    # 6. Add a custom model
    print("6. Adding a Custom Model:")
    custom_model = ModelConfig(
        name="custom-model",
        provider="custom",
        model_id="custom-v1",
        display_name="Custom Model v1",
        description="A custom model for specific use cases",
        model_type=ModelType.CHAT,
        tier=ModelTier.STANDARD,
        capabilities=ModelCapability(
            max_tokens=2048,
            max_input_tokens=4096,
            supports_streaming=True,
            context_window=4096
        ),
        pricing=ModelPricing(
            input_cost_per_1k_tokens=0.005,
            output_cost_per_1k_tokens=0.01,
            free_tier_limit=10
        ),
        tags=["custom", "specialized", "example"]
    )
    
    if model_library.add_model(custom_model):
        print("  Custom model added successfully!")
    else:
        print("  Custom model already exists or failed to add.")
    print()
    
    # 7. Update model configuration
    print("7. Updating Model Configuration:")
    if model_library.update_model("custom:custom-v1", is_active=False):
        print("  Custom model deactivated successfully!")
    print()
    
    # 8. Get statistics
    print("8. Model Library Statistics:")
    stats = model_library.get_model_statistics()
    print(f"  Total Models: {stats['total_models']}")
    print(f"  Active Models: {stats['active_models']}")
    print(f"  Local Models: {stats['local_models']}")
    print(f"  Free Models: {stats['free_models']}")
    print()
    
    # 9. Filter models by criteria
    print("9. Premium Models:")
    premium_models = model_library.get_models_by_tier(ModelTier.PREMIUM)
    for model in premium_models:
        print(f"  - {model.provider}:{model.model_id} - {model.display_name}")
    print()
    
    # 10. Export and import
    print("10. Export/Import Functionality:")
    export_path = "example_models_export.json"
    if model_library.export_library(export_path):
        print(f"  Model library exported to {export_path}")
        
        # Import the exported library
        if model_library.import_library(export_path, overwrite=False):
            print("  Model library imported successfully!")
        
        # Clean up
        Path(export_path).unlink(missing_ok=True)
    print()
    
    print("=== Example Complete ===")


def example_model_usage():
    """Example of how to use models in your application."""
    
    config_manager = ConfigManager()
    model_library = ModelLibrary(config_manager)
    
    # Get a specific model for your use case
    def get_best_model_for_task(task_type: str, budget: float = 0.01):
        """Get the best model for a specific task and budget."""
        
        if task_type == "simple":
            # For simple tasks, prefer free or local models
            models = model_library.get_free_models()
            if models:
                return models[0]  # Return first free model
        
        elif task_type == "complex":
            # For complex tasks, prefer premium models within budget
            premium_models = model_library.get_models_by_tier(ModelTier.PREMIUM)
            affordable_models = [
                m for m in premium_models 
                if m.pricing.input_cost_per_1k_tokens <= budget
            ]
            if affordable_models:
                return affordable_models[0]
        
        elif task_type == "coding":
            # For coding tasks, look for models with coding tags
            coding_models = model_library.get_models_by_tag("code")
            if coding_models:
                return coding_models[0]
        
        # Fallback to any available model
        active_models = model_library.get_active_models()
        return active_models[0] if active_models else None
    
    # Example usage
    print("=== Model Selection Examples ===")
    
    simple_model = get_best_model_for_task("simple")
    if simple_model:
        print(f"Best model for simple tasks: {simple_model.provider}:{simple_model.model_id}")
    
    complex_model = get_best_model_for_task("complex", budget=0.02)
    if complex_model:
        print(f"Best model for complex tasks: {complex_model.provider}:{complex_model.model_id}")
    
    coding_model = get_best_model_for_task("coding")
    if coding_model:
        print(f"Best model for coding tasks: {coding_model.provider}:{coding_model.model_id}")


if __name__ == "__main__":
    # Run the async example
    asyncio.run(main())
    
    # Run the sync example
    example_model_usage() 