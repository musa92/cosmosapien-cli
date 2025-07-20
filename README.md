# Cosmosapien CLI

A modular, extensible command-line interface for interacting with multiple Large Language Model (LLM) providers through a unified API. Built with Python, featuring smart routing, model library management, and multi-agent capabilities.

## Overview

Cosmosapien CLI provides a single interface to interact with various LLM providers including OpenAI, Google Gemini, Claude (Anthropic), Perplexity, and local models via Ollama. It features intelligent prompt routing, comprehensive model management, and support for multi-agent interactions.

## Key Features

### Core Capabilities
- **Multi-Provider Support**: OpenAI, Google Gemini, Claude, Perplexity, LLaMA (Ollama)
- **Smart Routing**: Automatic model selection based on prompt complexity and cost optimization
- **Auto-Distribution**: Intelligent task distribution across multiple models with load balancing
- **Free Tier Optimization**: Maximize usage of free tiers and local models with `squeeze`
- **Model Library**: Centralized model management with metadata and capabilities tracking
- **Multi-Agent System**: Run debates and conversations between multiple AI models
- **Local Model Support**: Integration with Ollama, LM Studio, and vLLM
- **Secure Authentication**: API keys stored securely using system keyring
- **Professional CLI**: Clean, emoji-free interface built with Typer and Rich

### Advanced Features
- **Plugin System**: Extensible architecture for custom providers and tools
- **Memory Management**: Conversation history and session persistence
- **Configuration Management**: TOML-based configuration with `.cosmosrc`
- **Usage Tracking**: Monitor API usage and costs across providers
- **Model Registration**: Easy model addition through interactive wizards and templates
- **Performance Monitoring**: Track model performance, response times, and success rates
- **Token Analytics**: Detailed token usage statistics and distribution analysis
- **Load Balancing**: Distribute workloads across available models efficiently
- **Fallback Mechanisms**: Automatic failover to alternative models on errors

## Screenshots

### Main Help Interface
```
Usage: cosmo [OPTIONS] COMMAND [ARGS]...

Cosmosapien CLI - A modular command-line tool for multiple LLM providers

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────╮
│ setup                     Set up local environment and install dependencies.                             │
│ agents                    Show available AI agents and their capabilities.                               │
│ collaborate               Get collaborative responses from multiple AI agents.                           │
│ solve                     Use multiple agents to solve a complex problem.                                │
│ hybrid                    Get responses from a mix of local and cloud agents.                            │
│ version                   Show version information.                                                      │
│ login                     Login to a provider by storing API key securely.                               │
│ logout                    Logout from a provider by removing API key.                                    │
│ status                    Show login status for all providers.                                           │
│ providers                 Show detailed information about all providers.                                 │
│ cosmic                    Launch the clean cosmic-themed interactive interface.                          │
│ ask                       Ask a question to any supported LLM provider.                                  │
│ chat                      Start an interactive chat session.                                             │
│ debate                    Run a debate between multiple AI models.                                       │
│ list-models               List available models.                                                         │
│ usage                     Show usage statistics and cost tracking.                                       │
│ reset-usage               Reset usage statistics.                                                        │
│ smart-route               Smart route a prompt to the most cost-efficient model.                         │
│ configure-smart-routing   Configure smart routing settings for a specific provider/model.                │
│ models                    List and manage models in the model library.                                   │
│ model-info                Show detailed information about a specific model.                              │
│ search-models             Search models by name, description, or tags.                                   │
│ model-stats               Show statistics about the model library.                                       │
│ export-models             Export the model library to a file.                                            │
│ import-models             Import models from a file.                                                     │
│ register                  Register a new model interactively.                                            │
│ register-quick            Quickly register a new model with minimal input.                               │
│ unregister                Unregister a model from the library.                                           │
│ register-template         Register a model using predefined templates.                                   │
│ register-from-file        Register a model from a JSON configuration file.                               │
│ register-help             Show help for model registration methods.                                      │
│ distribute                Distribute jobs intelligently across multiple models.                          │
│ squeeze                   Use all available free tiers and local models to process the task.             │
│ job-stats                 Show job distribution statistics.                                              │
│ reset-job-stats           Reset job distribution statistics.                                             │
│ token-stats               Show detailed token usage statistics across models.                            │
│ model-performance         Show comprehensive model performance statistics.                               │
│ config                    Show current configuration.                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Provider Information
```
Provider Information

╭──────────────────────────────────────────────── • OpenAI ────────────────────────────────────────────────╮
│ OpenAI                                                                                                   │
│ Individual model access with pay-per-use pricing                                                         │
│                                                                                                          │
│ 🌐 Website                                                                                               │
│ 📚 API Docs                                                                                              │
│ 💳 Subscription: Not Required                                                                            │
│ 🆓 Free Tier: Available                                                                                  │
│ 📦 Tier Type: Individual •                                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────────────────────── • Google Gemini ─────────────────────────────────────────────╮
│ Google Gemini                                                                                            │
│ Individual model access with generous free tier                                                          │
│                                                                                                          │
│ 🌐 Website                                                                                               │
│ 📚 API Docs                                                                                              │
│ 💳 Subscription: Not Required                                                                            │
│ 🆓 Free Tier: Available                                                                                  │
│ 📦 Tier Type: Individual •                                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Model Library
```
                                               Model Library

┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Provider    ┃ Model ID            ┃ Display Name        ┃ Tier  ┃ Type ┃ Tags                ┃ Active ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Openai      │ gpt-4               │ GPT-4               │ Premium│ Chat │ reasoning,          │ Yes    │
│             │                     │                     │       │      │ complex-tasks,      │        │
│             │                     │                     │       │      │ function-calling    │        │
│ Claude      │ claude-3-opus-2024… │ Claude 3 Opus       │ Premium│ Chat │ reasoning,          │ Yes    │
│             │                     │                     │       │      │ analysis,           │        │
│             │                     │                     │       │      │ long-context        │        │
│ Gemini      │ gemini-pro          │ Gemini Pro          │ Standard│ Chat │ google, free-tier,  │ Yes    │
│             │                     │                     │       │      │ general-purpose     │        │
│ Llama       │ llama3.2:8b         │ Llama 3.2 8B        │ Free   │ Chat │ local, free,        │ Yes    │
│             │                     │                     │       │      │ llama...            │        │
└─────────────┴─────────────────────┴─────────────────────┴───────┴──────┴─────────────────────┴────────┘
```

### Token Statistics
```
Token Usage Statistics

Overall Token Usage
Total Tokens: 0
Total Requests: 0
Average Tokens per Request: 0.0

Token Usage by Model
┏━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Model ┃ Total Tokens ┃ Input Tokens ┃ Output Tokens ┃ Avg/Request ┃ Requests ┃
┡━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━┩
└───────┴──────────────┴──────────────┴───────────────┴─────────────┴──────────┘
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/cosmosapien/cli.git
cd cosmosapien-cli

# Install in development mode
pip install -e .

# For development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Ask a question (uses smart routing by default)
cosmo ask "Explain quantum computing in simple terms"

# Start interactive chat with specific model
cosmo chat --provider openai --model gpt-4

# Use smart routing for complex tasks
cosmo ask "Complex reasoning task" --smart-route

# Auto-distribute jobs across models
cosmo distribute "Complex analysis task"

# Squeeze - use all free tiers and local models
cosmo squeeze "Process this task"

# Multi-agent debate
cosmo debate "Should AI be regulated?" --models openai:gpt-4 claude:claude-3-sonnet

# Register new models easily
cosmo register-template gpt4
cosmo register-quick myprovider mymodel --tier premium
```

## Project Structure

```
cosmosapien-cli/
├── cosmosapien/                    # Main package
│   ├── cli/                       # Command-line interface
│   │   ├── main.py               # CLI entry point and commands
│   │   └── cosmic_ui.py          # UI components and styling
│   ├── core/                      # Core functionality
│   │   ├── config.py             # Configuration management
│   │   ├── router.py             # Model routing and selection
│   │   ├── smart_router.py       # Intelligent prompt routing
│   │   ├── model_library.py      # Model library system
│   │   ├── agent_system.py       # Multi-agent interactions
│   │   └── local_manager.py      # Local model management
│   ├── models/                    # LLM provider implementations
│   │   ├── openai_model.py       # OpenAI API integration
│   │   ├── claude_model.py       # Claude API integration
│   │   ├── gemini_model.py       # Gemini API integration
│   │   ├── perplexity_model.py   # Perplexity API integration
│   │   └── llama_model.py        # LLaMA/Ollama integration
│   ├── auth/                      # Authentication system
│   │   └── manager.py            # API key management
│   ├── memory/                    # Conversation memory
│   │   └── manager.py            # Session and history management
│   ├── plugins/                   # Plugin system
│   │   └── manager.py            # Plugin loading and management
│   └── visual/                    # Visual generation (future)
│       └── generator.py          # Image generation capabilities
├── docs/                          # Documentation
│   └── MODEL_LIBRARY.md          # Model library documentation
├── examples/                      # Example scripts and templates
│   ├── model_library_usage.py    # Model library usage examples
│   └── model_template.json       # Model registration template
├── tests/                         # Test suite
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## Core Components

### 1. CLI Interface (`cosmosapien/cli/`)
- **main.py**: Entry point with all CLI commands
- **cosmic_ui.py**: Rich terminal UI components
- Commands: ask, chat, debate, login, models, register, etc.

### 2. Core System (`cosmosapien/core/`)
- **config.py**: Configuration management with TOML
- **router.py**: Model routing and provider selection
- **smart_router.py**: Intelligent prompt routing based on complexity
- **model_library.py**: Centralized model management system
- **agent_system.py**: Multi-agent interaction framework
- **local_manager.py**: Local model integration

### 3. Model Providers (`cosmosapien/models/`)
- Provider-specific implementations for each LLM service
- Async API integration with error handling
- Streaming response support
- Function calling capabilities

### 4. Authentication (`cosmosapien/auth/`)
- Secure API key storage using system keyring
- Provider-specific authentication flows
- Key rotation and management

## Commands Reference

### Authentication
```bash
cosmo login <provider>          # Authenticate with provider
cosmo logout <provider>         # Remove provider credentials
cosmo status                    # Show authentication status
```

### Core Interaction
```bash
cosmo ask <prompt>              # Single question to any model
cosmo chat                      # Interactive chat session
cosmo debate <topic>            # Multi-model debate
```

### Model Management
```bash
cosmo models                    # List all registered models
cosmo model-info <model-id>     # Show model details
cosmo search-models <query>     # Search models by criteria
cosmo model-stats               # Show model usage statistics
```

### Model Registration
```bash
cosmo register                  # Interactive registration wizard
cosmo register-quick <p> <m>    # Quick model registration
cosmo register-template <t>     # Register from template
cosmo register-from-file <f>    # Register from JSON file
cosmo unregister <model-id>     # Remove model from library
cosmo register-help             # Show registration help
```

### Smart Commands
```bash
# Smart routing (single best model)
cosmo ask <prompt> --smart-route    # Use intelligent routing
cosmo smart-route <prompt>          # Show routing decision only

# Auto-distribution (multiple models)
cosmo distribute <prompt>           # Auto-distribute across available models
cosmo distribute <prompt> --type parallel    # Parallel execution
cosmo distribute <prompt> --type pipeline    # Sequential processing
cosmo distribute <prompt> --explain          # Show distribution decision

# Free tier optimization
cosmo squeeze <prompt>              # Use all free tiers and local models
cosmo squeeze <prompt> --explain    # Show free tier decision

# Monitoring
cosmo usage                         # Show usage statistics
cosmo job-stats                     # Show distribution statistics
cosmo token-stats                   # Show detailed token usage
cosmo model-performance             # Show comprehensive performance stats
cosmo reset-usage                   # Reset usage counters
cosmo reset-job-stats               # Reset job statistics
```

### Configuration
```bash
cosmo config                      # Show current configuration
cosmo configure-smart-routing     # Configure smart routing
cosmo export-models <file>        # Export model library
cosmo import-models <file>        # Import model library
```

## Configuration

### Main Configuration (`~/.cosmosrc`)
```toml
default_provider = "openai"
default_model = "gpt-4"

[smart_routing]
free_tier_limits = { openai = 3, gemini = 15, claude = 0 }
custom_costs = { "openai:gpt-4" = 0.03, "claude:claude-3-sonnet" = 0.003 }

[providers.openai]
base_url = "https://api.openai.com/v1"
```

### Model Library (`~/.cosmo/model_library.json`)
Stores model metadata, capabilities, and pricing information.

### Usage Tracking (`~/.cosmo/usage.json`)
Tracks API usage and costs across providers.

## Supported Providers

| Provider | Models | Free Tier | Setup |
|----------|--------|-----------|-------|
| **OpenAI** | GPT-4, GPT-3.5 Turbo | Limited | `cosmo login openai` |
| **Claude** | Claude 3 Opus/Sonnet/Haiku | No | `cosmo login claude` |
| **Gemini** | Gemini Pro, Gemini 1.5 | 15 calls/month | `cosmo login gemini` |
| **Perplexity** | LLaMA 3.1, Mixtral | Limited | `cosmo login perplexity` |
| **Local** | LLaMA, Mistral, CodeLlama | Unlimited | Install Ollama first |

## Development

### Setup Development Environment
```bash
# Clone and install
git clone https://github.com/cosmosapien/cli.git
cd cosmosapien-cli
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests
```bash
pytest                    # Run all tests
pytest -v                # Verbose output
pytest --cov=cosmosapien # With coverage
```

### Code Quality
```bash
black cosmosapien/       # Code formatting
isort cosmosapien/       # Import sorting
flake8 cosmosapien/      # Linting
mypy cosmosapien/        # Type checking
```

### Adding New Providers

1. Create provider class in `cosmosapien/models/`
2. Implement required methods from `BaseModel`
3. Add authentication support in `cosmosapien/auth/`
4. Update router and configuration
5. Add tests and documentation

Example:
```python
# cosmosapien/models/my_provider.py
from cosmosapien.models.base_model import BaseModel

class MyProviderModel(BaseModel):
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        # Implementation here
        pass
```

## Contributing

We welcome contributions! Here's how to get started:

### Contribution Guidelines

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make your changes** following the code style
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Submit a pull request**

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and test
pytest
black cosmosapien/
mypy cosmosapien/

# Commit with clear message
git commit -m "Add new feature: description"

# Push and create PR
git push origin feature/new-feature
```

### Areas for Contribution

- **New Providers**: Add support for additional LLM services
- **Plugin System**: Create useful plugins and extensions
- **UI Improvements**: Enhance CLI interface and user experience
- **Documentation**: Improve docs, add examples, tutorials
- **Testing**: Add tests, improve coverage
- **Performance**: Optimize routing, caching, response times
- **Features**: Smart routing improvements, memory enhancements

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions
- Keep functions small and focused
- Add comprehensive tests
- No emojis in code or UI (professional interface)

## Architecture Decisions

### Design Principles
- **Modularity**: Easy to add new providers and features
- **Extensibility**: Plugin system for custom functionality
- **Professional**: Clean, emoji-free interface
- **Efficient**: Smart routing and cost optimization
- **Secure**: Proper API key management

### Technology Stack
- **Python 3.8+**: Core language
- **Typer**: CLI framework
- **Rich**: Terminal UI library
- **Pydantic**: Data validation
- **TOML**: Configuration format
- **Keyring**: Secure credential storage

## Roadmap

### Short Term
- [ ] Enhanced plugin marketplace
- [ ] Advanced memory features
- [ ] Web interface
- [ ] Docker support
- [ ] More local model integrations

### Long Term
- [ ] Multi-modal support (vision, audio)
- [ ] Advanced agent orchestration
- [ ] Enterprise features
- [ ] API server mode
- [ ] Mobile companion app

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/cosmosapien/cli/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cosmosapien/cli/discussions)
- **Documentation**: [docs/](docs/) directory

## Acknowledgments

- Inspired by Google's `gemini-cli`
- Built with modern Python tooling
- Community-driven development
