# Cosmosapien CLI

A modular command-line tool for interacting with multiple LLM providers, inspired by Google's `gemini-cli` but with enhanced capabilities and extensibility.

## Features

- **Multi-Provider Support**: OpenAI, Google Gemini, Claude (Anthropic), Perplexity, and LLaMA via Ollama
- **Secure Authentication**: API keys stored securely using keyring
- **Beautiful CLI**: Rich terminal output with Typer
- **Interactive Chat**: Real-time conversations with any model
- **Model Debates**: Run agent-to-agent interactions between multiple models
- **Modular Architecture**: Easy to add new providers and tools
- **Plugin System**: Extensible via custom plugins
- **Memory Management**: Conversation history and session management
- **Configuration**: TOML-based configuration with `.cosmosrc`

## Installation

### From Source

```bash
git clone https://github.com/cosmosapien/cli.git
cd cosmosapien-cli
pip install -e .
```

### Development Setup

```bash
pip install -e ".[dev]"
```

## Quick Start

1. **Login to a provider**:
   ```bash
   cosmo login openai
   # Enter your API key when prompted
   ```

2. **Ask a question**:
   ```bash
   cosmo ask "What is the meaning of life?"
   ```

3. **Start a chat session**:
   ```bash
   cosmo chat --provider openai --model gpt-4
   ```

4. **Run a debate between models**:
   ```bash
   cosmo debate "Should AI be regulated?" --models openai:gpt-4 claude:claude-3-sonnet-20240229
   ```

## Commands

### Authentication

- `cosmo login <provider>` - Login to a provider
- `cosmo logout <provider>` - Logout from a provider
- `cosmo status` - Show login status for all providers

### Core Commands

- `cosmo ask <prompt>` - Ask a question to any model
- `cosmo chat` - Start an interactive chat session
- `cosmo debate <topic>` - Run a debate between multiple models

### Information

- `cosmo list-models [--provider]` - List available models
- `cosmo config` - Show current configuration
- `cosmo version` - Show version information

## Configuration

Copy the example configuration file:

```bash
cp .cosmosrc.example ~/.cosmosrc
```

Edit `~/.cosmosrc` to customize your settings:

```toml
default_provider = "openai"
default_model = "gpt-4"

[providers.openai]
base_url = "https://api.openai.com/v1"
```

## Supported Providers

### OpenAI
- Models: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- Setup: `cosmo login openai`

### Google Gemini
- Models: Gemini Pro, Gemini Pro Vision, Gemini 1.5
- Setup: `cosmo login gemini`

### Claude (Anthropic)
- Models: Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
- Setup: `cosmo login claude`

### Perplexity
- Models: LLaMA 3.1 Sonar, Mixtral, CodeLlama
- Setup: `cosmo login perplexity`

### LLaMA (Ollama)
- Models: LLaMA 2, CodeLlama, Mistral, Mixtral
- Setup: Install [Ollama](https://ollama.ai) first, then use `cosmo ask --provider llama`

## Examples

### Basic Usage

```bash
# Ask a question with default provider
cosmo ask "Explain quantum computing in simple terms"

# Use specific provider and model
cosmo ask "Write a Python function to sort a list" --provider openai --model gpt-4

# Interactive chat
cosmo chat --provider claude --model claude-3-sonnet-20240229
```

### Advanced Features

```bash
# Debate between multiple models
cosmo debate "What's the best programming language for beginners?" \
  --models openai:gpt-4 claude:claude-3-sonnet-20240229 gemini:gemini-pro \
  --rounds 5

# List available models for a provider
cosmo list-models --provider openai

# Show configuration
cosmo config
```

## Architecture

```
cosmosapien/
├── cli/           # Command-line interface
├── core/          # Core functionality (config, router, models)
├── models/        # LLM provider implementations
├── auth/          # Authentication and key management
├── memory/        # Conversation history
├── visual/        # Image generation (future)
└── plugins/       # Plugin system
```

## Plugin Development

Create custom plugins by placing Python files in `~/.cosmosapien/plugins/`:

```python
# ~/.cosmosapien/plugins/my_plugin.py
from cosmosapien.core.models import BaseModel, ModelResponse

class MyCustomModel(BaseModel):
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        # Your implementation here
        pass

def setup():
    from cosmosapien.core.models import model_registry
    model_registry.register("my_provider", MyCustomModel)
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black cosmosapien/
isort cosmosapien/
```

### Type Checking

```bash
mypy cosmosapien/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Roadmap

- [ ] Image generation support
- [ ] Streaming responses
- [ ] Web interface
- [ ] More providers (Cohere, Hugging Face, etc.)
- [ ] Advanced memory features
- [ ] Plugin marketplace
- [ ] Docker support
