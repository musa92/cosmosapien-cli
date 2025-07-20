# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive token statistics and analytics
- New `cosmo token-stats` command for detailed token usage
- New `cosmo model-performance` command for comprehensive performance metrics
- Enhanced job distribution system with load balancing
- Free tier optimization with `cosmo squeeze` command
- Model library system with registration and management
- Smart routing with complexity-based model selection
- Professional CLI interface with Rich tables
- Job distribution statistics and monitoring

### Changed
- Enhanced `cosmo job-stats` with token information
- Improved README with comprehensive documentation
- Updated project structure for better organization

### Fixed
- Various bug fixes and improvements

## [0.1.0] - 2024-01-XX

### Added
- Initial release of Cosmosapien CLI
- Multi-provider support (OpenAI, Claude, Gemini, Perplexity, LLaMA)
- Basic CLI interface with Typer
- Authentication system with secure key storage
- Configuration management with TOML
- Basic model routing and selection
- Interactive chat functionality
- Multi-agent debate system
- Plugin system architecture
- Memory management for conversations
- Local model integration via Ollama

### Changed
- N/A (Initial release)

### Fixed
- N/A (Initial release)

## [0.0.1] - 2024-01-XX

### Added
- Project initialization
- Basic project structure
- Development environment setup
- Initial documentation

---

## Release Notes

### Version 0.1.0
This is the initial public release of Cosmosapien CLI, featuring a comprehensive command-line interface for interacting with multiple LLM providers. The release includes core functionality for authentication, model management, and intelligent routing.

### Key Features
- **Multi-Provider Support**: Seamless integration with OpenAI, Claude, Gemini, Perplexity, and local LLaMA models
- **Smart Routing**: Intelligent model selection based on task complexity and cost optimization
- **Professional CLI**: Clean, emoji-free interface built with Typer and Rich
- **Secure Authentication**: API key management using system keyring
- **Extensible Architecture**: Plugin system for custom providers and tools

### Breaking Changes
None (Initial release)

### Migration Guide
N/A (Initial release)

---

## Contributing

To add entries to this changelog:

1. Add your changes under the `[Unreleased]` section
2. Use the appropriate category: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, or `Security`
3. Write clear, concise descriptions
4. Reference issues and pull requests when applicable
5. Follow the existing format and style

## Links

- [GitHub Releases](https://github.com/cosmosapien/cli/releases)
- [Project Documentation](https://github.com/cosmosapien/cli#readme)
- [Contributing Guidelines](CONTRIBUTING.md) 