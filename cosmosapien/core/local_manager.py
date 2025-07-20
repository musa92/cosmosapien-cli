"""Local model manager for detecting and managing local AI model runners."""

import asyncio
import platform
import subprocess
from pathlib import Path
from typing import Dict, List


class LocalModelManager:
    """Manages local AI model runners like Ollama, LM Studio, etc."""

    def __init__(self):
        self.system = platform.system().lower()
        self.arch = platform.machine().lower()

    def detect_ollama(self) -> bool:
        """Check if Ollama is installed and running."""
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def detect_lm_studio(self) -> bool:
        """Check if LM Studio is installed."""
        if self.system == "darwin":  # macOS
            return Path("/Applications/LM Studio.app").exists()
        elif self.system == "windows":
            return Path(
                "C:/Users/AppData/Local/Programs/LM Studio/LM Studio.exe"
            ).exists()
        elif self.system == "linux":
            return (
                Path("/usr/local/bin/lmstudio").exists()
                or Path("/opt/lmstudio/lmstudio").exists()
            )
        return False

    def detect_vllm(self) -> bool:
        """Check if vLLM is available."""
        try:
            pass

            return True
        except ImportError:
            return False

    def get_installation_instructions(self, runner: str) -> Dict[str, str]:
        """Get installation instructions for different runners."""
        instructions = {
            "ollama": {
                "darwin": {
                    "url": "https://ollama.ai/download/mac",
                    "command": "curl -fsSL https://ollama.ai/install.sh | sh",
                    "description": "Download from https://ollama.ai or run: curl -fsSL https://ollama.ai/install.sh | sh",
                },
                "linux": {
                    "url": "https://ollama.ai/download/linux",
                    "command": "curl -fsSL https://ollama.ai/install.sh | sh",
                    "description": "Run: curl -fsSL https://ollama.ai/install.sh | sh",
                },
                "windows": {
                    "url": "https://ollama.ai/download/windows",
                    "command": "winget install Ollama.Ollama",
                    "description": "Download from https://ollama.ai or run: winget install Ollama.Ollama",
                },
            },
            "lm_studio": {
                "darwin": {
                    "url": "https://lmstudio.ai",
                    "command": "brew install --cask lmstudio",
                    "description": "Download from https://lmstudio.ai or run: brew install --cask lmstudio",
                },
                "linux": {
                    "url": "https://lmstudio.ai",
                    "command": "Download from website",
                    "description": "Download from https://lmstudio.ai",
                },
                "windows": {
                    "url": "https://lmstudio.ai",
                    "command": "winget install LMStudio.LMStudio",
                    "description": "Download from https://lmstudio.ai or run: winget install LMStudio.LMStudio",
                },
            },
        }
        return instructions.get(runner, {}).get(self.system, {})

    async def install_ollama(self) -> bool:
        """Attempt to install Ollama."""
        try:
            if self.system == "darwin":
                # Try Homebrew first
                result = subprocess.run(
                    ["brew", "install", "ollama"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    return True

                # Fallback to curl install
                result = subprocess.run(
                    ["curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"],
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                return result.returncode == 0

            elif self.system == "linux":
                result = subprocess.run(
                    ["curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"],
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                return result.returncode == 0

            elif self.system == "windows":
                result = subprocess.run(
                    ["winget", "install", "Ollama.Ollama"],
                    capture_output=True,
                    text=True,
                )
                return result.returncode == 0

        except Exception:
            print("Error installing Ollama: {e}")
            return False

        return False

    async def start_ollama(self) -> bool:
        """Start Ollama service."""
        try:
            if self.system == "darwin":
                # Try to start via launchctl
                subprocess.run(
                    ["launchctl", "start", "io.ollama.ollama"], capture_output=True
                )
            elif self.system == "linux":
                subprocess.run(["systemctl", "start", "ollama"], capture_output=True)
            elif self.system == "windows":
                subprocess.run(["ollama", "serve"], capture_output=True)

            # Wait a moment for service to start
            await asyncio.sleep(2)

            # Test if it's running
            return self.detect_ollama()

        except Exception:
            print("Error starting Ollama: {e}")
            return False

    def get_available_models(self, runner: str) -> List[str]:
        """Get available models for a specific runner."""
        if runner == "ollama" and self.detect_ollama():
            try:
                result = subprocess.run(
                    ["ollama", "list"], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")[1:]  # Skip header
                    models = []
                    for line in lines:
                        if line.strip():
                            model_name = line.split()[0]
                            models.append(model_name)
                    return models
            except Exception:
                pass

        # Return default models if detection fails
        return self.get_default_models(runner)

    def get_default_models(self, runner: str) -> List[str]:
        """Get default models for each runner."""
        defaults = {
            "ollama": [
                "llama3.2:3b",
                "llama3.2:8b",
                "llama3.2:70b",
                "llama3.1:8b",
                "llama3.1:70b",
                "mixtral:8x7b",
                "mixtral:8x22b",
                "codellama:7b",
                "codellama:13b",
                "codellama:34b",
                "mistral:7b",
                "mistral:8x7b",
                "neural-chat:7b",
                "neural-chat:8x7b",
                "dolphin-llama3:latest",
                "orca-mini:3b",
                "orca-mini:7b",
                "qwen2.5:7b",
                "qwen2.5:14b",
                "qwen2.5:32b",
                "phi3:mini",
                "phi3:medium",
                "phi3:large",
                "gemma2:2b",
                "gemma2:7b",
                "gemma2:27b",
            ],
            "lm_studio": [
                "llama-3.2-3b",
                "llama-3.2-8b",
                "llama-3.2-70b",
                "mixtral-8x7b",
                "mistral-7b",
                "codellama-7b",
                "codellama-13b",
                "codellama-34b",
            ],
            "vllm": [
                "meta-llama/Llama-3.2-3B",
                "meta-llama/Llama-3.2-8B",
                "meta-llama/Llama-3.2-70B",
                "mistralai/Mistral-7B-v0.1",
                "mistralai/Mixtral-8x7B-v0.1",
                "codellama/CodeLlama-7b-hf",
                "codellama/CodeLlama-13b-hf",
                "codellama/CodeLlama-34b-hf",
            ],
        }
        return defaults.get(runner, [])

    def get_recommended_models(self) -> List[Dict[str, str]]:
        """Get recommended models with descriptions."""
        return [
            {
                "name": "llama3.2:8b",
                "runner": "ollama",
                "description": "Latest Llama 3.2 8B - Good balance of performance and speed",
                "size": "~4.7GB",
                "performance": "High",
            },
            {
                "name": "mixtral:8x7b",
                "runner": "ollama",
                "description": "Mixture of Experts - Excellent reasoning and coding",
                "size": "~26GB",
                "performance": "Very High",
            },
            {
                "name": "codellama:13b",
                "runner": "ollama",
                "description": "Specialized for code generation and analysis",
                "size": "~7.6GB",
                "performance": "High",
            },
            {
                "name": "mistral:8x7b",
                "runner": "ollama",
                "description": "Mistral 8x7B - Great all-around performance",
                "size": "~26GB",
                "performance": "Very High",
            },
            {
                "name": "qwen2.5:7b",
                "runner": "ollama",
                "description": "Qwen 2.5 7B - Fast and capable",
                "size": "~4.1GB",
                "performance": "High",
            },
        ]

    async def setup_local_environment(self) -> Dict[str, bool]:
        """Set up local environment and return status."""
        status = {"ollama": False, "lm_studio": False, "vllm": False}

        # Check Ollama
        if self.detect_ollama():
            status["ollama"] = True
        else:
            print("Ollama not detected. Would you like to install it?")
            # In a real implementation, you'd ask user for confirmation
            # For now, we'll just note it's not available

        # Check LM Studio
        status["lm_studio"] = self.detect_lm_studio()

        # Check vLLM
        status["vllm"] = self.detect_vllm()

        return status

    def get_installation_help(self, runner: str) -> str:
        """Get installation help for a specific runner."""
        instructions = self.get_installation_instructions(runner)

        if not instructions:
            return (
                "No installation instructions available for {runner} on {self.system}"
            )

        help_text = """
Installation instructions for {runner}:

{instructions.get('description', 'No description available')}

URL: {instructions.get('url', 'No URL available')}
Command: {instructions.get('command', 'No command available')}

After installation, restart your terminal and run 'cosmo setup' to configure.
        """
        return help_text.strip()
