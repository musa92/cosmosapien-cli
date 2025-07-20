"""LLaMA model implementation using Ollama."""

import asyncio
import json
import subprocess
from typing import Any, Dict, List, Optional

from ..core.models import BaseModel, ChatMessage, ModelResponse


class LLaMA(BaseModel):
    """LLaMA model implementation using Ollama."""

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        model: str = "llama2",
        **kwargs,
    ):
        # LLaMA/Ollama doesn't need API keys - it runs locally
        super().__init__("", **kwargs)  # Pass empty API key
        self.model = model
        self.base_url = base_url or "http://localhost:11434"

    async def _run_ollama_command(self, command: List[str]) -> str:
        """Run an Ollama command and return the output."""
        try:
            process = await asyncio.create_subprocess_exec(
                "ollama",
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise Exception(f"Ollama command failed: {stderr.decode()}")

            return stdout.decode().strip()
        except FileNotFoundError:
            raise Exception(
                "Ollama not found. Please install Ollama first: https://ollama.ai"
            )

    async def _ensure_model_available(self):
        """Ensure the model is available in Ollama."""
        try:
            # Check if model exists
            await self._run_ollama_command(["list"])
        except Exception:
            # Try to pull the model
            print(f"Pulling model {self.model}...")
            await self._run_ollama_command(["pull", self.model])

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response using Ollama."""
        try:
            await self._ensure_model_available()

            # Prepare the request
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                **kwargs,
            }

            # Run the generate command
            result = await self._run_ollama_command(["run", self.model, prompt])

            return ModelResponse(
                content=result,
                model=self.model,
                provider="llama",
                usage=None,  # Ollama doesn't provide usage info
                metadata={
                    "provider": "ollama",
                    "model": self.model,
                },
            )
        except Exception as e:
            raise Exception(f"LLaMA/Ollama error: {str(e)}")

    async def chat(self, messages: List[ChatMessage], **kwargs) -> ModelResponse:
        """Generate a chat response using Ollama."""
        try:
            await self._ensure_model_available()

            # Convert messages to a single prompt
            prompt = ""
            for msg in messages:
                if msg.role == "system":
                    prompt += f"System: {msg.content}\n\n"
                elif msg.role == "user":
                    prompt += f"User: {msg.content}\n\n"
                elif msg.role == "assistant":
                    prompt += f"Assistant: {msg.content}\n\n"

            prompt += "Assistant: "

            # Generate response
            result = await self._run_ollama_command(["run", self.model, prompt])

            return ModelResponse(
                content=result,
                model=self.model,
                provider="llama",
                usage=None,
                metadata={
                    "provider": "ollama",
                    "model": self.model,
                },
            )
        except Exception as e:
            raise Exception(f"LLaMA/Ollama error: {str(e)}")

    def get_available_models(self) -> List[str]:
        """Get list of available LLaMA models (common Ollama models)."""
        return [
            "llama2",
            "llama2:7b",
            "llama2:13b",
            "llama2:70b",
            "llama2:7b-chat",
            "llama2:13b-chat",
            "llama2:70b-chat",
            "llama2:7b-code",
            "llama2:13b-code",
            "llama2:34b-code",
            "codellama",
            "codellama:7b",
            "codellama:13b",
            "codellama:34b",
            "mistral",
            "mistral:7b",
            "mixtral",
            "mixtral:8x7b",
            "neural-chat",
            "orca-mini",
            "orca-mini:3b",
            "orca-mini:7b",
        ]
