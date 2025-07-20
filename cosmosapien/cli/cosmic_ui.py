"""Clean cosmic-themed UI for Cosmosapien CLI."""

import random
from typing import Optional

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from ..auth.manager import AuthManager
from ..core.agent_system import AgentSystem
from ..core.config import ConfigManager
from ..core.local_manager import LocalModelManager
from ..core.models import ChatMessage
from ..core.router import Router
from ..core.provider_info import get_provider_display_name


class CosmicUI:
    """Clean cosmic-themed user interface for Cosmosapien CLI."""

    def __init__(self):
        self.console = Console()
        self.config_manager = ConfigManager()
        self.auth_manager = AuthManager(self.config_manager)
        self.router = Router(self.config_manager)
        self.local_manager = LocalModelManager()
        self.agent_system = AgentSystem(self.router, self.local_manager)

        # Clean cosmic elements
        self.cosmic_symbols = ["I", "B", "L", "A", "X", "Y", "Z", "Q"]

        # Default open-source models (no API key required)
        self.default_models = {
            "llama": "dolphin-llama3:latest",  # More powerful local model
            "huggingface": "gpt2",  # Free tier
        }

    def get_cosmic_header(self) -> str:
        """Generate a clean cosmic header."""
        header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    COSMOSAPIEN CLI                                                          ║
║                                                                              ║
║    Exploring the cosmos of AI models                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        return header

    def get_cosmic_tips(self) -> str:
        """Get clean cosmic-themed tips."""
        tips = [
            "Ask questions to explore the AI cosmos",
            "Be specific for the best responses",
            "/help for more information",
            "Using open-source models by default",
        ]
        return "\n".join([f"- {tip}" for tip in tips])

    def get_random_cosmic_symbol(self) -> str:
        """Get a random cosmic symbol."""
        return random.choice(self.cosmic_symbols)

    def create_clean_panel(
        self, content: str, title: str = "", color: str = "blue"
    ) -> Panel:
        """Create a clean panel."""
        return Panel(
            content, title=title, border_style=color, box=ROUNDED, padding=(1, 2)
        )

    def show_welcome_screen(self):
        """Show the clean cosmic welcome screen."""
        self.console.clear()

        # Cosmic header
        header = self.get_cosmic_header()
        self.console.print(Panel(header, border_style="bright_blue", box=ROUNDED))

        # Tips section
        tips = self.get_cosmic_tips()
        tips_panel = self.create_clean_panel(
            tips, title="Tips for getting started:", color="cyan"
        )
        self.console.print(tips_panel)

        # Status bar
        status_bar = self.create_status_bar()
        self.console.print(status_bar)

        self.console.print("\n")

    def create_status_bar(self) -> str:
        """Create a clean status bar."""
        providers = self.auth_manager.list_providers()
        sum(1 for p in providers if p["logged_in"])

        status_elements = [
            "Connected: {connected_count}/{len(providers)}",
            "Mode: Interactive",
            "Model: Open Source (llama2)",
        ]

        return " | ".join(status_elements)

    def show_clean_response(self, response: str, provider: str, model: str):
        """Show a clean response."""
        symbol = self.get_random_cosmic_symbol()
        provider_display = get_provider_display_name(provider)
        response_panel = self.create_clean_panel(
            f"{symbol} {response}",
            title=f"{provider_display} ({model})",
            color="magenta",
        )
        self.console.print(response_panel)

    def show_clean_thinking(self, message: str = "Exploring the cosmic knowledge..."):
        """Show a clean thinking animation."""
        with Progress(
            SpinnerColumn(style="bright_blue"),
            TextColumn("[bright_blue]{message}"),
            console=self.console,
            transient=True,
        ) as progress:
            pass

    def get_default_open_source_provider(self) -> tuple:
        """Get the best available open-source provider and model."""
        providers = self.auth_manager.list_providers()

        # Priority order: huggingface (free tier), llama (local)
        for provider_name in ["huggingface", "llama"]:
            for provider in providers:
                if provider["provider"] == provider_name:
                    if (
                        provider["logged_in"] or provider_name == "llama"
                    ):  # llama doesn't need login
                        return provider_name, self.default_models.get(
                            provider_name, "default"
                        )

        # Fallback to llama (local)
        return "llama", "codellama:latest"

    async def clean_chat(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ):
        """Start a clean chat session with open-source defaults."""
        self.show_welcome_screen()

        # Use open-source defaults if no provider specified
        if not provider:
            provider, model = self.get_default_open_source_provider()

        messages = []
        self.get_random_cosmic_symbol()

        self.console.print(
            "{symbol} Welcome to the cosmic chat. Using {provider} ({model}). Type 'quit' to exit.\n"
        )

        while True:
            # Get user input with clean prompt
            user_input = Prompt.ask("[bright_green]>[/bright_green]")

            if user_input.lower() in ["quit", "exit", "q", "bye"]:
                self.console.print("{symbol} [yellow]Goodbye![/yellow]")
                break

            if not user_input.strip():
                continue

            # Add user message
            messages.append(ChatMessage(role="user", content=user_input))

            try:
                # Show clean thinking
                with Progress(
                    SpinnerColumn(style="bright_blue"),
                    TextColumn("[bright_blue]Consulting AI models..."),
                    console=self.console,
                    transient=True,
                ) as progress:
                    pass

                    response = await self.router.chat(
                        messages=messages,
                        provider=provider,
                        model=model,
                    )

                # Add assistant response
                messages.append(ChatMessage(role="assistant", content=response.content))

                # Show clean response
                self.show_clean_response(
                    response.content, response.provider, response.model
                )

            except Exception:
                self.console.print("[red]Error: {str(e)}[/red]")
                self.console.print(
                    "[yellow]Trying to use a different open-source model...[/yellow]"
                )

                # Try to switch to a different open-source model
                if provider == "llama":
                    provider, model = "huggingface", "gpt2"
                elif provider == "huggingface":
                    provider, model = "llama", "llama2"
                else:
                    provider, model = "llama", "llama2"

                self.console.print("[cyan]Switched to {provider} ({model})[/cyan]")

                if messages and messages[-1].role == "user":
                    messages.pop()

    def show_clean_providers(self):
        """Show providers in a clean theme."""
        providers = self.auth_manager.list_providers()

        table = Table(title="AI Providers", box=ROUNDED)
        table.add_column("Provider", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Tier", style="magenta")
        table.add_column("Connected", style="yellow")

        for provider in providers:
            provider_name = provider["provider"]
            status_icon = "✓" if provider["logged_in"] else "✗"

            # Clean provider name
            clean_name = provider_name.title()

            # Determine tier with clean symbols
            tier_symbol = "I"  # Default for now

            table.add_row(
                clean_name,
                status_icon,
                tier_symbol,
                "Yes" if provider["logged_in"] else "No",
            )

        self.console.print(table)

    def show_clean_help(self):
        """Show clean help."""
        help_content = """
**Commands:**

- `cosmo ask <question>` - Ask AI models
- `cosmo chat` - Start an interactive conversation
- `cosmo status` - Check your connections
- `cosmo login <provider>` - Connect to an AI provider
- `cosmo providers` - Explore available models
- `cosmo debate <topic>` - Watch AI models debate

**Tips:**
- Be specific in your queries
- Different AI models have different knowledge
- Some models require API keys
- Local models don't need credentials

**Provider Types:**
I Individual (I) - Pay per interaction
B Bundled (B) - Multiple models with subscription
L Local (L) - Models that run locally
        """

        help_panel = self.create_clean_panel(
            help_content, title="Help Guide", color="bright_blue"
        )

        self.console.print(help_panel)


def main():
    """Main entry point for clean cosmic UI."""
    cosmic_ui = CosmicUI()
    cosmic_ui.show_welcome_screen()

    # For now, just show the welcome screen
    # In a full implementation, this would start an interactive session
    cosmic_ui.console.print("\n[dim]Cosmic UI ready for exploration...[/dim]")


if __name__ == "__main__":
    main()
