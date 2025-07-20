"""Main CLI application for Cosmosapien."""

import asyncio
import sys
from typing import Optional, List
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.text import Text

from ..core.config import ConfigManager
from ..core.router import Router
from ..core.models import ChatMessage, model_registry
from ..core.provider_info import get_provider_display_name, get_provider_info, get_all_providers
from ..auth.manager import AuthManager
from ..models import OpenAI, Gemini, Claude, Perplexity, LLaMA, Grok, HuggingFace

# Initialize Typer app
app = typer.Typer(
    name="cosmo",
    help="Cosmosapien CLI - A modular command-line tool for multiple LLM providers",
    add_completion=False,
)

# Initialize Rich console
console = Console()

# Global instances
config_manager = ConfigManager()
auth_manager = AuthManager(config_manager)
router = Router(config_manager)

# Register models
model_registry.register("openai", OpenAI)
model_registry.register("gemini", Gemini)
model_registry.register("claude", Claude)
model_registry.register("perplexity", Perplexity)
model_registry.register("llama", LLaMA)
model_registry.register("grok", Grok)
model_registry.register("huggingface", HuggingFace)


def print_error(message: str):
    """Print an error message."""
    console.print(f"[red]Error: {message}[/red]")


def print_success(message: str):
    """Print a success message."""
    console.print(f"[green]✓ {message}[/green]")


def get_default_open_source_provider() -> tuple:
    """Get the best available open-source provider and model."""
    providers = auth_manager.list_providers()
    
    # Priority order: llama (local), huggingface (free tier)
    for provider_name in ["llama", "huggingface"]:
        for provider in providers:
            if provider["provider"] == provider_name:
                if provider["logged_in"] or provider_name == "llama":  # llama doesn't need login
                    default_models = {
                        "llama": "llama2",
                        "huggingface": "gpt2",
                    }
                    return provider_name, default_models.get(provider_name, "default")
    
    # Fallback to llama (local)
    return "llama", "llama2"


@app.command()
def version():
    """Show version information."""
    from .. import __version__
    console.print(f"[bold blue]Cosmosapien CLI v{__version__}[/bold blue]")


@app.command()
def login(
    provider: str = typer.Argument(..., help=f"Provider name ({', '.join(get_all_providers())})"),
    api_key: Optional[str] = typer.Option(None, "--key", "-k", help="API key (will prompt if not provided)")
):
    """Login to a provider by storing API key securely."""
    if provider not in get_all_providers():
        print_error(f"Unknown provider: {provider}")
        print_error(f"Available providers: {', '.join(get_all_providers())}")
        raise typer.Exit(1)
    
    if auth_manager.login(provider, api_key):
        print_success(f"Logged in to {provider}")
    else:
        print_error(f"Failed to login to {provider}")
        raise typer.Exit(1)


@app.command()
def logout(
    provider: str = typer.Argument(..., help="Provider name to logout from")
):
    """Logout from a provider by removing API key."""
    if auth_manager.logout(provider):
        print_success(f"Logged out from {provider}")
    else:
        print_error(f"Failed to logout from {provider}")
        raise typer.Exit(1)


@app.command()
def status():
    """Show login status for all providers."""
    providers = auth_manager.list_providers()
    
    table = Table(title="Provider Status")
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Tier", style="magenta", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Logged In", style="yellow")
    
    for provider in providers:
        provider_name = provider["provider"]
        status_icon = "✓" if provider["logged_in"] else "✗"
        provider_info = get_provider_info(provider_name)
        
        # Get tier information
        tier_text = "Individual"
        tier_style = "blue"
        if provider_info:
            if provider_info.tier_type == "bundled":
                tier_text = "Bundled ⭐"
                tier_style = "yellow"
            elif provider_info.tier_type == "local":
                tier_text = "Local 🏠"
                tier_style = "green"
            elif provider_info.tier_type == "individual":
                tier_text = "Individual 🔑"
                tier_style = "blue"
        
        table.add_row(
            get_provider_display_name(provider_name),
            f"[{tier_style}]{tier_text}[/{tier_style}]",
            status_icon,
            "Yes" if provider["logged_in"] else "No"
        )
    
    console.print(table)
    
    # Add tier explanation
    console.print("\n[bold]Tier Types:[/bold]")
    console.print("• [blue]Individual[/blue] - Pay per model/usage")
    console.print("* [yellow]Bundled[/yellow] - Multiple models with subscription")
    console.print("○ [green]Local[/green] - Run locally, no API key needed")


@app.command()
def providers():
    """Show detailed information about all providers."""
    console.print("[bold blue]Provider Information[/bold blue]\n")
    
    for provider_name in get_all_providers():
        info = get_provider_info(provider_name)
        if not info:
            continue
            
        # Create provider card
        console.print(Panel(
            f"[bold]{info.display_name}[/bold]\n"
            f"[dim]{info.description}[/dim]\n\n"
            f"🌐 [link={info.website}]Website[/link]\n"
            f"📚 [link={info.api_docs}]API Docs[/link]\n"
            f"💳 Subscription: {'Required' if info.subscription_required else 'Not Required'}\n"
            f"🆓 Free Tier: {'Available' if info.free_tier_available else 'Not Available'}\n"
            f"📦 Tier Type: {info.tier_type.title()} {info.tier_icon}",
            title=f"{info.tier_icon} {info.display_name}",
            border_style="blue" if info.tier_type == "individual" else "yellow" if info.tier_type == "bundled" else "green"
        ))
        console.print("\n")


@app.command()
def cosmic():
    """Launch the clean cosmic-themed interactive interface."""
    from .cosmic_ui import CosmicUI
    
    async def _cosmic():
        cosmic_ui = CosmicUI()
        await cosmic_ui.clean_chat()
    
    asyncio.run(_cosmic())


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Your question or prompt"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Provider to use"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Maximum tokens to generate"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Temperature for generation"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream the response"),
):
    """Ask a question to any supported LLM provider."""
    
    async def _ask():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Generating response...", total=None)
                
                response = await router.generate(
                    prompt=prompt,
                    provider=provider,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            
            # Display the response
            console.print("\n")
            console.print(Panel(
                Markdown(response.content),
                title=f"[bold]{response.provider.title()} ({response.model})[/bold]",
                border_style="blue"
            ))
            
            # Show usage info if available
            if response.usage:
                usage_table = Table(title="Usage Information")
                usage_table.add_column("Metric", style="cyan")
                usage_table.add_column("Value", style="green")
                
                for key, value in response.usage.items():
                    usage_table.add_row(key.replace("_", " ").title(), str(value))
                
                console.print(usage_table)
                
        except Exception as e:
            print_error(str(e))
            raise typer.Exit(1)
    
    asyncio.run(_ask())


@app.command()
def chat(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Provider to use (defaults to open-source)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
):
    """Start an interactive chat session."""
    
    async def _chat():
        try:
            messages = []
            
            # Use open-source defaults if no provider specified
            chat_provider = provider
            chat_model = model
            
            if not chat_provider:
                chat_provider, chat_model = get_default_open_source_provider()
                console.print(f"[cyan]Using open-source model: {chat_provider} ({chat_model})[/cyan]\n")
            
            # Add system message if provided
            if system_prompt:
                messages.append(ChatMessage(role="system", content=system_prompt))
            
            console.print("[bold blue]Chat session started. Type 'quit' to exit.[/bold blue]\n")
            
            while True:
                # Get user input
                user_input = Prompt.ask("[bold green]You[/bold green]")
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                if not user_input.strip():
                    continue
                
                # Add user message
                messages.append(ChatMessage(role="user", content=user_input))
                
                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                        transient=True,
                    ) as progress:
                        task = progress.add_task("Thinking...", total=None)
                        
                        response = await router.chat(
                            messages=messages,
                            provider=chat_provider,
                            model=chat_model,
                        )
                    
                    # Add assistant response
                    messages.append(ChatMessage(role="assistant", content=response.content))
                    
                    # Display response
                    console.print(f"\n[bold blue]{response.provider.title()}[/bold blue]: {response.content}\n")
                    
                except Exception as e:
                    print_error(str(e))
                    # Remove the last user message if there was an error
                    if messages and messages[-1].role == "user":
                        messages.pop()
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Chat session interrupted.[/yellow]")
        except Exception as e:
            print_error(str(e))
            raise typer.Exit(1)
    
    asyncio.run(_chat())


@app.command()
def debate(
    prompt: str = typer.Argument(..., help="Debate topic or question"),
    models: List[str] = typer.Option(
        ["openai:gpt-4", "claude:claude-3-sonnet-20240229"],
        "--models", "-m",
        help="Models to participate in debate (format: provider:model)"
    ),
    rounds: int = typer.Option(3, "--rounds", "-r", help="Number of debate rounds"),
):
    """Run a debate between multiple AI models."""
    
    async def _debate():
        try:
            # Parse model configurations
            model_configs = []
            for model_str in models:
                if ":" in model_str:
                    provider, model = model_str.split(":", 1)
                    model_configs.append({"provider": provider, "model": model})
                else:
                    print_error(f"Invalid model format: {model_str}. Use provider:model format.")
                    raise typer.Exit(1)
            
            if len(model_configs) < 2:
                print_error("At least 2 models are required for a debate.")
                raise typer.Exit(1)
            
            console.print(f"[bold blue]Starting debate with {len(model_configs)} models for {rounds} rounds[/bold blue]\n")
            console.print(Panel(f"[bold]Topic:[/bold] {prompt}", border_style="blue"))
            console.print("\n")
            
            responses = await router.debate(
                prompt=prompt,
                models=model_configs,
                rounds=rounds,
            )
            
            # Display debate results
            for i, response in enumerate(responses):
                round_num = (i // len(model_configs)) + 1
                model_name = f"{response.provider.title()} ({response.model})"
                
                console.print(Panel(
                    Markdown(response.content),
                    title=f"[bold]Round {round_num} - {model_name}[/bold]",
                    border_style="green" if round_num % 2 == 1 else "yellow"
                ))
                console.print("\n")
                
        except Exception as e:
            print_error(str(e))
            raise typer.Exit(1)
    
    asyncio.run(_debate())


@app.command()
def list_models(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Show models for specific provider")
):
    """List available models."""
    if provider:
        if provider not in model_registry.list_models():
            print_error(f"Unknown provider: {provider}")
            raise typer.Exit(1)
        
        try:
            # Create a dummy instance to get available models
            model_class = model_registry.get(provider)
            if model_class:
                dummy_instance = model_class("dummy")
                models = dummy_instance.get_available_models()
                
                table = Table(title=f"Available Models for {provider.title()}")
                table.add_column("Model", style="cyan")
                
                for model in models:
                    table.add_row(model)
                
                console.print(table)
            else:
                print_error(f"Provider {provider} not found")
                raise typer.Exit(1)
                
        except Exception as e:
            print_error(f"Error listing models for {provider}: {str(e)}")
            raise typer.Exit(1)
    else:
        # List all providers
        table = Table(title="Available Providers")
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="green")
        
        for provider in model_registry.list_models():
            status = "✓" if auth_manager.is_logged_in(provider) else "✗"
            table.add_row(provider.title(), status)
        
        console.print(table)


@app.command()
def config():
    """Show current configuration."""
    config = config_manager.load()
    
    console.print(Panel(
        f"[bold]Default Provider:[/bold] {config.default_provider}\n"
        f"[bold]Default Model:[/bold] {config.default_model}\n"
        f"[bold]Memory Enabled:[/bold] {config.memory_enabled}\n"
        f"[bold]Memory Path:[/bold] {config.memory_path}\n"
        f"[bold]Plugins Path:[/bold] {config.plugins_path}",
        title="[bold]Configuration[/bold]",
        border_style="blue"
    ))


if __name__ == "__main__":
    app() 