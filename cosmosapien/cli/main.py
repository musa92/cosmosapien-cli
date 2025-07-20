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
from ..core.local_manager import LocalModelManager
from ..core.agent_system import AgentSystem, AgentRole
from ..core.smart_router import SmartRouter
from ..core.model_library import ModelLibrary, ModelType, ModelTier
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
local_manager = LocalModelManager()
agent_system = AgentSystem(router, local_manager)
smart_router = SmartRouter(config_manager, local_manager)
model_library = ModelLibrary(config_manager)

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
    
    # Priority order: huggingface (free tier), llama (local)
    for provider_name in ["huggingface", "llama"]:
        for provider in providers:
            if provider["provider"] == provider_name:
                if provider["logged_in"] or provider_name == "llama":  # llama doesn't need login
                    default_models = {
                        "llama": "dolphin-llama3:latest",  # More powerful than codellama
                        "huggingface": "gpt2",
                    }
                    return provider_name, default_models.get(provider_name, "default")
    
    # Fallback to llama (local)
    return "llama", "codellama:latest"


@app.command()
def setup():
    """Set up local environment and install dependencies."""
    async def _setup():
        console.print("[bold blue]Setting up Cosmosapien CLI environment...[/bold blue]\n")
        
        # Check local environment
        status = await local_manager.setup_local_environment()
        
        table = Table(title="Local Environment Status")
        table.add_column("Runner", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Action", style="yellow")
        
        for runner, is_available in status.items():
            status_icon = "✓" if is_available else "✗"
            action = "Ready" if is_available else "Install"
            table.add_row(
                runner.title(),
                status_icon,
                action
            )
        
        console.print(table)
        
        # Show installation help for missing runners
        for runner, is_available in status.items():
            if not is_available:
                console.print(f"\n[bold yellow]Installation help for {runner}:[/bold yellow]")
                help_text = local_manager.get_installation_help(runner)
                console.print(Panel(help_text, border_style="yellow"))
        
        # Show recommended models
        console.print("\n[bold blue]Recommended Models:[/bold blue]")
        recommended = local_manager.get_recommended_models()
        for model in recommended[:5]:  # Show top 5
            console.print(f"• [cyan]{model['name']}[/cyan] - {model['description']}")
            console.print(f"  Size: {model['size']} | Performance: {model['performance']}")
        
        console.print("\n[green]Setup complete! Run 'cosmo agents' to see available AI agents.[/green]")
    
    asyncio.run(_setup())


@app.command()
def agents():
    """Show available AI agents and their capabilities."""
    async def _agents():
        # Create default agents
        await agent_system.create_default_agents()
        
        agents = agent_system.get_available_agents()
        
        if not agents:
            console.print("[yellow]No agents available. Run 'cosmo setup' to configure local models.[/yellow]")
            return
        
        # Separate local and cloud agents
        local_agents = [a for a in agents if a.provider == "llama"]
        cloud_agents = [a for a in agents if a.provider != "llama"]
        
        # Show local agents
        if local_agents:
            console.print("[bold blue]🏠 Local Agents (No API Key Required)[/bold blue]")
            local_table = Table(title="Local AI Agents")
            local_table.add_column("Agent", style="cyan", no_wrap=True)
            local_table.add_column("Role", style="magenta")
            local_table.add_column("Model", style="blue")
            local_table.add_column("Capabilities", style="green")
            
            for agent in local_agents:
                capabilities = ", ".join(agent.capabilities[:3])
                if len(agent.capabilities) > 3:
                    capabilities += "..."
                
                local_table.add_row(
                    agent.name,
                    agent.role.value.title(),
                    agent.model,
                    capabilities
                )
            
            console.print(local_table)
            console.print()
        
        # Show cloud agents
        if cloud_agents:
            console.print("[bold blue]☁️ Cloud Agents (API Key Required)[/bold blue]")
            cloud_table = Table(title="Cloud AI Agents")
            cloud_table.add_column("Agent", style="cyan", no_wrap=True)
            cloud_table.add_column("Role", style="magenta")
            cloud_table.add_column("Provider", style="blue")
            cloud_table.add_column("Model", style="yellow")
            cloud_table.add_column("Capabilities", style="green")
            
            for agent in cloud_agents:
                capabilities = ", ".join(agent.capabilities[:3])
                if len(agent.capabilities) > 3:
                    capabilities += "..."
                
                cloud_table.add_row(
                    agent.name,
                    agent.role.value.title(),
                    agent.provider.title(),
                    agent.model,
                    capabilities
                )
            
            console.print(cloud_table)
            console.print()
        
        # Show total count
        total_agents = len(agents)
        local_count = len(local_agents)
        cloud_count = len(cloud_agents)
        
        console.print(f"[bold]Total Agents: {total_agents}[/bold] ([green]Local: {local_count}[/green], [blue]Cloud: {cloud_count}[/blue])")
        
        # Show login status for cloud providers
        if cloud_count == 0:
            console.print("\n[bold yellow]💡 To add cloud agents, login to providers:[/bold yellow]")
            console.print("• [cyan]cosmo login openai[/cyan] - Add GPT-4, GPT-4-Turbo agents")
            console.print("• [cyan]cosmo login gemini[/cyan] - Add Gemini-Pro, Gemini-Flash agents")
            console.print("• [cyan]cosmo login claude[/cyan] - Add Claude-3-Sonnet, Claude-3-Haiku agents")
            console.print("• [cyan]cosmo login grok[/cyan] - Add Grok agent")
            console.print("• [cyan]cosmo login perplexity[/cyan] - Add Perplexity agents")
            console.print("• [cyan]cosmo login huggingface[/cyan] - Add HuggingFace agents")
        
        console.print("\n[bold]Use these commands to interact with agents:[/bold]")
        console.print("• [cyan]cosmo collaborate <message>[/cyan] - Get responses from multiple agents")
        console.print("• [cyan]cosmo debate <topic>[/cyan] - Run a debate between agents")
        console.print("• [cyan]cosmo solve <problem>[/cyan] - Use agents to solve complex problems")
        console.print("• [cyan]cosmo hybrid <message>[/cyan] - Mix local and cloud agents")
    
    asyncio.run(_agents())


@app.command()
def collaborate(
    message: str = typer.Argument(..., help="Your message to the agents"),
    roles: List[str] = typer.Option(None, "--roles", "-r", help="Agent roles to use (generalist, coder, creative, etc.)")
):
    """Get collaborative responses from multiple AI agents."""
    async def _collaborate():
        # Create default agents
        await agent_system.create_default_agents()
        
        # Convert role strings to AgentRole enums
        agent_roles = None
        if roles:
            agent_roles = []
            for role_str in roles:
                try:
                    agent_roles.append(AgentRole(role_str.lower()))
                except ValueError:
                    console.print(f"[yellow]Unknown role: {role_str}[/yellow]")
        
        try:
            result = await agent_system.collaborative_chat(message, agent_roles)
            
            console.print(f"\n[bold blue]Collaborative Response[/bold blue]")
            console.print(f"[dim]Message: {message}[/dim]\n")
            
            # Show individual agent responses
            for agent_name, response in result["agents"].items():
                if response.get("error"):
                    console.print(Panel(
                        f"[red]Error: {response['content']}[/red]",
                        title=f"❌ {agent_name} ({response['role']})",
                        border_style="red"
                    ))
                else:
                    console.print(Panel(
                        response["content"],
                        title=f"🤖 {agent_name} ({response['role']})",
                        border_style="blue"
                    ))
                console.print()
            
            # Show summary
            if result["summary"]:
                console.print(Panel(
                    result["summary"],
                    title="📋 Summary",
                    border_style="green"
                ))
            
            # Show recommendations
            if result["recommendations"]:
                console.print("\n[bold]Recommendations:[/bold]")
                for rec in result["recommendations"]:
                    console.print(f"• {rec}")
                    
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
    
    asyncio.run(_collaborate())


@app.command()
def solve(
    problem: str = typer.Argument(..., help="Complex problem to solve"),
):
    """Use multiple agents to solve a complex problem."""
    async def _solve():
        # Create default agents
        await agent_system.create_default_agents()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Solving problem with AI agents...", total=None)
                
                result = await agent_system.solve_complex_problem(problem)
            
            console.print(f"\n[bold blue]Problem Solution[/bold blue]")
            console.print(f"[dim]Problem: {problem}[/dim]\n")
            
            # Show component solutions
            for component, solutions in result["solutions"].items():
                console.print(f"[bold]{component.title()}:[/bold]")
                for solution in solutions:
                    if solution.get("error"):
                        console.print(f"  ❌ {solution['agent']}: {solution['solution']}")
                    else:
                        console.print(f"  ✅ {solution['agent']}: {solution['solution'][:100]}...")
                console.print()
            
            # Show final solution
            if result["final_solution"]:
                console.print(Panel(
                    result["final_solution"],
                    title="🎯 Final Solution",
                    border_style="green"
                ))
                    
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
    
    asyncio.run(_solve())


@app.command()
def hybrid(
    message: str = typer.Argument(..., help="Your message to the hybrid agent team"),
    local_agents: int = typer.Option(1, "--local", "-l", help="Number of local agents to use"),
    cloud_agents: int = typer.Option(1, "--cloud", "-c", help="Number of cloud agents to use"),
):
    """Get responses from a mix of local and cloud agents."""
    async def _hybrid():
        # Create default agents
        await agent_system.create_default_agents()
        
        agents = agent_system.get_available_agents()
        
        if not agents:
            console.print("[yellow]No agents available. Run 'cosmo setup' to configure local models.[/yellow]")
            return
        
        # Separate local and cloud agents
        local_agent_list = [a for a in agents if a.provider == "llama"]
        cloud_agent_list = [a for a in agents if a.provider != "llama"]
        
        # Select agents
        selected_agents = []
        
        # Add local agents
        if local_agent_list:
            selected_local = local_agent_list[:min(local_agents, len(local_agent_list))]
            selected_agents.extend(selected_local)
            console.print(f"[green]Selected {len(selected_local)} local agent(s): {', '.join([a.name for a in selected_local])}[/green]")
        
        # Add cloud agents
        if cloud_agent_list:
            selected_cloud = cloud_agent_list[:min(cloud_agents, len(cloud_agent_list))]
            selected_agents.extend(selected_cloud)
            console.print(f"[blue]Selected {len(selected_cloud)} cloud agent(s): {', '.join([a.name for a in selected_cloud])}[/blue]")
        
        if not selected_agents:
            console.print("[red]No agents selected. Check your login status for cloud providers.[/red]")
            return
        
        try:
            console.print(f"\n[bold blue]🤖 Hybrid Agent Response[/bold blue]")
            console.print(f"[dim]Message: {message}[/dim]")
            console.print(f"[dim]Agents: {len(selected_agents)} total ({len([a for a in selected_agents if a.provider == 'llama'])} local, {len([a for a in selected_agents if a.provider != 'llama'])} cloud)[/dim]\n")
            
            # Get responses from each agent
            responses = {}
            for agent in selected_agents:
                try:
                    # Create a simple conversation for this agent
                    messages = [ChatMessage(role="user", content=message)]
                    
                    response = await agent_system.router.chat(
                        messages=messages,
                        provider=agent.provider,
                        model=agent.model
                    )
                    
                    responses[agent.name] = {
                        "content": response.content,
                        "role": agent.role.value,
                        "provider": agent.provider,
                        "model": agent.model,
                        "type": "local" if agent.provider == "llama" else "cloud"
                    }
                    
                except Exception as e:
                    responses[agent.name] = {
                        "content": f"Error: {str(e)}",
                        "role": agent.role.value,
                        "provider": agent.provider,
                        "model": agent.model,
                        "type": "local" if agent.provider == "llama" else "cloud",
                        "error": True
                    }
            
            # Show responses grouped by type
            if any(r["type"] == "local" for r in responses.values()):
                console.print("[bold green]🏠 Local Agents:[/bold green]")
                for agent_name, response in responses.items():
                    if response["type"] == "local":
                        if response.get("error"):
                            console.print(Panel(
                                f"[red]Error: {response['content']}[/red]",
                                title=f"❌ {agent_name} ({response['role']})",
                                border_style="red"
                            ))
                        else:
                            console.print(Panel(
                                response["content"],
                                title=f"🤖 {agent_name} ({response['role']})",
                                border_style="green"
                            ))
                console.print()
            
            if any(r["type"] == "cloud" for r in responses.values()):
                console.print("[bold blue]☁️ Cloud Agents:[/bold blue]")
                for agent_name, response in responses.items():
                    if response["type"] == "cloud":
                        if response.get("error"):
                            console.print(Panel(
                                f"[red]Error: {response['content']}[/red]",
                                title=f"❌ {agent_name} ({response['role']})",
                                border_style="red"
                            ))
                        else:
                            console.print(Panel(
                                response["content"],
                                title=f"☁️ {agent_name} ({response['role']})",
                                border_style="blue"
                            ))
                console.print()
            
            # Show comparison
            if len(responses) > 1:
                console.print("[bold yellow]📊 Agent Comparison:[/bold yellow]")
                for agent_name, response in responses.items():
                    if not response.get("error"):
                        console.print(f"• [cyan]{agent_name}[/cyan] ({response['type']}): {response['content'][:100]}...")
                    else:
                        console.print(f"• [red]{agent_name}[/red] ({response['type']}): Error")
                        
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
    
    asyncio.run(_hybrid())


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
    smart_route: bool = typer.Option(False, "--smart-route", "--squeeze", help="Use smart routing for cost efficiency"),
    explain_route: bool = typer.Option(False, "--explain-route", help="Show routing decision without making the call"),
):
    """Ask a question to any supported LLM provider."""
    
    async def _ask():
        try:
            # Use smart routing if requested
            if smart_route or explain_route:
                decision = smart_router.smart_route(prompt, explain_only=explain_route)
                
                # Show routing decision
                console.print(f"\n[bold blue]🧠 Smart Routing Decision[/bold blue]")
                console.print(f"[dim]Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}[/dim]")
                console.print(f"[bold]Complexity:[/bold] {decision.complexity.value.title()}")
                console.print(f"[bold]Selected:[/bold] {decision.selected_provider}/{decision.selected_model}")
                console.print(f"[bold]Reasoning:[/bold] {decision.reasoning}")
                console.print(f"[bold]Estimated Cost:[/bold] ${decision.estimated_cost:.4f}")
                
                if decision.alternatives:
                    console.print(f"\n[bold]Alternatives:[/bold]")
                    for alt_provider, alt_model, alt_reason in decision.alternatives[:3]:
                        console.print(f"• {alt_provider}/{alt_model} - {alt_reason}")
                
                if explain_route:
                    console.print(f"\n[green]Routing explanation complete. Use --smart-route to execute.[/green]")
                    return
                
                if decision.selected_provider == "none":
                    console.print(f"[red]No suitable provider found. Please login to providers or check local models.[/red]")
                    return
                
                # Use the selected provider and model
                selected_provider = decision.selected_provider
                selected_model = decision.selected_model
            else:
                # Use provided provider/model or defaults
                selected_provider = provider or get_default_open_source_provider()[0]
                selected_model = model or get_default_open_source_provider()[1]
            
            # Generate response
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Generating response...", total=None)
                
                response = await router.generate(
                    prompt=prompt,
                    provider=selected_provider,
                    model=selected_model,
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
            
            # Show usage info if smart routing was used
            if smart_route:
                usage = smart_router.get_usage_summary()
                if usage["total_calls"] > 0:
                    console.print(f"\n[dim]Usage: {usage['total_calls']} total calls, ${usage['estimated_cost']:.4f} estimated cost[/dim]")
            elif response.usage:
                usage_table = Table(title="Usage Information")
                usage_table.add_column("Metric", style="cyan")
                usage_table.add_column("Value", style="green")
                
                for key, value in response.usage.items():
                    usage_table.add_row(key.replace("_", " ").title(), str(value))
                
                console.print(usage_table)
                
        except Exception as e:
            print_error(str(e))
            if smart_route:
                console.print(f"[yellow]Smart routing failed. Try using a specific provider with --provider.[/yellow]")
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
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Show models for specific provider"),
    use_library: bool = typer.Option(True, "--library", help="Use model library instead of provider API"),
):
    """List available models."""
    if use_library:
        # Use model library
        models = router.list_library_models(provider=provider, active_only=True)
        
        if not models:
            console.print("[yellow]No models found in the library.[/yellow]")
            return
        
        table = Table(title="Model Library")
        table.add_column("Provider", style="cyan")
        table.add_column("Model ID", style="blue")
        table.add_column("Display Name", style="green")
        table.add_column("Tier", style="yellow")
        table.add_column("Type", style="magenta")
        
        for model in models:
            table.add_row(
                model.provider.title(),
                model.model_id,
                model.display_name,
                model.tier.value.title(),
                model.model_type.value.title()
            )
        
        console.print(table)
    else:
        # Use provider API (legacy method)
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
def usage():
    """Show usage statistics and cost tracking."""
    usage_summary = smart_router.get_usage_summary()
    
    console.print("[bold blue]Usage Statistics[/bold blue]\n")
    
    if not usage_summary["providers"]:
        console.print("[yellow]No usage data available. Start using smart routing with --smart-route flag.[/yellow]")
        return
    
    # Show summary
    console.print(f"[bold]Total Calls:[/bold] {usage_summary['total_calls']}")
    console.print(f"[bold]Estimated Cost:[/bold] ${usage_summary['estimated_cost']:.4f}")
    console.print()
    
    # Show provider details
    table = Table(title="Provider Usage")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="blue")
    table.add_column("Used Calls", style="green")
    table.add_column("Remaining", style="yellow")
    table.add_column("Cost", style="red")
    table.add_column("Type", style="magenta")
    
    for key, data in usage_summary["providers"].items():
        remaining = "∞" if data["is_local"] else str(data["remaining_calls"])
        cost = "$0.00" if data["is_local"] else f"${data['cost']:.4f}"
        provider_type = "Local" if data["is_local"] else "Cloud"
        
        table.add_row(
            data["provider"].title(),
            data["model"],
            str(data["used_calls"]),
            remaining,
            cost,
            provider_type
        )
    
    console.print(table)
    
    # Show recommendations
    console.print(f"\n[bold]💡 Recommendations:[/bold]")
    if usage_summary["estimated_cost"] > 0.01:
        console.print("• Consider using local models for simple tasks to reduce costs")
        console.print("• Use --smart-route flag for automatic cost optimization")
    
    local_usage = sum(1 for data in usage_summary["providers"].values() if data["is_local"])
    if local_usage == 0:
        console.print("• Try local models with Ollama for free, unlimited usage")


@app.command()
def reset_usage(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Reset usage for specific provider only")
):
    """Reset usage statistics."""
    if provider:
        smart_router.reset_usage(provider)
        console.print(f"[green]Reset usage for {provider}[/green]")
    else:
        smart_router.reset_usage()
        console.print("[green]Reset all usage statistics[/green]")


@app.command()
def smart_route(
    prompt: str = typer.Argument(..., help="Your question or prompt"),
    explain_only: bool = typer.Option(False, "--explain", "-e", help="Show routing decision without executing"),
):
    """Smart route a prompt to the most cost-efficient model."""
    decision = smart_router.smart_route(prompt, explain_only=explain_only)
    
    console.print(f"\n[bold blue]🧠 Smart Routing Decision[/bold blue]")
    console.print(f"[dim]Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}[/dim]")
    console.print(f"[bold]Complexity:[/bold] {decision.complexity.value.title()}")
    console.print(f"[bold]Selected:[/bold] {decision.selected_provider}/{decision.selected_model}")
    console.print(f"[bold]Reasoning:[/bold] {decision.reasoning}")
    console.print(f"[bold]Estimated Cost:[/bold] ${decision.estimated_cost:.4f}")
    
    if decision.alternatives:
        console.print(f"\n[bold]Alternatives:[/bold]")
        for alt_provider, alt_model, alt_reason in decision.alternatives[:3]:
            console.print(f"• {alt_provider}/{alt_model} - {alt_reason}")
    
    if explain_only:
        console.print(f"\n[green]Routing explanation complete. Use 'cosmo ask --smart-route' to execute.[/green]")
    elif decision.selected_provider == "none":
        console.print(f"\n[red]No suitable provider found. Please login to providers or check local models.[/red]")
    else:
        console.print(f"\n[green]Use 'cosmo ask --smart-route \"{prompt}\"' to execute this routing decision.[/green]")


@app.command()
def configure_smart_routing(
    provider: str = typer.Argument(..., help="Provider name"),
    model: str = typer.Argument(..., help="Model name"),
    free_tier_limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Free tier call limit"),
    cost_per_call: Optional[float] = typer.Option(None, "--cost", "-c", help="Cost per call in USD"),
):
    """Configure smart routing settings for a specific provider/model."""
    try:
        if free_tier_limit is not None:
            config_manager.set_free_tier_limit(provider, model, free_tier_limit)
            console.print(f"[green]Set free tier limit for {provider}/{model}: {free_tier_limit} calls[/green]")
        
        if cost_per_call is not None:
            config_manager.set_custom_cost(provider, model, cost_per_call)
            console.print(f"[green]Set cost for {provider}/{model}: ${cost_per_call:.4f} per call[/green]")
        
        if free_tier_limit is None and cost_per_call is None:
            console.print("[yellow]Please specify --limit or --cost to configure smart routing.[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error configuring smart routing: {str(e)}[/red]")


@app.command()
def models(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Filter by provider"),
    tier: Optional[str] = typer.Option(None, "--tier", "-t", help="Filter by tier (free, basic, standard, premium)"),
    type: Optional[str] = typer.Option(None, "--type", help="Filter by model type (chat, completion, embedding)"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Filter by tag"),
    active_only: bool = typer.Option(True, "--all", help="Show all models including inactive"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, csv)"),
):
    """List and manage models in the model library."""
    try:
        # Parse filters
        tier_enum = None
        if tier:
            try:
                tier_enum = ModelTier(tier.lower())
            except ValueError:
                console.print(f"[red]Invalid tier: {tier}. Valid options: free, basic, standard, premium, enterprise[/red]")
                return
        
        type_enum = None
        if type:
            try:
                type_enum = ModelType(type.lower())
            except ValueError:
                console.print(f"[red]Invalid type: {type}. Valid options: chat, completion, embedding, vision, audio, multimodal[/red]")
                return
        
        # Get filtered models
        models = model_library.list_models(
            provider=provider,
            tier=tier_enum,
            model_type=type_enum,
            active_only=active_only
        )
        
        if tag:
            models = [m for m in models if tag in m.tags]
        
        if not models:
            console.print("[yellow]No models found matching the specified criteria.[/yellow]")
            return
        
        # Display based on format
        if format == "json":
            import json
            data = [model.to_dict() for model in models]
            console.print(json.dumps(data, indent=2))
        elif format == "csv":
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Provider", "Model ID", "Display Name", "Tier", "Type", "Tags", "Active"])
            for model in models:
                writer.writerow([
                    model.provider,
                    model.model_id,
                    model.display_name,
                    model.tier.value,
                    model.model_type.value,
                    ", ".join(model.tags),
                    model.is_active
                ])
            console.print(output.getvalue())
        else:  # table format
            table = Table(title="Model Library")
            table.add_column("Provider", style="cyan")
            table.add_column("Model ID", style="blue")
            table.add_column("Display Name", style="green")
            table.add_column("Tier", style="yellow")
            table.add_column("Type", style="magenta")
            table.add_column("Tags", style="white")
            table.add_column("Active", style="red")
            
            for model in models:
                table.add_row(
                    model.provider.title(),
                    model.model_id,
                    model.display_name,
                    model.tier.value.title(),
                    model.model_type.value.title(),
                    ", ".join(model.tags[:3]) + ("..." if len(model.tags) > 3 else ""),
                    "Yes" if model.is_active else "No"
                )
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[red]Error listing models: {str(e)}[/red]")


@app.command()
def model_info(
    model_id: str = typer.Argument(..., help="Model ID (format: provider:model_id)"),
):
    """Show detailed information about a specific model."""
    try:
        model = model_library.get_model(model_id)
        if not model:
            console.print(f"[red]Model not found: {model_id}[/red]")
            return
        
        # Display model information
        console.print(Panel(
            f"[bold]Provider:[/bold] {model.provider.title()}\n"
            f"[bold]Model ID:[/bold] {model.model_id}\n"
            f"[bold]Display Name:[/bold] {model.display_name}\n"
            f"[bold]Description:[/bold] {model.description}\n"
            f"[bold]Type:[/bold] {model.model_type.value.title()}\n"
            f"[bold]Tier:[/bold] {model.tier.value.title()}\n"
            f"[bold]Active:[/bold] {'Yes' if model.is_active else 'No'}\n"
            f"[bold]Local:[/bold] {'Yes' if model.is_local else 'No'}\n"
            f"[bold]Tags:[/bold] {', '.join(model.tags)}",
            title=f"[bold]Model Information[/bold]",
            border_style="blue"
        ))
        
        # Capabilities
        console.print(Panel(
            f"[bold]Max Tokens:[/bold] {model.capabilities.max_tokens or 'Unlimited'}\n"
            f"[bold]Max Input Tokens:[/bold] {model.capabilities.max_input_tokens or 'Unlimited'}\n"
            f"[bold]Context Window:[/bold] {model.capabilities.context_window or 'Unknown'}\n"
            f"[bold]Training Data Cutoff:[/bold] {model.capabilities.training_data_cutoff or 'Unknown'}\n"
            f"[bold]Supports Streaming:[/bold] {'Yes' if model.capabilities.supports_streaming else 'No'}\n"
            f"[bold]Supports Function Calling:[/bold] {'Yes' if model.capabilities.supports_function_calling else 'No'}\n"
            f"[bold]Supports Vision:[/bold] {'Yes' if model.capabilities.supports_vision else 'No'}\n"
            f"[bold]Supports Audio:[/bold] {'Yes' if model.capabilities.supports_audio else 'No'}\n"
            f"[bold]Supports Embeddings:[/bold] {'Yes' if model.capabilities.supports_embeddings else 'No'}",
            title="[bold]Capabilities[/bold]",
            border_style="green"
        ))
        
        # Pricing
        console.print(Panel(
            f"[bold]Input Cost (per 1K tokens):[/bold] ${model.pricing.input_cost_per_1k_tokens:.4f}\n"
            f"[bold]Output Cost (per 1K tokens):[/bold] ${model.pricing.output_cost_per_1k_tokens:.4f}\n"
            f"[bold]Free Tier Limit:[/bold] {model.pricing.free_tier_limit if model.pricing.free_tier_limit != float('inf') else 'Unlimited'}\n"
            f"[bold]Free Tier Reset:[/bold] {model.pricing.free_tier_reset_period.title()}\n"
            f"[bold]Currency:[/bold] {model.pricing.currency}",
            title="[bold]Pricing[/bold]",
            border_style="yellow"
        ))
        
        # Metadata
        console.print(Panel(
            f"[bold]Created:[/bold] {model.created_at}\n"
            f"[bold]Updated:[/bold] {model.updated_at}",
            title="[bold]Metadata[/bold]",
            border_style="magenta"
        ))
        
    except Exception as e:
        console.print(f"[red]Error showing model info: {str(e)}[/red]")


@app.command()
def search_models(
    query: str = typer.Argument(..., help="Search query"),
):
    """Search models by name, description, or tags."""
    try:
        results = model_library.search_models(query)
        
        if not results:
            console.print(f"[yellow]No models found matching '{query}'[/yellow]")
            return
        
        console.print(f"[bold]Search Results for '{query}'[/bold]")
        console.print(f"Found {len(results)} model(s)\n")
        
        table = Table(title="Search Results")
        table.add_column("Provider", style="cyan")
        table.add_column("Model ID", style="blue")
        table.add_column("Display Name", style="green")
        table.add_column("Description", style="white")
        table.add_column("Tags", style="yellow")
        
        for model in results:
            table.add_row(
                model.provider.title(),
                model.model_id,
                model.display_name,
                model.description[:50] + "..." if len(model.description) > 50 else model.description,
                ", ".join(model.tags[:3]) + ("..." if len(model.tags) > 3 else "")
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error searching models: {str(e)}[/red]")


@app.command()
def model_stats():
    """Show statistics about the model library."""
    try:
        stats = model_library.get_model_statistics()
        
        console.print(Panel(
            f"[bold]Total Models:[/bold] {stats['total_models']}\n"
            f"[bold]Active Models:[/bold] {stats['active_models']}\n"
            f"[bold]Local Models:[/bold] {stats['local_models']}\n"
            f"[bold]Free Models:[/bold] {stats['free_models']}",
            title="[bold]Model Library Statistics[/bold]",
            border_style="blue"
        ))
        
        # Providers breakdown
        if stats['providers']:
            console.print("\n[bold]Models by Provider:[/bold]")
            provider_table = Table()
            provider_table.add_column("Provider", style="cyan")
            provider_table.add_column("Count", style="green")
            
            for provider, count in sorted(stats['providers'].items()):
                provider_table.add_row(provider.title(), str(count))
            
            console.print(provider_table)
        
        # Tiers breakdown
        if stats['tiers']:
            console.print("\n[bold]Models by Tier:[/bold]")
            tier_table = Table()
            tier_table.add_column("Tier", style="yellow")
            tier_table.add_column("Count", style="green")
            
            for tier, count in sorted(stats['tiers'].items()):
                tier_table.add_row(tier.title(), str(count))
            
            console.print(tier_table)
        
        # Types breakdown
        if stats['types']:
            console.print("\n[bold]Models by Type:[/bold]")
            type_table = Table()
            type_table.add_column("Type", style="magenta")
            type_table.add_column("Count", style="green")
            
            for type_name, count in sorted(stats['types'].items()):
                type_table.add_row(type_name.title(), str(count))
            
            console.print(type_table)
        
    except Exception as e:
        console.print(f"[red]Error showing model statistics: {str(e)}[/red]")


@app.command()
def export_models(
    file_path: str = typer.Argument(..., help="Output file path"),
):
    """Export the model library to a file."""
    try:
        if model_library.export_library(file_path):
            console.print(f"[green]Model library exported to {file_path}[/green]")
        else:
            console.print(f"[red]Failed to export model library to {file_path}[/red]")
    except Exception as e:
        console.print(f"[red]Error exporting models: {str(e)}[/red]")


@app.command()
def import_models(
    file_path: str = typer.Argument(..., help="Input file path"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing models"),
):
    """Import models from a file."""
    try:
        if model_library.import_library(file_path, overwrite):
            console.print(f"[green]Models imported from {file_path}[/green]")
        else:
            console.print(f"[red]Failed to import models from {file_path}[/red]")
    except Exception as e:
        console.print(f"[red]Error importing models: {str(e)}[/red]")


@app.command()
def register():
    """Register a new model interactively."""
    try:
        console.print("[bold blue]Model Registration Wizard[/bold blue]")
        console.print("This wizard will help you register a new model in the library.\n")
        
        # Basic Information
        console.print("[bold]Basic Information[/bold]")
        provider = Prompt.ask("Provider name", default="custom")
        model_id = Prompt.ask("Model ID")
        display_name = Prompt.ask("Display name", default=model_id)
        description = Prompt.ask("Description", default="Custom model")
        
        # Model Type
        console.print(f"\n[bold]Model Type[/bold]")
        console.print("Available types: chat, completion, embedding, vision, audio, multimodal")
        model_type_input = Prompt.ask("Model type", default="chat")
        try:
            model_type = ModelType(model_type_input.lower())
        except ValueError:
            console.print(f"[red]Invalid model type: {model_type_input}[/red]")
            return
        
        # Model Tier
        console.print(f"\n[bold]Model Tier[/bold]")
        console.print("Available tiers: free, basic, standard, premium, enterprise")
        tier_input = Prompt.ask("Model tier", default="standard")
        try:
            tier = ModelTier(tier_input.lower())
        except ValueError:
            console.print(f"[red]Invalid model tier: {tier_input}[/red]")
            return
        
        # Capabilities
        console.print(f"\n[bold]Model Capabilities[/bold]")
        max_tokens = Prompt.ask("Max output tokens", default="4096")
        max_input_tokens = Prompt.ask("Max input tokens", default="8192")
        context_window = Prompt.ask("Context window size", default="8192")
        supports_streaming = Confirm.ask("Supports streaming?", default=True)
        supports_function_calling = Confirm.ask("Supports function calling?", default=False)
        supports_vision = Confirm.ask("Supports vision?", default=False)
        supports_audio = Confirm.ask("Supports audio?", default=False)
        supports_embeddings = Confirm.ask("Supports embeddings?", default=False)
        training_data_cutoff = Prompt.ask("Training data cutoff (YYYY-MM)", default="2023-12")
        
        # Pricing
        console.print(f"\n[bold]Pricing Information[/bold]")
        input_cost = Prompt.ask("Input cost per 1K tokens ($)", default="0.0")
        output_cost = Prompt.ask("Output cost per 1K tokens ($)", default="0.0")
        free_tier_limit = Prompt.ask("Free tier limit (calls per month)", default="0")
        free_tier_reset = Prompt.ask("Free tier reset period", choices=["daily", "weekly", "monthly"], default="monthly")
        
        # Additional Information
        console.print(f"\n[bold]Additional Information[/bold]")
        is_local = Confirm.ask("Is this a local model?", default=False)
        is_active = Confirm.ask("Is this model active?", default=True)
        
        # Tags
        console.print(f"\n[bold]Tags[/bold]")
        console.print("Enter tags separated by commas (e.g., coding, fast, reliable)")
        tags_input = Prompt.ask("Tags", default="custom")
        tags = [tag.strip() for tag in tags_input.split(",")]
        
        # Create model configuration
        from cosmosapien.core.model_library import ModelConfig, ModelCapability, ModelPricing
        
        model_config = ModelConfig(
            name=f"{provider}-{model_id}",
            provider=provider,
            model_id=model_id,
            display_name=display_name,
            description=description,
            model_type=model_type,
            tier=tier,
            capabilities=ModelCapability(
                max_tokens=int(max_tokens) if max_tokens.isdigit() else None,
                max_input_tokens=int(max_input_tokens) if max_input_tokens.isdigit() else None,
                supports_streaming=supports_streaming,
                supports_function_calling=supports_function_calling,
                supports_vision=supports_vision,
                supports_audio=supports_audio,
                supports_embeddings=supports_embeddings,
                context_window=int(context_window) if context_window.isdigit() else None,
                training_data_cutoff=training_data_cutoff if training_data_cutoff != "2023-12" else None
            ),
            pricing=ModelPricing(
                input_cost_per_1k_tokens=float(input_cost),
                output_cost_per_1k_tokens=float(output_cost),
                free_tier_limit=int(free_tier_limit) if free_tier_limit.isdigit() else 0,
                free_tier_reset_period=free_tier_reset
            ),
            tags=tags,
            is_active=is_active,
            is_local=is_local
        )
        
        # Show summary
        console.print(f"\n[bold]Model Configuration Summary[/bold]")
        console.print(Panel(
            f"[bold]Provider:[/bold] {model_config.provider}\n"
            f"[bold]Model ID:[/bold] {model_config.model_id}\n"
            f"[bold]Display Name:[/bold] {model_config.display_name}\n"
            f"[bold]Description:[/bold] {model_config.description}\n"
            f"[bold]Type:[/bold] {model_config.model_type.value}\n"
            f"[bold]Tier:[/bold] {model_config.tier.value}\n"
            f"[bold]Local:[/bold] {'Yes' if model_config.is_local else 'No'}\n"
            f"[bold]Active:[/bold] {'Yes' if model_config.is_active else 'No'}\n"
            f"[bold]Tags:[/bold] {', '.join(model_config.tags)}",
            title="[bold]Configuration Summary[/bold]",
            border_style="blue"
        ))
        
        # Confirm registration
        if Confirm.ask("Register this model?"):
            model_id_full = f"{provider}:{model_id}"
            if model_library.add_model(model_config):
                console.print(f"[green]Model '{model_id_full}' registered successfully![/green]")
                console.print(f"[dim]You can now use this model with: cosmo ask --provider {provider} --model {model_id}[/dim]")
            else:
                console.print(f"[red]Failed to register model. It may already exist.[/red]")
        else:
            console.print("[yellow]Model registration cancelled.[/yellow]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Model registration cancelled.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during registration: {str(e)}[/red]")


@app.command()
def register_quick(
    provider: str = typer.Argument(..., help="Provider name"),
    model_id: str = typer.Argument(..., help="Model ID"),
    display_name: str = typer.Option(None, "--name", help="Display name"),
    description: str = typer.Option(None, "--desc", help="Model description"),
    tier: str = typer.Option("standard", "--tier", help="Model tier (free, basic, standard, premium, enterprise)"),
    model_type: str = typer.Option("chat", "--type", help="Model type (chat, completion, embedding, vision, audio, multimodal)"),
    max_tokens: int = typer.Option(4096, "--max-tokens", help="Maximum output tokens"),
    context_window: int = typer.Option(8192, "--context", help="Context window size"),
    input_cost: float = typer.Option(0.0, "--input-cost", help="Input cost per 1K tokens"),
    output_cost: float = typer.Option(0.0, "--output-cost", help="Output cost per 1K tokens"),
    free_limit: int = typer.Option(0, "--free-limit", help="Free tier limit"),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags"),
    is_local: bool = typer.Option(False, "--local", help="Is local model"),
):
    """Quickly register a new model with minimal input."""
    try:
        # Parse inputs
        try:
            model_type_enum = ModelType(model_type.lower())
            tier_enum = ModelTier(tier.lower())
        except ValueError as e:
            console.print(f"[red]Invalid model type or tier: {str(e)}[/red]")
            return
        
        # Set defaults
        display_name = display_name or model_id
        description = description or f"{display_name} model"
        tags_list = [tag.strip() for tag in tags.split(",")] if tags else ["custom"]
        
        # Create model configuration
        from cosmosapien.core.model_library import ModelConfig, ModelCapability, ModelPricing
        
        model_config = ModelConfig(
            name=f"{provider}-{model_id}",
            provider=provider,
            model_id=model_id,
            display_name=display_name,
            description=description,
            model_type=model_type_enum,
            tier=tier_enum,
            capabilities=ModelCapability(
                max_tokens=max_tokens,
                max_input_tokens=context_window,
                supports_streaming=True,
                supports_function_calling=False,
                supports_vision=False,
                supports_audio=False,
                supports_embeddings=False,
                context_window=context_window
            ),
            pricing=ModelPricing(
                input_cost_per_1k_tokens=input_cost,
                output_cost_per_1k_tokens=output_cost,
                free_tier_limit=free_limit
            ),
            tags=tags_list,
            is_active=True,
            is_local=is_local
        )
        
        # Register model
        model_id_full = f"{provider}:{model_id}"
        if model_library.add_model(model_config):
            console.print(f"[green]Model '{model_id_full}' registered successfully![/green]")
            console.print(f"[dim]Use: cosmo ask --provider {provider} --model {model_id}[/dim]")
        else:
            console.print(f"[red]Failed to register model. It may already exist.[/red]")
            
    except Exception as e:
        console.print(f"[red]Error during registration: {str(e)}[/red]")


@app.command()
def unregister(
    model_id: str = typer.Argument(..., help="Model ID (format: provider:model_id)"),
    confirm: bool = typer.Option(False, "--confirm", help="Skip confirmation prompt"),
):
    """Unregister a model from the library."""
    try:
        if not confirm:
            if not Confirm.ask(f"Are you sure you want to unregister '{model_id}'?"):
                console.print("[yellow]Unregistration cancelled.[/yellow]")
                return
        
        if model_library.remove_model(model_id):
            console.print(f"[green]Model '{model_id}' unregistered successfully![/green]")
        else:
            console.print(f"[red]Model '{model_id}' not found in library.[/red]")
            
    except Exception as e:
        console.print(f"[red]Error during unregistration: {str(e)}[/red]")


@app.command()
def register_template(
    template: str = typer.Argument(..., help="Template name (gpt4, claude, gemini, local, custom)"),
    provider: str = typer.Option(None, "--provider", help="Override provider name"),
    model_id: str = typer.Option(None, "--model-id", help="Override model ID"),
    display_name: str = typer.Option(None, "--name", help="Override display name"),
):
    """Register a model using predefined templates."""
    try:
        from cosmosapien.core.model_library import ModelConfig, ModelCapability, ModelPricing
        
        # Define templates
        templates = {
            "gpt4": {
                "provider": "openai",
                "model_id": "gpt-4",
                "display_name": "GPT-4",
                "description": "Most capable GPT model for complex reasoning tasks",
                "model_type": ModelType.CHAT,
                "tier": ModelTier.PREMIUM,
                "capabilities": ModelCapability(
                    max_tokens=8192,
                    max_input_tokens=8192,
                    supports_streaming=True,
                    supports_function_calling=True,
                    context_window=8192,
                    training_data_cutoff="2023-04"
                ),
                "pricing": ModelPricing(
                    input_cost_per_1k_tokens=0.03,
                    output_cost_per_1k_tokens=0.06,
                    free_tier_limit=0
                ),
                "tags": ["reasoning", "complex-tasks", "function-calling"]
            },
            "gpt35": {
                "provider": "openai",
                "model_id": "gpt-3.5-turbo",
                "display_name": "GPT-3.5 Turbo",
                "description": "Fast and efficient model for most tasks",
                "model_type": ModelType.CHAT,
                "tier": ModelTier.BASIC,
                "capabilities": ModelCapability(
                    max_tokens=4096,
                    max_input_tokens=4096,
                    supports_streaming=True,
                    supports_function_calling=True,
                    context_window=4096,
                    training_data_cutoff="2021-09"
                ),
                "pricing": ModelPricing(
                    input_cost_per_1k_tokens=0.0015,
                    output_cost_per_1k_tokens=0.002,
                    free_tier_limit=3
                ),
                "tags": ["fast", "efficient", "general-purpose"]
            },
            "claude": {
                "provider": "claude",
                "model_id": "claude-3-sonnet-20240229",
                "display_name": "Claude 3 Sonnet",
                "description": "Balanced Claude model for most tasks",
                "model_type": ModelType.CHAT,
                "tier": ModelTier.STANDARD,
                "capabilities": ModelCapability(
                    max_tokens=4096,
                    max_input_tokens=200000,
                    supports_streaming=True,
                    supports_function_calling=True,
                    context_window=200000,
                    training_data_cutoff="2023-08"
                ),
                "pricing": ModelPricing(
                    input_cost_per_1k_tokens=0.003,
                    output_cost_per_1k_tokens=0.015,
                    free_tier_limit=0
                ),
                "tags": ["balanced", "analysis", "long-context"]
            },
            "gemini": {
                "provider": "gemini",
                "model_id": "gemini-pro",
                "display_name": "Gemini Pro",
                "description": "Google's advanced language model",
                "model_type": ModelType.CHAT,
                "tier": ModelTier.STANDARD,
                "capabilities": ModelCapability(
                    max_tokens=2048,
                    max_input_tokens=30720,
                    supports_streaming=True,
                    supports_function_calling=False,
                    context_window=30720,
                    training_data_cutoff="2023-02"
                ),
                "pricing": ModelPricing(
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                    free_tier_limit=15
                ),
                "tags": ["google", "free-tier", "general-purpose"]
            },
            "local": {
                "provider": "llama",
                "model_id": "llama3.2:8b",
                "display_name": "Llama 3.2 8B",
                "description": "Local Llama 3.2 model (8B parameters)",
                "model_type": ModelType.CHAT,
                "tier": ModelTier.FREE,
                "capabilities": ModelCapability(
                    max_tokens=4096,
                    max_input_tokens=8192,
                    supports_streaming=True,
                    supports_function_calling=False,
                    context_window=8192,
                    training_data_cutoff="2023-12"
                ),
                "pricing": ModelPricing(
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                    free_tier_limit=float('inf')
                ),
                "tags": ["local", "free", "llama", "8b"],
                "is_local": True
            },
            "custom": {
                "provider": "custom",
                "model_id": "custom-v1",
                "display_name": "Custom Model",
                "description": "Custom model for specific use cases",
                "model_type": ModelType.CHAT,
                "tier": ModelTier.STANDARD,
                "capabilities": ModelCapability(
                    max_tokens=2048,
                    max_input_tokens=4096,
                    supports_streaming=True,
                    context_window=4096
                ),
                "pricing": ModelPricing(
                    input_cost_per_1k_tokens=0.005,
                    output_cost_per_1k_tokens=0.01,
                    free_tier_limit=10
                ),
                "tags": ["custom", "specialized"]
            }
        }
        
        if template.lower() not in templates:
            console.print(f"[red]Unknown template: {template}[/red]")
            console.print(f"Available templates: {', '.join(templates.keys())}")
            return
        
        # Get template
        template_config = templates[template.lower()]
        
        # Override with user-provided values
        if provider:
            template_config["provider"] = provider
        if model_id:
            template_config["model_id"] = model_id
        if display_name:
            template_config["display_name"] = display_name
        
        # Create model configuration
        model_config = ModelConfig(
            name=f"{template_config['provider']}-{template_config['model_id']}",
            provider=template_config["provider"],
            model_id=template_config["model_id"],
            display_name=template_config["display_name"],
            description=template_config["description"],
            model_type=template_config["model_type"],
            tier=template_config["tier"],
            capabilities=template_config["capabilities"],
            pricing=template_config["pricing"],
            tags=template_config["tags"],
            is_active=True,
            is_local=template_config.get("is_local", False)
        )
        
        # Show summary
        console.print(f"[bold]Registering model from template: {template}[/bold]")
        console.print(Panel(
            f"[bold]Provider:[/bold] {model_config.provider}\n"
            f"[bold]Model ID:[/bold] {model_config.model_id}\n"
            f"[bold]Display Name:[/bold] {model_config.display_name}\n"
            f"[bold]Description:[/bold] {model_config.description}\n"
            f"[bold]Type:[/bold] {model_config.model_type.value}\n"
            f"[bold]Tier:[/bold] {model_config.tier.value}\n"
            f"[bold]Tags:[/bold] {', '.join(model_config.tags)}",
            title="[bold]Template Configuration[/bold]",
            border_style="green"
        ))
        
        # Register model
        model_id_full = f"{model_config.provider}:{model_config.model_id}"
        if model_library.add_model(model_config):
            console.print(f"[green]Model '{model_id_full}' registered successfully from template![/green]")
            console.print(f"[dim]Use: cosmo ask --provider {model_config.provider} --model {model_config.model_id}[/dim]")
        else:
            console.print(f"[red]Failed to register model. It may already exist.[/red]")
            
    except Exception as e:
        console.print(f"[red]Error during template registration: {str(e)}[/red]")


@app.command()
def register_from_file(
    file_path: str = typer.Argument(..., help="JSON file path containing model configuration"),
    provider: str = typer.Option(None, "--provider", help="Override provider name"),
    model_id: str = typer.Option(None, "--model-id", help="Override model ID"),
):
    """Register a model from a JSON configuration file."""
    try:
        import json
        
        # Read configuration file
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        # Override with user-provided values
        if provider:
            config_data["provider"] = provider
        if model_id:
            config_data["model_id"] = model_id
        
        # Create model configuration
        from cosmosapien.core.model_library import ModelConfig
        
        model_config = ModelConfig.from_dict(config_data)
        
        # Show summary
        console.print(f"[bold]Registering model from file: {file_path}[/bold]")
        console.print(Panel(
            f"[bold]Provider:[/bold] {model_config.provider}\n"
            f"[bold]Model ID:[/bold] {model_config.model_id}\n"
            f"[bold]Display Name:[/bold] {model_config.display_name}\n"
            f"[bold]Description:[/bold] {model_config.description}\n"
            f"[bold]Type:[/bold] {model_config.model_type.value}\n"
            f"[bold]Tier:[/bold] {model_config.tier.value}\n"
            f"[bold]Tags:[/bold] {', '.join(model_config.tags)}",
            title="[bold]File Configuration[/bold]",
            border_style="yellow"
        ))
        
        # Register model
        model_id_full = f"{model_config.provider}:{model_config.model_id}"
        if model_library.add_model(model_config):
            console.print(f"[green]Model '{model_id_full}' registered successfully from file![/green]")
            console.print(f"[dim]Use: cosmo ask --provider {model_config.provider} --model {model_config.model_id}[/dim]")
        else:
            console.print(f"[red]Failed to register model. It may already exist.[/red]")
            
    except FileNotFoundError:
        console.print(f"[red]File not found: {file_path}[/red]")
    except json.JSONDecodeError:
        console.print(f"[red]Invalid JSON format in file: {file_path}[/red]")
    except Exception as e:
        console.print(f"[red]Error during file registration: {str(e)}[/red]")


@app.command()
def register_help():
    """Show help for model registration methods."""
    console.print("[bold blue]Model Registration Methods[/bold blue]\n")
    
    # Interactive registration
    console.print("[bold]1. Interactive Registration (Recommended for beginners)[/bold]")
    console.print("   [cyan]cosmo register[/cyan]")
    console.print("   Guided wizard that asks for all required information step by step.\n")
    
    # Quick registration
    console.print("[bold]2. Quick Registration (For simple models)[/bold]")
    console.print("   [cyan]cosmo register-quick <provider> <model-id> [options][/cyan]")
    console.print("   Register a model with minimal input using command-line options.\n")
    
    # Template registration
    console.print("[bold]3. Template Registration (For common models)[/bold]")
    console.print("   [cyan]cosmo register-template <template> [options][/cyan]")
    console.print("   Available templates: gpt4, gpt35, claude, gemini, local, custom")
    console.print("   Example: [cyan]cosmo register-template gpt4[/cyan]\n")
    
    # File registration
    console.print("[bold]4. File-based Registration (For complex models)[/bold]")
    console.print("   [cyan]cosmo register-from-file <json-file> [options][/cyan]")
    console.print("   Register from a JSON configuration file.")
    console.print("   Example: [cyan]cosmo register-from-file examples/model_template.json[/cyan]\n")
    
    # Management
    console.print("[bold]5. Model Management[/bold]")
    console.print("   [cyan]cosmo unregister <model-id>[/cyan] - Remove a model")
    console.print("   [cyan]cosmo models[/cyan] - List all registered models")
    console.print("   [cyan]cosmo model-info <model-id>[/cyan] - Show model details\n")
    
    # Examples
    console.print("[bold]Quick Examples:[/bold]")
    console.print("   [cyan]cosmo register[/cyan] - Start interactive wizard")
    console.print("   [cyan]cosmo register-quick myprovider mymodel --tier premium[/cyan]")
    console.print("   [cyan]cosmo register-template claude --provider myclaude[/cyan]")
    console.print("   [cyan]cosmo register-from-file my_model.json[/cyan]\n")
    
    console.print("[dim]For detailed examples, see: docs/MODEL_LIBRARY.md[/dim]")


@app.command()
def distribute(
    prompt: str = typer.Argument(..., help="Prompt to distribute"),
    job_type: str = typer.Option("auto", "--type", help="Job type: auto, single, parallel, pipeline, load_balanced, fallback"),
    models: str = typer.Option(None, "--models", help="Comma-separated list of models (provider:model)"),
    priority: int = typer.Option(3, "--priority", help="Job priority (1-5, 5=highest)"),
    timeout: int = typer.Option(30, "--timeout", help="Timeout in seconds"),
    explain: bool = typer.Option(False, "--explain", help="Show distribution decision without execution"),
):
    """Distribute jobs intelligently across multiple models."""
    try:
        from cosmosapien.core.job_distributor import JobDistributor, JobType
        
        # Initialize job distributor
        job_distributor = JobDistributor(config_manager, model_library, local_manager)
        
        # Parse job type
        job_type_map = {
            "auto": JobType.LOAD_BALANCED,  # Default to auto distribution
            "single": JobType.SINGLE_TASK,
            "parallel": JobType.PARALLEL_TASK,
            "pipeline": JobType.PIPELINE_TASK,
            "load_balanced": JobType.LOAD_BALANCED,
            "fallback": JobType.FALLBACK_CHAIN
        }
        
        if job_type not in job_type_map:
            console.print(f"[red]Invalid job type: {job_type}[/red]")
            console.print(f"Available types: {', '.join(job_type_map.keys())}")
            return
        
        # Parse models
        model_list = None
        if models:
            model_list = [m.strip() for m in models.split(",")]
        
        # Create job request
        job_request = job_distributor.create_job(
            prompt=prompt,
            job_type=job_type_map[job_type],
            models=model_list,
            priority=priority,
            timeout=timeout
        )
        
        if explain:
            # Show distribution decision
            console.print(f"[bold]Job Distribution Decision[/bold]")
            console.print(f"Job ID: {job_request.job_id}")
            console.print(f"Job Type: {job_request.job_type.value}")
            console.print(f"Models: {', '.join(job_request.models) if job_request.models else 'Auto-selected'}")
            console.print(f"Priority: {job_request.priority}")
            console.print(f"Timeout: {job_request.timeout}s")
            
            # Show model status
            stats = job_distributor.get_distribution_stats()
            console.print(f"\n[bold]Model Status[/bold]")
            for model_key, status in stats["model_status"].items():
                console.print(f"  {model_key}: Load={status['current_load']}, Success={status['success_rate']:.2f}, Response={status['avg_response_time']:.2f}s")
            
            return
        
        # Execute job
        console.print(f"[bold]Executing job: {job_request.job_id}[/bold]")
        console.print(f"Type: {job_request.job_type.value}")
        console.print(f"Models: {', '.join(job_request.models) if job_request.models else 'Auto-selected'}")
        
        result = job_distributor.distribute_job(job_request)
        
        # Display result
        if result.success:
            console.print(f"\n[green]Job completed successfully![/green]")
            console.print(f"Model used: {result.model_used}")
            console.print(f"Execution time: {result.execution_time:.2f}s")
            console.print(f"Tokens used: {result.tokens_used}")
            
            console.print(f"\n[bold]Response:[/bold]")
            console.print(result.response)
        else:
            console.print(f"\n[red]Job failed![/red]")
            console.print(f"Error: {result.error}")
            console.print(f"Model attempted: {result.model_used}")
            console.print(f"Execution time: {result.execution_time:.2f}s")
        
        # Save stats
        job_distributor.save_stats()
        
    except Exception as e:
        console.print(f"[red]Error during job distribution: {str(e)}[/red]")


@app.command()
def squeeze(
    prompt: str = typer.Argument(..., help="Prompt to process"),
    explain: bool = typer.Option(False, "--explain", help="Show routing decision without execution"),
):
    """Use all available free tiers and local models to process the task."""
    try:
        from cosmosapien.core.job_distributor import JobDistributor, JobType
        
        # Initialize job distributor
        job_distributor = JobDistributor(config_manager, model_library, local_manager)
        
        # Get all free tier and local models
        free_models = []
        models = model_library.list_models()
        
        for model in models:
            if model.is_active:
                # Check if it's a free tier model or local model
                if (model.pricing.free_tier_limit > 0 or 
                    model.is_local or 
                    model.pricing.input_cost_per_1k_tokens == 0.0):
                    free_models.append(f"{model.provider}:{model.model_id}")
        
        if not free_models:
            console.print("[yellow]No free tier models available. Using smart routing instead.[/yellow]")
            # Fallback to smart routing
            routing_decision = smart_router.smart_route(prompt, explain_only=explain)
            if explain:
                console.print(f"[bold]Smart Routing Decision[/bold]")
                console.print(f"Selected: {routing_decision.selected_provider}:{routing_decision.selected_model}")
                console.print(f"Reasoning: {routing_decision.reasoning}")
                console.print(f"Complexity: {routing_decision.complexity.value}")
                return
            
            # Execute with smart routing
            from ..models import get_model_instance
            model_instance = get_model_instance(routing_decision.selected_provider, routing_decision.selected_model)
            response = asyncio.run(model_instance.generate(prompt))
            console.print(response.content)
            return
        
        # Create job request for free tier distribution
        job_request = job_distributor.create_job(
            prompt=prompt,
            job_type=JobType.PARALLEL_TASK,  # Try all free models in parallel
            models=free_models,
            priority=1,  # Low priority for free tier usage
            timeout=60  # Longer timeout for free models
        )
        
        if explain:
            console.print(f"[bold]Squeeze Distribution Decision[/bold]")
            console.print(f"Free models available: {len(free_models)}")
            console.print(f"Models: {', '.join(free_models)}")
            console.print(f"Strategy: Parallel execution across all free tiers")
            return
        
        # Execute with free tier distribution
        console.print(f"[bold]Squeezing across {len(free_models)} free models...[/bold]")
        console.print(f"Models: {', '.join(free_models)}")
        
        result = job_distributor.distribute_job(job_request)
        
        # Display result
        if result.success:
            console.print(f"\n[green]Task completed using free tier![/green]")
            console.print(f"Model used: {result.model_used}")
            console.print(f"Execution time: {result.execution_time:.2f}s")
            console.print(f"Cost: $0.00 (free tier)")
            
            console.print(f"\n[bold]Response:[/bold]")
            console.print(result.response)
        else:
            console.print(f"\n[red]All free tiers failed![/red]")
            console.print(f"Error: {result.error}")
            console.print(f"Attempted models: {', '.join(free_models)}")
        
        # Save stats
        job_distributor.save_stats()
        
    except Exception as e:
        console.print(f"[red]Error during squeeze execution: {str(e)}[/red]")


@app.command()
def job_stats():
    """Show job distribution statistics."""
    try:
        from cosmosapien.core.job_distributor import JobDistributor
        
        # Initialize job distributor
        job_distributor = JobDistributor(config_manager, model_library, local_manager)
        
        # Get statistics
        stats = job_distributor.get_distribution_stats()
        
        console.print("[bold blue]Job Distribution Statistics[/bold blue]\n")
        
        # Overall stats
        console.print(f"[bold]Overall Performance[/bold]")
        console.print(f"Active Jobs: {stats['active_jobs']}")
        console.print(f"Completed Jobs: {stats['completed_jobs']}")
        console.print(f"Average Response Time: {stats['performance']['avg_response_time']:.2f}s")
        console.print(f"Total Success Rate: {stats['performance']['total_success_rate']:.2%}")
        console.print(f"Total Errors: {stats['performance']['total_errors']}\n")
        
        # Model status
        console.print(f"[bold]Model Status[/bold]")
        if stats['model_status']:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Model", style="cyan")
            table.add_column("Load", justify="center")
            table.add_column("Success Rate", justify="center")
            table.add_column("Avg Response", justify="center")
            table.add_column("Errors", justify="center")
            table.add_column("Last Used", justify="center")
            
            for model_key, status in stats['model_status'].items():
                table.add_row(
                    model_key,
                    str(status['current_load']),
                    f"{status['success_rate']:.2%}",
                    f"{status['avg_response_time']:.2f}s",
                    str(status['error_count']),
                    status['last_used'][:19] if status['last_used'] else "Never"
                )
            
            console.print(table)
        else:
            console.print("No model statistics available.")
        
    except Exception as e:
        console.print(f"[red]Error getting job statistics: {str(e)}[/red]")


@app.command()
def reset_job_stats():
    """Reset job distribution statistics."""
    try:
        from cosmosapien.core.job_distributor import JobDistributor
        
        # Initialize job distributor
        job_distributor = JobDistributor(config_manager, model_library, local_manager)
        
        # Reset stats
        job_distributor.save_stats()
        
        console.print("[green]Job distribution statistics reset successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Error resetting job statistics: {str(e)}[/red]")


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
    
    # Show smart routing configuration
    smart_config = config_manager.get_smart_routing_config()
    console.print(Panel(
        f"[bold]Smart Routing Enabled:[/bold] {smart_config.enabled}\n"
        f"[bold]Prefer Local Models:[/bold] {smart_config.prefer_local}\n"
        f"[bold]Cost Threshold:[/bold] ${smart_config.cost_threshold:.4f}\n"
        f"[bold]Custom Limits:[/bold] {len(smart_config.free_tier_limits)} providers\n"
        f"[bold]Custom Costs:[/bold] {len(smart_config.custom_costs)} providers",
        title="[bold]Smart Routing Configuration[/bold]",
        border_style="green"
    ))
    
    # Show custom limits and costs
    if smart_config.free_tier_limits or smart_config.custom_costs:
        console.print("\n[bold]Custom Smart Routing Settings:[/bold]")
        
        if smart_config.free_tier_limits:
            console.print("\n[bold]Free Tier Limits:[/bold]")
            for provider, models in smart_config.free_tier_limits.items():
                for model, limit in models.items():
                    console.print(f"  • {provider}/{model}: {limit} calls")
        
        if smart_config.custom_costs:
            console.print("\n[bold]Custom Costs:[/bold]")
            for provider, models in smart_config.custom_costs.items():
                for model, cost in models.items():
                    console.print(f"  • {provider}/{model}: ${cost:.4f}/call")


if __name__ == "__main__":
    app() 