"""Multi-agent system for coordinating AI models."""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from .models import ChatMessage, ModelResponse
from .router import Router
from .local_manager import LocalModelManager


class AgentRole(Enum):
    """Different roles for AI agents."""
    GENERALIST = "generalist"
    SPECIALIST = "specialist"
    CODER = "coder"
    ANALYST = "analyst"
    CREATIVE = "creative"
    REASONER = "reasoner"
    VALIDATOR = "validator"


@dataclass
class Agent:
    """Represents an AI agent with specific capabilities."""
    name: str
    role: AgentRole
    provider: str
    model: str
    description: str
    capabilities: List[str]
    is_active: bool = True


class AgentSystem:
    """Coordinates multiple AI agents for collaborative problem solving."""
    
    def __init__(self, router: Router, local_manager: LocalModelManager):
        self.router = router
        self.local_manager = local_manager
        self.agents: List[Agent] = []
        self.conversation_history: List[ChatMessage] = []
        
    def register_agent(self, agent: Agent):
        """Register a new agent."""
        self.agents.append(agent)
    
    def get_agents_by_role(self, role: AgentRole) -> List[Agent]:
        """Get all agents with a specific role."""
        return [agent for agent in self.agents if agent.role == role and agent.is_active]
    
    def get_available_agents(self) -> List[Agent]:
        """Get all available agents."""
        return [agent for agent in self.agents if agent.is_active]
    
    async def create_default_agents(self):
        """Create default agents based on available models."""
        # Check what models are available
        ollama_available = self.local_manager.detect_ollama()
        
        if ollama_available:
            # Get available Ollama models
            available_models = self.local_manager.get_available_models("ollama")
            
            # Create agents based on available models
            if "llama3.2:8b" in available_models or "llama3.2:3b" in available_models:
                model = "llama3.2:8b" if "llama3.2:8b" in available_models else "llama3.2:3b"
                self.register_agent(Agent(
                    name="Llama",
                    role=AgentRole.GENERALIST,
                    provider="llama",
                    model=model,
                    description="General purpose AI assistant",
                    capabilities=["conversation", "reasoning", "analysis"]
                ))
            
            if "codellama:13b" in available_models or "codellama:7b" in available_models:
                model = "codellama:13b" if "codellama:13b" in available_models else "codellama:7b"
                self.register_agent(Agent(
                    name="CodeLlama",
                    role=AgentRole.CODER,
                    provider="llama",
                    model=model,
                    description="Specialized in code generation and analysis",
                    capabilities=["coding", "debugging", "code_review", "algorithm_design"]
                ))
            
            if "mixtral:8x7b" in available_models:
                self.register_agent(Agent(
                    name="Mixtral",
                    role=AgentRole.REASONER,
                    provider="llama",
                    model="mixtral:8x7b",
                    description="Advanced reasoning and problem solving",
                    capabilities=["complex_reasoning", "mathematics", "logic", "planning"]
                ))
            
            if "dolphin-llama3:latest" in available_models:
                self.register_agent(Agent(
                    name="Dolphin",
                    role=AgentRole.CREATIVE,
                    provider="llama",
                    model="dolphin-llama3:latest",
                    description="Creative and helpful AI assistant",
                    capabilities=["creative_writing", "brainstorming", "storytelling"]
                ))
        
        # Add cloud-based agents if available
        await self._add_cloud_agents()
    
    async def _add_cloud_agents(self):
        """Add cloud-based agents for logged-in providers."""
        # Import auth manager to check login status
        from ..auth.manager import AuthManager
        from ..core.config import ConfigManager
        
        config_manager = ConfigManager()
        auth_manager = AuthManager(config_manager)
        
        # Check each cloud provider and add agents
        providers = auth_manager.list_providers()
        
        for provider_info in providers:
            if provider_info["logged_in"]:
                provider_name = provider_info["provider"]
                
                if provider_name == "openai":
                    # Add GPT agents
                    self.register_agent(Agent(
                        name="GPT-4",
                        role=AgentRole.GENERALIST,
                        provider="openai",
                        model="gpt-4",
                        description="Advanced general-purpose AI with strong reasoning",
                        capabilities=["conversation", "reasoning", "analysis", "creative_writing", "coding"]
                    ))
                    
                    self.register_agent(Agent(
                        name="GPT-4-Turbo",
                        role=AgentRole.SPECIALIST,
                        provider="openai",
                        model="gpt-4-turbo-preview",
                        description="Fast and efficient AI for real-time tasks",
                        capabilities=["fast_response", "real_time_analysis", "quick_coding"]
                    ))
                
                elif provider_name == "gemini":
                    # Add Gemini agents
                    self.register_agent(Agent(
                        name="Gemini-Pro",
                        role=AgentRole.GENERALIST,
                        provider="gemini",
                        model="gemini-pro",
                        description="Google's advanced AI with strong reasoning capabilities",
                        capabilities=["conversation", "reasoning", "analysis", "multimodal"]
                    ))
                    
                    self.register_agent(Agent(
                        name="Gemini-Flash",
                        role=AgentRole.SPECIALIST,
                        provider="gemini",
                        model="gemini-1.5-flash",
                        description="Fast and efficient Gemini model",
                        capabilities=["fast_response", "efficient_processing", "quick_analysis"]
                    ))
                
                elif provider_name == "claude":
                    # Add Claude agents
                    self.register_agent(Agent(
                        name="Claude-3-Sonnet",
                        role=AgentRole.ANALYST,
                        provider="claude",
                        model="claude-3-sonnet-20240229",
                        description="Anthropic's analytical AI with strong reasoning",
                        capabilities=["analysis", "reasoning", "research", "writing"]
                    ))
                    
                    self.register_agent(Agent(
                        name="Claude-3-Haiku",
                        role=AgentRole.SPECIALIST,
                        provider="claude",
                        model="claude-3-haiku-20240307",
                        description="Fast and efficient Claude model",
                        capabilities=["fast_response", "quick_analysis", "efficient_processing"]
                    ))
                
                elif provider_name == "grok":
                    # Add Grok agents
                    self.register_agent(Agent(
                        name="Grok",
                        role=AgentRole.CREATIVE,
                        provider="grok",
                        model="grok-beta",
                        description="xAI's creative and witty AI assistant",
                        capabilities=["creative_writing", "humor", "brainstorming", "entertainment"]
                    ))
                
                elif provider_name == "perplexity":
                    # Add Perplexity agents
                    self.register_agent(Agent(
                        name="Perplexity",
                        role=AgentRole.ANALYST,
                        provider="perplexity",
                        model="llama-3.1-sonar-small-128k-online",
                        description="Web-connected AI with real-time information",
                        capabilities=["web_search", "real_time_info", "research", "analysis"]
                    ))
                    
                    self.register_agent(Agent(
                        name="Perplexity-Sonar",
                        role=AgentRole.SPECIALIST,
                        provider="perplexity",
                        model="llama-3.1-sonar-medium-128k-online",
                        description="Advanced Perplexity model with enhanced capabilities",
                        capabilities=["advanced_analysis", "deep_research", "comprehensive_search"]
                    ))
                
                elif provider_name == "huggingface":
                    # Add Hugging Face agents
                    self.register_agent(Agent(
                        name="HuggingFace",
                        role=AgentRole.SPECIALIST,
                        provider="huggingface",
                        model="gpt2",
                        description="Open-source AI model from Hugging Face",
                        capabilities=["text_generation", "experimentation", "open_source"]
                    ))
    
    async def collaborative_chat(self, user_message: str, agent_roles: Optional[List[AgentRole]] = None) -> Dict[str, Any]:
        """Run a collaborative chat session with multiple agents."""
        if not agent_roles:
            # Default to generalist and specialist agents
            agent_roles = [AgentRole.GENERALIST, AgentRole.SPECIALIST]
        
        # Get relevant agents
        agents = []
        for role in agent_roles:
            agents.extend(self.get_agents_by_role(role))
        
        if not agents:
            # Fallback to any available agent
            agents = self.get_available_agents()
        
        if not agents:
            raise ValueError("No agents available for collaboration")
        
        # Add user message to history
        self.conversation_history.append(ChatMessage(role="user", content=user_message))
        
        # Get responses from each agent
        responses = {}
        for agent in agents:
            try:
                response = await self.router.chat(
                    messages=self.conversation_history,
                    provider=agent.provider,
                    model=agent.model
                )
                responses[agent.name] = {
                    "content": response.content,
                    "role": agent.role.value,
                    "capabilities": agent.capabilities,
                    "model": agent.model
                }
                
                # Add agent response to history
                self.conversation_history.append(ChatMessage(
                    role="assistant", 
                    content=f"[{agent.name}]: {response.content}"
                ))
                
            except Exception as e:
                responses[agent.name] = {
                    "content": f"Error: {str(e)}",
                    "role": agent.role.value,
                    "capabilities": agent.capabilities,
                    "model": agent.model,
                    "error": True
                }
        
        return {
            "agents": responses,
            "summary": await self._generate_summary(responses),
            "recommendations": await self._generate_recommendations(user_message, responses)
        }
    
    async def _generate_summary(self, responses: Dict[str, Any]) -> str:
        """Generate a summary of all agent responses."""
        if not responses:
            return "No responses generated."
        
        # Create a summary prompt
        summary_prompt = "Summarize the following AI agent responses:\n\n"
        for agent_name, response in responses.items():
            if not response.get("error"):
                summary_prompt += f"{agent_name} ({response['role']}): {response['content']}\n\n"
        
        try:
            # Use the first available agent to generate summary
            agents = self.get_available_agents()
            if agents:
                summary_response = await self.router.generate(
                    prompt=summary_prompt,
                    provider=agents[0].provider,
                    model=agents[0].model
                )
                return summary_response.content
        except Exception:
            pass
        
        # Fallback summary
        return f"Generated responses from {len(responses)} agents."
    
    async def _generate_recommendations(self, user_message: str, responses: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on agent responses."""
        recommendations = []
        
        # Analyze responses and generate recommendations
        if len(responses) > 1:
            recommendations.append("Multiple agents provided different perspectives - consider combining their insights.")
        
        # Check for coding-related queries
        if any("coding" in response.get("capabilities", []) for response in responses.values()):
            recommendations.append("Consider using the CodeLlama agent for detailed code analysis.")
        
        # Check for creative queries
        if any("creative" in response.get("capabilities", []) for response in responses.values()):
            recommendations.append("The creative agent can help with brainstorming and ideation.")
        
        return recommendations
    
    async def debate_topic(self, topic: str, num_rounds: int = 3) -> List[Dict[str, Any]]:
        """Run a debate between agents on a given topic."""
        agents = self.get_available_agents()
        if len(agents) < 2:
            raise ValueError("Need at least 2 agents for a debate")
        
        debate_history = []
        current_topic = topic
        
        for round_num in range(num_rounds):
            round_responses = {}
            
            for agent in agents:
                try:
                    # Create debate prompt
                    debate_prompt = f"""
Topic: {current_topic}

Previous responses:
{self._format_debate_history(debate_history)}

Please provide your perspective on this topic, considering the previous responses.
                    """
                    
                    response = await self.router.generate(
                        prompt=debate_prompt,
                        provider=agent.provider,
                        model=agent.model
                    )
                    
                    round_responses[agent.name] = {
                        "content": response.content,
                        "role": agent.role.value,
                        "round": round_num + 1
                    }
                    
                    # Add to debate history
                    debate_history.append({
                        "agent": agent.name,
                        "content": response.content,
                        "round": round_num + 1
                    })
                    
                except Exception as e:
                    round_responses[agent.name] = {
                        "content": f"Error: {str(e)}",
                        "role": agent.role.value,
                        "round": round_num + 1,
                        "error": True
                    }
            
            # Update topic for next round
            current_topic = f"{topic} (Round {round_num + 2})"
        
        return debate_history
    
    def _format_debate_history(self, history: List[Dict[str, Any]]) -> str:
        """Format debate history for prompts."""
        if not history:
            return "No previous responses."
        
        formatted = ""
        for entry in history[-6:]:  # Last 6 responses
            formatted += f"{entry['agent']} (Round {entry['round']}): {entry['content']}\n\n"
        
        return formatted
    
    async def solve_complex_problem(self, problem: str) -> Dict[str, Any]:
        """Use multiple agents to solve a complex problem."""
        # Break down the problem into components
        components = await self._analyze_problem(problem)
        
        # Assign agents to different components
        assignments = await self._assign_agents_to_components(components)
        
        # Solve each component
        solutions = {}
        for component, agents in assignments.items():
            component_solutions = []
            for agent in agents:
                try:
                    response = await self.router.generate(
                        prompt=f"Solve this component of the problem: {component}",
                        provider=agent.provider,
                        model=agent.model
                    )
                    component_solutions.append({
                        "agent": agent.name,
                        "solution": response.content,
                        "role": agent.role.value
                    })
                except Exception as e:
                    component_solutions.append({
                        "agent": agent.name,
                        "solution": f"Error: {str(e)}",
                        "role": agent.role.value,
                        "error": True
                    })
            solutions[component] = component_solutions
        
        # Synthesize final solution
        final_solution = await self._synthesize_solutions(problem, solutions)
        
        return {
            "problem": problem,
            "components": components,
            "solutions": solutions,
            "final_solution": final_solution
        }
    
    async def _analyze_problem(self, problem: str) -> List[str]:
        """Analyze a problem into components."""
        # This would use an agent to break down the problem
        # For now, return a simple breakdown
        return ["analysis", "solution_design", "implementation", "validation"]
    
    async def _assign_agents_to_components(self, components: List[str]) -> Dict[str, List[Agent]]:
        """Assign agents to problem components."""
        assignments = {}
        agents = self.get_available_agents()
        
        for component in components:
            if component == "analysis":
                assignments[component] = [a for a in agents if a.role in [AgentRole.ANALYST, AgentRole.GENERALIST]]
            elif component == "solution_design":
                assignments[component] = [a for a in agents if a.role in [AgentRole.REASONER, AgentRole.SPECIALIST]]
            elif component == "implementation":
                assignments[component] = [a for a in agents if a.role in [AgentRole.CODER, AgentRole.SPECIALIST]]
            elif component == "validation":
                assignments[component] = [a for a in agents if a.role in [AgentRole.VALIDATOR, AgentRole.ANALYST]]
            else:
                assignments[component] = agents
        
        return assignments
    
    async def _synthesize_solutions(self, problem: str, solutions: Dict[str, List[Dict[str, Any]]]) -> str:
        """Synthesize individual solutions into a final solution."""
        synthesis_prompt = f"""
Original Problem: {problem}

Component Solutions:
"""
        
        for component, component_solutions in solutions.items():
            synthesis_prompt += f"\n{component.upper()}:\n"
            for solution in component_solutions:
                if not solution.get("error"):
                    synthesis_prompt += f"- {solution['agent']}: {solution['solution']}\n"
        
        synthesis_prompt += "\nPlease synthesize these solutions into a comprehensive final solution."
        
        try:
            agents = self.get_available_agents()
            if agents:
                response = await self.router.generate(
                    prompt=synthesis_prompt,
                    provider=agents[0].provider,
                    model=agents[0].model
                )
                return response.content
        except Exception:
            pass
        
        return "Unable to synthesize solutions due to errors." 