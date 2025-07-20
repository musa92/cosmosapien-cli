"""Intelligent job distribution system for multi-model task execution."""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import ConfigManager
from .local_manager import LocalModelManager
from .model_library import ModelLibrary
from .smart_router import SmartRouter


class JobType(Enum):
    """Types of jobs that can be distributed."""

    SINGLE_TASK = "single_task"  # Single prompt to single model
    PARALLEL_TASK = "parallel_task"  # Same task to multiple models
    PIPELINE_TASK = "pipeline_task"  # Sequential processing chain
    LOAD_BALANCED = "load_balanced"  # Distribute across available models
    FALLBACK_CHAIN = "fallback_chain"  # Try models in order until success


@dataclass
class JobRequest:
    """Job request specification."""

    job_id: str
    job_type: JobType
    prompt: str
    models: List[str]  # List of model IDs (provider:model)
    priority: int = 1  # 1=low, 5=high
    timeout: int = 30  # seconds
    max_retries: int = 2
    context: Dict[str, Any] = None
    callback: Optional[Callable] = None


@dataclass
class JobResult:
    """Result of a job execution."""

    job_id: str
    model_used: str
    response: str
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0
    cost: float = 0.0
    tokens_used: int = 0


@dataclass
class ModelStatus:
    """Current status of a model."""

    provider: str
    model: str
    is_available: bool
    current_load: int = 0
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    last_used: datetime = None
    error_count: int = 0
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_tokens_per_request: float = 0.0
    total_requests: int = 0


class JobDistributor:
    """Intelligent job distribution system."""

    def __init__(
        self,
        config_manager: ConfigManager,
        model_library: ModelLibrary,
        local_manager: LocalModelManager,
    ):
        self.config_manager = config_manager
        self.model_library = model_library
        self.local_manager = local_manager
        self.smart_router = SmartRouter(config_manager, local_manager)

        # Job tracking
        self.active_jobs: Dict[str, JobRequest] = {}
        self.completed_jobs: Dict[str, JobResult] = {}
        self.model_status: Dict[str, ModelStatus] = {}

        # Performance tracking
        self.stats_file = Path.home() / ".cosmo" / "job_stats.json"
        self.stats_file.parent.mkdir(exist_ok=True)
        self.load_stats()

        # Initialize model status
        self._initialize_model_status()

    def _initialize_model_status(self):
        """Initialize status for all available models."""
        models = self.model_library.list_models()

        for model in models:
            if model.is_active:
                key = "{model.provider}:{model.model_id}"
                self.model_status[key] = ModelStatus(
                    provider=model.provider,
                    model=model.model_id,
                    is_available=True,
                    last_used=datetime.now(),
                )

    def load_stats(self):
        """Load job execution statistics."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, "r") as f:
                    stats = json.load(f)
                    # Load model performance data
                    for model_key, data in stats.get("models", {}).items():
                        if model_key in self.model_status:
                            self.model_status[model_key].avg_response_time = data.get(
                                "avg_response_time", 0.0
                            )
                            self.model_status[model_key].success_rate = data.get(
                                "success_rate", 1.0
                            )
                            self.model_status[model_key].error_count = data.get(
                                "error_count", 0
                            )
                            self.model_status[model_key].total_tokens = data.get(
                                "total_tokens", 0
                            )
                            self.model_status[model_key].total_input_tokens = data.get(
                                "total_input_tokens", 0
                            )
                            self.model_status[model_key].total_output_tokens = data.get(
                                "total_output_tokens", 0
                            )
                            self.model_status[model_key].avg_tokens_per_request = (
                                data.get("avg_tokens_per_request", 0.0)
                            )
                            self.model_status[model_key].total_requests = data.get(
                                "total_requests", 0
                            )
            except (json.JSONDecodeError, IOError):
                pass

    def save_stats(self):
        """Save job execution statistics."""
        stats = {"models": {}, "last_updated": datetime.now().isoformat()}

        for model_key, status in self.model_status.items():
            stats["models"][model_key] = {
                "avg_response_time": status.avg_response_time,
                "success_rate": status.success_rate,
                "error_count": status.error_count,
                "total_jobs": status.current_load,
                "total_tokens": status.total_tokens,
                "total_input_tokens": status.total_input_tokens,
                "total_output_tokens": status.total_output_tokens,
                "avg_tokens_per_request": status.avg_tokens_per_request,
                "total_requests": status.total_requests,
            }

        try:
            with open(self.stats_file, "w") as f:
                json.dump(stats, f, indent=2)
        except IOError:
            pass

    def distribute_job(self, job_request: JobRequest) -> JobResult:
        """Distribute a job based on its type and current system state."""

        if job_request.job_type == JobType.SINGLE_TASK:
            return self._execute_single_task(job_request)
        elif job_request.job_type == JobType.PARALLEL_TASK:
            return self._execute_parallel_task(job_request)
        elif job_request.job_type == JobType.PIPELINE_TASK:
            return self._execute_pipeline_task(job_request)
        elif job_request.job_type == JobType.LOAD_BALANCED:
            return self._execute_load_balanced(job_request)
        elif job_request.job_type == JobType.FALLBACK_CHAIN:
            return self._execute_fallback_chain(job_request)
        else:
            raise ValueError("Unknown job type: {job_request.job_type}")

    def _execute_single_task(self, job_request: JobRequest) -> JobResult:
        """Execute a single task with the best available model."""
        # Use smart router to select the best model
        routing_decision = self.smart_router.smart_route(job_request.prompt)

        if routing_decision.selected_provider == "none":
            return JobResult(
                job_id=job_request.job_id,
                model_used="none",
                response="",
                success=False,
                error="No available models",
            )

        # Execute with selected model
        model_key = (
            "{routing_decision.selected_provider}:{routing_decision.selected_model}"
        )
        return self._execute_with_model(job_request, model_key)

    def _execute_parallel_task(self, job_request: JobRequest) -> JobResult:
        """Execute the same task with multiple models in parallel."""
        results = []

        # Filter available models
        available_models = [
            model for model in job_request.models if self._is_model_available(model)
        ]

        if not available_models:
            return JobResult(
                job_id=job_request.job_id,
                model_used="none",
                response="",
                success=False,
                error="No available models",
            )

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=len(available_models)) as executor:
            future_to_model = {
                executor.submit(self._execute_with_model, job_request, model): model
                for model in available_models
            }

            for future in as_completed(future_to_model, timeout=job_request.timeout):
                model = future_to_model[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result.success:
                        # Return first successful result
                        return result
                except Exception as e:
                    self._update_model_error(model, str(e))

        # Return best result or error
        if results:
            # Return the result with highest success rate
            best_result = max(
                results,
                key=lambda r: self.model_status.get(
                    r.model_used, ModelStatus("", "", True)
                ).success_rate,
            )
            return best_result

        return JobResult(
            job_id=job_request.job_id,
            model_used="none",
            response="",
            success=False,
            error="All models failed",
        )

    def _execute_pipeline_task(self, job_request: JobRequest) -> JobResult:
        """Execute a pipeline of tasks through multiple models."""
        if len(job_request.models) < 2:
            return self._execute_single_task(job_request)

        current_prompt = job_request.prompt
        final_result = None

        for model in job_request.models:
            if not self._is_model_available(model):
                continue

            # Create sub-job for this step
            sub_job = JobRequest(
                job_id="{job_request.job_id}_step_{model}",
                job_type=JobType.SINGLE_TASK,
                prompt=current_prompt,
                models=[model],
                priority=job_request.priority,
                timeout=job_request.timeout // len(job_request.models),
            )

            result = self._execute_with_model(sub_job, model)

            if not result.success:
                return JobResult(
                    job_id=job_request.job_id,
                    model_used=model,
                    response="",
                    success=False,
                    error="Pipeline failed at {model}: {result.error}",
                )

            # Use result as input for next step
            current_prompt = result.response
            final_result = result

        return final_result or JobResult(
            job_id=job_request.job_id,
            model_used="none",
            response="",
            success=False,
            error="Pipeline execution failed",
        )

    def _execute_load_balanced(self, job_request: JobRequest) -> JobResult:
        """Execute with load balancing across available models."""
        available_models = [
            model for model in job_request.models if self._is_model_available(model)
        ]

        if not available_models:
            return JobResult(
                job_id=job_request.job_id,
                model_used="none",
                response="",
                success=False,
                error="No available models",
            )

        # Select model with lowest load and highest success rate
        best_model = min(
            available_models,
            key=lambda m: (
                self.model_status.get(m, ModelStatus("", "", True)).current_load,
                -self.model_status.get(m, ModelStatus("", "", True)).success_rate,
            ),
        )

        return self._execute_with_model(job_request, best_model)

    def _execute_fallback_chain(self, job_request: JobRequest) -> JobResult:
        """Execute with fallback chain - try models in order until success."""
        for model in job_request.models:
            if not self._is_model_available(model):
                continue

            result = self._execute_with_model(job_request, model)
            if result.success:
                return result

        return JobResult(
            job_id=job_request.job_id,
            model_used="none",
            response="",
            success=False,
            error="All models in fallback chain failed",
        )

    def _execute_with_model(self, job_request: JobRequest, model_key: str) -> JobResult:
        """Execute a job with a specific model."""
        start_time = time.time()

        try:
            # Update model status
            if model_key in self.model_status:
                self.model_status[model_key].current_load += 1
                self.model_status[model_key].last_used = datetime.now()

            # Parse model key
            provider, model = model_key.split(":", 1)

            # Import and execute with appropriate model
            from ..models import get_model_instance

            model_instance = get_model_instance(provider, model)

            # Execute the request
            response = asyncio.run(model_instance.generate(job_request.prompt))

            execution_time = time.time() - start_time

            # Update model performance
            if model_key in self.model_status:
                status = self.model_status[model_key]
                status.current_load = max(0, status.current_load - 1)
                status.avg_response_time = (
                    status.avg_response_time + execution_time
                ) / 2
                status.success_rate = min(1.0, status.success_rate + 0.01)

                # Update token statistics
                tokens_used = (
                    response.usage.get("total_tokens", 0) if response.usage else 0
                )
                input_tokens = (
                    response.usage.get("prompt_tokens", 0) if response.usage else 0
                )
                output_tokens = (
                    response.usage.get("completion_tokens", 0) if response.usage else 0
                )

                status.total_tokens += tokens_used
                status.total_input_tokens += input_tokens
                status.total_output_tokens += output_tokens
                status.total_requests += 1
                status.avg_tokens_per_request = (
                    status.total_tokens / status.total_requests
                )

            return JobResult(
                job_id=job_request.job_id,
                model_used=model_key,
                response=response.content,
                success=True,
                execution_time=execution_time,
                tokens_used=(
                    response.usage.get("total_tokens", 0) if response.usage else 0
                ),
            )

        except Exception as e:
            execution_time = time.time() - start_time

            # Update model error status
            self._update_model_error(model_key, str(e))

            return JobResult(
                job_id=job_request.job_id,
                model_used=model_key,
                response="",
                success=False,
                error=str(e),
                execution_time=execution_time,
            )

    def _is_model_available(self, model_key: str) -> bool:
        """Check if a model is available for use."""
        if model_key not in self.model_status:
            return False

        status = self.model_status[model_key]

        # Check if model is overloaded
        if status.current_load > 5:  # Max 5 concurrent jobs per model
            return False

        # Check if model has too many recent errors
        if status.error_count > 3:
            return False

        return status.is_available

    def _update_model_error(self, model_key: str, error: str):
        """Update model error statistics."""
        if model_key in self.model_status:
            status = self.model_status[model_key]
            status.error_count += 1
            status.success_rate = max(0.0, status.success_rate - 0.1)
            status.current_load = max(0, status.current_load - 1)

    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get current distribution statistics."""
        stats = {
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "model_status": {},
            "performance": {
                "avg_response_time": 0.0,
                "total_success_rate": 0.0,
                "total_errors": 0,
                "total_tokens": 0,
                "total_requests": 0,
            },
        }

        total_response_time = 0.0
        total_success_rate = 0.0
        total_errors = 0
        total_tokens = 0
        total_requests = 0
        active_models = 0

        for model_key, status in self.model_status.items():
            if status.is_available:
                active_models += 1
                total_response_time += status.avg_response_time
                total_success_rate += status.success_rate
                total_errors += status.error_count
                total_tokens += status.total_tokens
                total_requests += status.total_requests

                stats["model_status"][model_key] = {
                    "current_load": status.current_load,
                    "avg_response_time": status.avg_response_time,
                    "success_rate": status.success_rate,
                    "error_count": status.error_count,
                    "last_used": (
                        status.last_used.isoformat() if status.last_used else None
                    ),
                    "total_tokens": status.total_tokens,
                    "total_input_tokens": status.total_input_tokens,
                    "total_output_tokens": status.total_output_tokens,
                    "avg_tokens_per_request": status.avg_tokens_per_request,
                    "total_requests": status.total_requests,
                }

        if active_models > 0:
            stats["performance"]["avg_response_time"] = (
                total_response_time / active_models
            )
            stats["performance"]["total_success_rate"] = (
                total_success_rate / active_models
            )
            stats["performance"]["total_errors"] = total_errors
            stats["performance"]["total_tokens"] = total_tokens
            stats["performance"]["total_requests"] = total_requests

        return stats

    def create_job(
        self,
        prompt: str,
        job_type: JobType = JobType.SINGLE_TASK,
        models: Optional[List[str]] = None,
        **kwargs,
    ) -> JobRequest:
        """Create a new job request."""
        job_id = "job_{int(time.time())}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"

        if models is None:
            # Use smart routing to select models
            routing_decision = self.smart_router.smart_route(prompt)
            if routing_decision.selected_provider != "none":
                models = [
                    "{routing_decision.selected_provider}:{routing_decision.selected_model}"
                ]
            else:
                models = []

        return JobRequest(
            job_id=job_id, job_type=job_type, prompt=prompt, models=models, **kwargs
        )
