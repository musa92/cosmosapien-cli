import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
import json
import re
from ..context import model_library, router
from ..main import ensure_ollama_ready

app = typer.Typer()

@app.command()
def orchestrate(
    prompt: str = typer.Argument(..., help="A complex question to be decomposed, distributed, and synthesized by models."),
    planner_model: str = typer.Option(None, "--planner", help="Model to use for decomposition (provider:model), defaults to best local model"),
    synthesizer_model: str = typer.Option(None, "--synthesizer", help="Model to use for synthesis/paraphrase, defaults to best local model"),
):
    """Meta-agent: Use a model to decompose, distribute, and synthesize answers from a single complex question."""
    console = Console()
    if not ensure_ollama_ready():
        print("Local model setup was not completed. Exiting.")
        return
    preferred_models = [
        "llama:llama3.2:3b",
        "llama:llama2:latest",
        "llama:codellama:13b",
        "llama:mixtral:8x7b",
        "llama:mistral:7b",
    ]
    def get_best_local_model():
        local_models = model_library.get_local_models()
        available = [f"{m.provider}:{m.model_id}" for m in local_models]
        for m in preferred_models:
            if m in available:
                return m
        raise RuntimeError("No preferred local models available. Please install and activate at least one of: " + ", ".join(preferred_models))
    async def get_plan(prompt, planner_model, available_models):
        provider, model = planner_model.split(":", 1)
        planner = router.get_model_instance(provider, model)
        available_models_str = ", ".join(available_models)
        planner_prompt = (
            "You are an expert AI task planner. Given the following complex question, break it down into a set of sub-tasks. "
            "For each sub-task, assign the most suitable model from this list: [" + available_models_str + "] (choose one per sub-task). "
            "Return your answer as a JSON array, with each item like: {\"subtask\": \"...\", \"model\": \"...\"}. "
            "Do not include any explanation or text outside the JSON array.\n\n"
            "Question: " + prompt
        )
        response = await planner.generate(planner_prompt)
        try:
            json_start = response.content.find("[")
            json_end = response.content.rfind("]") + 1
            plan_json = response.content[json_start:json_end]
            plan = json.loads(plan_json)
            return plan
        except Exception:
            text = response.content.strip()
            pattern = r"\d+\.\s*(.+?):?\n?\s*\* Model: ([^\n]+)"
            matches = re.findall(pattern, text)
            if matches:
                plan = [{"subtask": subtask.strip(), "model": model.strip()} for subtask, model in matches]
                return plan
            console.print("[yellow]Planner model did not return valid JSON or a parseable list. Raw output:[/yellow]")
            console.print(Panel(text, title="Planner Output", border_style="red"))
            raise RuntimeError("Planner model did not return valid JSON or a parseable list.")
    async def get_answer(idx, subtask, model_key):
        provider, model = model_key.split(":", 1)
        try:
            model_instance = router.get_model_instance(provider, model)
            table_rows[idx][2] = "Thinking..."
            response = await model_instance.generate(subtask)
            table_rows[idx][2] = "Done"
            table_rows[idx][3] = response.content[:60] + ("..." if len(response.content) > 60 else "")
            return (idx, subtask, model_key, response.content)
        except Exception as e:
            table_rows[idx][2] = "Error"
            table_rows[idx][3] = str(e)
            return (idx, subtask, model_key, f"Error: {str(e)}")
    def build_table():
        table = Table(title="Meta-Agent Progress", show_lines=True)
        table.add_column("Sub-Task", style="cyan", no_wrap=True)
        table.add_column("Model", style="magenta")
        table.add_column("Status", style="yellow")
        table.add_column("Answer", style="green")
        for row in table_rows:
            table.add_row(*row)
        return table
    async def _orchestrate():
        local_models = model_library.get_local_models()
        available_model_keys = [f"{m.provider}:{m.model_id}" for m in local_models if f"{m.provider}:{m.model_id}" in preferred_models]
        if not available_model_keys:
            console.print("[red]No preferred local models available. Please install and activate at least one of: " + ", ".join(preferred_models) + "[/red]")
            return
        planner_key = planner_model or get_best_local_model()
        synthesizer_key = synthesizer_model or get_best_local_model()
        console.print(f"[bold blue]Using planner:[/bold blue] {planner_key}")
        console.print(f"[bold blue]Using synthesizer:[/bold blue] {synthesizer_key}")
        console.print("[bold blue]Decomposing and assigning tasks using planner model...[/bold blue]")
        plan = await get_plan(prompt, planner_key, available_model_keys)
        if not plan or not isinstance(plan, list):
            console.print("[red]Planner did not return a valid plan.[/red]")
            return
        console.print("\n[bold]Breakdown of sub-tasks and model assignments:[/bold]")
        for i, item in enumerate(plan):
            console.print(f"  [cyan]Task {i+1}:[/cyan] {item['subtask']} [magenta]({item['model']})[/magenta] - Asking model...")
        global table_rows
        table_rows = []
        for i, item in enumerate(plan):
            table_rows.append([f"Task {i+1}", item['model'], "Thinking...", ""])
        tasks = [get_answer(i, item['subtask'], item['model']) for i, item in enumerate(plan)]
        results = [None] * len(plan)
        with Live(build_table(), refresh_per_second=4, console=console) as live:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                idx = result[0]
                live.update(build_table())
                results[idx] = result
        answers_for_summary = "\n".join([
            f"Task: {plan[idx]['subtask']}\nAnswer: {answer}" for idx, _, _, answer in results
        ])
        provider, model = synthesizer_key.split(":", 1)
        synthesizer = router.get_model_instance(provider, model)
        summary_prompt = (
            "Given these answers to sub-tasks, combine and paraphrase them into a single, clear, and concise answer to the original question.\n\n" + answers_for_summary
        )
        console.print("[bold blue]Synthesizing final answer using synthesizer model...[/bold blue]")
        summary_response = await synthesizer.generate(summary_prompt)
        for idx, subtask, model_key, answer in results:
            console.print(Panel(answer, title=f"Task {idx+1} ({model_key})", border_style="blue"))
        console.print("[green]Final Paraphrased Answer:[/green]")
        console.print(Panel(summary_response.content, title="Meta-Agent Synthesis", border_style="green"))
    import asyncio
    asyncio.run(_orchestrate()) 