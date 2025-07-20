from typer import Typer

app = Typer()

@app.command()
def list_local_models():
    """List all available local models detected by Ollama."""
    import subprocess
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        lines = result.stdout.strip().split("\n")
        if len(lines) <= 1:
            print("No local models found in Ollama.")
            return
        # Try to use Rich for a nice table
        try:
            from rich.table import Table
            from rich.console import Console
            console = Console()
            table = Table(title="Available Local Models (Ollama)")
            table.add_column("Model Name", style="cyan")
            table.add_column("ID", style="magenta")
            table.add_column("Size", style="green")
            table.add_column("Last Modified", style="yellow")
            for line in lines[1:]:
                parts = [p.strip() for p in line.split() if p.strip()]
                if len(parts) >= 4:
                    name, id_, size, *modified = parts
                    table.add_row(name, id_, size, " ".join(modified))
                elif len(parts) >= 3:
                    name, id_, size = parts[:3]
                    table.add_row(name, id_, size, "")
            console.print(table)
        except ImportError:
            # Fallback to plain text
            print("Available Local Models (Ollama):")
            print(f"{'Model Name':20} {'ID':12} {'Size':8} {'Last Modified'}")
            for line in lines[1:]:
                parts = [p.strip() for p in line.split() if p.strip()]
                if len(parts) >= 4:
                    name, id_, size, *modified = parts
                    print(f"{name:20} {id_:12} {size:8} {' '.join(modified)}")
                elif len(parts) >= 3:
                    name, id_, size = parts[:3]
                    print(f"{name:20} {id_:12} {size:8}")
    except Exception as e:
        print(f"Failed to list local models: {e}") 