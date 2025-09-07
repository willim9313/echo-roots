"""Main CLI entry point for echo-roots."""

import typer
from pathlib import Path

app = typer.Typer(
    name="echo-roots",
    help="Practical taxonomy construction and semantic enrichment framework",
)


@app.command()
def version():
    """Show the version information."""
    from echo_roots import __version__
    
    typer.echo(f"echo-roots version {__version__}")


@app.command()
def status():
    """Show system status and configuration."""
    typer.echo("🌱 Echo-Roots Framework")
    typer.echo("Status: Scaffolding Complete")
    typer.echo("Ready for implementation...")


@app.command()
def init(
    output_dir: str = "./workspace",
):
    """Initialize a new echo-roots workspace."""
    output_path = Path(output_dir)
    typer.echo(f"🚀 Initializing echo-roots workspace at {output_path}")
    
    # Create workspace structure
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (output_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (output_path / "exports").mkdir(parents=True, exist_ok=True)
    (output_path / "logs").mkdir(parents=True, exist_ok=True)
    
    typer.echo("✅ Workspace initialized successfully")
    typer.echo("\nNext steps:")
    typer.echo("1. Place your data files in ./workspace/data/raw/")
    typer.echo("2. Run: echo-roots process --workspace ./workspace")


if __name__ == "__main__":
    app()
