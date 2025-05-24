import typer

from xtl.config.settings import XTLSettings


app = typer.Typer()


@app.command('generate')
def cli_config_generate():
    if XTLSettings.global_config.exists():
        typer.echo(f'Global config file already exists: {XTLSettings.global_config}')
        return typer.Exit()

    settings = XTLSettings()
    settings.to_toml(XTLSettings.global_config)
    typer.echo(f'Global config file generated: {XTLSettings.global_config}')
    return typer.Exit()