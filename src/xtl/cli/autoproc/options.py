import rich.box
import typer

from xtl.cli.cliio import Console, epilog
import xtl.cli.autoproc.cli_utils as apu


app = typer.Typer()


@app.command('options', help='Show available autoPROC configuration options', epilog=epilog)
def cli_autoproc_options():
    cli = Console()

    table_kwargs = {
        'title_style': 'bold italic white on cornflower_blue',
        'box': rich.box.HORIZONTALS,
        'expand': True
    }

    cli.print('The following parameters can be passed as headers in the [b i u]datasets.csv[/b i u] file.')
    cli.print()
    cli.print_table(table=apu.get_attributes_dataset(),
                    headers=['XTL parameter', 'Type', 'Description'],
                    column_kwargs=[
                        {'style': 'cornflower_blue'},
                        {'style': 'italic'},
                        {'style': 'bright_black'}
                    ],
                    table_kwargs=table_kwargs | {'title': 'Dataset options',
                                                 # 'caption': 'An additional \'dataset_group\' parameter can be added to '
                                                 #            'the [u]datasets.csv[/u] file to process and merge multiple'
                                                 #            ' datasets together ([i]e.g.[/i] multi-sweep data)'
                                                 }
                    )
    cli.print()
    cli.print_table(table=apu.get_attributes_config(),
                    headers=['XTL parameter', 'autoPROC parameter', 'Type', 'Description'],
                    column_kwargs=[
                        {'style': 'cornflower_blue'},
                        {'style': 'medium_orchid'},
                        {'style': 'italic'},
                        {'style': 'bright_black'}
                    ],
                    table_kwargs=table_kwargs | {'title': 'autoPROC configuration options',
                                                 'caption': 'Parameters in purple are the equivalent autoPROC '
                                                            'parameters that will be passed to the process command. '
                                                            'Additional parameters can be passed as a dictionary in '
                                                            'the \'extra_params\' argument. A full list of the '
                                                            'available autoPROC parameters can be found '
                                                            '[link=https://www.globalphasing.com/autoproc/manual/appendix1.html]'
                                                            'here[/link].'}
                    )

    return typer.Exit(code=0)