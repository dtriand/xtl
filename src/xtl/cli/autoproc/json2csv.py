import json
from pathlib import Path

import pandas as pd
import typer

import xtl.cli.autoproc.cli_utils as apu
from xtl.cli.cliio import Console, epilog


app = typer.Typer()


@app.command('json2csv', short_help='Create summary CSV from many JSON files', epilog=epilog)
def cli_autoproc_json_to_csv(
        datasets_file: Path = typer.Argument(metavar='<DATASETS.CSV>', help='Path to a CSV file containing dataset names'),
        out_dir: Path = typer.Option(Path('./'), '-o', '--out-dir', help='Path to the output directory'),
        debug: bool = typer.Option(False, '--debug', help='Print debug information')
):
    '''
    Reads a datasets_output.csv file and collects all the xtl_autoPROC.json files from the job directories to create a
    single summary CSV file.

    Note that the CSV file must contain the following columns: 'job_dir', 'sweep_id', 'autoproc_id'.
    '''
    cli = Console()
    # Check if csv file exists
    if not datasets_file.exists():
        cli.print(f'File {datasets_file} does not exist', style='red')
        raise typer.Abort()

    cli.print(f'Parsing dataset names from {datasets_file}... ')
    datasets = apu.parse_csv(datasets_file, extra_headers=['job_dir', 'sweep_id', 'autoproc_id'])
    cli.print(f'Found {len(datasets["extra"]["job_dir"])} datasets')

    data = []
    if debug:
        cli.print('# dataset_subdir, rename_dataset_subdir, autoproc_dir', style='magenta')
    for i, (j_dir, sweep_id, autoproc_id) in enumerate(zip(datasets["extra"]['job_dir'], datasets["extra"]['sweep_id'],
                                                           datasets["extra"]['autoproc_id'])):
        if j_dir:
            j = Path(j_dir) / 'xtl_autoPROC.json'
            if j.exists():
                d = {
                    'id': i,
                    'sweep_id': sweep_id,
                    'job_dir': j.parent.as_uri(),
                    'autoproc_id': autoproc_id
                }
                d.update(json.loads(j.read_text()))
                data.append(d)

    cli.print(f'Found {len(data)} JSON files')
    if not data:
        return typer.Exit(code=0)

    df = pd.json_normalize(data)
    df = apu.df_stringify(df)
    csv_file = out_dir / f'xtl_autoPROC_summary.csv'
    df.to_csv(csv_file, index=False)
    cli.print(f'Wrote summary to {csv_file}')