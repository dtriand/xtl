from os import get_terminal_size
from datetime import datetime
import warnings

import click

from xtl import cfg
from xtl.cli.utils import GroupWithCommandOptions, OutputQueue, update_docstring

from xtl.pdbapi import Client
from xtl.pdbapi.queries import free_text, has_uniprot_id
from xtl.pdbapi.search.options import ResultsContentType
from xtl.pdbapi.data.options import DataService
from xtl.pdbapi.schema import SearchSchema, DataSchema


@click.group(cls=GroupWithCommandOptions, context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-d', '--debug', is_flag=True, default=False, help='Run on debug mode.')
@click.option('-v', '--verbose', count=True, type=click.IntRange(0, 3, clamp=True), help='Display additional info.')
@click.pass_context
def rest_search_group(ctx: click.core.Context, debug: bool, verbose: bool):
    if debug:
        click.secho(f'Debug mode is on.', fg='magenta')
    if debug and verbose:
        click.secho(f'Verbosity set to {verbose}.', fg='magenta')

    ctx.ensure_object(dict)
    ctx.obj = {
        'debug': debug,
        'verbose': verbose
    }


@rest_search_group.command(short_help='Perform a search.')
@click.pass_context
@click.argument('text', nargs=1)
@click.option('--all' ,'all_' , is_flag=True, default=False, help='Return all entries.')
@click.option('--csm', is_flag=True, default=False, help='Include Computed Structure Models in results.')
@click.option('-t', '--title', is_flag=True, default=False, help='Return entry titles.')
def search(ctx: click.core.Context, text: str, all_: bool, csm: bool, title: bool):
    """
    Perform a search on the PDB API

    \f
    :param ctx:
    :param text:
    :param all_:
    :param csm:
    :param title:
    :return:
    """
    debug: bool = ctx.obj["debug"]
    verbose: bool = ctx.obj["verbose"]

    q = OutputQueue(debug)
    api = Client()

    if all_:
        api.request_options.return_all_hits = True
    if csm:
        api.request_options.results_content_types = [ResultsContentType.EXPERIMENTAL, ResultsContentType.COMPUTATIONAL]

    query = free_text(text)
    if debug:
        click.secho(f"Query JSON: {query.to_dict()}", fg='magenta')

    results = api.search(query)
    pdbs = results.pdbs

    terminal_size = 80
    try:
        terminal_size = get_terminal_size()[0]
    except OSError:
        pass

    if title:
        attrd = DataSchema(verbose=True, service=DataService.ENTRY)
        data = api.data(pdbs, attributes=[attrd.rcsb_id, attrd.struct.title])
        table, titles = data.tabulate('entries')
        q.append_table(table=table, tablefmt='plain', maxcolwidths=[4, terminal_size-6], headers=titles)
        q.pop_from_queue()
        q.print()
    else:
        entries_per_line = terminal_size // (4+1)
        lines = -(len(pdbs) // -entries_per_line)  # equivalent to math.ceil()
        for l in range(lines):
            click.secho(" ".join(pdbs[l*entries_per_line:(l+1)*entries_per_line]))

    click.secho(f"Total number of entries: {len(pdbs)}", nl=False)
    if all_:
        click.secho()
    else:
        click.secho(f'/{results.total_count} [Use --all to get all entries]')


@rest_search_group.command(short_help='Inspect the RCSB API schema.')
@click.pass_context
@click.option('-t', '--test', is_flag=True, default=False)
@click.option('--search', 'search_only', is_flag=True, default=False)
@click.option('--data', 'data_only', is_flag=True, default=False)
def schema(ctx: click.core.Context, test: bool, search_only: bool, data_only: bool):
    """
    Inspect the RCSB API for proper buildup.

    \f
    :param ctx:
    :param test:
    :param search_only:
    :param data_only:
    :return:
    """
    debug: bool = ctx.obj["debug"]
    verbose: bool = ctx.obj["verbose"]
    ctx.obj['test'] = test

    schemas = {
        'search': [SearchSchema, []],
        'data.entry': [DataSchema, [DataService.ENTRY]],
        'data.polymer_entity': [DataSchema, [DataService.POLYMER_ENTITY]],
        'data.branched_entity': [DataSchema, [DataService.BRANCHED_ENTITY]],
        'data.nonpolymer_entity': [DataSchema, [DataService.NON_POLYMER_ENTITY]],
        'data.polymer_instance': [DataSchema, [DataService.POLYMER_INSTANCE]],
        'data.branched_instance': [DataSchema, [DataService.BRANCHED_INSTANCE]],
        'data.nonpolymer_instance': [DataSchema, [DataService.NON_POLYMER_INSTANCE]],
        'data.assembly': [DataSchema, [DataService.ASSEMBLY]],
        'data.chem_comp': [DataSchema, [DataService.CHEMICAL_COMPONENT]],
    }

    click.secho('Hi from here!')
    if search_only:
        ctx.invoke(schema_interrogate, schema_cls=SearchSchema)
    elif data_only:
        ctx.invoke(schema_interrogate, schema_cls=DataSchema)
    else:
        ctx.invoke(schema_interrogate, schema_cls=SearchSchema)
        ctx.invoke(schema_interrogate, schema_cls=DataSchema)


@click.command()
@click.pass_context
def schema_interrogate(ctx: click.core.Context, schema_cls: SearchSchema or DataSchema):
    test = ctx.obj['test']
    debug = ctx.obj['debug']
    if test:
        click.secho(f'Testing {schema_cls.__name__}...', fg='cyan')
        t1 = datetime.now()
    from xtl.pdbapi.data.options import DataService
    attr = schema_cls(DataService.CHEMICAL_COMPONENT)
    if test:
        t2 = datetime.now()
        click.secho(f'Fetched schema v.{attr.schema_version} from {attr._schema_url} and processed in '
                    f'{round((t2 - t1).total_seconds(), 3)} sec', fg='cyan')
        click.secho(f'Schema contains {len(attr.__dict__)} entries', fg='cyan')
        click.secho(f'Number of unparsed entries: {len(attr._dubious_entries)} ', fg='cyan')
        if debug:
            click.secho('\n'.join(f"    {obj}" for obj in attr._dubious_entries), fg='magenta')
    click.secho(f'A: {len(attr.attributes)} + AG: {len(attr.attribute_groups)} = {len(attr.attributes) + len(attr.attribute_groups)}')
    click.secho(f'    {len(attr.__dict__) - (len(attr.attributes) + len(attr.attribute_groups))} additional attributes\n')

    # from xtl.pdbapi.attributes import _Attribute, _AttributeGroup
    # from pprint import pprint
    # for i, attribute in enumerate(attr.attributes):
    #     a = getattr(attr, attribute)
    #     if issubclass(a.__class__, _Attribute):
    #         click.secho(f'{i:04d}   {a}')
    #     elif issubclass(a.__class__, _AttributeGroup):
    #         click.secho(f'{i:04d}', fg='cyan')
    #         pprint(a)
    #         print('\n')


@click.command(cls=click.CommandCollection, sources=[rest_search_group], invoke_without_command=True,
               context_settings=dict(help_option_names=['-h', '--help']))
@click.pass_context
@click.option('-d', '--debug', is_flag=True, default=False, help='Run on debug mode.')
@click.option('-v', '--verbose', count=True, type=click.IntRange(0, 3, clamp=True), help='Display additional info.')
@update_docstring(field='{version}', value=cfg['xtl']['version'].value)
def cli_pdbapi(ctx: click.core.Context, debug: bool, verbose: int):
    """
    \b
    Access to RCSB PDB Search and Data APIs.
    Installed by xtl (version {version})

    \f
    :param ctx:
    :param debug:
    :param verbose:
    :return:
    """
    if not ctx.invoked_subcommand:
        print(ctx.get_help())


if __name__ == '__main__':
    cli_pdbapi(obj={})


