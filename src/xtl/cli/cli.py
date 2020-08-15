import click

from xtl import cfg
from xtl.cli.utils import update_docstring

# Verbosity settings
#
# 0: Errors
# 1: Warnings
# 2: Minimal info
# 3: Excessive info


# Note: If this cli does nothing but forward commands, then debug and verbose are not needed.
@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-d', '--debug', is_flag=True, default=False, help='Run on debug mode.')
@click.option('-v', '--verbose', is_flag=True, default=False, help='Display additional info.')
@click.pass_context
@update_docstring(field='{version}', value=cfg['xtl']['version'].value)
def cli_main(ctx: click.core.Context, debug: bool, verbose: bool):
    """
    \b
    XTL: Crystallographic Tools Library
    (version {version})

    \f
    :param ctx: Context object
    :param debug: Debug mode
    :param verbose: Verbose level
    :returns:
    """
    if debug:
        click.secho(f'Debug mode is on.', fg='magenta')
    if debug and verbose:
        click.secho(f'Verbose mode is on.', fg='magenta')

    # Note: Passing parameters in ctx.obj might not be necessary after all, since each cli and its subcommands accept
    #  the same options (i.e. debug, verbose).
    ctx.ensure_object(dict)  # creates ctx.obj if function is called without obj={}
    ctx.obj = {
        'debug': debug,
        'verbose': verbose
    }


@cli_main.command(short_help='Utilities for manipulating GSAS2 .gpx files.',
                  context_settings=dict(ignore_unknown_options=True, allow_interspersed_args=False))
@click.argument('argopts', nargs=-1, type=click.UNPROCESSED)  # consumes all arguments and options
@click.pass_context
def gsas2(ctx: click.core.Context, argopts):
    """
    \b
    Alias for command: gsas2.
    For help try: gsas2 -h
    \f
    :param ctx:
    :param argopts:
    :return:
    """

    if not argopts:
        # from xtl.cli.gsas2_commands import cli_gsas
        # print(cli_gsas.get_help(ctx))
        print(ctx.get_help())
        return

    # Call the relevant command with all arguments and options
    from subprocess import call
    cmd = ['gsas2'] + list(argopts)
    call(cmd)


if __name__ == '__main__':
    cli_main(obj={})
