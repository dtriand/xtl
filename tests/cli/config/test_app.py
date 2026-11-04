def test_no_args_shows_help(cli):
    result = cli()
    assert result.exit_code == 2  # click >= 8.2 exits with 2 for no_args_is_help
    assert 'Usage' in result.output
    for command in ('where', 'show', 'create', 'validate'):
        assert command in result.output
