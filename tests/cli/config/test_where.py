import json

from tests.cli.config.conftest import INVALID, VALID, WARNINGS


class TestWhere:
    @staticmethod
    def _json(cli) -> dict:
        result = cli('where', '--json')
        assert result.exit_code == 0, result.output
        return json.loads(result.stdout)

    @staticmethod
    def _statuses(data: dict) -> dict:
        return {loc['name']: loc['status'] for loc in data['locations']}

    def test_no_files(self, cli, locs):
        data = self._json(cli)
        assert data['active'] is None
        assert data['will_create'] == 'user'
        assert self._statuses(data) == {
            'env': 'not set',
            'local': 'missing',
            'user': 'will create',
            'global': 'missing',
        }
        assert [loc['name'] for loc in data['locations']] == ['env', 'local', 'user', 'global']
        assert data['locations'][0]['path'] is None
        assert data['locations'][1]['path'] == str(locs.local)

        result = cli('where')
        assert result.exit_code == 0
        assert 'XTL_CONFIG not set' in result.stdout
        assert 'One will be created upon first use' in result.stdout

    def test_will_create_env(self, cli, locs):
        locs.set_env()
        data = self._json(cli)
        assert data['will_create'] == 'env'
        assert self._statuses(data)['env'] == 'will create'
        assert self._statuses(data)['user'] == 'missing'

    def test_nothing_writable(self, cli, monkeypatch):
        monkeypatch.setattr('xtl.cli.config.where._is_writable', lambda path: False)
        data = self._json(cli)
        assert data['active'] is None
        assert data['will_create'] is None
        assert 'will create' not in self._statuses(data).values()
        assert 'no writable location' in cli('where').stdout

    def test_active_and_shadowed(self, cli, locs):
        locs.write(locs.local, VALID)
        locs.write(locs.global_, VALID)
        data = self._json(cli)
        assert data['active'] == 'local'
        assert data['will_create'] is None
        assert self._statuses(data) == {'env': 'not set', 'local': 'active', 'user': 'missing', 'global': 'shadowed'}

    def test_env_active(self, cli, locs):
        locs.write(locs.set_env(), VALID)
        locs.write(locs.user, VALID)
        data = self._json(cli)
        assert data['active'] == 'env'
        assert self._statuses(data) == {'env': 'active', 'local': 'missing', 'user': 'shadowed', 'global': 'missing'}

    def test_env_directory(self, cli, locs):
        env_dir = locs.root / 'env'
        env_dir.mkdir()
        locs.set_env(env_dir)
        data = self._json(cli)
        assert data['locations'][0] == {
            'name': 'env',
            'path': str(env_dir),
            'exists': False,
            'creatable': False,
            'writable': None,
            'status': 'ignored',
            'valid': None,
            'errors': [],
            'warnings': [],
        }
        assert self._statuses(data)['user'] == 'will create'
        assert 'not a file' in cli('where').stdout

    def test_valid_column(self, cli, locs):
        locs.write(locs.local, VALID)
        locs.write(locs.user, WARNINGS)
        locs.write(locs.global_, INVALID)
        locations = {loc['name']: loc for loc in self._json(cli)['locations']}
        assert locations['env']['valid'] is None
        assert (locations['local']['valid'], len(locations['local']['warnings'])) == (True, 0)
        assert (locations['user']['valid'], len(locations['user']['warnings'])) == (True, 2)
        assert locations['global']['valid'] is False
        assert locations['global']['errors'][0].startswith('units.temperature: ')

        result = cli('where')
        assert '2 warnings' in result.stdout
        assert 'is invalid' not in result.stdout  # the active file is valid

    def test_invalid_active(self, cli, locs):
        locs.write(locs.local, INVALID)
        result = cli('where')
        assert result.exit_code == 0
        assert 'Active config file local is invalid' in result.stdout
        assert 'xtl.config validate' in result.stdout
