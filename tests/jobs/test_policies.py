import pytest

from xtl.jobs.policies import BaseCommandPolicy, CommandPolicy, DefaultCommandPolicy, NiceCommandPolicy


class TestCommandPolicy:
    def test_get(self):
        assert isinstance(CommandPolicy.DEFAULT.get(), DefaultCommandPolicy)
        assert isinstance(CommandPolicy.NICE.get(), NiceCommandPolicy)
        assert CommandPolicy.NICE.get() is not CommandPolicy.NICE.get()  # a new instance each time

    def test_equal_to_its_class_and_instances(self):
        assert CommandPolicy.NICE == NiceCommandPolicy
        assert CommandPolicy.NICE == NiceCommandPolicy()
        assert CommandPolicy.NICE != DefaultCommandPolicy
        assert CommandPolicy.NICE != DefaultCommandPolicy()

    def test_strings_and_hash(self):
        assert CommandPolicy('nice') is CommandPolicy.NICE
        assert CommandPolicy.DEFAULT == 'default'
        assert {CommandPolicy.DEFAULT: 1}['default'] == 1


class TestPolicies:
    def test_base_is_abstract(self):
        with pytest.raises(TypeError):
            BaseCommandPolicy()

    def test_default_changes_nothing(self):
        policy = DefaultCommandPolicy()
        assert policy.name == 'default'
        for intercept in (
            policy.intercept_preamble,
            policy.intercept_command,
            policy.intercept_content,
            policy.intercept_postamble,
        ):
            assert intercept('echo "a b"\n') == 'echo "a b"\n'

    def test_nice_prefixes_commands_only(self):
        policy = NiceCommandPolicy()
        assert policy.name == 'nice'
        assert policy.intercept_command('xds_par') == 'nice -n 10 xds_par'
        for intercept in (policy.intercept_preamble, policy.intercept_content, policy.intercept_postamble):
            assert intercept('module load xds') == 'module load xds'

    def test_nice_value(self):
        class Nicer(NiceCommandPolicy):
            nice_value = 19

        assert Nicer().intercept_command('xds_par') == 'nice -n 19 xds_par'
