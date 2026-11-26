from pathlib import Path
import tempfile


class TestTempDirs:
    def test_tempfile_in_session_directory(self):
        assert Path(tempfile.gettempdir()).name.startswith('xtl-tests-')

    def test_pytest_directories_outside(self, tmp_path):
        # Removed with the session's directory otherwise, failed tests' too
        assert Path(tempfile.gettempdir()) not in tmp_path.parents
