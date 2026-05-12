from os import chmod
from pathlib import Path
from typing import Any

from xtl.common.compatibility import OS_POSIX
from xtl.common.options import Option
from xtl.common.os import FilePermissions
from xtl.jobs import Job, JobConfig


class CreateDirectoryJobConfig(JobConfig):
    directory: Path = \
        Option(
            ...,
            desc='Path to directory to create',
            path_exists=False,
        )
    permissions: FilePermissions | None = \
        Option(
            default=None,
            desc='Permissions for the created directory'
        )


class CreateDirectoryJob(Job[CreateDirectoryJobConfig]):
    """
    Create a directory if it doesn't exist and optionally update its
    permissions.
    """

    async def _execute(self) -> dict[str, Any] | None:
        if not (config := self.config):
            raise ValueError(f'{CreateDirectoryJob.__name__} requires a config object')

        # Create directory if it doesn't exist
        if not config.directory.exists():
            try:
                # Ensure only one process creates the directory
                async with self.lock():
                    config.directory.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
            except OSError as e:
                raise RuntimeError(f'Failed to create directory {config.directory}') from e

        # Skip permissions update on non-POSIX systems
        if not OS_POSIX:
            return {'directory': config.directory, 'permissions': None}

        # Update directory permissions
        if (permissions := config.permissions) is not None:
            try:
                # Ensure only one process updates permissions
                async with self.lock():
                    chmod(config.directory, permissions.decimal)
            except OSError as e:
                raise RuntimeError(f'Failed to set permissions for {config.directory}') from e

            # Check that permission were actually correctly set
            if FilePermissions.from_path(config.directory) != permissions:
                raise RuntimeError(f'Permissions for {config.directory} were not set to {permissions.octal[2:]}, '
                                   f'check permissions of parent directories and umask settings')
            return {'directory': config.directory, 'permissions': permissions}
        else:
            return {'directory': config.directory, 'permissions': None}
