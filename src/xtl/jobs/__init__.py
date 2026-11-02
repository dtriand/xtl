from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xtl.common.lazy_exports import lazy_getattr, lazy_dir

__all__ = [
    'BatchFile',
    'JobConfig', 'BatchJobConfig', 'SteppedJobConfig',
    'Job', 'BatchJob', 'SteppedJob',
    'CommandPolicy', 'CommandPolicyType',
    'JobPool', 'SimplePool', 'AsyncPool', 'ThreadedPool', 'MultiprocessPool',
    'Shell', 'ShellType', 'BashShell', 'CmdShell', 'PowerShell',
    'ComputeSite', 'ComputeSiteType', 'LocalSite', 'ModulesSite',
    'StepSpec',
]

if TYPE_CHECKING:
    from .batchfiles import BatchFile
    from .config import JobConfig, BatchJobConfig, SteppedJobConfig
    from .jobs import Job, BatchJob, SteppedJob
    from .policies import CommandPolicy, CommandPolicyType
    from .pools import JobPool, SimplePool, AsyncPool, ThreadedPool, MultiprocessPool
    from .shells import Shell, ShellType, BashShell, CmdShell, PowerShell
    from .sites import ComputeSite, ComputeSiteType, LocalSite, ModulesSite
    from .steps import StepSpec


# symbol -> (module, attr)
_EXPORTS = {
    'BatchFile':         ('.batchfiles', 'BatchFile'),
    'JobConfig':         ('.config',     'JobConfig'),
    'BatchJobConfig':    ('.config',     'BatchJobConfig'),
    'SteppedJobConfig':  ('.config',     'SteppedJobConfig'),
    'Job':               ('.jobs',       'Job'),
    'BatchJob':          ('.jobs',       'BatchJob'),
    'SteppedJob':        ('.jobs',       'SteppedJob'),
    'CommandPolicy':     ('.policies',   'CommandPolicy'),
    'CommandPolicyType': ('.policies',   'CommandPolicyType'),
    'JobPool':           ('.pools',      'JobPool'),
    'SimplePool':        ('.pools',      'SimplePool'),
    'AsyncPool':         ('.pools',      'AsyncPool'),
    'ThreadedPool':      ('.pools',      'ThreadedPool'),
    'MultiprocessPool':  ('.pools',      'MultiprocessPool'),
    'Shell':             ('.shells',     'Shell'),
    'ShellType':         ('.shells',     'ShellType'),
    'BashShell':         ('.shells',     'BashShell'),
    'CmdShell':          ('.shells',     'CmdShell'),
    'PowerShell':        ('.shells',     'PowerShell'),
    'ComputeSite':       ('.sites',       'ComputeSite'),
    'ComputeSiteType':   ('.sites',       'ComputeSiteType'),
    'LocalSite':         ('.sites',       'LocalSite'),
    'ModulesSite':       ('.sites',       'ModulesSite'),
    'StepSpec':          ('.steps',       'StepSpec'),
}


def __getattr__(name: str) -> Any:
    return lazy_getattr(__name__, _EXPORTS, globals(), name)


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
