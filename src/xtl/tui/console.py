from typing import TYPE_CHECKING

import rich.console

if TYPE_CHECKING:
    from xtl.jobs import JobPool
    from xtl.tui.live_pool import LivePool


class ConsoleIO(rich.console.Console):

    def __init__(self, *args, verbose: int = 0, debug: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.debug = debug

    def get_pool(self, pool_type: 'JobPool | str' = None, max_jobs: int = 1) -> 'LivePool':
        from xtl.tui.live_pool import LivePool

        return LivePool(
            pool_type=pool_type,
            max_jobs=max_jobs,
            console=self,
        )

