from typing import Any

import numpy as np

from xtl.common.options import Option
from xtl.jobs.jobs import Job
from xtl.jobs.config import JobConfig
from xtl.math.clustering import find_maximal_cliques


class CliqueSearchJobConfig(JobConfig):

    data: np.ndarray = \
        Option(
            ...,
            desc='Boolean adjacency matrix with False on the diagonal'
        )

    max_cliques: int | None = \
        Option(
            default=None,
            desc='Maximum number of cliques to return, or None for all'
        )


class CliqueSearchJob(Job[CliqueSearchJobConfig]):

    async def _execute(self) -> Any | None:

        adjacency = self.config.data
        cliques = find_maximal_cliques(adjacency)
        if self.config.max_cliques is not None:
            cliques = cliques[:min(self.config.max_cliques, len(cliques))]
        return cliques