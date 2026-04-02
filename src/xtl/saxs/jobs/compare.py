import asyncio
import copy
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np

from xtl import settings
from xtl.common.options import Option
from xtl.exceptions.base import StderrError
from xtl.jobs.jobs import Job
from xtl.jobs.config import JobConfig, BatchConfig, JobStepsConfig
from xtl.jobs.pools import JobPool
from xtl.math.uuid import UUIDFactory
from xtl.saxs.jobs.atsas import DatcmpBatchJob, DatcmpBatchJobConfig
from xtl.saxs.jobs.atsas_utils import DatcmpOptions, DatcmpMode, DatcmpTest, \
    DatcmpAdjustment


uuid = UUIDFactory()


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

    def _execute(self) -> Any | None:

        adjacency = self.config.data
        all_cliques = []

        def bron_kerbosch(R, P, X):
            if not P and not X:
                all_cliques.append(sorted(R[:]))
                return

            # pruning: can't beat current best
            if all_cliques and len(R) + len(P) <= len(all_cliques[0]):
                return

            pivot = max(P + X, key=lambda u: np.sum(adjacency[u][P]))
            candidates = [v for v in P if not adjacency[pivot, v]]

            for v in candidates:
                bron_kerbosch(
                    R + [v],
                    [u for u in P if adjacency[v, u]],
                    [u for u in X if adjacency[v, u]],
                )
                P.remove(v)
                X.append(v)

        bron_kerbosch([], list(range(len(adjacency))), [])
        return sorted(all_cliques, key=len, reverse=True)


class SAXSCompareJobStepsConfig(JobStepsConfig, total=False):
    datcmp_batch: DatcmpBatchJobConfig


class SAXSCompareJobConfig(JobConfig):
    files: list[Path] = \
        Option(
            ...,
            desc='List of data files to compare',
            min_length=2,
            path_exists=True
        )
    steps: SAXSCompareJobStepsConfig = \
        Option(
            default_factory=lambda: SAXSCompareJobStepsConfig(
                datcmp_batch=DatcmpBatchJobConfig(
                    options=DatcmpOptions(
                        test=DatcmpTest.CORMAP,
                        adjust=DatcmpAdjustment.FWER,
                        alpha=0.01,
                        mode=DatcmpMode.PAIRWISE,
                        format='CSV'
                    ),
                )
            ),
            desc='Configuration for each step of the job.'
        )


@dataclass
class SAXSComparisonMatrix:
    correlation_length: np.ndarray
    p_value: np.ndarray
    adjusted_p_value: np.ndarray

@dataclass
class SAXSComparisonResults:
    cliques: list[list[int]]
    matrix: SAXSComparisonMatrix
    datasets: list[Path]


class SAXSCompareJob(Job[SAXSCompareJobConfig]):
    """
    Job to compare SAXS datasets using datcmp.
    """

    async def _execute(self):
        no_steps = len(self.config.steps_list)

        ############################
        # Step 1: datcmp batch job #
        ############################
        i = 0
        step = self.config.steps_list[i]
        self.logger.info('Executing step %(i)d/%(n)d: %(steps)s', {'i': i + 1, 'n': no_steps, 'steps': step})

        # Prepare batch job
        self.logger.debug(f'Preparing {DatcmpBatchJob.__name__}')
        batch_config: DatcmpBatchJobConfig = self.config.steps[step]
        batch_config.options.format = 'CSV'  # Ensure CSV format for parsing
        batch_job = DatcmpBatchJob.with_config(
            job_id=f'{self.job_id}.{i + 1}',
            batch_args=[str(f) for f in self.config.files],
            config=batch_config,
            **{
                'ATSAS_KWARGS': batch_config.get_args(),
            }
        )

        # Run batch job
        self.logger.info('Running datcmp batch job in: %(dir)s', {'dir': batch_config.job_directory})
        results = await batch_job.run()
        self.logger.debug(f'Batch job completed')
        if results.error:
            raise RuntimeError(f'datcmp batch job failed with error: {results.error}')
        else:
            if stderr := results.data.get('stderr', None):
                for line in stderr.splitlines():
                    # Skip known warnings from datcmp that do not indicate a failure
                    if 'warning: data shortened to common range' in line.lower():
                        continue
                    elif 'warning: data rebinned to common grid' in line.lower():
                        continue
                    else:
                        raise StderrError('datcmp batch job failed', stderr=line)

        # Parse results
        no_files = len(self.config.files)
        data = SAXSComparisonMatrix(
            correlation_length=np.zeros((no_files, no_files)),
            p_value=np.ones((no_files, no_files)),
            adjusted_p_value=np.ones((no_files, no_files))
        )
        if stdout := results.data.get('stdout', None):
            for i, line in enumerate(stdout.splitlines()):
                if not line.startswith('Correlation Map test'):
                    continue
                parts = line.split(',')
                if len(parts) != 6:
                    continue
                try:
                    index_1 = int(parts[1]) - 1
                    index_2 = int(parts[2]) - 1
                    correlation_length = float(parts[3])
                    p_value = float(parts[4])
                    adjusted_p_value = float(parts[5])
                except ValueError:
                    self.logger.warning(f'Failed to parse line {i + 1} of datcmp output: {line}')
                    self.logger.warning(f'   parts: {parts}')
                    continue

                data.correlation_length[index_1, index_2] = correlation_length
                data.p_value[index_1, index_2] = p_value
                data.adjusted_p_value[index_1, index_2] = adjusted_p_value

            # Symmetrize matrices since datcmp only outputs upper triangle
            data.correlation_length = np.triu(data.correlation_length) + np.triu(data.correlation_length, k=1).T
            data.p_value = np.triu(data.p_value) + np.triu(data.p_value, k=1).T
            data.adjusted_p_value = np.triu(data.adjusted_p_value) + np.triu(data.adjusted_p_value, k=1).T

        #################
        # Clique search #
        #################

        adjacent = data.p_value >= self.config.steps['datcmp_batch'].options.alpha
        np.fill_diagonal(adjacent, False)  # No self-connections
        cliques = self.find_cliques(adjacent)

        return SAXSComparisonResults(
            cliques=cliques,
            matrix=data,
            datasets=self.config.files
        )

    @staticmethod
    def find_cliques(adjacency: np.ndarray) -> list[list[int]]:
        """
        Find all maximal cliques in an undirected graph using Bron-Kerbosch with pivoting and pruning, sorted by size
        descending.

        :param adjacency: Symmetric boolean (N, N) adjacency matrix with False on the diagonal.
        :return: List of cliques, each a sorted list of node indices, ordered largest first.
        """
        all_cliques = []

        def bron_kerbosch(R, P, X):
            if not P and not X:
                all_cliques.append(sorted(R[:]))
                return

            # pruning: can't beat current best
            if all_cliques and len(R) + len(P) <= len(all_cliques[0]):
                return

            pivot = max(P + X, key=lambda u: np.sum(adjacency[u][P]))
            candidates = [v for v in P if not adjacency[pivot, v]]

            for v in candidates:
                bron_kerbosch(
                    R + [v],
                    [u for u in P if adjacency[v, u]],
                    [u for u in X if adjacency[v, u]],
                )
                P.remove(v)
                X.append(v)

        bron_kerbosch([], list(range(len(adjacency))), [])
        return sorted(all_cliques, key=len, reverse=True)

    def _cleanup_temp_files(self):
        """
        Clean up temporary files created during the job execution.
        This method is called when the job is finished or if an error occurs.
        """
        if not settings.automate.keep_temp:
            shutil.rmtree(self.config.job_directory, ignore_errors=True)
            self.logger.debug('Temporary files cleaned up.')
        else:
            from xtl.common.os import chmod_recursively
            self.logger.debug('Updating permissions for temporary files')
            chmod_recursively(
                self.config.job_directory,
                files_permissions=settings.automate.permissions.files,
                directories_permissions=settings.automate.permissions.directories
            )
            self.logger.debug('Temporary files retained due to keep_temp setting.')