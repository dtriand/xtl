import asyncio
import copy
from dataclasses import dataclass
from pathlib import Path
import re
import shutil

import numpy as np

from xtl import settings
from xtl.common.options import Option
from xtl.jobs.jobs import Job
from xtl.jobs.config import JobConfig, BatchConfig
from xtl.jobs.pools import JobPool
from xtl.math.uuid import UUIDFactory
from xtl.saxs.jobs.atsas import DatcmpJob, DatcmpJobConfig
from xtl.saxs.jobs.atsas_utils import DatcmpOptions, DatcmpMode, DatcmpTest, \
    DatcmpAdjustment


uuid = UUIDFactory()


class SAXSCompareTreeJobConfig(JobConfig):
    files: list[Path] = \
        Option(
            ...,
            desc='List of data files to compare',
            min_length=2,
            path_exists=True)
    datcmp: DatcmpOptions = \
        Option(
            default_factory=lambda: DatcmpOptions(
                test=DatcmpTest.CORMAP,
                adjust=DatcmpAdjustment.FWER,
                alpha=0.01,
                format='FULL'
            ),
            desc='Options for `datcmp`')
    max_jobs: int = \
        Option(
            10, ge=0,
            desc='Maximum number of parallel jobs to run. '
                 'If set to 0, all jobs will be run in parallel.'
        )


@dataclass
class SAXSCompareTreeMatrix:
    correlation_length: np.array
    p_value: np.array
    adjusted_p_value: np.array

@dataclass
class SAXSCompareTreeResults:
    lineages: list[list[Path]]
    matrix: SAXSCompareTreeMatrix
    datasets: list[Path]


class SAXSCompareTreeJob(Job[SAXSCompareTreeJobConfig]):
    """
    Job to compare SAXS datasets using datcmp.
    """

    _job_prefix = 'saxs_compare_tree'
    _dependencies = {'atsas'}

    async def _execute(self):
        # Check for config
        if self.config is None:
            self.logger.error('Job is not configured.')
            raise ValueError('Job is not configured.')

        # Generate all permutations first
        permutations = self._get_permutations(self.config.files)

        # Run all datcmp jobs in parallel
        with JobPool(max_jobs=self.config.max_jobs or len(permutations),
                     logger_config=self._logging_config) as pool:

            # Create configs for each permutation
            job_ids, configs = [], []
            for i, files in enumerate(permutations):
                job_id = uuid.random(settings.automate.job_digits)
                config = DatcmpJobConfig(
                    job_directory=self.config.job_directory,
                    files=files,
                    options=self.config.datcmp,
                    batch=BatchConfig(
                        compute_site=self.config.batch.compute_site,
                        filename=f'datcmp_{job_id}',
                        dependencies=self.config.batch.dependencies,
                        permissions=self.config.batch.permissions
                    ),
                )
                config.batch.shell = self.config.batch.shell
                config.batch._strict_resolution = self.config.batch._strict_resolution

                configs.append(config)
                job_ids.append(f'{DatcmpJob.__name__}|{i+1}')

            # Submit jobs to the pool
            jobs = pool.submit(DatcmpJob, configs=configs, job_ids=job_ids)
            self.logger.debug('Running datcmp for %d permutations in batches of %d',
                              len(permutations), pool._max_jobs)
            results = await pool.launch()

        # Check if any of the datcmp jobs failed
        for r in results:
            if not r.data:
                self._cleanup_temp_files()
                raise RuntimeError('A datcmp job failed or returned no data.')

        # Parse logs
        self.logger.debug('Parsing datcmp logs')
        matrix = await self._parse_matrix(results[0].data)
        tasks = [self._parse_lineage(r.data) for r in results if r is not None]
        lineages = await asyncio.gather(*tasks)

        # Get the unique lineages
        unique_lineages = []
        for lineage in lineages:
            lineage = sorted(lineage)
            if lineage not in unique_lineages:
                unique_lineages.append(lineage)

        # Cleanup temporary files
        self._cleanup_temp_files()

        # Return results
        return SAXSCompareTreeResults(
            lineages=unique_lineages,
            matrix=SAXSCompareTreeMatrix(
                correlation_length=np.array(matrix['correlation_length']),
                p_value=np.array(matrix['p_value']),
                adjusted_p_value=np.array(matrix['adjusted_p_value'])
            ),
            datasets=permutations[0]
        )

    @staticmethod
    def _get_permutations(files: list[Path]) -> list[list[Path]]:
        """
        Generate all permutations of the given list of files.

        :param files: List of Paths representing the datasets to compare.
        """
        if len(files) <= 2:
            return [files]

        permutations = []
        for i in range(len(files)):
            permutations.append(files[i:] + files[:i])
        return permutations

    async def _parse_lineage(self, log: str) -> list[Path]:
        """
        Parse the log output from datcmp to extract a list of datasets that passed the
        test.

        :param log: The log output from the datcmp command.
        :return: A list of Paths representing the datasets that passed the test.
        """
        if not log:
            self.logger.warning('Log is empty, no lineage to parse.')
            return []

        starred_regex = re.compile(r'\s+(\d+)(\*?)\s+(.+)')
        selected = []
        for line in log.splitlines():
            match = starred_regex.match(line)
            if match:
                d, starred, file = match.groups()
                if starred == '*':
                    selected.append(Path(file))
        return selected

    async def _parse_matrix(self, log: str) -> dict:
        """
        Parse the log output from datcmp to extract the correlation matrix and p-values.

        :param log: The log output from the datcmp command.
        :return: A dictionary containing the correlation lengths, p-values, and adjusted
            p-values matrices.
        """
        D1 = []
        D2 = []
        correlation_lengths = []
        p_values = []
        adjusted_p_values = []

        matrix_regex = re.compile(
            r'\s+(\d+)\s+vs\.\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)(\s|\*)'
        )
        for line in log.splitlines():
            match = matrix_regex.match(line)
            if match:
                d1, d2, corr_length, p_value, adj_p_value, starred = match.groups()
                D1.append(int(d1))
                D2.append(int(d2))
                correlation_lengths.append(float(corr_length))
                p_values.append(float(p_value))
                adjusted_p_values.append(float(adj_p_value))

        if not D1:
            self.logger.warning('No valid data found in the log for matrix parsing.')
            return {}

        no_datasets = int(D2[-1])
        empty_matrix: list = [[0.] * no_datasets for _ in range(no_datasets)]
        matrix = {
            'correlation_length': copy.deepcopy(empty_matrix),
            'p_value': copy.deepcopy(empty_matrix),
            'adjusted_p_value': copy.deepcopy(empty_matrix),
        }
        for d1, d2, corr_length, p_value, adj_p_value in zip(
                D1, D2, correlation_lengths, p_values, adjusted_p_values):
            matrix['correlation_length'][d1 - 1][d2 - 1] = corr_length
            matrix['p_value'][d1 - 1][d2 - 1] = p_value
            matrix['adjusted_p_value'][d1 - 1][d2 - 1] = adj_p_value

        return matrix

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