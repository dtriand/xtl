from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xtl.common.options import Option
from xtl.jobs import SteppedJob, SteppedJobConfig, StepSpec
from xtl.math.jobs.clustering import CliqueSearchJob, CliqueSearchJobConfig
from xtl.saxs.jobs.atsas import DatcmpBatchJob, DatcmpBatchJobConfig
from xtl.saxs.jobs.atsas_utils import DatcmpOptions, DatcmpMode, DatcmpTest, \
    DatcmpAdjustment


class SAXSCompareJobConfig(SteppedJobConfig):
    files: list[Path] = \
        Option(
            ...,
            desc='List of data files to compare',
            min_length=2,
            path_exists=True
        )


@dataclass
class SAXSComparisonMatrix:
    correlation_length: np.ndarray
    p_value: np.ndarray
    p_value_adjusted: np.ndarray


@dataclass
class SAXSComparisonResults:
    cliques: list[list[int]]
    matrix: SAXSComparisonMatrix
    datasets: list[Path]


class SAXSCompareJob(SteppedJob[SAXSCompareJobConfig]):
    """
    Job to compare SAXS datasets using ATSAS `datcmp`.
    """
    _steps = (
        StepSpec(
            name='datcmp_batch',
            desc='Run datcmp batch job to compute pairwise comparisons',
            job_cls=DatcmpBatchJob,
            config_cls=DatcmpBatchJobConfig,
            defaults={
                'options': DatcmpOptions(
                    test=DatcmpTest.CORMAP,
                    adjust=DatcmpAdjustment.FWER,
                    alpha=0.01,
                    mode=DatcmpMode.PAIRWISE,
                    format='CSV'
                ).to_dict()
            },
            dynamic_defaults=lambda ctx: {
                'input': ctx.parent.config.files
            },
            post_processor=lambda results, ctx: {
                'alpha': ctx.step.config.options.alpha,
                'matrix': SAXSCompareJob.datcmp_to_matrix(results.data.stdout, len(ctx.step.config.input))
            }
        ),
        StepSpec(
            name='clique_search',
            desc='Find cliques of similar datasets based on datcmp results',
            job_cls=CliqueSearchJob,
            config_cls=CliqueSearchJobConfig,
            defaults={
                'max_cliques': None
            },
            dynamic_defaults=lambda ctx: {
                'data': SAXSCompareJob.similarity_to_adjacency(ctx.data['matrix'].p_value, ctx.data['alpha'])
            },
            post_processor=lambda results, ctx: {
                'cliques': [[ctx.parent.config.files[i] for i in clique] for clique in results.data]
            }
        )
    )

    @staticmethod
    def datcmp_to_matrix(csv_output: str, no_files: int) -> SAXSComparisonMatrix:
        """
        Parse the CSV output from `datcmp` and construct a symmetric matrix of correlation lengths, p-values, and
        adjusted p-values.

        :param csv_output: The CSV output from `datcmp` as a string.
        :param no_files: The number of files that were compared, which determines the size of the matrices.
        :return: A SAXSComparisonMatrix containing the correlation lengths, p-values, and adjusted p-values.
        """
        data = SAXSComparisonMatrix(
            correlation_length=np.zeros((no_files, no_files)),
            p_value=np.ones((no_files, no_files)),
            p_value_adjusted=np.ones((no_files, no_files))
        )
        for i, line in enumerate(csv_output.splitlines()):
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
                continue

            data.correlation_length[index_1, index_2] = correlation_length
            data.p_value[index_1, index_2] = p_value
            data.p_value_adjusted[index_1, index_2] = adjusted_p_value

        # Symmetrize matrices since `datcmp` only outputs upper triangle
        data.correlation_length = np.triu(data.correlation_length) + np.triu(data.correlation_length, k=1).T
        data.p_value = np.triu(data.p_value) + np.triu(data.p_value, k=1).T
        data.p_value_adjusted = np.triu(data.p_value_adjusted) + np.triu(data.p_value_adjusted, k=1).T

        return data

    @staticmethod
    def similarity_to_adjacency(matrix: np.ndarray, threshold: float) -> np.ndarray:
        """
        Convert a similarity matrix to a boolean adjacency matrix based on a threshold. Values above the threshold are
        considered adjacent (similar), while values below are not. The diagonal is set to False since we do not consider
        self-connections.

        :param matrix: A 2D numpy array representing pairwise similarities (e.g., p-values).
        :param threshold: A float threshold to determine adjacency.
        :return: A boolean 2D numpy array where True indicates adjacency (similarity above threshold) and False
            otherwise.
        """
        adjacency = matrix >= threshold
        np.fill_diagonal(adjacency, False)  # No self-connections
        return adjacency
