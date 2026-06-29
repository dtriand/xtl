from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Callable

import matplotlib
import matplotlib.pyplot as plt

from xtl import settings
from xtl.common.compatibility import OS_POSIX
from .base import GraphBackend, GraphRenderable, GraphEnumMapper, RenderContext
from ..axes import AxesType, AxisScaleType, AxisName, AxisOptions, SymLogScale
from ..graph import Graph
from ..layouts import GridLayout


@dataclass
class TraceArtists:
    ...


class MatplotlibRenderable(GraphRenderable):

    def __init__(self, graph: Graph, *, ctx: RenderContext | None = None) -> None:
        super().__init__(graph, ctx=ctx)
        self._axes: dict[str, plt.Axes] = {}
        self._fig: plt.Figure = self._build_figure()
        self._gs: plt.GridSpec = self._build_gridspec()
        self._ax: dict[str, plt.Axes] = self._build_axes()
        self._traces = self._build_traces()
        self._link_axes()
        self._configure_axes()

    def _build_figure(self) -> plt.Figure:
        fig = plt.figure(
            figsize=self._graph.dimensions.to_inches,
            dpi=self._graph.dimensions.dpi
        )
        return fig

    def _build_gridspec(self) -> plt.GridSpec:
        layout = self._graph.layout
        if not isinstance(layout, GridLayout):
            raise NotImplementedError('Only GridLayout is supported for Matplotlib backend')
        gs = self._fig.add_gridspec(
            nrows=layout.rows,
            ncols=layout.cols,
        )
        return gs

    def _build_axes(self) -> dict[str, plt.Axes]:
        axes: dict[str, plt.Axes] = {}
        cells = self._graph.layout.cells
        for cell in cells:
            ax = self._fig.add_subplot(
                self._gs[
                    cell.idx[1]:cell.idx[1]+cell.span[1], # row
                    cell.idx[0]:cell.idx[0]+cell.span[0]  # column
                ],
                projection=self._ctx.map(cell.panel.axes.kind, None)
            )
            axes[cell.panel.id] = ax
        return axes

    def _link_axes(self) -> None:
        for link in self._graph.layout.links:
            ax = self._ax[link.panel_ids[0]]
            for panel_id in link.panel_ids[1:]:
                for axis in link.axes:
                    if axis in {AxisName.X, AxisName.THETA}:
                        ax.sharex(self._ax[panel_id])
                    elif axis in {AxisName.Y, AxisName.R}:
                        ax.sharey(self._ax[panel_id])

    def _configure_axes(self) -> None:

        def _configure_axis(ax_: plt.Axes, kind: Literal['x', 'y'], options: AxisOptions) -> None:
            _func: dict[str, Callable] = {
                'label': getattr(ax_, f'set_{kind}label'),
                'limit': getattr(ax_, f'set_{kind}lim'),
                'scale': getattr(ax_, f'set_{kind}scale')
            }

            # Set label
            if options.label:
                _func['label'](options.label)

            # Set axis limits
            _func['limit'](options.limits.min, options.limits.max)

            # Set axis scale
            scale_kwargs = {}
            if isinstance(options.scale, SymLogScale):
                scale_kwargs['linthresh'] = options.scale.threshold
            _func['scale'](self._ctx.map(options.scale.kind, default=None), **scale_kwargs)

        for panel in self._graph.layout.panels.values():
            ax = self._ax[panel.id]
            axes = panel.axes
            for name, options in axes.axes.items():
                if name in {AxisName.X, AxisName.THETA}:
                    _configure_axis(ax, kind='x', options=options)
                elif name in {AxisName.Y, AxisName.R}:
                    _configure_axis(ax, kind='y', options=options)

    def _build_traces(self) -> dict[str, Any]:
        ...

    @property
    def figure(self) -> plt.Figure:
        return self._fig

    def show(self) -> None:
        plt.show(block=True)

    def save(self, filename: str | Path, overwrite: bool = False) -> None:
        f = Path(filename)
        if f.exists():
            if not overwrite:
                raise FileExistsError(f'File already exists: {filename}')
            f.unlink()

        f.parent.mkdir(parents=True, exist_ok=True, mode=settings.jobs.permissions.directories.decimal)
        self._fig.savefig(f, dpi=self._fig.dpi)
        if OS_POSIX:
            f.chmod(settings.jobs.permissions.files.decimal)


class MatplotlibBackend(GraphBackend[MatplotlibRenderable]):

    _MAPPER: GraphEnumMapper = GraphEnumMapper(
        {
            AxisScaleType: {
                AxisScaleType.LINEAR: 'linear',
                AxisScaleType.LOG:    'log',
                AxisScaleType.SYMLOG: 'symlog',
            },
            AxesType: {
                AxesType.CARTESIAN: 'rectilinear',
                AxesType.POLAR:     'polar',
            }
        },
        warn=True
    )

    def __init__(self, backend: str | None = None) -> None:
        if backend is not None:
            matplotlib.use(backend)
        self._ctx = RenderContext(MatplotlibBackend._MAPPER)

    def render(self, graph: Graph) -> MatplotlibRenderable:
        return MatplotlibRenderable(graph, ctx=self._ctx)

