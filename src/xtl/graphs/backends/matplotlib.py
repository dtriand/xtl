from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Callable
import warnings

import matplotlib
import matplotlib.pyplot as plt

from xtl import settings
from xtl.common.compatibility import OS_POSIX
from xtl.common.misc import slice_to_str
from .base import GraphBackend, GraphRenderable, GraphEnumMapper, RenderContext
from ..axes import AxesType, AxisScaleType, AxisName, AxisOptions, SymLogScale
from ..colors import Colormap
from ..data import XYData
from ..graph import Graph
from ..layouts import GridLayout
from ..styles import LineStyleType, PointStyleType
from ..traces import Trace, LineTrace, ScatterTrace


@dataclass
class MatplotlibArtistsCollection:
    primary: Any
    extra: list[Any] = field(default_factory=list)
    raw: Any = field(default_factory=None)

    @property
    def artists(self) -> list[Any]:
        return [self.primary, *self.extra]


class MatplotlibRenderable(GraphRenderable):

    def __init__(self, graph: Graph, *, ctx: RenderContext | None = None) -> None:
        super().__init__(graph, ctx=ctx)
        self._axes: dict[str, plt.Axes] = {}
        self._fig: plt.Figure = self._build_figure()
        self._gs: plt.GridSpec = self._build_gridspec()
        self._ax: dict[str, plt.Axes] = self._build_axes()
        self._traces: dict[str, MatplotlibArtistsCollection] = self._build_traces()
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

    def _trace_line(self, trace: LineTrace, *, ax: plt.Axes) -> MatplotlibArtistsCollection:
        options = {
            'linestyle': self._ctx.map(trace.style.line.style, default='-'),
            'linewidth': trace.style.line.width,
            'color':     trace.style.line.color.to_rgba(),
            'marker':    self._ctx.map(trace.style.points.style, default='o'),
            'markerfacecolor': trace.style.points.color.to_rgba(),
            'markeredgecolor': trace.style.points.color.to_rgba(),
        }
        series = trace.data
        match series:
            case XYData():
                lines = ax.plot(series.x.data, series.y.data, **options)
                return MatplotlibArtistsCollection(primary=lines[0], raw=lines)
            case _:
                raise NotImplementedError(f'Data type {series.kind!r} is not supported for Matplotlib backend')

    def _trace_scatter(self, trace: ScatterTrace, *, ax: plt.Axes) -> MatplotlibArtistsCollection:
        options = {
            'color':     trace.style.points.color.to_rgba(),
            'cmap':      self._ctx.map(trace.style.points.colormap, default=None),
            'marker':    self._ctx.map(trace.style.points.style, default='o'),
        }
        series = trace.data
        match series:
            case XYData():
                lines = ax.scatter(series.x.data, series.y.data, **options)
                return MatplotlibArtistsCollection(primary=lines, raw=lines)
            case _:
                raise NotImplementedError(f'Data type {series.kind!r} is not supported for Matplotlib backend')

    def _build_traces(self) -> dict[str, MatplotlibArtistsCollection]:

        traces = {}
        for panel in self._graph.layout.panels.values():
            ax = self._ax[panel.id]
            for trace in panel.traces:
                match trace:
                    case LineTrace():
                        traces[trace.id] = self._trace_line(trace, ax=ax)
                    case ScatterTrace():
                        traces[trace.id] = self._trace_scatter(trace, ax=ax)
                    case _:
                        raise NotImplementedError(f'Trace type {trace.kind!r} is not supported for Matplotlib backend')

        return traces

    def _build_grid_labels(self):
        for panel_id, ax in self._ax.items():
            ss = ax.get_subplotspec()
            if ss is None:
                warnings.warn(f'No subplot spec found for panel {panel_id!r}, skipping debug labels')
                continue
            loc = (f'axs[{slice_to_str(slice(ss.rowspan.start, ss.rowspan.stop))}, '
                   f'{slice_to_str(slice(ss.colspan.start, ss.colspan.stop))}]')
            ax.annotate(
                f'{panel_id}\n{loc}', (0.5, 0.5),
                transform=ax.transAxes, ha='center', va='center',
                color='darkgrey', backgroundcolor='white'
            )

    def _enable_debug(self) -> None:
        self._build_grid_labels()

    @property
    def figure(self) -> plt.Figure:
        return self._fig

    def display(self, debug: bool = False) -> None:
        if debug:
            self._enable_debug()
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
            },
            Colormap: {
                Colormap.NONE:    '',
                Colormap.VIRIDIS: 'viridis'
            },
            LineStyleType: {
                LineStyleType.NONE:       '',
                LineStyleType.SOLID:      '-',
                LineStyleType.DASHED:     '--',
                LineStyleType.DOTTED:     ':',
                LineStyleType.DASHDOTTED: '-.',
            },
            PointStyleType: {
                PointStyleType.NONE:            '',
                PointStyleType.CIRCLE:          'o',
                PointStyleType.SQUARE:          's',
                PointStyleType.DIAMOND:         'D',
                PointStyleType.TRIANGLE_UP:     '^',
                PointStyleType.TRIANGLE_DOWN:   'v',
                PointStyleType.TRIANGLE_LEFT:   '<',
                PointStyleType.TRIANGLE_RIGHT:  '>',
                PointStyleType.PENTAGON:        'p',
                PointStyleType.STAR:            '*',
                PointStyleType.HEXAGON:         'H',
                PointStyleType.HEXAGON_UP:      'h',
                PointStyleType.OCTAGON:         '8',
                PointStyleType.CROSS:           '+',
                PointStyleType.CROSS_SIDE:      'x',
                PointStyleType.TICK_VERTICAL:   '|',
                PointStyleType.TICK_HORIZONTAL: '_',
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

