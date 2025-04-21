from dataclasses import dataclass
from datetime import datetime
from functools import partial
import json
from pathlib import Path
from typing import Optional

import typer
import xlsxwriter
from xlsxwriter.format import Format
import xlsxwriter.worksheet
from xlsxwriter.utility import xl_rowcol_to_cell as rc2a1

from xtl import __version__
import xtl.cli.autoproc.cli_utils as apu
from xtl.cli.cliio import Console, epilog
from xtl.exceptions.utils import Catcher


app = typer.Typer()


@dataclass
class MultiRowHeader:
    title: str
    subtitle: Optional[str] = None
    group: Optional[str] = None
    subgroup: Optional[str] = None
    units: Optional[str] = None


JobHeader = partial(MultiRowHeader, subtitle=None, group='job', subgroup=None, units=None)
TruncateHeader = partial(MultiRowHeader, group='autoproc.truncate')
TruncateOverallHeader = partial(MultiRowHeader, group='autoproc.truncate', subgroup='statistics.overall')
TruncateInnerHeader = partial(MultiRowHeader, group='autoproc.truncate', subgroup='statistics.inner_shell')
TruncateOuterHeader = partial(MultiRowHeader, group='autoproc.truncate', subgroup='statistics.outer_shell')
StaranisoHeader = partial(MultiRowHeader, group='autoproc.staraniso')
StaranisoOverallHeader = partial(MultiRowHeader, group='autoproc.staraniso', subgroup='statistics.overall')
StaranisoInnerHeader = partial(MultiRowHeader, group='autoproc.staraniso', subgroup='statistics.inner_shell')
StaranisoOuterHeader = partial(MultiRowHeader, group='autoproc.staraniso', subgroup='statistics.outer_shell')
StaranisoAnisoHeader = partial(MultiRowHeader, group='autoproc.staraniso', subgroup='anisotropic_cutoff')
DatasetHeader = partial(MultiRowHeader, group='dataset', subgroup=None)
ImginfoHeader = partial(MultiRowHeader, group='autoproc.imginfo', subgroup=None)
CorrectHeader = partial(MultiRowHeader, group='xds.correct', subgroup=None)
CorrectInputHeader = partial(MultiRowHeader, group='xds.correct', subgroup='input_params')
CorrectRefinedHeader = partial(MultiRowHeader, group='xds.correct', subgroup='refined_params')
ValidationHeader1 = partial(MultiRowHeader, group='xtl.validation', units=None, subgroup='autoproc.imginfo vs. xds.correct.input_params')
ValidationHeader2 = partial(MultiRowHeader, group='xtl.validation', units=None, subgroup='xds.correct.input_params vs. xds.correct.refined_params')


@dataclass
class MultiRowHeaderStyle:
    title: Optional[Format] = None
    subtitle: Optional[Format] = None
    group: Optional[Format] = None
    subgroup: Optional[Format] = None
    units: Optional[Format] = None


@dataclass
class HeaderOptions:
    compact: bool = False
    colors: bool = True
    show_units: bool = True
    rotated_headers: bool = True
    conditional_formatting: bool = True


@dataclass
class ColorScale3:
    min: str
    mid: str
    max: str


ryg = ColorScale3(min='#f8696b', mid='#ffeb84', max='#63be7b')
rwr = ColorScale3(min='#f8696b', mid='#efefef', max='#f8696b')


class Worksheet:
    def __init__(self, wb: xlsxwriter.Workbook, name: str, o: Optional[HeaderOptions] = None):
        self.wb = wb
        self.ws = wb.add_worksheet(name)
        self.o = HeaderOptions() if o is None else o
        self._headers: list[MultiRowHeader] = []
        self._header_styles: list[MultiRowHeaderStyle] = []
        self._merged_header_ranges: dict = {
            'group': [],
            'subgroup': [],
            'title': [],
        }
        self._data: list[list] = []

    @property
    def name(self):
        return self.ws.name

    def _register_style(self, wb_style: dict) -> Format:
        # Apply header options
        if not self.o.colors:
            wb_style.pop('fg_color', None)

        f = self.wb.add_format(wb_style)
        return f

    def register_body_style(self, wb_style: dict) -> Format:
        return self._register_style(wb_style)

    def register_header_style(self, wb_style: dict) -> MultiRowHeaderStyle:
        # Get style dicts
        title_style = wb_style.get('title', {})
        subtitle_style = wb_style.get('subtitle', {})
        group_style = wb_style.get('group', {})
        subgroup_style = wb_style.get('subgroup', {})
        units_style = wb_style.get('units', {})

        # Apply header options
        if self.o.rotated_headers:
            if self.o.compact:
                group_style['rotation'] = -90
            else:
                title_style['rotation'] = -90
                subtitle_style['rotation'] = -90

        # Register styles
        title_style = self._register_style(title_style)
        subtitle_style = self._register_style(subtitle_style)
        group_style = self._register_style(group_style)
        subgroup_style = self._register_style(subgroup_style)
        units_style = self._register_style(units_style)

        return MultiRowHeaderStyle(
            title=title_style,
            subtitle=subtitle_style,
            group=group_style,
            subgroup=subgroup_style,
            units=units_style
        )

    def add_headers(self, headers: MultiRowHeader | list[MultiRowHeader],
                    styles: MultiRowHeaderStyle | list[MultiRowHeaderStyle] = None,):
        if not isinstance(headers, list):
            headers = [headers]
        if not isinstance(styles, list):
            styles = [styles] * len(headers)
        if len(headers) != len(styles):
            raise ValueError('Number of headers must match number of styles')

        for i, (h, s) in enumerate(zip(headers, styles)):
            if not isinstance(h, MultiRowHeader):
                raise ValueError(f'Invalid header type: {h!r}')
            if not isinstance(s, MultiRowHeaderStyle) and s is not None:
                raise ValueError(f'Invalid style type: {s!r}')
            self._headers.append(h)
            self._header_styles.append(s)

    def _find_merge_ranges(self):
        old_contents = {'group': None, 'subgroup': None, 'title': None}
        old_styles = {'group': None, 'subgroup': None, 'title': None}
        r_start = {'group': 0, 'subgroup': 0, 'title': 0}
        r_end = {'group': 0, 'subgroup': 0, 'title': 0}

        # Iterate over different rows of header cells (i.e. group, subgroup, title)
        for group in ['group', 'subgroup', 'title']:
            # Iterate over the headers
            for i in range(len(self._headers) + 1):
                # Get header/style at index
                try:
                    h = self._headers[i]
                    s = self._header_styles[i]
                except IndexError:
                    # when i = len(self._headers) + 1 loop back to the first column
                    h = self._headers[0]
                    s = self._header_styles[0]

                # Get the cell content
                new_content = getattr(h, group, None)
                old_content = old_contents[group]

                # Get the cell style
                new_style = getattr(s, group, None)
                old_style = old_styles[group]

                # Update range end if cell content/style is same as before or both are None
                update_condition = \
                    (
                        (
                            # Contents are the same
                            (new_content == old_content) or
                            (new_content is None and old_content is None)
                        ) and
                        (
                            # Styles are the same
                            new_style is old_style
                        )
                    )

                if update_condition:
                    r_end[group] = i
                else:
                    # Add a new range if the start and end indices are different
                    if r_end[group] != r_start[group]:
                        self._merged_header_ranges[group].append(
                            {
                                'range': [r_start[group], r_end[group]],
                                'content': old_content,
                                'style': old_style
                            }
                        )
                    # Update range start/end
                    r_start[group] = i
                    r_end[group] = i
                # Update previous value
                old_contents[group] = new_content
                old_styles[group] = new_style

    def _merge_header_ranges(self):
        if self.o.compact:
            return
        self._find_merge_ranges()
        rows = {'group': 0, 'subgroup': 1, 'title': 2, 'subtitle': 3}
        for group, ranges in self._merged_header_ranges.items():
            for r in ranges:
                self.ws.merge_range(rows[group], r['range'][0], rows[group], r['range'][1],
                                    r['content'], r['style'])

    def _write_header(self, h: MultiRowHeader, s: MultiRowHeaderStyle,
                      r: int = 0, c: int = 0):
        if self.o.compact:
            text = ''
            for component in [h.group, h.subgroup, h.title, h.subtitle]:
                if component:
                    if text:
                        text += '.'
                    text += component
            self.ws.write(r, c, text, s.group if s else None)
        else:
            self.ws.write(r, c, h.group, s.group if s else None)

            r += 1
            self.ws.write(r, c, h.subgroup, s.subgroup if s else None)

            r += 1
            self.ws.write(r, c, h.title, s.title if s else None)

            r += 1
            self.ws.write(r, c, h.subtitle, s.subtitle if s else None)

        if self.o.show_units:
            r += 1
            self.ws.write(r, c, h.units, s.units if s else None)

        return r, c

    def write_headers(self, r: int = 0, c: int = 0):
        for i, (h, s) in enumerate(zip(self._headers, self._header_styles)):
            last_row, last_column = self._write_header(h, s, r=r, c=c+i)
        self._merge_header_ranges()
        return last_row, last_column

    def conditional_format_color_scale(self, first_row, first_col, last_row, last_col,
                                       reverse=False, color_scale: ColorScale3 = None,
                                       value_mid=None, percentile_mid=None):
        # Choose mid value/type
        if value_mid is None and percentile_mid is None:
            type_mid = 'percentile'
            value_mid = 50
        elif value_mid:
            type_mid = 'number'
            value_mid = value_mid
        else:
            type_mid = 'percentile'
            value_mid = percentile_mid

        # Choose colors
        if color_scale is None:
            color_scale = ryg
        color_min = color_scale.min
        color_mid = color_scale.mid
        color_max = color_scale.max
        if reverse:
            color_min, color_max = color_max, color_min

        # Format cells
        options = {
            'type': '3_color_scale',
            'min_type': 'min', 'min_color': color_min,
            'mid_value': value_mid, 'mid_type': type_mid, 'mid_color': color_mid,
            'max_type': 'max', 'max_color': color_max,
        }
        self.ws.conditional_format(first_row, first_col, last_row, last_col, options)

    def add_chart_scatter(self, range_x, range_y, location, title: str = None,
                          x_label: str = None, y_label: str = None, legend: bool = False):
        c = self.wb.add_chart({'type': 'scatter'})
        c.add_series(
            {
                'categories': range_x,
                'values': range_y
            }
        )
        if title:
            c.set_title({'name': title})
        if x_label:
            c.set_x_axis({'name': x_label})
        if y_label:
            c.set_y_axis({'name': y_label})
        if not legend:
            c.set_legend({'none': True})
        self.ws.insert_chart(*location, c)
        return c


def get_workbook_formats(ws: Worksheet) -> dict:
    formats = {'data': {}, 'header': {}}

    # Formats
    bold, italics = {'bold': True}, {'italic': True}
    gray_0, gray_1, gray_2 = {'fg_color': '#b7b7b7'}, {'fg_color': '#d9d9d9'}, {'fg_color': '#efefef'}
    blue_0, blue_1, blue_2 = {'fg_color': '#6d9eeb'}, {'fg_color': '#a4c2f4'}, {'fg_color': '#c9daf8'}
    green_0, green_1, green_2 = {'fg_color': '#93c47d'}, {'fg_color': '#b6d7a8'}, {'fg_color': '#d9ead3'}
    yellow_0, yellow_1, yellow_2 = {'fg_color': '#ffd966'}, {'fg_color': '#ffe599'}, {'fg_color': '#fff2cc'}
    purple_0, purple_1, purple_2 = {'fg_color': '#8e7cc3'}, {'fg_color': '#d9d2e9'}, {'fg_color': '#b4a7d6'}

    # Data formats
    formats['data']['gray'] = [ws.register_body_style(x) for x in [gray_2, gray_1]]
    formats['data']['blue'] = [ws.register_body_style(x) for x in [blue_2, blue_1]]
    formats['data']['green'] = [ws.register_body_style(x) for x in [green_2, green_1]]
    formats['data']['yellow'] = [ws.register_body_style(x) for x in [yellow_2, yellow_1]]
    formats['data']['purple'] = [ws.register_body_style(x) for x in [purple_2, purple_1]]

    # Header formats
    h = lambda x, y: {
        'title': bold | y,
        'subtitle': bold | y,
        'group': bold | x,
        'subgroup': bold | y,
        'units': italics | y
    }

    formats['header']['gray'] = [ws.register_header_style(h(gray_0, x)) for x in [gray_1, gray_2]]
    formats['header']['blue'] = [ws.register_header_style(h(blue_0, x)) for x in [blue_1, blue_2]]
    formats['header']['green'] = [ws.register_header_style(h(green_0, x)) for x in [green_1, green_2]]
    formats['header']['yellow'] = [ws.register_header_style(h(yellow_0, x)) for x in [yellow_1, yellow_2]]
    formats['header']['purple'] = [ws.register_header_style(h(purple_0, x)) for x in [purple_1, purple_2]]
    return formats


def create_headers(s0: Worksheet, s1: Worksheet, formats: dict):
    # Table 1 sheet
    s0.add_headers(headers=[
        JobHeader('#'),  # col 0
        JobHeader('job_dir'),
        JobHeader('sweep_id'),
        JobHeader('autoproc_id'),
    ], styles=formats['header']['gray'][0])

    s0.add_headers(headers=[
        TruncateHeader('_file'),
        TruncateHeader('_file_exists'),  # col 5
        TruncateHeader('_is_parsed'),
        TruncateHeader('_is_processed'),
        TruncateHeader('processing_time'),
        TruncateHeader('space_group'),
        TruncateHeader('unit_cell', subtitle='a', units='\u212b'),  # col 10
        TruncateHeader('unit_cell', subtitle='b', units='\u212b'),
        TruncateHeader('unit_cell', subtitle='c', units='\u212b'),
        TruncateHeader('unit_cell', subtitle='\u03b1', units='\u00b0'),
        TruncateHeader('unit_cell', subtitle='\u03b2', units='\u00b0'),
        TruncateHeader('unit_cell', subtitle='\u03b3', units='\u00b0'),  # col 15
        TruncateHeader('wavelength', units='\u212b'),
    ], styles=formats['header']['blue'][0])

    s0.add_headers(headers=[
        TruncateOverallHeader('resolution', subtitle='low', units='\u212b'),
        TruncateOverallHeader('resolution', subtitle='high', units='\u212b'),
        TruncateOverallHeader('r_merge'),
        TruncateOverallHeader('r_meas_within_i_plus_minus'),  # col 20
        TruncateOverallHeader('r_meas_all_i_plus_i_minus'),
        TruncateOverallHeader('r_pim_within_i_plus_minus'),
        TruncateOverallHeader('r_pim_all_i_plus_i_minus'),
        TruncateOverallHeader('no_observations', subtitle='all'),
        TruncateOverallHeader('no_observations', subtitle='unique'),  # col 25
        TruncateOverallHeader('i_over_sigma_mean'),
        TruncateOverallHeader('completeness'),
        TruncateOverallHeader('multiplicity'),
        TruncateOverallHeader('cc_half'),
        TruncateOverallHeader('anomalous_completeness'),  # col 30
        TruncateOverallHeader('anomalous_multiplicity'),
        TruncateOverallHeader('anomalous_cc'),
        TruncateOverallHeader('dano_over_sigma_dano')
    ], styles=formats['header']['blue'][1])

    s0.add_headers(headers=[
        TruncateInnerHeader('resolution', subtitle='low', units='\u212b'),
        TruncateInnerHeader('resolution', subtitle='high', units='\u212b'),  # col 35
        TruncateInnerHeader('r_merge'),
        TruncateInnerHeader('r_meas_within_i_plus_minus'),
        TruncateInnerHeader('r_meas_all_i_plus_i_minus'),
        TruncateInnerHeader('r_pim_within_i_plus_minus'),
        TruncateInnerHeader('r_pim_all_i_plus_i_minus'),  # col 40
        TruncateInnerHeader('no_observations', subtitle='all'),
        TruncateInnerHeader('no_observations', subtitle='unique'),
        TruncateInnerHeader('i_over_sigma_mean'),
        TruncateInnerHeader('completeness'),
        TruncateInnerHeader('multiplicity'),  # col 45
        TruncateInnerHeader('cc_half'),
        TruncateInnerHeader('anomalous_completeness'),
        TruncateInnerHeader('anomalous_multiplicity'),
        TruncateInnerHeader('anomalous_cc'),
        TruncateInnerHeader('dano_over_sigma_dano')  # col 50
    ], styles=formats['header']['blue'][0])

    s0.add_headers(headers=[
        TruncateOuterHeader('resolution', subtitle='low', units='\u212b'),
        TruncateOuterHeader('resolution', subtitle='high', units='\u212b'),
        TruncateOuterHeader('r_merge'),
        TruncateOuterHeader('r_meas_within_i_plus_minus'),
        TruncateOuterHeader('r_meas_all_i_plus_i_minus'),  # col 55
        TruncateOuterHeader('r_pim_within_i_plus_minus'),
        TruncateOuterHeader('r_pim_all_i_plus_i_minus'),
        TruncateOuterHeader('no_observations', subtitle='all'),
        TruncateOuterHeader('no_observations', subtitle='unique'),
        TruncateOuterHeader('i_over_sigma_mean'),  # col 60
        TruncateOuterHeader('completeness'),
        TruncateOuterHeader('multiplicity'),
        TruncateOuterHeader('cc_half'),
        TruncateOuterHeader('anomalous_completeness'),
        TruncateOuterHeader('anomalous_multiplicity'),  # col 65
        TruncateOuterHeader('anomalous_cc'),
        TruncateOuterHeader('dano_over_sigma_dano')
    ], styles=formats['header']['blue'][1])

    s0.add_headers(headers=[
        StaranisoHeader('_file'),
        StaranisoHeader('_file_exists'),
        StaranisoHeader('_is_parsed'),  # col 70
        StaranisoHeader('_is_processed'),
        StaranisoHeader('processing_time'),
        StaranisoHeader('space_group'),
        StaranisoHeader('unit_cell', subtitle='a', units='\u212b'),
        StaranisoHeader('unit_cell', subtitle='b', units='\u212b'),  # col 75
        StaranisoHeader('unit_cell', subtitle='c', units='\u212b'),
        StaranisoHeader('unit_cell', subtitle='\u03b1', units='\u00b0'),
        StaranisoHeader('unit_cell', subtitle='\u03b2', units='\u00b0'),
        StaranisoHeader('unit_cell', subtitle='\u03b3', units='\u00b0'),
        StaranisoHeader('wavelength', units='\u212b'),  # col 80
    ], styles=formats['header']['green'][0])

    s0.add_headers(headers=[
        StaranisoOverallHeader('resolution', subtitle='low', units='\u212b'),
        StaranisoOverallHeader('resolution', subtitle='high', units='\u212b'),
        StaranisoOverallHeader('r_merge'),
        StaranisoOverallHeader('r_meas_within_i_plus_minus'),
        StaranisoOverallHeader('r_meas_all_i_plus_i_minus'),  # col 85
        StaranisoOverallHeader('r_pim_within_i_plus_minus'),
        StaranisoOverallHeader('r_pim_all_i_plus_i_minus'),
        StaranisoOverallHeader('no_observations', subtitle='all'),
        StaranisoOverallHeader('no_observations', subtitle='unique'),
        StaranisoOverallHeader('i_over_sigma_mean'),  # col 90
        StaranisoOverallHeader('completeness'),
        StaranisoOverallHeader('multiplicity'),
        StaranisoOverallHeader('cc_half'),
        StaranisoOverallHeader('anomalous_completeness'),
        StaranisoOverallHeader('anomalous_multiplicity'),  # col 95
        StaranisoOverallHeader('anomalous_cc'),
        StaranisoOverallHeader('dano_over_sigma_dano')
    ], styles=formats['header']['green'][1])

    s0.add_headers(headers=[
        StaranisoInnerHeader('resolution', subtitle='low', units='\u212b'),
        StaranisoInnerHeader('resolution', subtitle='high', units='\u212b'),
        StaranisoInnerHeader('r_merge'),  # col 100
        StaranisoInnerHeader('r_meas_within_i_plus_minus'),
        StaranisoInnerHeader('r_meas_all_i_plus_i_minus'),
        StaranisoInnerHeader('r_pim_within_i_plus_minus'),
        StaranisoInnerHeader('r_pim_all_i_plus_i_minus'),
        StaranisoInnerHeader('no_observations', subtitle='all'),  # col 105
        StaranisoInnerHeader('no_observations', subtitle='unique'),
        StaranisoInnerHeader('i_over_sigma_mean'),
        StaranisoInnerHeader('completeness'),
        StaranisoInnerHeader('multiplicity'),
        StaranisoInnerHeader('cc_half'),  # col 110
        StaranisoInnerHeader('anomalous_completeness'),
        StaranisoInnerHeader('anomalous_multiplicity'),
        StaranisoInnerHeader('anomalous_cc'),
        StaranisoInnerHeader('dano_over_sigma_dano')
    ], styles=formats['header']['green'][0])

    s0.add_headers(headers=[
        StaranisoOuterHeader('resolution', subtitle='low', units='\u212b'),  # col 115
        StaranisoOuterHeader('resolution', subtitle='high', units='\u212b'),
        StaranisoOuterHeader('r_merge'),
        StaranisoOuterHeader('r_meas_within_i_plus_minus'),
        StaranisoOuterHeader('r_meas_all_i_plus_i_minus'),
        StaranisoOuterHeader('r_pim_within_i_plus_minus'),  # col 120
        StaranisoOuterHeader('r_pim_all_i_plus_i_minus'),
        StaranisoOuterHeader('no_observations', subtitle='all'),
        StaranisoOuterHeader('no_observations', subtitle='unique'),
        StaranisoOuterHeader('i_over_sigma_mean'),
        StaranisoOuterHeader('completeness'),  # col 125
        StaranisoOuterHeader('multiplicity'),
        StaranisoOuterHeader('cc_half'),
        StaranisoOuterHeader('anomalous_completeness'),
        StaranisoOuterHeader('anomalous_multiplicity'),
        StaranisoOuterHeader('anomalous_cc'),  # col 130
        StaranisoOuterHeader('dano_over_sigma_dano')
    ], styles=formats['header']['green'][1])

    s0.add_headers(headers=[
        StaranisoAnisoHeader('ellipsoid_axes', subtitle='11'),
        StaranisoAnisoHeader('ellipsoid_axes', subtitle='12'),
        StaranisoAnisoHeader('ellipsoid_axes', subtitle='13'),
        StaranisoAnisoHeader('ellipsoid_axes', subtitle='21'),  # col 135
        StaranisoAnisoHeader('ellipsoid_axes', subtitle='22'),
        StaranisoAnisoHeader('ellipsoid_axes', subtitle='23'),
        StaranisoAnisoHeader('ellipsoid_axes', subtitle='31'),
        StaranisoAnisoHeader('ellipsoid_axes', subtitle='32'),
        StaranisoAnisoHeader('ellipsoid_axes', subtitle='33'),  # col 140
        StaranisoAnisoHeader('resolution_limit', subtitle='1', units='\u212b'),
        StaranisoAnisoHeader('resolution_limit', subtitle='2', units='\u212b'),
        StaranisoAnisoHeader('resolution_limit', subtitle='3', units='\u212b'),
    ], styles=formats['header']['green'][0])

    # Datasets sheet
    s1.add_headers(headers=[
        JobHeader('#'),  # col 0
        JobHeader('job_dir'),
        JobHeader('sweep_id'),
        JobHeader('autoproc_id'),
    ], styles=formats['header']['gray'][0])

    s1.add_headers(headers=[
        DatasetHeader('dataset_name'),
        DatasetHeader('dataset_dir'),  # col 5
        DatasetHeader('first_image'),
        DatasetHeader('raw_data_dir'),
        DatasetHeader('processed_data_dir'),
        DatasetHeader('output_dir'),
    ], styles=formats['header']['gray'][1])

    s1.add_headers(headers=[
        ImginfoHeader('_file'),  # col 10
        ImginfoHeader('_file_exists'),
        ImginfoHeader('_is_parsed'),
        ImginfoHeader('_is_processed'),
        ImginfoHeader('collection_time', units='datetime'),
        ImginfoHeader('exposure_time', units='s'),  # col 15
        ImginfoHeader('detector_distance', units='m'),
        ImginfoHeader('wavelength', units='\u212b'),
        ImginfoHeader('phi_angle', units='\u00b0'),
        ImginfoHeader('omega_angle', subtitle='start', units='\u00b0'),
        ImginfoHeader('omega_angle', subtitle='end', units='\u00b0'),  # col 20
        ImginfoHeader('omega_angle', subtitle='step', units='\u00b0'),
        ImginfoHeader('kappa_angle', units='\u00b0'),
        ImginfoHeader('two_theta_angle', units='\u00b0'),
        ImginfoHeader('beam_center', subtitle='x', units='px'),
        ImginfoHeader('beam_center', subtitle='y', units='px'),  # col 25
        ImginfoHeader('no_images'),
        ImginfoHeader('image_first'),
        ImginfoHeader('image_last'),
    ], styles=formats['header']['yellow'][0])

    s1.add_headers(headers=[
        CorrectHeader('_file'),
        CorrectHeader('_file_exists'),  # col 30
        CorrectHeader('_is_parsed'),
        CorrectHeader('_is_processed')
    ], styles=formats['header']['purple'][0])

    s1.add_headers(headers=[
        CorrectInputHeader('space_group_number'),
        CorrectInputHeader('unit_cell', subtitle='a', units='\u212b'),
        CorrectInputHeader('unit_cell', subtitle='b', units='\u212b'),  # col 35
        CorrectInputHeader('unit_cell', subtitle='c', units='\u212b'),
        CorrectInputHeader('unit_cell', subtitle='\u03b1', units='\u00b0'),
        CorrectInputHeader('unit_cell', subtitle='\u03b2', units='\u00b0'),
        CorrectInputHeader('unit_cell', subtitle='\u03b3', units='\u00b0'),
        CorrectInputHeader('friedels_law'),  # col 40
        CorrectInputHeader('image_template'),
        CorrectInputHeader('data_range', subtitle='start'),
        CorrectInputHeader('data_range', subtitle='end'),
        CorrectInputHeader('rotation_axis', subtitle='x'),
        CorrectInputHeader('rotation_axis', subtitle='y'),  # col 45
        CorrectInputHeader('rotation_axis', subtitle='z'),
        CorrectInputHeader('oscillation_angle', units='\u00b0'),
        CorrectInputHeader('wavelength', units='\u212b'),
        CorrectInputHeader('polarization_fraction'),
        CorrectInputHeader('detector'),  # col 50
        CorrectInputHeader('no_pixels', subtitle='x'),
        CorrectInputHeader('no_pixels', subtitle='y'),
        CorrectInputHeader('pixel_size', subtitle='x', units='mm'),
        CorrectInputHeader('pixel_size', subtitle='y', units='mm'),
        CorrectInputHeader('beam_center', subtitle='x', units='px'),  # col 55
        CorrectInputHeader('beam_center', subtitle='y', units='px'),
        CorrectInputHeader('detector_distance', units='mm'),
        CorrectInputHeader('beam_divergence_esd', units='\u00b0'),
        CorrectInputHeader('reflecting_range_esd', units='\u00b0'),
        CorrectInputHeader('spot_position_error_max', units='px'),  # col 60
        CorrectInputHeader('spindle_position_error_max', units='\u00b0'),
    ], styles=formats['header']['purple'][0])

    s1.add_headers(headers=[
        CorrectRefinedHeader('indexed_spots'),
        CorrectRefinedHeader('deviation_spot_position', units='px'),
        CorrectRefinedHeader('deviation_spindle_position', units='\u00b0'),
        CorrectRefinedHeader('space_group_number'),  # col 65
        CorrectRefinedHeader('unit_cell', subtitle='a', units='\u212b'),
        CorrectRefinedHeader('unit_cell', subtitle='b', units='\u212b'),
        CorrectRefinedHeader('unit_cell', subtitle='c', units='\u212b'),
        CorrectRefinedHeader('unit_cell', subtitle='\u03b1', units='\u00b0'),
        CorrectRefinedHeader('unit_cell', subtitle='\u03b2', units='\u00b0'),  # col 70
        CorrectRefinedHeader('unit_cell', subtitle='\u03b3', units='\u00b0'),
        CorrectRefinedHeader('unit_cell_esd', subtitle='a', units='\u212b'),
        CorrectRefinedHeader('unit_cell_esd', subtitle='b', units='\u212b'),
        CorrectRefinedHeader('unit_cell_esd', subtitle='c', units='\u212b'),
        CorrectRefinedHeader('unit_cell_esd', subtitle='\u03b1', units='\u00b0'),
        # col 75
        CorrectRefinedHeader('unit_cell_esd', subtitle='\u03b2', units='\u00b0'),
        CorrectRefinedHeader('unit_cell_esd', subtitle='\u03b3', units='\u00b0'),
        CorrectRefinedHeader('mosaicity', units='\u00b0'),
        CorrectRefinedHeader('beam_center', subtitle='x', units='px'),
        CorrectRefinedHeader('beam_center', subtitle='y', units='px'),  # col 80
        CorrectRefinedHeader('detector_distance', units='mm'),
    ], styles=formats['header']['purple'][1])

    s1.add_headers(headers=[
        CorrectHeader('ISa', subtitle='value'),
        CorrectHeader('ISa', subtitle='a'),
        CorrectHeader('ISa', subtitle='b'),
    ], styles=formats['header']['purple'][0])

    s1.add_headers(headers=[
        ValidationHeader1('beam_center_difference_abs', subtitle='x', units='px'),
        ValidationHeader1('beam_center_difference_abs', subtitle='y', units='px'),
        ValidationHeader1('detector_distance_difference_abs', units='mm'),
    ], styles=formats['header']['gray'][0])

    s1.add_headers(headers=[
        ValidationHeader2('beam_center_difference_abs', subtitle='x', units='px'),
        ValidationHeader2('beam_center_difference_abs', subtitle='y', units='px'),
        ValidationHeader2('detector_distance_difference_abs', units='mm'),
    ], styles=formats['header']['gray'][0])


def write_data_line_jobs(ws: xlsxwriter.worksheet.Worksheet, job: dict, i: int, j: int, formats: dict):
    f = formats['data']['gray'][0]
    ws.write_number(i, j+0, job['id'], f)
    ws.write_string(i, j+1, job['job_dir'], f)
    ws.write_number(i, j+2, int(job['sweep_id']), f)
    ws.write_string(i, j+3, job['autoproc_id'], f)
    return i, j+3


def write_data_line_statistics_shell(ws: xlsxwriter.worksheet.Worksheet, job: dict, i: int, j: int, f: Format):
    ws.write(i, j+0, job['resolution_low'], f)
    ws.write(i, j+1, job['resolution_high'], f)
    ws.write(i, j+2, job['r_merge'], f)
    ws.write(i, j+3, job['r_meas_within_i_plus_minus'], f)
    ws.write(i, j+4, job['r_meas_all_i_plus_i_minus'], f)
    ws.write(i, j+5, job['r_pim_within_i_plus_minus'], f)
    ws.write(i, j+6, job['r_pim_all_i_plus_i_minus'], f)
    ws.write(i, j+7, job['no_observations'], f)
    ws.write(i, j+8, job['no_observations_unique'], f)
    ws.write(i, j+9, job['i_over_sigma_mean'], f)
    ws.write(i, j+10, job['completeness'], f)
    ws.write(i, j+11, job['multiplicity'], f)
    ws.write(i, j+12, job['cc_half'], f)
    ws.write(i, j+13, job['anomalous_completeness'], f)
    ws.write(i, j+14, job['anomalous_multiplicity'], f)
    ws.write(i, j+15, job['anomalous_cc'], f)
    ws.write(i, j+16, job['dano_over_sigma_dano'], f)
    return i, j+16


def write_data_line_statistics(ws: xlsxwriter.worksheet.Worksheet, job: dict, i: int, j: int, flist: list):
    f0 = flist[0]
    f1 = flist[1]

    ws.write_string(i,  j+0, job['_file'], f0)
    ws.write_boolean(i, j+1, bool(job['_file_exists']), f0)
    ws.write_boolean(i, j+2, bool(job['_is_parsed']), f0)
    ws.write_boolean(i, j+3, bool(job['_is_processed']), f0)

    ws.write(i, j+4, job['processing_time'], f0)

    ws.write(i, j+5, job['space_group'], f0)
    ws.write(i, j+6, job['unit_cell'][0], f0)
    ws.write(i, j+7, job['unit_cell'][1], f0)
    ws.write(i, j+8, job['unit_cell'][2], f0)
    ws.write(i, j+9, job['unit_cell'][3], f0)
    ws.write(i, j+10, job['unit_cell'][4], f0)
    ws.write(i, j+11, job['unit_cell'][5], f0)

    ws.write(i, j+12, job['wavelength'], f0)

    i, j = write_data_line_statistics_shell(ws=ws, job=job['statistics']['overall'],
                                            i=i, j=j+13, f=f1)
    i, j = write_data_line_statistics_shell(ws=ws, job=job['statistics']['inner'],
                                            i=i, j=j+1, f=f0)
    i, j = write_data_line_statistics_shell(ws=ws, job=job['statistics']['outer'],
                                            i=i, j=j+1, f=f1)
    return i, j


def write_data_line_truncate(ws: xlsxwriter.worksheet.Worksheet, job: dict, i: int, j: int, formats: dict):
    f0 = formats['data']['blue'][0]
    f1 = formats['data']['blue'][1]

    truncate = job['autoproc.truncate']
    i, j = write_data_line_statistics(ws=ws, job=truncate, i=i, j=j, flist=[f0, f1])
    return i, j


def write_data_line_staraniso(ws: xlsxwriter.worksheet.Worksheet, job: dict, i: int, j: int, formats: dict):
    f0 = formats['data']['green'][0]
    f1 = formats['data']['green'][1]

    staraniso = job['autoproc.staraniso']
    i, j = write_data_line_statistics(ws=ws, job=staraniso, i=i, j=j, flist=[f0, f1])

    ws.write(i, j+1, staraniso['resolution_ellipsoid_axes'][0][0], f0)
    ws.write(i, j+2, staraniso['resolution_ellipsoid_axes'][0][1], f0)
    ws.write(i, j+3, staraniso['resolution_ellipsoid_axes'][0][2], f0)
    ws.write(i, j+4, staraniso['resolution_ellipsoid_axes'][1][0], f0)
    ws.write(i, j+5, staraniso['resolution_ellipsoid_axes'][1][1], f0)
    ws.write(i, j+6, staraniso['resolution_ellipsoid_axes'][1][2], f0)
    ws.write(i, j+7, staraniso['resolution_ellipsoid_axes'][2][0], f0)
    ws.write(i, j+8, staraniso['resolution_ellipsoid_axes'][2][1], f0)
    ws.write(i, j+9, staraniso['resolution_ellipsoid_axes'][2][2], f0)
    ws.write(i, j+10, staraniso['resolution_limits'][0], f0)
    ws.write(i, j+11, staraniso['resolution_limits'][1], f0)
    ws.write(i, j+12, staraniso['resolution_limits'][2], f0)

    return i, j+12


def write_data_line_s0(s0: Worksheet, job: dict, i: int, j: int, formats: dict):
    # Table 1 sheet
    i, j = write_data_line_jobs(ws=s0.ws, job=job, i=i, j=j, formats=formats)
    i, j = write_data_line_truncate(ws=s0.ws, job=job, i=i, j=j+1, formats=formats)
    i, j = write_data_line_staraniso(ws=s0.ws, job=job, i=i, j=j+1, formats=formats)
    return i, j


def write_data_line_dataset(ws: xlsxwriter.worksheet.Worksheet, dataset: dict, i: int, j: int, formats: dict):
    f = formats['data']['gray'][1]
    ws.write_string(i, j+0, dataset['dataset_name'], f)
    ws.write_string(i, j+1, dataset['dataset_dir'], f)
    ws.write_string(i, j+2, dataset['first_image'], f)
    ws.write_string(i, j+3, dataset['raw_data_dir'], f)
    ws.write_string(i, j+4, dataset['processed_data_dir'], f)
    ws.write_string(i, j+5, dataset['output_dir'], f)
    return i, j+5


def write_data_line_imginfo(ws: xlsxwriter.worksheet.Worksheet, imginfo: dict,
                            i: int, j: int, formats: dict):
    f = formats['data']['yellow'][0]
    ws.write_string(i, j+0, imginfo['_file'], f)
    ws.write_boolean(i, j+1, bool(imginfo['_file_exists']), f)
    ws.write_boolean(i, j+2, bool(imginfo['_is_parsed']), f)
    ws.write_boolean(i, j+3, bool(imginfo['_is_processed']), f)
    ws.write(i, j+4, imginfo['collection_time'], f)
    ws.write(i, j+5, imginfo['exposure_time'], f)
    ws.write(i, j+6, imginfo['detector_distance'], f)
    ws.write(i, j+7, imginfo['wavelength'], f)
    ws.write(i, j+8, imginfo['phi_angle'], f)
    ws.write(i, j+9, imginfo['omega_angle_start'], f)
    ws.write(i, j+10, imginfo['omega_angle_end'], f)
    ws.write(i, j+11, imginfo['omega_angle_step'], f)
    ws.write(i, j+12, imginfo['kappa_angle'], f)
    ws.write(i, j+13, imginfo['two_theta_angle'], f)
    ws.write(i, j+14, imginfo['beam_center_x'], f)
    ws.write(i, j+15, imginfo['beam_center_y'], f)
    ws.write(i, j+16, imginfo['no_images'], f)
    ws.write(i, j+17, imginfo['image_first'], f)
    ws.write(i, j+18, imginfo['image_last'], f)
    return i, j+18


def write_data_line_correct(ws: xlsxwriter.worksheet.Worksheet, xdscorrect: dict,
                            i: int, j: int, formats: dict):
    f0 = formats['data']['purple'][0]
    f1 = formats['data']['purple'][1]

    # xds.correct.input_params
    ws.write_string(i, j+0, xdscorrect['_file'], f0)
    ws.write_boolean(i, j+1, bool(xdscorrect['_file_exists']), f0)
    ws.write_boolean(i, j+2, bool(xdscorrect['_is_parsed']), f0)
    ws.write_boolean(i, j+3, bool(xdscorrect['_is_processed']), f0)

    ws.write(i, j+4, xdscorrect['input_params']['space_group_number'], f1)
    ws.write(i, j+5, xdscorrect['input_params']['unit_cell'][0], f1)
    ws.write(i, j+6, xdscorrect['input_params']['unit_cell'][1], f1)
    ws.write(i, j+7, xdscorrect['input_params']['unit_cell'][2], f1)
    ws.write(i, j+8, xdscorrect['input_params']['unit_cell'][3], f1)
    ws.write(i, j+9, xdscorrect['input_params']['unit_cell'][4], f1)
    ws.write(i, j+10, xdscorrect['input_params']['unit_cell'][5], f1)
    ws.write_boolean(i, j+11, bool(xdscorrect['input_params']['friedels_law']), f1)
    ws.write(i, j+12, xdscorrect['input_params']['image_template'], f1)
    ws.write(i, j+13, xdscorrect['input_params']['data_range'][0], f1)
    ws.write(i, j+14, xdscorrect['input_params']['data_range'][1], f1)
    ws.write(i, j+15, xdscorrect['input_params']['rotation_axis'][0], f1)
    ws.write(i, j+16, xdscorrect['input_params']['rotation_axis'][1], f1)
    ws.write(i, j+17, xdscorrect['input_params']['rotation_axis'][2], f1)
    ws.write(i, j+18, xdscorrect['input_params']['oscillation_angle'], f1)
    ws.write(i, j+19, xdscorrect['input_params']['wavelength'], f1)
    ws.write(i, j+20, xdscorrect['input_params']['polarization_fraction'], f1)
    ws.write(i, j+21, xdscorrect['input_params']['detector'], f1)
    ws.write(i, j+22, xdscorrect['input_params']['no_pixels_x'], f1)
    ws.write(i, j+23, xdscorrect['input_params']['no_pixels_y'], f1)
    ws.write(i, j+24, xdscorrect['input_params']['pixel_size_x'], f1)
    ws.write(i, j+25, xdscorrect['input_params']['pixel_size_y'], f1)
    ws.write(i, j+26, xdscorrect['input_params']['beam_center_x'], f1)
    ws.write(i, j+27, xdscorrect['input_params']['beam_center_y'], f1)
    ws.write(i, j+28, xdscorrect['input_params']['detector_distance'], f1)
    ws.write(i, j+29, xdscorrect['input_params']['beam_divergence_esd'], f1)
    ws.write(i, j+30, xdscorrect['input_params']['reflecting_range_esd'], f1)
    ws.write(i, j+31, xdscorrect['input_params']['spot_position_error_max'], f1)
    ws.write(i, j+32, xdscorrect['input_params']['spindle_position_error_max'], f1)

    # xds.correct.refined_params
    ws.write(i, j+33, xdscorrect['refined_params']['indexed_spots'], f0),
    ws.write(i, j+34, xdscorrect['refined_params']['deviation_spot_position'], f0),
    ws.write(i, j+35, xdscorrect['refined_params']['deviation_spindle_position'], f0),
    ws.write(i, j+36, xdscorrect['refined_params']['space_group_number'], f0),
    ws.write(i, j+37, xdscorrect['refined_params']['unit_cell'][0], f0),
    ws.write(i, j+38, xdscorrect['refined_params']['unit_cell'][1], f0),
    ws.write(i, j+39, xdscorrect['refined_params']['unit_cell'][2], f0),
    ws.write(i, j+40, xdscorrect['refined_params']['unit_cell'][3], f0),
    ws.write(i, j+41, xdscorrect['refined_params']['unit_cell'][4], f0),
    ws.write(i, j+42, xdscorrect['refined_params']['unit_cell'][5], f0),
    ws.write(i, j+43, xdscorrect['refined_params']['unit_cell_esd'][0], f0),
    ws.write(i, j+44, xdscorrect['refined_params']['unit_cell_esd'][1], f0),
    ws.write(i, j+45, xdscorrect['refined_params']['unit_cell_esd'][2], f0),
    ws.write(i, j+46, xdscorrect['refined_params']['unit_cell_esd'][3], f0),
    ws.write(i, j+47, xdscorrect['refined_params']['unit_cell_esd'][4], f0),
    ws.write(i, j+48, xdscorrect['refined_params']['unit_cell_esd'][5], f0),
    ws.write(i, j+49, xdscorrect['refined_params']['mosaicity'], f0),
    ws.write(i, j+50, xdscorrect['refined_params']['beam_center_x'], f0),
    ws.write(i, j+51, xdscorrect['refined_params']['beam_center_y'], f0),
    ws.write(i, j+52, xdscorrect['refined_params']['detector_distance'], f0),

    # xds.correct.isa
    ws.write(i, j+53, xdscorrect['isa']['value'], f1),
    ws.write(i, j+54, xdscorrect['isa']['params'][0], f1),
    ws.write(i, j+55, xdscorrect['isa']['params'][1], f1),
    return i, j+55


def write_data_line_validation(ws: xlsxwriter.worksheet.Worksheet, i: int, j: int, formats: dict):
    f = formats['data']['gray'][0]

    # xtl.validation
    # autoproc.imginfo vs xds.correct.input_params
    ws.write_formula(i, j+0, f'=ABS({rc2a1(i, 55)}-{rc2a1(i, 24)})', f)  # beam_center_x
    ws.write_formula(i, j+1, f'=ABS({rc2a1(i, 56)}-{rc2a1(i, 25)})', f)  # beam_center_y
    ws.write_formula(i, j+2, f'=ABS({rc2a1(i, 57)}-1000*{rc2a1(i, 16)})', f)  # detector_distance
    # xds.correct.input_params vs xds.correct.refined_params
    ws.write_formula(i, j+3, f'=ABS({rc2a1(i, 79)}-{rc2a1(i, 55)})', f)  # beam_center_x
    ws.write_formula(i, j+4, f'=ABS({rc2a1(i, 80)}-{rc2a1(i, 56)})', f)  # beam_center_y
    ws.write_formula(i, j+5, f'=ABS({rc2a1(i, 81)}-{rc2a1(i, 57)})', f)  # detector_distance
    return i, j+2


def write_data_line_s1(s1: Worksheet, i: int, j: int, job: dict, dataset: dict,
                       imginfo: dict, xdscorrect: dict, formats: dict):
    # Dataset sheet
    i, j = write_data_line_jobs(ws=s1.ws, job=job, i=i, j=j, formats=formats)
    i, j = write_data_line_dataset(ws=s1.ws, dataset=dataset, i=i, j=j+1, formats=formats)
    i, j = write_data_line_imginfo(ws=s1.ws, imginfo=imginfo, i=i, j=j+1, formats=formats)
    i, j = write_data_line_correct(ws=s1.ws, xdscorrect=xdscorrect, i=i, j=j+1, formats=formats)
    i, j = write_data_line_validation(ws=s1.ws, i=i, j=j+1, formats=formats)
    return i, j


def do_conditional_formatting(s0: Worksheet, s1: Worksheet, t: int, d: int, ht: int, hd: int):
    # autoproc.truncate [resolution.high, r_merge]
    # autoproc.staraniso [resolution.high, r_merge]
    # autoproc.staraniso [anisotropic_resolution]
    for col in [
        18, 19, 35, 36, 52, 53,
        82, 83, 99, 100, 116, 117,
        141, 142, 143
    ]:
        s0.conditional_format_color_scale(ht, col, t, col, percentile_mid=50,
                                          reverse=True)

    # autoproc.truncate [I/sigma, completeness, multiplicity, CC1/2]
    # autoproc.staraniso [I/sigma, completeness, multiplicity, CC1/2]
    for col in [
        26, 27, 28, 29, 43, 44, 45, 46, 60, 61, 62, 63,
        90, 91, 92, 93, 107, 108, 109, 110, 124, 125, 126, 127
    ]:
        s0.conditional_format_color_scale(ht, col, t, col, percentile_mid=50)

    # xds.correct.refined_params [deviation_spot_position, deviation_spindle_position, mosaicity]
    for col in [63, 64, 78]:
        s1.conditional_format_color_scale(hd, col, d, col, percentile_mid=50,
                                          reverse=True)

    # xds.correct.refined_params [ISa]
    s1.conditional_format_color_scale(hd, 82, d, 82, percentile_mid=25)

    # xtl.validation
    for col in [85, 86, 87, 88, 89, 90]:
        s1.conditional_format_color_scale(hd, col, d, col, percentile_mid=50,
                                          color_scale=rwr)


def write_data_line_s2(s2: Worksheet, i: int, j: int, eq_range: str):
    s2.ws.write_formula(i, j+0, f'=MIN({eq_range})')
    s2.ws.write_formula(i, j+1, f'=AVERAGE({eq_range})')
    s2.ws.write_formula(i, j+2, f'=MEDIAN({eq_range})')
    s2.ws.write_formula(i, j+3, f'=MAX({eq_range})')
    s2.ws.write_formula(i, j+4, f'=STDEV({eq_range})')


def write_sheet_s2(s0: Worksheet, s1: Worksheet, s2: Worksheet, formats: dict,
                   t: int, d: int, ht: int, hd: int):
    ### Statistics sheet
    bold = s0.register_body_style({'bold': True})
    italics = s0.register_body_style({'italic': True})
    s2.ws.write(0, 2, 'Min', bold)
    s2.ws.write(0, 3, 'Mean', bold)
    s2.ws.write(0, 4, 'Median', bold)
    s2.ws.write(0, 5, 'Max', bold)
    s2.ws.write(0, 6, 'Std', bold)

    s2.ws.write(1, 0, 'Resolution cutoff', bold)
    s2.ws.write(1, 1, 'isotropic', italics)
    write_data_line_s2(s2=s2, i=1, j=2,
                       eq_range=f'{s0.name}!{rc2a1(ht, 18)}:{rc2a1(t, 18)}')

    s2.ws.write(2, 1, 'anisotropic', italics)
    write_data_line_s2(s2=s2, i=2, j=2,
                       eq_range=f'{s0.name}!{rc2a1(ht, 82)}:{rc2a1(t, 82)}')
    s2.ws.write(1, 7, '\u212b')
    s2.ws.write(2, 7, '\u212b')

    s2.ws.write(3, 0, '⟨I/σ⟩', bold)
    s2.ws.write(3, 1, 'isotropic', italics)
    write_data_line_s2(s2=s2, i=3, j=2,
                       eq_range=f'{s0.name}!{rc2a1(ht, 26)}:{rc2a1(t, 26)}')

    s2.ws.write(4, 1, 'anisotropic', italics)
    write_data_line_s2(s2=s2, i=4, j=2,
                       eq_range=f'{s0.name}!{rc2a1(ht, 90)}:{rc2a1(t, 90)}')

    s2.ws.write(5, 0, 'Completeness', bold)
    s2.ws.write(5, 1, 'isotropic', italics)
    write_data_line_s2(s2=s2, i=5, j=2,
                       eq_range=f'{s0.name}!{rc2a1(ht, 27)}:{rc2a1(t, 27)}')

    s2.ws.write(6, 1, 'anisotropic', italics)
    write_data_line_s2(s2=s2, i=6, j=2,
                       eq_range=f'{s0.name}!{rc2a1(ht, 91)}:{rc2a1(t, 91)}')

    s2.ws.write(7, 0, 'Multiplicity', bold)
    s2.ws.write(7, 1, 'isotropic', italics)
    write_data_line_s2(s2=s2, i=7, j=2,
                       eq_range=f'{s0.name}!{rc2a1(ht, 28)}:{rc2a1(t, 28)}')

    s2.ws.write(8, 1, 'anisotropic', italics)
    write_data_line_s2(s2=s2, i=8, j=2,
                       eq_range=f'{s0.name}!{rc2a1(ht, 92)}:{rc2a1(t, 92)}')

    s2.ws.write(9, 0, 'CC1/2', bold)
    s2.ws.write(9, 1, 'isotropic', italics)
    write_data_line_s2(s2=s2, i=9, j=2,
                       eq_range=f'{s0.name}!{rc2a1(ht, 29)}:{rc2a1(t, 29)}')

    s2.ws.write(10, 1, 'anisotropic', italics)
    write_data_line_s2(s2=s2, i=10, j=2,
                       eq_range=f'{s0.name}!{rc2a1(ht, 93)}:{rc2a1(t, 93)}')

    s2.ws.write(11, 0, 'ISa', bold)
    write_data_line_s2(s2=s2, i=11, j=2,
                       eq_range=f'{s1.name}!{rc2a1(ht, 82)}:{rc2a1(t, 82)}')

    s2.ws.write(12, 0, 'Mosaicity', bold)
    write_data_line_s2(s2=s2, i=12, j=2,
                       eq_range=f'{s1.name}!{rc2a1(ht, 78)}:{rc2a1(t, 78)}')
    s2.ws.write(12, 7, '\u00b0')

    s2.ws.write(13, 0, 'Spot deviation', bold)
    write_data_line_s2(s2=s2, i=13, j=2,
                       eq_range=f'{s1.name}!{rc2a1(ht, 63)}:{rc2a1(t, 63)}')
    s2.ws.write(13, 7, 'px')

    s2.ws.write(14, 0, 'Spindle deviation', bold)
    write_data_line_s2(s2=s2, i=14, j=2,
                       eq_range=f'{s1.name}!{rc2a1(ht, 64)}:{rc2a1(t, 64)}')
    s2.ws.write(14, 7, '\u00b0')


@app.command('summarize', short_help='Create a summary file from multiple JSON files', epilog=epilog)
def cli_autoproc_summarize(
        datasets_file: Path = typer.Argument(metavar='<DATASETS.CSV>', help='Path to a CSV file containing dataset names',
                                             rich_help_panel='Input'),
        out_dir: Path = typer.Option(Path('./'), '-o', '--out-dir', help='Path to the output directory',
                                     rich_help_panel='Output'),
        csv_output: bool = typer.Option(False, '--csv', help='Export CSV file instead of XSLX',
                                        rich_help_panel='Output'),
        compact: bool = typer.Option(False, '-c', '--compact', help='Compact headers',
                                     rich_help_panel='XLSX formatting'),
        simple: bool = typer.Option(False, '-s', '--simple', help='Disable colored tables',
                                    rich_help_panel='XLSX formatting'),
        no_conditional: bool = typer.Option(False, '-f', '--no-conditional', help='Disable conditional formatting',
                                         rich_help_panel='XLSX formatting'),
        rotated: bool = typer.Option(False, '-r', '--rotated', help='Rotated headers',
                                     rich_help_panel='XLSX formatting'),
        no_units: bool = typer.Option(False, '-u', '--no-units', help='Disable units',
                                      rich_help_panel='XLSX formatting'),
        verbose: int = typer.Option(0, '-v', '--verbose', count=True, help='Print additional information',
                                    rich_help_panel='Debugging'),
        debug: bool = typer.Option(False, '--debug', help='Print debug information',
                                   rich_help_panel='Debugging')
):
    '''
    Reads a datasets_output.csv file and collects all the xtl_autoPROC.json files from the job directories to create a
    single summary XLSX file.

    Note that the CSV file must contain the following columns: 'job_dir', 'sweep_id', 'autoproc_id'.
    '''
    cli = Console(verbose=verbose, debug=debug)
    # Check if csv file exists
    if not datasets_file.exists():
        cli.print(f'File {datasets_file} does not exist', style='red')
        raise typer.Abort()

    cli.print(f'Parsing dataset names from {datasets_file}... ')
    datasets = apu.parse_csv(datasets_file, extra_headers=['job_dir', 'sweep_id', 'autoproc_id'])
    cli.print(f'Found {len(datasets["extra"]["job_dir"])} datasets')

    data = []
    # Debug will print the exceptions
    with Catcher(echo_func=cli.print, traceback_func=cli.print_traceback, silent=not debug) as catcher:
        for i, (j_dir, sweep_id, autoproc_id) in \
            enumerate(zip(datasets["extra"]['job_dir'], datasets["extra"]['sweep_id'],
                          datasets["extra"]['autoproc_id'])):
            if j_dir:
                j = Path(j_dir) / 'xtl_autoPROC.json'
                if j.exists():
                    job = {
                        'id': i,
                        'sweep_id': sweep_id,
                        'job_dir': j.parent.as_uri(),
                        'autoproc_id': autoproc_id
                    }
                    job.update(json.loads(j.read_text()))
                    data.append(job)

    if catcher.raised:
        if not debug:
            cli.print(f':police_car_light: An error occurred while parsing: {j}', style='red')
            error = catcher.errors[0]
            exc_type, exc_value, exc_tb = error
            cli.print(f'[b]{exc_type.__name__}[/]: {exc_value}', style='red')
        raise typer.Abort()

    cli.print(f'Found {len(data)} JSON files')
    if not data:
        return typer.Exit(code=0)

    # Create CSV file
    if csv_output:
        import pandas as pd
        # ToDo: Write custom dictionary flattener / remove pandas
        df = pd.json_normalize(data)
        df = apu.df_stringify(df)
        csv_file = out_dir / f'xtl_autoPROC_summary.csv'
        # ToDo: Split output to two CSV files, similar to XLSX
        df.to_csv(csv_file, index=False)
        cli.print(f'Wrote summary to {csv_file}')
        return typer.Exit(code=0)

    # Create XLSX file
    wb = xlsxwriter.Workbook(out_dir / 'xtl_autoPROC_summary.xlsx')
    header_options = HeaderOptions(
        compact=compact,
        colors=not simple,
        show_units=not no_units,
        rotated_headers=rotated,
        conditional_formatting=not no_conditional and not simple,
    )

    ### Create worksheets
    s0 = Worksheet(wb=wb, name='table1', o=header_options)
    s1 = Worksheet(wb=wb, name='datasets', o=header_options)
    s2 = Worksheet(wb=wb, name='statistics', o=header_options)
    # s3 = Worksheet(wb=wb, name='graphs', o=header_options)

    ### Define formats
    formats = get_workbook_formats(ws=s0)

    ### Add headers
    create_headers(s0=s0, s1=s1, formats=formats)

    ### Write headers to worksheets
    ht, _ = s0.write_headers()
    hd, _ = s1.write_headers()

    ### Add data
    t = ht + 1
    d = hd + 1
    for job in data:
        if len(job['datasets']) == 1:
            imginfos = {'0': job['autoproc.imginfo']}
            xdscorrects = {'0': job['xds.correct']}
        else:
            imginfos = job['autoproc.imginfo']
            xdscorrects = job['xds.correct']

        with Catcher(echo_func=cli.print, traceback_func=cli.print_traceback, silent=not debug) as catcher:
            write_data_line_s0(s0=s0, i=t, j=0, job=job, formats=formats)
            t += 1
        if catcher.raised:
            if not debug:
                cli.print(f':police_car_light: An error occurred while writing line {t} '
                          f'on the {s0.name} sheet', style='red')
                error = catcher.errors[0]
                exc_type, exc_value, exc_tb = error
                cli.print(f'[b]{exc_type.__name__}[/]: {exc_value}', style='red')
            raise typer.Abort()

        ### Datasets sheet
        for dataset, imginfo, xdscorrect in zip(job['datasets'], imginfos.values(), xdscorrects.values()):
            with Catcher(echo_func=cli.print, traceback_func=cli.print_traceback,
                         silent=not debug) as catcher:
                write_data_line_s1(s1=s1, i=d, j=0, job=job, dataset=dataset, imginfo=imginfo,
                                   xdscorrect=xdscorrect, formats=formats)
                d += 1
            if catcher.raised:
                if not debug:
                    cli.print(f':police_car_light: An error occurred while writing line {d} '
                        f'on the {s1.name} sheet', style='red')
                    error = catcher.errors[0]
                    exc_type, exc_value, exc_tb = error
                    cli.print(f'[b]{exc_type.__name__}[/]: {exc_value}', style='red')
                raise typer.Abort()

    # ### Sheet formatting
    # Hidden columns  [_file, _file_exists, _is_parsed, _is_processed]
    s0.ws.set_column(4, 7, None, None, {'hidden': True})    # autoproc.truncate
    s0.ws.set_column(68, 71, None, None, {'hidden': True})  # autoproc.staraniso
    s1.ws.set_column(10, 13, None, None, {'hidden': True})  # autoproc.imginfo
    s1.ws.set_column(29, 32, None, None, {'hidden': True})  # xds.correct

    # Freeze header rows and first column
    s0.ws.freeze_panes(ht + 1, 1)
    s1.ws.freeze_panes(hd + 1, 1)

    # Conditional formatting
    if s0.o.conditional_formatting:
        do_conditional_formatting(s0=s0, s1=s1, t=t, d=d, ht=ht, hd=hd)

    with Catcher(echo_func=cli.print, traceback_func=cli.print_traceback,
                 silent=not debug) as catcher:
        write_sheet_s2(s0=s0, s1=s1, s2=s2, formats=formats, t=t, d=d, ht=ht, hd=hd)
    if catcher.raised:
        if not debug:
            cli.print(f':police_car_light: An error occurred while writing '
                      f'{s2.name} sheet', style='red')
            error = catcher.errors[0]
            exc_type, exc_value, exc_tb = error
            cli.print(f'[b]{exc_type.__name__}[/]: {exc_value}', style='red')
        raise typer.Abort()

    # ### Graphs
    # c0 = s3.add_chart_scatter(
    #     range_x=[s0.name, ht, 0, t, 0],
    #     range_y=[s0.name, ht, 74, t, 74],
    #     location=(1, 1),
    #     title='Unit-cell parameter a', x_label='Job #', y_label='a (\u212b)'
    # )

    ### Set workbook properties
    wb.set_properties({
        'title': 'xtl_autoPROC_summary',
        'comments': f'Created from {datasets_file} by xtl {__version__} on {datetime.now()}',
    })

    # Save XLSX file
    while True:
        try:
            wb.close()
        except xlsxwriter.exceptions.FileCreateError:
            cli.print(f'Cannot write to {wb.filename}. Please close the file if open.', style='red')
            if cli.confirm('Try again?'):
                continue
        break

    cli.print(f'Wrote summary to {wb.filename}')
    return typer.Exit()
