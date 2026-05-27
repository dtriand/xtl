from __future__ import annotations

from datetime import datetime

from xtl.common.options import Option, Options
from xtl.scattering.files import Scattering1DFileType


class ScatteringProfileMetadata(Options):
    """
    Metadata for a 1D scattering profile that were extracted from a file.
    """

    origin_file_type: Scattering1DFileType | None = \
        Option(
            default=None, desc='Type of the file where the profile originates from'
        )
    name: str | None = \
        Option(
            default=None, desc='Dataset title'
        )
    wavelength: float | None = \
        Option(
            default=None, ge=0.0, desc='Wavelength in Angstroms'
        )


class AtsasDatScatteringProfileMetadata(ScatteringProfileMetadata):
    """
    Metadata for a 1D scattering profile extracted from an ATSAS DAT file.
    """

    origin_file_type: Scattering1DFileType = Option(
        default=Scattering1DFileType.DAT, desc='Type of the file where the profile originates from'
    )
    description: str | None = Option(
        default=None, desc='Sample description',
        alias='Sample description'
    )
    parents: list[str] = Option(
        default_factory=list, desc='Parent scattering profiles',
        alias='Parent(s)'
    )
    working_directory: str | None = Option(
        default=None, desc='Working directory for data reduction and processing',
        alias='working-directory'
    )
    angular_axis_file: str | None = Option(
        default=None, desc='File containing angular axis information',
        alias='angular-axis-file'
    )
    mask_file: str | None = Option(
        default=None, desc='Mask file for data reduction',
        alias='beamstop-mask-file'
    )
    detector_response_file: str | None = Option(
        default=None, desc='File containing detector response information',
        alias='detector-response-file'
    )
    flatfield_file: str | None = Option(
        default=None, desc='File containing flat-field information',
        alias='flatfield-file'
    )
    beam_center_x: float | None = Option(
        default=None, desc='Beam center x coordinate',
        alias='beam-center-x'
    )
    beam_center_y: float | None = Option(
        default=None, desc='Beam center y coordinate',
        alias='beam-center-y'
    )
    unit_time: float | None = Option(
        default=None, alias='unit-time'
    )
    channel_first: int | None = Option(
        default=None, alias='channel-first'
    )
    channel_last: int | None = Option(
        default=None, alias='channel-last'
    )
    angular_unit: str | None = Option(
        default=None, desc='Units for radial axis',
        alias='angular-unit'
    )
    petra_current: float | None = Option(
        default=None, desc='PETRA III current in mA',
        alias='PETRA Current'
    )
    bragg_angle: float | None = Option(
        default=None, alias='Bragg Angle [°]'
    )
    wavelength: float | None = Option(
        default=None, desc='Wavelength in nm',
        alias='Wavelength [nm]'
    )
    energy: float | None = Option(
        default=None, desc='Energy in eV',
        alias='Energy [eV]'
    )
    qbpm_1: float | None = Option(
        default=None, alias='QBPM 1'
    )
    diode_1: float | None = Option(
        default=None, alias='Diode 1'
    )
    diode_2: float | None = Option(
        default=None, alias='Diode 2'
    )
    piezo_x: float | None = Option(
        default=None, desc='SEU2B piezo motor x coordinate in mm',
        alias='Piezo motor X position [mm]'
    )
    piezo_y: float | None = Option(
        default=None, desc='SEU2B piezo motor y coordinate in mm',
        alias='Piezo motor Y position [mm]'
    )
    piezo_z: float | None = Option(
        default=None, desc='SEU2B piezo motor z coordinate in mm',
        alias='Piezo motor Z position [mm]'
    )
    detector_distance: float | None = Option(
        default=None, desc='Sample to detector distance in m',
        alias='Detector Distance [m]'
    )
    sample_code: str | None = Option(
        default=None, desc='Sample code',
        alias='Code'
    )
    exposure_delay: float | None = Option(
        default=None, desc='Exposure delay in s',
        alias='Exposure Delay [s]'
    )
    concentration: float | None = Option(
        default=None, desc='Concentration in mg/mL',
        alias='Concentration [mg/ml]'
    )
    contrast: float | None = Option(
        default=None, desc='Sample contrast in 10^10 cm^-2',
        alias='Contrast [10^10cm^-2]'
    )
    exposure_period: float | None = Option(
        default=None, desc='Exposure period in s',
        alias='Exposure period [s]'
    )
    buffer: str | None = Option(
        default=None, desc='Buffer description',
        alias='Buffer'
    )
    run_no: int | None = Option(
        default=None, desc='Run number',
        alias='Run Number'
    )
    detector: str | None = Option(
        default=None, desc='Detector name',
        alias='Detector Name'
    )
    partial_specific_volume: float | None = Option(
        default=None, desc='Partial specific volume in cm^3/g',
        alias='Partial Specific Volume [cm^3/g]'
    )
    exposure_time: float | None = Option(
        default=None, desc='Exposure time in s',
        alias='Exposure time [s]'
    )
    transmission: float | None = Option(
        default=None, desc='Beam transmission in %',
        alias='Transmitted Beam (mean)'
    )
    time_start: datetime | None = Option(
        default=None, desc='Start time of data collection',
        alias='Timestamp (begin)'
    )
    time_end: datetime | None = Option(
        default=None, desc='End time of data collection',
        alias='Timestamp (end)'
    )
    creator: str | None = Option(
        default=None, desc='Program that created the file',
        alias='creator'
    )
    creator_version: str | None = Option(
        default=None, desc='Version of creator program',
        alias='creator-version'
    )
    frame_no: int | None = Option(
        default=None, desc='Frame number',
        alias='Frame Number'
    )
    extra: list[tuple[str, str]] = Option(
        default_factory=list, desc='Extra metadata lines that could not be parsed'
    )

    @classmethod
    def from_dat_kwargs(cls, metadata: dict) -> AtsasDatScatteringProfileMetadata:
        """
        Initialize a AtsasDatScatteringProfileMetadata instance from a dictionary of key-valued metadata extracted from
        an ATSAS DAT file.

        :param metadata: Dictionary of key-valued metadata extracted from an ATSAS DAT file, where keys are the original
            metadata names
        :return:
        """

        aliased = {field.serialization_alias: name for name, field in cls.__pydantic_fields__.items()
                   if field.serialization_alias is not None}
        kwargs = { 'extra': [] }
        for key, value in metadata.items():
            if key in aliased:
                kwargs[aliased[key]] = value
            else:
                if key in ['Sample', 'parent']:
                    # Skip these two keys, since they can be reconstructed from other metadata:
                    # - concentration + code -> Sample
                    # - Parent(s) -> parent
                    continue
                kwargs['extra'].append((key, value))

        return cls(**kwargs)


ScatteringProfileMetadataType = ScatteringProfileMetadata | AtsasDatScatteringProfileMetadata