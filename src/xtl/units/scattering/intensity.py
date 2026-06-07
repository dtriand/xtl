from xtl.units.base import Units, UnitsDescription


class IntensityUnits(Units):
    ARB = (
        'arbitrary',
        UnitsDescription(
            name='intensity_arbitrary',
            desc='Intensity in arbitrary units',
            repr='Intensity (A.U.)',
            pretty='I (A.U.)',
            latex=r'$\mathrm{Intensity\ (A.U.)}$',
        )
    )

    ABS = (
        'absolute',
        UnitsDescription(
            name='intensity_absolute',
            desc='Intensity in absolute units',
            repr='Intensity (Abs)',
            pretty='I (Abs)',
            latex=r'$\mathrm{Intensity\ (Abs)}$',
        )
    )
