# XTL: Crystallographic Tools Library

XTL is a collection of library code for everyday tasks in the workflow of the macromolecular crystallographer.

## Features
### API
- [`xtl.diffraction`](https://github.com/dtriand/xtl/tree/master/src/xtl/diffraction) : Interacting with diffraction images
  - [`Image`](https://github.com/dtriand/xtl/blob/master/src/xtl/diffraction/images/images.py) : Main interface for diffraction images
  - [`ImageMask`](https://github.com/dtriand/xtl/blob/master/src/xtl/diffraction/images/images.py) : Image masks
  - [`AzimuthalIntegrator1D/2D`](https://github.com/dtriand/xtl/blob/master/src/xtl/diffraction/images/integrators.py) : 1D and 2D azimuthal integration
  - [`AzimuthalCrossCorrelatorQQ_1`](https://github.com/dtriand/xtl/blob/master/src/xtl/diffraction/images/correlators.py) : Azimuthal cross-correlation
- [`xtl.pdbapi`](https://github.com/dtriand/xtl/tree/master/src/xtl/pdbapi) : REST API for RCSB PDB
  - [`Client`](https://github.com/dtriand/xtl/tree/master/src/xtl/pdbapi/client.py) : Client for search or data queries to RCSB PDB
- [`xtl.files`](https://github.com/dtriand/xtl/tree/master/src/xtl/files) : Custom file formats
  - [`NpxFile`](https://github.com/dtriand/xtl/tree/master/src/xtl/files/npx.py) : NPX file format (storing Numpy arrays w/ metadata)
- [`xtl.common`](https://github.com/dtriand/xtl/tree/master/src/xtl/common) : Shared utilities
  - [`Options`](https://dtriand.github.io/xtl/common/options.html) : API around [`pydantic.BaseModel`](https://docs.pydantic.dev/latest/concepts/models/)
  - [`Table`](https://dtriand.github.io/xtl/common/tables.html) : Structured tabular data

### CLIs
- [`xtl.autoproc`](https://dtriand.github.io/xtl/cli/xtl.autoproc.html) : Run multiple [autoPROC](https://www.globalphasing.com/autoproc/) jobs in parallel
- `xtl.diffraction` : Plot diffraction images, perform azimuthal integrations and cross-correlations
- `xtl.math spacing` : Convert between various definitions of scattering angle (2θ, d, q, etc.)

## Requirements
- Python 3.10

## Installation
XTL can be installed using:
```shell
pip install xtl
```

or directly from GitHub:
```shell
git clone https://github.com/dtriand/xtl .
pip install .
```

## Documentation
An, admittedly incomplete, API and CLI documentation is available at 
[dtriand.github.io/xtl](https://dtriand.github.io/xtl/index.html).

## Contribute

- Source code: [github.com/dtriand/xtl](https://github.com/dtriand/xtl)
- Issue tracker: [github.com/dtriand/xtl/issues](https://github.com/dtriand/xtl/issues)

## License
XTL is licensed under GNU GPLv3.
