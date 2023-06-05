# Fiber LP modes

Compute LP modes from GRIN fibers. The modes are computed using analytical formulas, under the weak guidance assumption.

## Installation

Use in virtual environment or you own environment:

```bash
$ python -m pip install -r requirements.txt
$ pip install -e .
```

## Main features

- Fiber: GRIN fiber with parameters to compute theoretical LP modes effective refractive index.
- Grid: 2D grid supporting LP modes 2D scalar fields computation.
- Speckle: Sum of GRIN fiber supported LP modes with complex random weights (energy is conserved), composition or decomposition is available.
- Beams: Beam definition on a 2D grid support (Gaussian, Bessel, Bessel with Gaussian envelope).
- Devices: Essentially supporting macropixel phase-shifts and support constraints to the input field. Aims at mimicking an ideal deformable mirror.
- Coupling: Couples an optical field to the GRIN fiber using LP modes decomposition. Scrambles the mode through a LP modes coupling matrix.
- Datasets: Generates datasets on support 2D grid for later usage: Pure GRIN LP modes, GRIN speckles (limited to N modes), or Propagated GRIN speckles (using defined mode coupling matrix).
- Transforms: Fourier and Fresnel transforms.

## Examples

- See [./notebooks/tutorial.ipynb](./notebooks/tutorial.ipynb) for basic library usage.
- See [./notebooks/generate_dataset.ipynb](./notebooks/generate_dataset.ipynb) for basic dataset generation.

## To-Do

- Refactor architecture: Speckle is a sum of scalar fields, Beam is a scalar field, MockDeformableMirror is a particular scalar field.
- Profile code and speed-up time consuming functions (particularly for heavy dataset generation).
- Add support for Step-Index fibers.

## Notes

- Tested with Python 3.9.
- Matlab files are outdated and not supported anymore. They will slowly get deleted once their features are implemented on Python.
