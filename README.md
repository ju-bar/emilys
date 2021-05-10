# emilys

Electron Microscopy Image anaLYSis tools

Version: 0.1.5

## Authors and Copyright

Juri Barthel, 
Forschungszentrum Jülich GmbH, 52425 Jülich, Germany

Copyright (c) 2019 - 2021 - Forschungszentrum Jülich GmbH
   
Published under the GNU General Public License, version 3,
see <http://www.gnu.org/licenses/> and LICENSE!

## Installation

If you want to play in the code, copy this source tree to some place, where your Python environment is able to find it.

If you just want to use it, install via 

    pip install emilys

## Changes

* Version 0.1.5:
added debug to emilys.image.peakfit.fit_local_gauss_2d
added sub-package structure for handling atomic structure models used in TEM simulations
* Version 0.1.4:
modified some interfaces and routines which were causing deprecation warnings with numba.jit decorators
  This kind of adoption will need to be updated step-by-step whenever it occurs. 
  Deprecated numba functions will not be supported in later versions of numba.
* Version 0.1.3:
added peak fitting to shift measurements as additional method
modified linear search for minimum on data
changed some functions related to fitting output
added image resampling code to imagedata
* Version 0.1.2:
removed ineffective numba jit decorators from routines in image.polar, 
renamed image.polar.polar_radpol3_trasform to polar_radpol3_rebin
added image.polar_radpol2_resample
added numerics.roots.py with some primitive root finding functions
* Version 0.1.1:
added diffraction pattern data for STO_110_* with json file containing meta data, 
modified image.polar.polar_resample,
renamed and modiefied image.polar.polar_transform to image.polar.polar_rebin,
added a test Jupyter notebook showing how to use polar_resample from a non-isotropic input grid,
updated PiPy upload with install requirements
* Version 0.1.0:
packaging of some initial functions, uploaded to PyPi
