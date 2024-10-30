# PyRAM

## Introduction

PyRAM is a Python adaptation of the Range-dependent Acoustic Model (RAM).
RAM was created by Michael D. Collins at the US Naval Research Laboratory.
This adaptation is of RAM v1.5, available from the [Ocean Acoustics Library](https://oalib-acoustics.org/models-and-software/parabolic-equation/).

The purpose of PyRAM is to provide a version of RAM which can be used within a Python interpreter environment (e.g. Spyder or the Jupyter notebook) and is
easier to understand, extend, and integrate into other applications than the Fortran version.
It is written in pure Python and achieves speeds comparable to native code by using the [Numba](https://numba.pydata.org) library for just-in-time (JIT) compilation.

The PyRAM package contains functions which largely correspond to the original Fortran subroutines and functions (including retaining the same names).
The variable names are also mostly the same. However some of the original code (e.g. subroutine `zread`) is unnecessary when the same purpose can be achieved using available Python library functions (e.g., from [NumPy](https://numpy.org) or [SciPy](https://scipy.org)) and has therefore been replaced.

A difference in functionality is that sound speed profile updates with range are decoupled from seabed parameter updates, which provides more flexibility in specifying the environment (e.g., if the data come from different sources).

PyRAM also provides various conveniences, e.g., automatic calculation of range and depth steps (though these can be overridden using keyword arguments).

## Installation

To install PyRAM, use pip:
```bash
pip install git+https://github.com/TritonOA/pyram.git
```

## Usage

To use PyRAM, you can import it in your Python script or Jupyter notebook:
```python
import pyram
```