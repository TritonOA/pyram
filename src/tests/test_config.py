#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import numpy
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import pyram.pyram as rpr
import pyram.config as config


def main():
    frequency = 50.0
    source_depth = 50.0
    receiver_depth = 50.0
    water_env = config.WaterEnvironment(
        ssp_depths=numpy.array([0, 100.0, 400]),
        ssp_ranges=numpy.array([0, 25000.0]),
        ssp=numpy.array([[1480, 1530.0], [1520, 1530.0], [1530, 1560.0]]),
    )
    bottom_env = config.BottomEnvironment(
        bathy_ranges=numpy.array([0.0, 40000.0]),
        bathy_depths=numpy.array([200.0, 400.0]),
        bottom_depths=numpy.array([[400.0]]),
        bottom_ranges=numpy.array([0, 40000.0]),
        bottom_ssp=numpy.array([[1700.0, 1700.0]]),
        bottom_density=numpy.array([[1.5, 1.5]]),
        bottom_attenuation=numpy.array([[0.5, 0.5]]),
    )
    cfg = config.Configuration(
        frequency=frequency,
        source_depth=source_depth,
        receiver_depth=receiver_depth,
        water_env=water_env,
        bottom_env=bottom_env,
    )
    print(cfg.is_range_dependent)
    return


if __name__ == "__main__":
    main()
