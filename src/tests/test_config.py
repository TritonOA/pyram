#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import time

import numpy
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pyram import plotting
import pyram.config as config
import pyram.pyram as pr


def main():
    frequency = 50.0
    source_depth = 50.0
    receiver_depth = 50.0
    water_env = config.WaterEnvironment(
        depths=numpy.array([0, 100.0, 400]),
        ranges=numpy.array([0, 25000.0]),
        ssp=numpy.array([[1480, 1510.0], [1520, 1530.0], [1530, 1560.0]]),
    )
    bottom_env = config.BottomEnvironment(
        bathy_ranges=numpy.array([0.0, 40000.0]),
        bathy_depths=numpy.array([200.0, 400.0]),
        bottom_depths=numpy.array([[200.0, 400.0]]),
        bottom_ranges=numpy.array([0, 40000.0]),
        ssp=numpy.array([[1700.0, 1700.0]]),
        density=numpy.array([[1.5, 1.5]]),
        attenuation=numpy.array([[0.5, 0.5]]),
    )
    cfg = config.Configuration(
        frequency=frequency,
        source_depth=source_depth,
        receiver_depth=receiver_depth,
        water_env=water_env,
        bottom_env=bottom_env,
        dr=50.0,
        dz=2.0,
    )

    cfg.save("config.json")
    cfg2 = config.Configuration.load("config.json")

    for _ in range(4):
        now = time.time()
        result = pr.run(cfg2)
        print("proc_time", time.time() - now)

    fig = plotting.plot_result(cfg2, result)
    plt.show()


if __name__ == "__main__":
    main()
