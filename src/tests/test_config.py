#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import time

import numpy
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import pyram.pyram as pr
import pyram.config as config


def main():
    frequency = 50.0
    source_depth = 50.0
    receiver_depth = 50.0
    water_env = config.WaterEnvironment(
        depths=numpy.array([0, 100.0, 400]),
        ranges=numpy.array([0, 25000.0]),
        ssp=numpy.array([[1480, 1530.0], [1520, 1530.0], [1530, 1560.0]]),
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
        dr=500.0,
        dz=2.0,
    )

    # vr, vz, tlg, tll, cpg, cpl, c0, proc_time = pr.run(cfg)
    for i in range(4):
        now = time.time()
        vr, vz, tlg, tll, cpg, cpl, c0, proc_time = pr.run(cfg)
        print("proc_time", time.time() - now)

    k0 = 2 * numpy.pi * frequency / c0
    # pyram Fourier convention is S(omega) = int s(t) e^{i omega t} dt,
    # my preference is e^{-i omega t} so i take conjugate
    cpg = cpg.conj()
    cpg *= numpy.exp(-1j * vr * k0)  # this follows my preferred convention
    cpg = (
        -cpg / numpy.sqrt(vr * 8 * numpy.pi) * numpy.exp(-1j * numpy.pi / 4) / numpy.pi
    )  # add cylindrical spreading and scalings for comparison with KRAKEN

    tl = 20 * numpy.log10(
        numpy.abs(numpy.squeeze(cpg)) / numpy.max(numpy.abs(cpg))
    )

    plt.figure()
    plt.pcolormesh(vr, vz, tl, cmap="jet")
    plt.plot(bottom_env.bathy_ranges, bottom_env.bathy_depths)
    plt.axvline(25e3, color="w", linestyle="--")
    plt.colorbar()
    plt.gca().invert_yaxis()

    plt.show()


if __name__ == "__main__":
    main()
