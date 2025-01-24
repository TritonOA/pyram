# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from pyram.config import Configuration
from pyram.pyram import Result


def plot_bathymetry(
    x: np.ndarray, z: np.ndarray, ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    ax.plot(x, z, **kwargs)
    return ax


def plot_transmission_loss(
    x: np.ndarray, z: np.ndarray, tl: np.ndarray, ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    im = ax.pcolormesh(x, z, tl, cmap="jet", **kwargs)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Transmission Loss (dB)")

    return ax


def plot_water_ssp(
    x: np.ndarray, z: np.ndarray, c: np.ndarray, ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    im = ax.pcolormesh(x, z, c, cmap="jet", **kwargs)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sound Speed (m/s)")

    return ax


# TODO: Plot bottom SSP
def plot_bottom_ssp():
    return


# TODO: Plot both water and bottom SSPs
def plot_ssp():
    return


def plot_result(
    config: Configuration, result: Result, figsize: tuple = (10, 8), dbvmin: float = None, dbvmax: float = None
) -> plt.Figure:
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize)

    ax = axs[0]
    ax.invert_yaxis()
    ax = plot_water_ssp(
        config.water_env.ranges, config.water_env.depths, config.water_env.ssp, ax=ax
    )
    ax = plot_bathymetry(
        config.bottom_env.bathy_ranges,
        config.bottom_env.bathy_depths,
        ax=ax,
        color="k",
        linestyle="--",
    )
    ax.set_xlim(0, config.rmax)
    ax.set_ylim(config.bottom_env.bottom_depths.max(), 0)

    ax = axs[1]
    ax.invert_yaxis()
    ax = plot_transmission_loss(result.vr, result.vz, result.tl, ax=ax, vmin=dbvmin, vmax=dbvmax)
    ax = plot_bathymetry(
        config.bottom_env.bathy_ranges,
        config.bottom_env.bathy_depths,
        ax=ax,
        color="k",
        linestyle="--",
    )
    ax.set_xlim(0, config.rmax)
    ax.set_ylim(config.bottom_env.bottom_depths.max(), 0)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Depth (m)")

    return fig
