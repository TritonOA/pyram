# -*- coding: utf-8 -*-

from dataclasses import dataclass
import json

import numpy as np


class DimensionMismatchError(Exception):
    pass


@dataclass
class BottomEnvironment:
    """Ocean bottom environment definition.

    Attributes:
        bathy_ranges: An `Nx1` array containing ranges (km) corresponding to `bathy_depths`.
        bathy_depths: An `Nx1` array of bathymetry (m) at the `bathy_ranges`.
        bottom_depths: An `Mx1` array of depths (m) for the bottom profile.
        bottom_ranges: An `Nx1` array of ranges (km) for the bottom profile.
        bottom_ssp: An `MxN` array of sound speed values corresponding to the `M`
            depths of `bottom_depths` and `N` ranges of `bottom_ranges` (m/s).
        bottom_density: An `MxN` array of density values corresponding to the `M`
            depths of `bottom_depths` and `N` ranges of `bottom_ranges` (kg/m^3).
        bottom_attenuation: An `MxN` array of attenuation values corresponding to
            the `M` depths of `bottom_depths` and `N` ranges of `bottom_ranges`
            (dB/m).
    """

    bathy_ranges: np.ndarray
    bathy_depths: np.ndarray
    bottom_depths: np.ndarray
    bottom_ranges: np.ndarray
    bottom_ssp: np.ndarray
    bottom_density: np.ndarray
    bottom_attenuation: np.ndarray

    def __post_init__(self) -> None:
        self._validate_inputs()

    @property
    def bathy_is_range_dependent(self) -> bool:
        return self.bathy_ranges.size > 1

    @property
    def bottom_is_range_dependent(self) -> bool:
        return self.bottom_ranges.size > 1

    @property
    def is_range_dependent(self) -> bool:
        return any((self.bathy_is_range_dependent, self.bottom_is_range_dependent))

    def _validate_inputs(self) -> None:
        self._validate_profile(self.bottom_ssp)
        self._validate_profile(self.bottom_density)
        self._validate_profile(self.bottom_attenuation)

    def _validate_profile(self, field: np.ndarray) -> None:
        num_depths = self.bottom_depths.shape[0]
        num_ranges = self.bottom_ranges.size
        field_shape = field.shape
        if (field_shape[0] != num_depths) or (field_shape[1] != num_ranges):
            raise DimensionMismatchError(
                f"Field shape {field_shape} must match bottom depths "
                f"{num_depths} and ranges {num_ranges}"
            )


@dataclass
class WaterEnvironment:
    """Underwater environment definition.

    Attributes:
        ssp_depths: An `Mx1` array of depths for the sound speed profile (m).
        ssp_ranges: An `Nx1` array of ranges for the sound speed profile (km).
        ssp: An `MxN` array of sound speed values corresponding to the `M`
            depths of `ssp_depth` and `N` ranges of `ssp_range` (m/s).
    """

    ssp_depths: np.ndarray
    ssp_ranges: np.ndarray
    ssp: np.ndarray

    def __post_init__(self) -> None:
        self._validate_inputs()

    @property
    def is_range_dependent(self) -> bool:
        return self.ssp_ranges.size > 1

    def _validate_inputs(self) -> None:
        if self.ssp_depths.size != self.ssp.shape[0]:
            raise DimensionMismatchError(
                f"`ssp_depths` size ({self.ssp_depths.size}) must match "
                f"`ssp.shape[0]` ({self.ssp.shape[0]})"
            )
        if self.ssp_ranges.size != self.ssp.shape[1]:
            raise DimensionMismatchError(
                f"`ssp_ranges` size ({self.ssp_ranges.size}) must match "
                f"`ssp.shape[1]` ({self.ssp.shape[1]})"
            )

    @property
    def c0(self) -> float:
        cw = self.ssp
        return np.mean(cw[:, 0]) if len(cw.shape) > 1 else np.mean(cw)


@dataclass
class Configuration:
    """RAM configuration.

    Attributes:
        frequency: The frequency of the acoustic signal (Hz).
        source_depth: The depth of the source (m).
        receiver_depth: The depth of the receiver (m).
        water_env: An instance of `WaterEnvironment` representing the water
            column environment.
        bottom_env: An instance of `BottomEnvironment` representing the
            seabed environment.
    """

    frequency: float
    source_depth: float
    receiver_depth: float
    water_env: WaterEnvironment
    bottom_env: BottomEnvironment
    dz: float = 2.0
    ndz: int = 1
    dr: float = 500.0
    ndr: int = 1
    nump: int = 8
    dzf: float = 0.1
    ns_default: int = 1
    lyrw_default: int = 20
    run_id_default: int = 0

    def __post_init__(self) -> None:
        self._validate_bathymetry()
        self._validate_depths()

    @property
    def is_range_dependent(self) -> bool:
        return any(
            (self.water_env.is_range_dependent, self.bottom_env.is_range_dependent)
        )

    def _validate_bathymetry(self) -> None:
        if self.bottom_env.bathy_depths.max() > self.water_env.ssp_depths[-1]:
            raise ValueError(
                "Deepest sound speed point must be at or below deepest bathymetry point."
            )

    def _validate_depths(self) -> None:
        self._validate_depth(self.source_depth)
        self._validate_depth(self.receiver_depth)

    def _validate_depth(self, z: float) -> None:
        z_ss = self.water_env.ssp_depths
        if not z_ss[0] <= z <= z_ss[-1]:
            raise ValueError(
                f"Depth {z} not within min/max sound speed depths {z_ss[0]}/{z_ss[-1]}."
            )
        return

    @property
    def wavelength(self) -> float:
        return self.water_env.c0 / self.frequency
