# -*- coding: utf-8 -*-

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

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
        ssp: An `MxN` array of sound speed values corresponding to the `M`
            depths of `bottom_depths` and `N` ranges of `bottom_ranges` (m/s).
        density: An `MxN` array of density values corresponding to the `M`
            depths of `bottom_depths` and `N` ranges of `bottom_ranges` (kg/m^3).
        attenuation: An `MxN` array of attenuation values corresponding to
            the `M` depths of `bottom_depths` and `N` ranges of `bottom_ranges`
            (dB/m).
    """

    bathy_ranges: np.ndarray
    bathy_depths: np.ndarray
    bottom_depths: np.ndarray
    bottom_ranges: np.ndarray
    ssp: np.ndarray
    density: np.ndarray
    attenuation: np.ndarray

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

    @property
    def zmplt(self) -> float:
        return self.bathy_depths.max()

    def to_json(self) -> dict:
        return {
            "bathy_ranges": self.bathy_ranges.tolist(),
            "bathy_depths": self.bathy_depths.tolist(),
            "bottom_depths": self.bottom_depths.tolist(),
            "bottom_ranges": self.bottom_ranges.tolist(),
            "ssp": self.ssp.tolist(),
            "density": self.density.tolist(),
            "attenuation": self.attenuation.tolist(),
        }

    def _validate_inputs(self) -> None:
        self._validate_profile(self.ssp)
        self._validate_profile(self.density)
        self._validate_profile(self.attenuation)

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
        depths: An `Mx1` array of depths for the sound speed profile (m).
        ranges: An `Nx1` array of ranges for the sound speed profile (km).
        ssp: An `MxN` array of sound speed values corresponding to the `M`
            depths of `ssp_depth` and `N` ranges of `ssp_range` (m/s).
    """

    depths: np.ndarray
    ranges: np.ndarray
    ssp: np.ndarray

    def __post_init__(self) -> None:
        self._validate_inputs()

    @property
    def c0(self) -> float:
        cw = self.ssp
        return np.mean(cw[:, 0]) if len(cw.shape) > 1 else np.mean(cw)

    @property
    def is_range_dependent(self) -> bool:
        return self.ranges.size > 1

    def to_json(self) -> dict:
        return {
            "depths": self.depths.tolist(),
            "ranges": self.ranges.tolist(),
            "ssp": self.ssp.tolist(),
        }

    def _validate_inputs(self) -> None:
        if self.depths.size != self.ssp.shape[0]:
            raise DimensionMismatchError(
                f"`ssp_depths` size ({self.depths.size}) must match "
                f"`ssp.shape[0]` ({self.ssp.shape[0]})"
            )
        if self.ranges.size != self.ssp.shape[1]:
            raise DimensionMismatchError(
                f"`ssp_ranges` size ({self.ranges.size}) must match "
                f"`ssp.shape[1]` ({self.ssp.shape[1]})"
            )


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
        dz: Calculation depth step (m). Defaults to `dz_factor * wavelength`.
        ndz: Number of depth steps. Defaults to 1.
        dr: Calculation range step (km). Defaults to `num_pade * wavelength`.
        ndr: Number of range steps. Defaults to 1.
        num_pade: Number of Pade terms. Defaults to _np_default.
        dz_factor: Depth step factor. Defaults to 0.1.
        ns:
        lyrw:

    Properties:
        is_range_dependent: True if either the water or bottom environment is range dependent.
        rmax: Maximum range.
        rs: Source range.
        wavelength: Wavelength of the acoustic signal.
    """

    frequency: float
    source_depth: float
    receiver_depth: float
    water_env: WaterEnvironment
    bottom_env: BottomEnvironment
    dz: Optional[float] = None
    dr: Optional[float] = None
    ndz: int = 1
    ndr: int = 1
    num_pade: int = 8
    dz_factor: float = 0.1
    ns: int = 1
    lyrw: int = 20

    def __post_init__(self) -> None:
        self._validate_bathymetry()
        self._validate_depths()
        self.dr = self._set_dr() if self.dr is None else self.dr
        self.dz = self._set_dz() if self.dz is None else self.dz

    @property
    def is_range_dependent(self) -> bool:
        return any(
            (self.water_env.is_range_dependent, self.bottom_env.is_range_dependent)
        )

    @property
    def rmax(self) -> float:
        return max(
            self.water_env.ranges.max(),
            self.bottom_env.bathy_ranges.max(),
            self.bottom_env.bottom_ranges.max(),
        )

    @property
    def rs(self) -> float:
        return self.rmax + self.dr

    @property
    def wavelength(self) -> float:
        return self.water_env.c0 / self.frequency

    @classmethod
    def load(cls, file: Path) -> "Configuration":
        with open(file, "r") as f:
            data = json.load(f)

        water_env = WaterEnvironment(
            depths=np.array(data["water_env"]["depths"]),
            ranges=np.array(data["water_env"]["ranges"]),
            ssp=np.array(data["water_env"]["ssp"]),
        )

        bottom_env = BottomEnvironment(
            bathy_ranges=np.array(data["bottom_env"]["bathy_ranges"]),
            bathy_depths=np.array(data["bottom_env"]["bathy_depths"]),
            bottom_depths=np.array(data["bottom_env"]["bottom_depths"]),
            bottom_ranges=np.array(data["bottom_env"]["bottom_ranges"]),
            ssp=np.array(data["bottom_env"]["ssp"]),
            density=np.array(data["bottom_env"]["density"]),
            attenuation=np.array(data["bottom_env"]["attenuation"]),
        )

        return cls(
            frequency=data["frequency"],
            source_depth=data["source_depth"],
            receiver_depth=data["receiver_depth"],
            water_env=water_env,
            bottom_env=bottom_env,
            dz=data.get("dz"),
            dr=data.get("dr"),
            ndz=data["ndz"],
            ndr=data["ndr"],
            num_pade=data["num_pade"],
            dz_factor=data["dz_factor"],
            ns=data["ns"],
            lyrw=data["lyrw"],
        )

    def save(self, file: Path) -> None:
        with open(file, "w") as f:
            json.dump(self.to_json(), f, indent=True)

    def to_json(self) -> dict:
        return {
            "frequency": self.frequency,
            "source_depth": self.source_depth,
            "receiver_depth": self.receiver_depth,
            "water_env": self.water_env.to_json(),
            "bottom_env": self.bottom_env.to_json(),
            "dz": self.dz,
            "dr": self.dr,
            "ndz": self.ndz,
            "ndr": self.ndr,
            "num_pade": self.num_pade,
            "dz_factor": self.dz_factor,
            "ns": self.ns,
            "lyrw": self.lyrw,
        }

    def _set_dr(self) -> float:
        return self.num_pade * self.wavelength

    def _set_dz(self) -> float:
        return self.dz_factor * self.wavelength

    def _validate_bathymetry(self) -> None:
        if self.bottom_env.bathy_depths.max() > self.water_env.depths[-1]:
            raise ValueError(
                "Deepest sound speed point must be at or below deepest bathymetry point."
            )

    def _validate_depth(self, z: float) -> None:
        z_ss = self.water_env.depths
        if not z_ss[0] <= z <= z_ss[-1]:
            raise ValueError(
                f"Depth {z} not within min/max sound speed depths {z_ss[0]}/{z_ss[-1]}."
            )
        return

    def _validate_depths(self) -> None:
        self._validate_depth(self.source_depth)
        self._validate_depth(self.receiver_depth)
