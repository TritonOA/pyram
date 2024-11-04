# -*- coding: utf-8 -*-

from dataclasses import dataclass
import json

import numpy as np




@dataclass
class Environment:
    """Class representing the underwater environment for acoustic modeling.

    Attributes:
        frequency: The frequency of the sound source (Hz).
        source_depth: The depth of the sound source (m).
        receiver_depth: The depth of the receiver (m).
        ssp_depth: An `Mx1` array of depths for the sound speed profile (m).
        ssp_range: An `Nx1` array of ranges for the sound speed profile (km).
        ssp: An `MxN` array of sound speed values corresponding to the `M` 
            depths of `ssp_depth` and `N` ranges of `ssp_range` (m/s).
        bathymetry: An `Nx2` array where is the first column contains ranges 
            (km) and the second column contains depths (m) of the seabed.
        


    """


    frequency: float
    source_depth: float
    receiver_depth: float
    ssp_depth: np.ndarray
    ssp_range: np.ndarray
    ssp: np.ndarray
    bathymetry: np.ndarray
    seabed_range: np.ndarray
    seabed_


    def __post_init__(self): ...

    def to_json(self):
        return json.dumps(
            {
                "seabed": {
                    "depth": self.seabed.depth.tolist(),
                    "range": self.seabed.range.tolist(),
                },
                "water": {},  # Assuming Water has no attributes for now
                "source": {
                    "freq": self.source.freq,
                    "depth": self.source.depth,
                },
                "receiver": {
                    "depth": self.receiver.depth,
                },
            }
        )

    @staticmethod
    def from_json(json_str):
        data = json.loads(json_str)
        ...
