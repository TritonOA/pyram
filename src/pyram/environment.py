# -*- coding: utf-8 -*-

from dataclasses import dataclass

import numpy as np

@dataclass
class Seabed:
    depth: np.ndarray
    range: np.ndarray

@dataclass
class Water:
    ...

@dataclass
class Source:
    freq: float
    depth: float


@dataclass
class Receiver:
    depth: float
