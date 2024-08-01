"""outpt function definition"""

from numba import jit, int64, float64, complex128
import numpy as np
import numpy.typing as npt


@jit(
    int64[:](
        float64,
        int64,
        int64,
        int64,
        int64,
        float64[:],
        complex128[:],
        float64,
        int64,
        float64[:],
        float64[:, :],
        complex128[:],
        complex128[:, :],
    ),
    nopython=True,
)
def outpt(
    r: float,
    mdr: int,
    ndr: int,
    ndz: int,
    tlc: int,
    f3: npt.ArrayLike,
    u: npt.ArrayLike,
    _dir: float,
    ir: int,
    tll: npt.ArrayLike,
    tlg: npt.ArrayLike,
    cpl: npt.ArrayLike,
    cpg: npt.ArrayLike,
) -> np.ndarray:
    """Output transmission loss and complex pressure.

    Note: Complex pressure does not include cylindrical spreading term 1/sqrt(r)
    or phase term exp(-j*k0*r).
    """

    eps = 1e-20

    mdr += 1
    if mdr == ndr:
        mdr = 0
        tlc += 1
        cpl[tlc] = (1 - _dir) * f3[ir] * u[ir] + _dir * f3[ir + 1] * u[ir + 1]
        temp = 10 * np.log10(r + eps)
        tll[tlc] = -20 * np.log10(np.abs(cpl[tlc]) + eps) + temp

        for i in range(tlg.shape[0]):
            j = (i + 1) * ndz
            cpg[i, tlc] = u[j] * f3[j]
            tlg[i, tlc] = -20 * np.log10(np.abs(cpg[i, tlc]) + eps) + temp

    return np.array([mdr, tlc], dtype=np.int64)
