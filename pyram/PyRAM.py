"""
Description:
    A jitted function that takes in the mesh parameters and computes the parabolic equation solution 

Date:
    09/24/2024

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy
from pyram.matrc import matrc
from pyram.solve import solve
from pyram.outpt import outpt
from numba import jit, njit, jit_module


def solve_field(
    freq, zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb, dr, dz
):

    # Check input dims and get flags indicating which model aspects are range-dependent
    rd_ss, rd_sb, rd_bt = check_inputs(
        zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb
    )

    # get model parameters (num pade coefficients, approximate wavelength, max plot depth, ...
    np, c0, lam, zmplt, rmax, ns, rs, lyrw, run_id = get_params(
        freq, zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb, dr, dz
    )

    ndr, ndz = 1, 1  # stride for saving outputs

    (
        u,
        v,
        ksq,
        ksqw,
        ksqb,
        alpw,
        alpb,
        p_rhob,
        r1,
        r2,
        r3,
        s1,
        s2,
        s3,
        f1,
        f2,
        f3,
        pd1,
        pd2,
        vr,
        vz,
        tll,
        tlg,
        cpl,
        cpg,
        tlc,
        ss_ind,
        sb_ind,
        bt_ind,
        zmax,
        nz,
        iz,
        nzplt,
        r,
        mdr,
        ir,
        ddir,
    ) = setup(
        freq,
        zs,
        zr,
        z_ss,
        rp_ss,
        cw,
        z_sb,
        rp_sb,
        cb,
        rhob,
        attn,
        rbzb,
        dr,
        dz,
        ndr,
        ndz,
        np,
        c0,
        lam,
        zmplt,
        rmax,
        ns,
        rs,
        lyrw,
        run_id,
    )

    nr = int(numpy.round(rmax / dr)) - 1

    for rn in range(nr):
        iz = updat(
            freq,
            zs,
            zr,
            z_ss,
            rp_ss,
            cw,
            z_sb,
            rp_sb,
            cb,
            rhob,
            attn,
            rbzb,
            dr,
            dz,
            zmax,
            np,
            c0,
            lam,
            zmplt,
            rmax,
            ns,
            rs,
            lyrw,
            run_id,
            u,
            v,
            ksq,
            ksqw,
            ksqb,
            alpw,
            alpb,
            p_rhob,
            r1,
            r2,
            r3,
            s1,
            s2,
            s3,
            f1,
            f2,
            f3,
            pd1,
            pd2,
            vr,
            vz,
            tll,
            tlg,
            cpl,
            cpg,
            tlc,
            ss_ind,
            sb_ind,
            bt_ind,
            rd_ss,
            rd_sb,
            rd_bt,
            nz,
            iz,
            nzplt,
            r,
        )

        solve(u, v, s1, s2, s3, r1, r2, r3, iz, nz, np)

        r = (rn + 2) * dr

        mdr, tlc = outpt(r, mdr, ndr, ndz, tlc, f3, u, ddir, ir, tll, tlg, cpl, cpg)[:]

    proc_time = 0.0
    return vr, vz, tlg, tll, cpg, cpl, c0, proc_time


def check_inputs(zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb):
    """Basic checks on dimensions of inputs"""

    _status_ok = True

    # Source and receiver depths
    if not z_ss[0] <= zs <= z_ss[-1]:
        _status_ok = False
        raise ValueError("Source depth outside sound speed depths")
    if not z_ss[0] <= zr <= z_ss[-1]:
        _status_ok = False
        raise ValueError("Receiver depth outside sound speed depths")
    if _status_ok:
        z_ss = z_ss

    # Water sound speed profiles
    num_depths = z_ss.size
    num_ranges = rp_ss.size
    cw_dims = cw.shape
    if (cw_dims[0] == num_depths) and (cw_dims[1] == num_ranges):
        rp_ss, cw = rp_ss, cw
    else:
        raise ValueError("Dimensions of z_ss, rp_ss and cw must be consistent.")

    # Seabed profiles
    num_depths = z_sb.shape[0]
    num_ranges = rp_sb.size
    for prof in [cb, rhob, attn]:
        prof_dims = prof.shape
        if (prof_dims[0] != num_depths) or (prof_dims[1] != num_ranges):
            _status_ok = False
    if _status_ok:
        rp_sb, cb, rhob, attn = rp_sb, cb, rhob, attn
    else:
        raise ValueError(
            "Dimensions of z_sb, rp_sb, cb, rhob and attn must be consistent."
        )

    if rbzb[:, 1].max() <= z_ss[-1]:
        rbzb = rbzb
    else:
        _status_ok = False
        raise ValueError(
            "Deepest sound speed point must be at or below deepest bathymetry point."
        )

    # Set flags for range-dependence (water SSP, seabed profile, bathymetry)
    rd_ss = True if rp_ss.size > 1 else False
    rd_sb = True if rp_sb.size > 1 else False
    rd_bt = True if rbzb.shape[0] > 1 else False
    return rd_ss, rd_sb, rd_bt


# @njit
def get_params(
    freq, zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb, dr, dz
):
    np_default = 8
    dzf = 0.1
    ns_default = 1
    lyrw_default = 20
    run_id_default = 0

    np = np_default

    c0 = numpy.mean(cw[:, 0]) if len(cw.shape) > 1 else numpy.mean(cw)

    lam = c0 / freq

    zmplt = rbzb[:, 1].max()
    tmp_arr = numpy.zeros(3)
    tmp_arr[0] = rp_ss.max()
    tmp_arr[1] = rp_sb.max()
    tmp_arr[2] = rbzb[:, 0].max()
    rmax = numpy.max(tmp_arr)
    ns = ns_default
    rs = rmax + dr

    lyrw = lyrw_default

    run_id = run_id_default

    return np, c0, lam, zmplt, rmax, ns, rs, lyrw, run_id


# @njit
def setup(
    freq,
    zs,
    zr,
    z_ss,
    rp_ss,
    cw,
    z_sb,
    rp_sb,
    cb,
    rhob,
    attn,
    rbzb,
    dr,
    dz,
    ndr,
    ndz,
    np,
    c0,
    lam,
    zmplt,
    rmax,
    ns,
    rs,
    lyrw,
    run_id,
):
    """Initialise the parameters, acoustic field, and matrices"""

    if rbzb[-1, 0] < rmax:
        rbzb = numpy.append(rbzb, numpy.array([[rmax, rbzb[-1, 1]]]), axis=0)

    eta = 1 / (40 * numpy.pi * numpy.log10(numpy.exp(1)))
    ib = 0  # Bathymetry pair index
    mdr = 0  # Output range counter
    r = dr
    omega = 2 * numpy.pi * freq
    ri = zr / dz
    ir = int(numpy.floor(ri))  # Receiver depth index
    ddir = ri - ir  # Offset
    k0 = omega / c0
    # _z_sb += _z_ss[-1]  # Make seabed profiles relative to deepest water profile point
    zmax = z_sb.max() + lyrw * lam
    nz = int(numpy.floor(zmax / dz)) - 1  # Number of depth grid points - 2
    nzplt = int(numpy.floor(zmplt / dz))  # Deepest output grid point
    iz = int(numpy.floor(rbzb[0, 1] / dz))  # First index below seabed
    iz = max(1, iz)
    iz = min(nz - 1, iz)

    u = numpy.zeros(nz + 2, dtype=numpy.complex128)
    v = numpy.zeros(nz + 2, dtype=numpy.complex128)
    ksq = numpy.zeros(nz + 2, dtype=numpy.complex128)
    ksqb = numpy.zeros(nz + 2, dtype=numpy.complex128)
    r1 = numpy.zeros((nz + 2, np), dtype=numpy.complex128)
    r2 = numpy.zeros((nz + 2, np), dtype=numpy.complex128)
    r3 = numpy.zeros((nz + 2, np), dtype=numpy.complex128)
    s1 = numpy.zeros((nz + 2, np), dtype=numpy.complex128)
    s2 = numpy.zeros((nz + 2, np), dtype=numpy.complex128)
    s3 = numpy.zeros((nz + 2, np), dtype=numpy.complex128)
    pd1 = numpy.zeros(np, dtype=numpy.complex128)
    pd2 = numpy.zeros(np, dtype=numpy.complex128)

    alpw = numpy.zeros(nz + 2)
    alpb = numpy.zeros(nz + 2)
    f1 = numpy.zeros(nz + 2)
    f2 = numpy.zeros(nz + 2)
    f3 = numpy.zeros(nz + 2)
    ksqw = numpy.zeros(nz + 2)
    tmp_val = rmax / (dr * ndr)
    nvr = int(numpy.round(rmax / (dr * ndr)))
    rmax = nvr * dr * ndr
    nvz = int(numpy.floor(nzplt / ndz))
    vr = numpy.arange(1, nvr + 1) * dr * ndr
    vz = numpy.arange(1, nvz + 1) * dz * ndz
    tll = numpy.zeros(nvr)
    tlg = numpy.zeros((nvz, nvr))
    cpl = numpy.zeros(nvr) * 1j
    cpg = numpy.zeros((nvz, nvr)) * 1j
    tlc = -1  # TL output range counter

    ss_ind = 0  # Sound speed profile range index
    sb_ind = 0  # Seabed parameters range index
    bt_ind = 0  # Bathymetry range index

    # The initial profiles and starting field
    ksqw, ksqb, p_rhob, alpw, alpb = profl(
        freq,
        c0,
        z_ss,
        rp_ss,
        cw,
        z_sb,
        rp_sb,
        cb,
        rhob,
        attn,
        rbzb,
        dr,
        dz,
        zmax,
        ss_ind,
        sb_ind,
        nz,
        lyrw,
        lam,
        ksqw,
        ksqb,
        alpw,
        alpb,
    )

    # The self-starter
    selfs(
        freq,
        c0,
        zs,
        dz,
        dr,
        np,
        nz,
        iz,
        ns,
        u,
        v,
        s1,
        s2,
        s3,
        r1,
        r2,
        r3,
        f1,
        f2,
        f3,
        ksq,
        ksqw,
        ksqb,
        p_rhob,
        alpw,
        alpb,
        pd1,
        pd2,
    )

    mdr, tlc = outpt(r, mdr, ndr, ndz, tlc, f3, u, ddir, ir, tll, tlg, cpl, cpg)[:]

    # The propagation matrices
    epade(np, 1, k0, dr, ns, pd1, pd2)
    matrc(
        k0,
        dz,
        iz,
        iz,
        nz,
        np,
        f1,
        f2,
        f3,
        ksq,
        alpw,
        alpb,
        ksqw,
        ksqb,
        p_rhob,
        r1,
        r2,
        r3,
        s1,
        s2,
        s3,
        pd1,
        pd2,
    )
    return (
        u,
        v,
        ksq,
        ksqw,
        ksqb,
        alpw,
        alpb,
        p_rhob,
        r1,
        r2,
        r3,
        s1,
        s2,
        s3,
        f1,
        f2,
        f3,
        pd1,
        pd2,
        vr,
        vz,
        tll,
        tlg,
        cpl,
        cpg,
        tlc,
        ss_ind,
        sb_ind,
        bt_ind,
        zmax,
        nz,
        iz,
        nzplt,
        r,
        mdr,
        ir,
        ddir,
    )


# @njit
def interp(z, z_arr, val_arr, val_l, val_h):
    out = numpy.zeros(z.size)
    for i in range(z.size):
        if z[i] < z_arr[0]:
            out[i] = val_l
        elif z[i] > z_arr[-1]:
            out[i] = val_h
        else:
            out[i] = numpy.interp(z[i], z_arr, val_arr)
    return out


# @njit
def profl(
    freq,
    c0,
    z_ss,
    rp_ss,
    cw,
    z_sb,
    rp_sb,
    cb,
    rhob,
    attn,
    rbzb,
    dr,
    dz,
    zmax,
    ss_ind,
    sb_ind,
    nz,
    lyrw,
    lam,
    ksqw,
    ksqb,
    alpw,
    alpb,
):
    """Set up the profiles"""

    eta = 1 / (40 * numpy.pi * numpy.log10(numpy.exp(1)))

    attnf = 10  # 10dB/wavelength at floor
    omega = 2 * numpy.pi * freq
    k0 = omega / c0

    z = numpy.linspace(0, zmax, nz + 2)
    p_cw = interp(z, z_ss, cw[:, ss_ind], cw[0, ss_ind], cw[-1, ss_ind])
    p_cb = interp(z, z_sb[:, sb_ind], cb[:, sb_ind], cb[0, sb_ind], cb[-1, sb_ind])
    p_rhob = interp(
        z, z_sb[:, sb_ind], rhob[:, sb_ind], rhob[0, sb_ind], rhob[-1, sb_ind]
    )
    attnlyr = numpy.concatenate(
        (attn[:, sb_ind], numpy.array([attn[-1, sb_ind], attnf]))
    )
    zlyr = numpy.concatenate(
        (
            z_sb[:, sb_ind],
            numpy.array(
                [z_sb[-1, sb_ind] + 0.75 * lyrw * lam, z_sb[-1, sb_ind] + lyrw * lam]
            ),
        )
    )
    p_attn = interp(z, zlyr, attnlyr, attn[0, sb_ind], attnf)

    for i in range(nz + 2):
        ksqw[i] = (omega / p_cw[i]) ** 2 - k0**2
        ksqb[i] = ((omega / p_cb[i]) * (1 + 1j * eta * p_attn[i])) ** 2 - k0**2
        alpw[i] = numpy.sqrt(p_cw[i] / c0)
        alpb[i] = numpy.sqrt(p_rhob[i] * p_cb[i] / c0)
    return ksqw, ksqb, p_rhob, alpw, alpb


def updat(
    freq,
    zs,
    zr,
    z_ss,
    rp_ss,
    cw,
    z_sb,
    rp_sb,
    cb,
    rhob,
    attn,
    rbzb,
    dr,
    dz,
    zmax,
    np,
    c0,
    lam,
    zmplt,
    rmax,
    ns,
    rs,
    lyrw,
    run_id,
    u,
    v,
    ksq,
    ksqw,
    ksqb,
    alpw,
    alpb,
    p_rhob,
    r1,
    r2,
    r3,
    s1,
    s2,
    s3,
    f1,
    f2,
    f3,
    pd1,
    pd2,
    vr,
    vz,
    tll,
    tlg,
    cpl,
    cpg,
    tlc,
    ss_ind,
    sb_ind,
    bt_ind,
    rd_ss,
    rd_sb,
    rd_bt,
    nz,
    iz,
    nzplt,
    r,
):
    omega = 2 * numpy.pi * freq
    k0 = omega / c0

    """Matrix updates"""

    # Varying bathymetry
    if rd_bt:
        npt = rbzb.shape[0]
        while (bt_ind < npt - 1) and (r >= rbzb[bt_ind + 1, 0]):
            bt_ind += 1
        jz = iz
        z = rbzb[bt_ind, 1] + (r + 0.5 * dr - rbzb[bt_ind, 0]) * (
            rbzb[bt_ind + 1, 1] - rbzb[bt_ind, 1]
        ) / (rbzb[bt_ind + 1, 0] - rbzb[bt_ind, 0])
        iz = int(numpy.floor(z / dz))  # First index below seabed
        iz = max(1, iz)
        iz = min(nz - 1, iz)
        if iz != jz:
            matrc(
                k0,
                dz,
                iz,
                jz,
                nz,
                np,
                f1,
                f2,
                f3,
                ksq,
                alpw,
                alpb,
                ksqw,
                ksqb,
                p_rhob,
                r1,
                r2,
                r3,
                s1,
                s2,
                s3,
                pd1,
                pd2,
            )

    # Varying sound speed profile
    if rd_ss:
        npt = rp_ss.size
        ss_ind_o = ss_ind
        while (ss_ind < npt - 1) and (r >= rp_ss[ss_ind + 1]):
            ss_ind += 1
        if ss_ind != ss_ind_o:
            ksqw, ksqb, p_rhob, alpw, alpb = profl(
                freq,
                c0,
                z_ss,
                rp_ss,
                cw,
                z_sb,
                rp_sb,
                cb,
                rhob,
                attn,
                rbzb,
                dr,
                dz,
                zmax,
                ss_ind,
                sb_ind,
                nz,
                lyrw,
                lam,
                ksqw,
                ksqb,
                alpw,
                alpb,
            )
            matrc(
                k0,
                dz,
                iz,
                iz,
                nz,
                np,
                f1,
                f2,
                f3,
                ksq,
                alpw,
                alpb,
                ksqw,
                ksqb,
                p_rhob,
                r1,
                r2,
                r3,
                s1,
                s2,
                s3,
                pd1,
                pd2,
            )

    # Varying seabed profile
    if rd_sb:
        npt = rp_sb.size
        sb_ind_o = sb_ind
        while (sb_ind < npt - 1) and (r >= rp_sb[sb_ind + 1]):
            sb_ind += 1
        if sb_ind != sb_ind_o:
            ksqw, ksqb, p_rhob, alpw, alpb = profl(
                freq,
                c0,
                z_ss,
                rp_ss,
                cw,
                z_sb,
                rp_sb,
                cb,
                rhob,
                attn,
                rbzb,
                dr,
                dz,
                zmax,
                ss_ind,
                sb_ind,
                nz,
                lyrw,
                lam,
                ksqw,
                ksqb,
                alpw,
                alpb,
            )
        matrc(
            k0,
            dz,
            iz,
            iz,
            nz,
            np,
            f1,
            f2,
            f3,
            ksq,
            alpw,
            alpb,
            ksqw,
            ksqb,
            p_rhob,
            r1,
            r2,
            r3,
            s1,
            s2,
            s3,
            pd1,
            pd2,
        )

    # Turn off the stability constraints
    if r >= rs:
        ns = 0
        rs = rmax + dr
        epade(np, 1, k0, dr, ns, pd1, pd2)
        matrc(
            k0,
            dz,
            iz,
            iz,
            nz,
            np,
            f1,
            f2,
            f3,
            ksq,
            alpw,
            alpb,
            ksqw,
            ksqb,
            p_rhob,
            r1,
            r2,
            r3,
            s1,
            s2,
            s3,
            pd1,
            pd2,
        )

    return iz


def selfs(
    freq,
    c0,
    zs,
    dz,
    dr,
    np,
    nz,
    iz,
    ns,
    u,
    v,
    s1,
    s2,
    s3,
    r1,
    r2,
    r3,
    f1,
    f2,
    f3,
    ksq,
    ksqw,
    ksqb,
    p_rhob,
    alpw,
    alpb,
    pd1,
    pd2,
):
    """The self-starter"""
    omega = 2 * numpy.pi * freq
    k0 = omega / c0

    # Conditions for the delta function

    si = zs / dz
    _is = int(numpy.floor(si))  # Source depth index
    dis = si - _is  # Offset

    u[_is] = (1 - dis) * numpy.sqrt(2 * numpy.pi / k0) / (dz * alpw[_is])
    u[_is + 1] = dis * numpy.sqrt(2 * numpy.pi / k0) / (dz * alpw[_is])

    # Divide the delta function by (1-X)**2 to get a smooth rhs

    pd1[0] = 0
    pd2[0] = -1

    matrc(
        k0,
        dz,
        iz,
        iz,
        nz,
        1,
        f1,
        f2,
        f3,
        ksq,
        alpw,
        alpb,
        ksqw,
        ksqb,
        p_rhob,
        r1,
        r2,
        r3,
        s1,
        s2,
        s3,
        pd1,
        pd2,
    )
    for _ in range(2):
        solve(u, v, s1, s2, s3, r1, r2, r3, iz, nz, 1)

    # Apply the operator (1-X)**2*(1+X)**(-1/4)*exp(ci*k0*r*sqrt(1+X))

    epade(np, 2, k0, dr, ns, pd1, pd2)
    matrc(
        k0,
        dz,
        iz,
        iz,
        nz,
        np,
        f1,
        f2,
        f3,
        ksq,
        alpw,
        alpb,
        ksqw,
        ksqb,
        p_rhob,
        r1,
        r2,
        r3,
        s1,
        s2,
        s3,
        pd1,
        pd2,
    )
    solve(u, v, s1, s2, s3, r1, r2, r3, iz, nz, np)
    return


def epade(np, ip, k0, dr, ns, pd1, pd2):
    """The coefficients of the rational approximation"""

    n = 2 * np
    _bin = numpy.zeros((n + 1, n + 1))
    a = numpy.zeros((n + 1, n + 1), dtype=numpy.complex128)
    b = numpy.zeros(n, dtype=numpy.complex128)
    dg = numpy.zeros(n + 1, dtype=numpy.complex128)
    dh1 = numpy.zeros(n, dtype=numpy.complex128)
    dh2 = numpy.zeros(n, dtype=numpy.complex128)
    dh3 = numpy.zeros(n, dtype=numpy.complex128)
    fact = numpy.zeros(n + 1)
    sig = k0 * dr

    if ip == 1:
        nu, alp = 0, 0
    else:
        nu, alp = 1, -0.25

    # The factorials
    fact[0] = 1
    for i in range(1, n):
        fact[i] = (i + 1) * fact[i - 1]

    # The binomial coefficients
    for i in range(n + 1):
        _bin[i, 0] = 1
        _bin[i, i] = 1
    for i in range(2, n + 1):
        for j in range(1, i):
            _bin[i, j] = _bin[i - 1, j - 1] + _bin[i - 1, j]

    # The accuracy constraints
    dg, dh1, dh2, dh3 = deriv(n, sig, alp, dg, dh1, dh2, dh3, _bin, nu)
    for i in range(n):
        b[i] = dg[i + 1]
    for i in range(n):
        if 2 * i <= n - 1:
            a[i, 2 * i] = fact[i]
        for j in range(i + 1):
            if 2 * j + 1 <= n - 1:
                a[i, 2 * j + 1] = -_bin[i + 1, j + 1] * fact[j] * dg[i - j]

    # The stability constraints

    if ns >= 1:
        z1 = -3 + 0j
        b[n - 1] = -1
        for j in range(np):
            a[n - 1, 2 * j] = z1 ** (j + 1)
            a[n - 1, 2 * j + 1] = 0

    if ns >= 2:
        z1 = -1.5 + 0j
        b[n - 2] = -1
        for j in range(np):
            a[n - 2, 2 * j] = z1 ** (j + 1)
            a[n - 2, 2 * j + 1] = 0

    a, b = gauss(n, a, b, pivot)

    dh1[0] = 1
    for j in range(np):
        dh1[j + 1] = b[2 * j]
    dh1, dh2 = fndrt(dh1, np, dh2, guerre)
    for j in range(np):
        pd1[j] = -1 / dh2[j]

    dh1[0] = 1
    for j in range(np):
        dh1[j + 1] = b[2 * j + 1]
    dh1, dh2 = fndrt(dh1, np, dh2, guerre)
    for j in range(np):
        pd2[j] = -1 / dh2[j]

    return pd1, pd2


def deriv(n, sig, alp, dg, dh1, dh2, dh3, _bin, nu):
    """The derivatives of the operator function at x=0"""

    dh1[0] = 0.5 * 1j * sig
    exp1 = -0.5
    dh2[0] = alp
    exp2 = -1
    dh3[0] = -2 * nu
    exp3 = -1
    for i in range(1, n):
        dh1[i] = dh1[i - 1] * exp1
        exp1 -= 1
        dh2[i] = dh2[i - 1] * exp2
        exp2 -= 1
        dh3[i] = -nu * dh3[i - 1] * exp3
        exp3 -= 1

    dg[0] = 1
    dg[1] = dh1[0] + dh2[0] + dh3[0]
    for i in range(1, n):
        dg[i + 1] = dh1[i] + dh2[i] + dh3[i]
        for j in range(i):
            dg[i + 1] += _bin[i, j] * (dh1[j] + dh2[j] + dh3[j]) * dg[i - j]

    return dg, dh1, dh2, dh3


def gauss(n, a, b, pivot):
    """Gaussian elimination"""

    # Downward elimination
    for i in range(n):
        if i < n - 1:
            a, b = pivot(n, i, a, b)
        a[i, i] = 1 / a[i, i]
        b[i] *= a[i, i]
        if i < n - 1:
            for j in range(i + 1, n + 1):
                a[i, j] *= a[i, i]
            for k in range(i + 1, n):
                b[k] -= a[k, i] * b[i]
                for j in range(i + 1, n):
                    a[k, j] -= a[k, i] * a[i, j]

    # Back substitution
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            b[i] -= a[i, j] * b[j]

    return a, b


def pivot(n, i, a, b):
    """Rows are interchanged for stability"""

    i0 = i
    amp0 = numpy.abs(a[i, i])
    for j in range(i + 1, n):
        amp = numpy.abs(a[j, i])
        if amp > amp0:
            i0 = j
            amp0 = amp

    if i0 != i:
        b[i0], b[i] = b[i], b[i0]
        for j in range(i, n + 1):
            a[i0, j], a[i, j] = a[i, j], a[i0, j]

    return a, b


def fndrt(a, n, z, guerre):
    """The root finding subroutine"""

    if n == 1:
        z[0] = -a[0] / a[1]
        return a, z

    if n != 2:
        for k in range(n - 1, 1, -1):
            # Obtain an approximate root
            root = 0
            err = 1e-12
            a, root, err = guerre(a, k + 1, root, err, 1000)
            # Refine the root by iterating five more times
            err = 0
            a, root, err = guerre(a, k + 1, root, err, 5)
            z[k] = root
            # Divide out the factor (z-root).
            for i in range(k, -1, -1):
                a[i] += root * a[i + 1]
            for i in range(k + 1):
                a[i] = a[i + 1]

    z[1] = 0.5 * (-a[1] + numpy.sqrt(a[1] ** 2 - 4 * a[0] * a[2])) / a[2]
    z[0] = 0.5 * (-a[1] - numpy.sqrt(a[1] ** 2 - 4 * a[0] * a[2])) / a[2]

    return a, z


def guerre(a, n, z, err, nter):
    """This subroutine finds a root of a polynomial of degree n > 2 by Laguerre's method"""

    az = numpy.zeros(n, dtype=numpy.complex128)
    azz = numpy.zeros(n - 1, dtype=numpy.complex128)

    eps = 1e-20
    # The coefficients of p'(z) and p''(z)
    for i in range(n):
        az[i] = (i + 1) * a[i + 1]
    for i in range(n - 1):
        azz[i] = (i + 1) * az[i + 1]

    _iter = 0
    jter = 0  # Missing from original code - assume this is correct
    dz = numpy.Inf

    while (numpy.abs(dz) > err) and (_iter < nter - 1):
        p = a[n - 1] + a[n] * z
        for i in range(n - 2, -1, -1):
            p = a[i] + z * p
        if numpy.abs(p) < eps:
            return a, z, err

        pz = az[n - 2] + az[n - 1] * z
        for i in range(n - 3, -1, -1):
            pz = az[i] + z * pz

        pzz = azz[n - 3] + azz[n - 2] * z
        for i in range(n - 4, -1, -1):
            pzz = azz[i] + z * pzz

        # The Laguerre perturbation
        f = pz / p
        g = f**2 - pzz / p
        h = numpy.sqrt((n - 1) * (n * g - f**2))
        amp1 = numpy.abs(f + h)
        amp2 = numpy.abs(f - h)
        if amp1 > amp2:
            dz = -n / (f + h)
        else:
            dz = -n / (f - h)

        _iter += 1

        # Rotate by 90 degrees to avoid limit cycles
        jter += 1
        if jter == 9:
            jter = 0
            dz *= 1j
        z += dz

        if _iter == 100:
            raise ValueError(
                "Laguerre method not converging. Try a different combination of DR and NP."
            )

    return a, z, err


jit_module(nopython=True)
