'''
PyRAM: Python adaptation of the Range-dependent Acoustic Model (RAM).
RAM was created by Michael D Collins at the US Naval Research Laboratory.
This adaptation is of RAM v1.5, available from the Ocean Acoustics Library at
http://oalib.hlsresearch.com/PE/RAM/
The purpose of PyRAM is to provide a version of RAM which can be used within a
Python interpreter environment (e.g. Spyder or the Jupyter notebook) and is
easier to understand, extend and integrate into other applications than the
Fortran version. It is written in pure Python and achieves speeds comparable to
native code by using the Numba library for JIT compilation and parallelisation.

The PyRAM class contains methods which largely correspond to the original
Fortran subroutines and functions (including retaining the same names). The
variable names are also mostly the same. However some of the original code
(e.g. subroutine zread) is unnecessary when the same purpose can be achieved
using available Python library functions (e.g. from NumPy or SciPy) and has
therefore been replaced.

A difference in functionality is that sound speed profile updates with range
are decoupled from seabed parameter updates, which provides more flexibility
in specifying the environment (e.g. if the data comes from different sources).

PyRAM also provides various conveniences, e.g. automatic calculation of range
and depth steps (though these can be overridden using keyword arguments).
Additional functionality including status outputs and run time calculation is
included.
'''

import numpy


class PyRAM:

    _np_default = 8
    _dzf = 0.1
    _ndr_default = 1
    _ndz_default = 1
    _ns_default = 1
    _lyrw_default = 30

    def __init__(self, freq, zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob,
                 attn, rbzb, **kwargs):

        '''
        args...
        -------
        freq: Frequency (Hz).
        zs: Source depth (m).
        zr: Receiver depth (m).
        z_ss: Water sound speed profile depths (m), NumPy 1D array.
        rp_ss: Water sound speed profile update ranges (m), NumPy 1D array.
        cw: Water sound speed values (m/s),
            Numpy 2D array, dimensions z_ss.size by rp_ss.size.
        z_sb: Seabed parameter profile depths (m), NumPy 1D array.
        rp_sb: Seabed parameter update ranges (m), NumPy 1D array.
        cb: Seabed sound speed values (m/s),
            NumPy 2D array, dimensions z_sb.size by rp_sb.size.
        rhob: Seabed density values (g/cm3), same dimensions as cb
        attn: Seabed attenuation values (dB/wavelength), same dimensions as cb
        rbzb: Bathymetry (m), Numpy 2D array with columns of ranges and depths
        ---------
        kwargs...
        ---------
        np: Number of Pade terms. Defaults to _np_default.
        c0: Reference sound speed (m/s). Defaults to mean of 1st profile.
        dr: Calculation range step (m). Defaults to np times the wavelength.
        dz: Calculation depth step (m). Defaults to _dzf*wavelength.
        ndr: Number of range steps between outputs. Defaults to _ndr_default.
        ndz: Number of depth steps between outputs. Defaults to _ndz_default.
        zmplt: Maximum output depth (m). Defaults to maximum depth in rbzb.
        rmax: Maximum calculation range (m). Defaults to max in rp_ss or rp_sb.
        ns: Number of stability constraints. Defaults to _ns_default.
        rs: Maximum range of the stability constraints (m). Defaults to rmax.
        lyrw: Absorbing layer width (wavelengths). Defaults to _lyrw_default.
        NB: original zmax input not needed due to lyrw.
        '''

        self._freq, self._zs, self._zr = freq, zs, zr

        self.check_inputs(z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb)

        self.get_params(**kwargs)

        self.setup()

        while self.r < self._rmax:
            self.updat()
            self.solve()
            self.r += self._dr
            self.outpt()

    def check_inputs(self, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb):

        '''Basic checks on dimensions of inputs'''

        self.status_ok = True

        # Source and receiver depths
        if not z_ss[0] <= self._zs <= z_ss[-1]:
            self.status_ok = False
            raise ValueError('Source depth outside sound speed depths')
        if not z_ss[0] <= self._zr <= z_ss[-1]:
            self.status_ok = False
            raise ValueError('Receiver depth outside sound speed depths')
        if self.status_ok:
            self._z_ss = z_ss

        # Water sound speed profiles
        num_depths = self._z_ss.size
        num_ranges = rp_ss.size
        cw_dims = cw.shape
        if (cw_dims[0] == num_depths) and (cw_dims[1] == num_ranges):
            self._rp_ss, self._cw = rp_ss, cw
        else:
            raise ValueError('Dimensions of z_ss, rp_ss and cw must be consistent.')

        # Seabed profiles
        self._z_sb = z_sb
        num_depths = self._z_sb.size
        num_ranges = rp_sb.size
        for prof in [cb, rhob, attn]:
            prof_dims = prof.shape
            if (prof_dims[0] != num_depths) or (prof_dims[1] != num_ranges):
                self.status_ok = False
        if self.status_ok:
            self._rp_sb, self._cb, self._rhob, self._attn = \
                rp_sb, cb, rhob, attn
        else:
            raise ValueError('Dimensions of z_sb, rp_sb, cb, rhob and attn must be consistent.')

        if rbzb[:, 1].max() == self._z_ss[-1]:
            self._rbzb = rbzb
        else:
            self.status_ok = False
            raise ValueError('Deepest bathymetry point must be at same depth as deepest sound speed')

        # Set flags for range-dependence (water SSP, seabed profile, bathymetry)
        self.rd_ss = True if self._rp_ss.size > 1 else False
        self.rd_sb = True if self._rp_sb.size > 1 else False
        self.rd_bt = True if self._rbzb.shape[0] > 1 else False

    def get_params(self, **kwargs):

        '''Get the parameters from the keyword arguments'''

        self._np = kwargs.get('np', PyRAM._np_default)

        self._c0 = kwargs.get('c0', numpy.mean(self._cw[:, 0])
                              if len(self._cw.shape) > 1 else
                              numpy.mean(self._cw))

        self._lambda = self._c0/self._freq

        # dr and dz are based on 1500m/s to get sensible output steps
        self._dr = kwargs.get('dr', self._np*1500/self._freq)
        self._dz = kwargs.get('dz', PyRAM._dzf*1500/self._freq)

        self._ndr = kwargs.get('ndr', PyRAM._ndr_default)
        self._ndz = kwargs.get('ndz', PyRAM._ndz_default)

        self._zmplt = kwargs.get('zmplt', self._rbzb[:, 1].max())

        self._rmax = kwargs.get('rmax', numpy.max([self._rp_ss.max(),
                                                   self._rp_sb.max()]))
        self.vr = numpy.linspace(self._dr, self._rmax,
                                 int(numpy.round(self._rmax/self._dr)))
#        self._rmax += self._dr  # Output at last range

        self._ns = kwargs.get('ns', PyRAM._ns_default)
        self._rs = kwargs.get('rs', self._rmax + self._dr)

        self._lyrw = kwargs.get('lyrw', PyRAM._lyrw_default)

    def setup(self):

        '''Initialise the parameters, acoustic field, and matrices'''

        if self._rbzb[-1, 0] < self._rmax:
            self._rbzb = numpy.append(self._rbzb,
                                      [[self._rmax, self._rbzb[-1, 1]]],
                                      axis=0)

        self.eta = 1/(40*numpy.pi*numpy.log10(numpy.exp(1)))
        self.eps = 1e-20
        self.ib = 0  # Bathymetry pair index
        self.mdr = 0  # Output range counter
        self.r = self._dr
        self.omega = 2*numpy.pi*self._freq
        ri = self._zr/self._dz
        self.ir = int(numpy.floor(ri))  # Receiver depth index
        self.dir = ri - self.ir  # Offset
        self.k0 = self.omega/self._c0
        self._zmax = self._z_ss.max() + self._lyrw*self._lambda
        self.nz = int(numpy.floor(self._zmax/self._dz)) - 1  # Number of depth grid points - 2
        self.nzplt = int(numpy.floor(self._zmplt/self._dz)) - 1  # Deepest output grid point
        self.iz = int(numpy.floor(self._rbzb[0, 1]/self._dz))  # First index below seabed
        self.iz = max(1, self.iz)
        self.iz = min(self.nz-1, self.iz)

        self.u = numpy.zeros(self.nz+2, dtype=numpy.complex)
        self.v = numpy.zeros(self.nz+2, dtype=numpy.complex)
        self.ksq = numpy.zeros(self.nz+2, dtype=numpy.complex)
        self.ksqb = numpy.zeros(self.nz+2, dtype=numpy.complex)
        self.r1 = numpy.zeros([self.nz+2, self._np], dtype=numpy.complex)
        self.r2 = numpy.zeros([self.nz+2, self._np], dtype=numpy.complex)
        self.r3 = numpy.zeros([self.nz+2, self._np], dtype=numpy.complex)
        self.s1 = numpy.zeros([self.nz+2, self._np], dtype=numpy.complex)
        self.s2 = numpy.zeros([self.nz+2, self._np], dtype=numpy.complex)
        self.s3 = numpy.zeros([self.nz+2, self._np], dtype=numpy.complex)
        self.pd1 = numpy.zeros(self._np, dtype=numpy.complex)
        self.pd2 = numpy.zeros(self._np, dtype=numpy.complex)

        self.alpw = numpy.zeros(self.nz+2)
        self.alpb = numpy.zeros(self.nz+2)
        self.f1 = numpy.zeros(self.nz+2)
        self.f2 = numpy.zeros(self.nz+2)
        self.f3 = numpy.zeros(self.nz+2)
        self.ksqw = numpy.zeros(self.nz+2)
        self.tll = numpy.zeros(int(numpy.floor(self._rmax/(self._dr*self._ndr))))
        self.tlg = numpy.zeros([int(numpy.floor(self.nz+2/self._ndz)),
                               self.tll.size])
        self.tlc = -1  # TL output range counter

        self.ss_ind = 0  # Sound speed profile range index
        self.sb_ind = 0  # Seabed parameters range index
        self.bt_ind = 0  # Bathymetry range index

        # The initial profiles and starting field
        self.profl()
        self.selfs()
        self.outpt()

        # The propagation matrices
        self.epade()
        self.matrc()

    def profl(self):

        attnf = 10  # 10dB/wavelength at floor

        '''Set up the profiles'''

        z = numpy.linspace(0, self._zmax, self.nz+2)
        self.cw = numpy.interp(z, self._z_ss, self._cw[:, self.ss_ind],
                               left=self._cw[0, self.ss_ind],
                               right=self._cw[-1, self.ss_ind])
        self.cb = numpy.interp(z, self._z_sb, self._cb[:, self.sb_ind],
                               left=self._cb[0, self.sb_ind],
                               right=self._cb[-1, self.sb_ind])
        self.rhob = numpy.interp(z, self._z_sb, self._rhob[:, self.sb_ind],
                                 left=self._rhob[0, self.sb_ind],
                                 right=self._rhob[-1, self.sb_ind])
        attnlyr = numpy.concatenate((self._attn[:, self.sb_ind],
                                     [self._attn[-1, self.sb_ind], attnf]))
        zlyr = numpy.concatenate((self._z_sb,
                                  [self._z_sb[-1] + 0.75*self._lyrw*self._lambda,
                                   self._z_sb[-1] + self._lyrw*self._lambda]))
        self.attn = numpy.interp(z, zlyr, attnlyr,
                                 left=self._attn[0, self.sb_ind],
                                 right=attnf)

        for i in range(self.nz+2):
            self.ksqw[i] = (self.omega/self.cw[i])**2 - self.k0**2
            self.ksqb[i] = ((self.omega/self.cb[i]) *
                            (1 + 1j*self.eta*self.attn[i]))**2 - self.k0**2
            self.alpw[i] = numpy.sqrt(self.cw[i]/self._c0)
            self.alpb[i] = numpy.sqrt(self.rhob[i]*self.cb[i]/self._c0)

    def matrc(self, jz=None, np=None):

        '''The tridiagonal matrices'''

        if jz is None:
            jz = self.iz
        if np is None:
            np = self._np

        a1 = self.k0**2/6
        a2 = 2*self.k0**2/3
        a3 = a1
        cfact = 0.5/self._dz**2
        dfact = 1/12

        iz = self.iz

        # New matrices when iz == jz
        if iz == jz:
            i1 = 1
            i2 = self.nz
            for i in range(iz+1):
                self.f1[i] = 1/self.alpw[i]
                self.f2[i] = 1
                self.f3[i] = self.alpw[i]
                self.ksq[i] = self.ksqw[i]
            for i in range(iz+1, self.nz+2):
                self.f1[i] = self.rhob[i]/self.alpb[i]
                self.f2[i] = 1/self.rhob[i]
                self.f3[i] = self.alpb[i]
                self.ksq[i] = self.ksqb[i]
        # Updated matrices when iz != jz
        elif iz > jz:
            i1 = jz
            i2 = iz+1
            for i in range(jz+1, iz+1):
                self.f1[i] = 1/self.alpw[i]
                self.f2[i] = 1
                self.f3[i] = self.alpw[i]
                self.ksq[i] = self.ksqw[i]
        elif iz < jz:
            i1 = iz
            i2 = jz+1
            for i in range(iz+1, jz+1):
                self.f1[i] = self.rhob[i]/self.alpb[i]
                self.f2[i] = 1/self.rhob[i]
                self.f3[i] = self.alpb[i]
                self.ksq[i] = self.ksqb[i]

        # Discretization by Galerkin's method

        for i in range(i1, i2+1):

            c1 = cfact*self.f1[i]*(self.f2[i-1] + self.f2[i])*self.f3[i-1]
            c2 = -cfact*self.f1[i] * \
                (self.f2[i-1] + 2*self.f2[i] + self.f2[i+1])*self.f3[i]
            c3 = cfact*self.f1[i]*(self.f2[i] + self.f2[i+1])*self.f3[i+1]
            d1 = c1 + dfact*(self.ksq[i-1] + self.ksq[i])
            d2 = c2 + dfact*(self.ksq[i-1] + 6*self.ksq[i] + self.ksq[i+1])
            d3 = c3 + dfact*(self.ksq[i] + self.ksq[i+1])

            for j in range(np):
                self.r1[i, j] = a1 + self.pd2[j]*d1
                self.r2[i, j] = a2 + self.pd2[j]*d2
                self.r3[i, j] = a3 + self.pd2[j]*d3
                self.s1[i, j] = a1 + self.pd1[j]*d1
                self.s2[i, j] = a2 + self.pd1[j]*d2
                self.s3[i, j] = a3 + self.pd1[j]*d3

        # The matrix decomposition
        for j in range(np):

            for i in range(i1, iz+1):
                rfact = 1/(self.r2[i, j] - self.r1[i, j]*self.r3[i-1, j])
                self.r1[i, j] *= rfact
                self.r3[i, j] *= rfact
                self.s1[i, j] *= rfact
                self.s2[i, j] *= rfact
                self.s3[i, j] *= rfact

            for i in range(i2, iz+1, -1):
                rfact = 1/(self.r2[i, j] - self.r3[i, j]*self.r1[i+1, j])
                self.r1[i, j] *= rfact
                self.r3[i, j] *= rfact
                self.s1[i, j] *= rfact
                self.s2[i, j] *= rfact
                self.s3[i, j] *= rfact

            self.r2[iz+1, j] -= self.r1[iz+1, j]*self.r3[iz, j]
            self.r2[iz+1, j] -= self.r3[iz+1, j]*self.r1[iz+2, j]
            self.r2[iz+1, j] = 1/self.r2[iz+1, j]

    def solve(self, np=None):

        '''The tridiagonal solver'''

        if np is None:
            np = self._np

        eps = 1e-30

        for j in range(np):
            # The right side
            for i in range(1, self.nz+1):
                self.v[i] = self.s1[i, j]*self.u[i-1] + \
                    self.s2[i, j]*self.u[i] + self.s3[i, j]*self.u[i+1] + eps

            # The elimination steps
            for i in range(2, self.iz+1):
                self.v[i] -= self.r1[i, j]*self.v[i-1] + eps
            for i in range(self.nz-1, self.iz+1, -1):
                self.v[i] -= self.r3[i, j]*self.v[i+1] + eps

            self.u[self.iz+1] = (self.v[self.iz+1] -
                                 self.r1[self.iz+1, j]*self.v[self.iz] -
                                 self.r3[self.iz+1, j]*self.v[self.iz+2]) * \
                self.r2[self.iz+1, j] + eps

            # The back substitution steps
            for i in range(self.iz, -1, -1):
                self.u[i] = self.v[i] - self.r3[i, j]*self.u[i+1] + eps
            for i in range(self.iz+2, self.nz+1):
                self.u[i] = self.v[i] - self.r1[i, j]*self.u[i-1] + eps

    def updat(self):

        '''Matrix updates'''

        # Varying bathymetry
        if self.rd_bt and (self.bt_ind < self._rbzb.shape[0]-2):
            if self.r >= self._rbzb[self.bt_ind+1, 0]:
                self.bt_ind += 1
            jz = self.iz
            z = self._rbzb[self.bt_ind, 1] + \
                (self.r + 0.5*self._dr - self._rbzb[self.bt_ind, 0]) * \
                (self._rbzb[self.bt_ind+1, 1] - self._rbzb[self.bt_ind, 1]) / \
                (self._rbzb[self.bt_ind+1, 0] - self._rbzb[self.bt_ind, 0])
            self.iz = int(numpy.floor(z/self._dz))  # First index below seabed
            self.iz = max(1, self.iz)
            self.iz = min(self.nz-1, self.iz)
            if (self.iz != jz):
                self.matrc(jz=jz)

        # Varying sound speed profile
        if self.rd_ss and (self.ss_ind < self._rp_ss.size-1):
            if self.r >= self._rp_ss[self.ss_ind+1]:
                self.ss_ind += 1
                self.profl()
                self.matrc()

        # Varying seabed profile
        if self.rd_sb and (self.sb_ind < self._rp_sb.size-1):
            if self.r >= self._rp_sb[self.sb_ind+1]:
                self.sb_ind += 1
                self.profl()
                self.matrc()

        # Turn off the stability constraints
        if self.r >= self._rs:
            self._ns = 0
            self._rs = self._rmax + self._dr
            self.epade()
            self.matrc()

    def selfs(self):

        '''The self-starter'''

        # Conditions for the delta function

        si = self._zs/self._dz
        _is = int(numpy.floor(si))  # Source depth index
        dis = si - _is  # Offset

        self.u[_is] = (1 - dis)*numpy.sqrt(2*numpy.pi/self.k0) / \
            (self._dz*self.alpw[_is])
        self.u[_is+1] = dis*numpy.sqrt(2*numpy.pi/self.k0) / \
            (self._dz*self.alpw[_is])

        # Divide the delta function by (1-X)**2 to get a smooth rhs

        self.pd1[0] = 0
        self.pd2[0] = -1

        self.matrc(np=1)
        self.solve(np=1)
        self.solve(np=1)

        # Apply the operator (1-X)**2*(1+X)**(-1/4)*exp(ci*k0*r*sqrt(1+X))

        self.epade(ip=2)
        self.matrc()
        self.solve()

    def outpt(self):

        '''Output transmission loss'''

        self.mdr += 1
        if self.mdr == self._ndr:
            self.mdr = 0
            self.tlc += 1
            ur = (1 - self.dir)*self.f3[self.ir]*self.u[self.ir] + \
                self.dir*self.f3[self.ir+1]*self.u[self.ir+1]
            self.tll[self.tlc] = -20*numpy.log10(numpy.abs(ur) + self.eps) + \
                10*numpy.log10(self.r + self.eps)

            j = 0
            for i in range(self._ndz-1, self.nzplt+1, self._ndz):
                ur = self.u[i]*self.f3[i]
                j += 1
                self.tlg[j, self.tlc] = \
                    -20*numpy.log10(numpy.abs(ur) + self.eps) + \
                    10*numpy.log10(self.r + self.eps)

    def epade(self, ip=1):

        '''The coefficients of the rational approximation'''

        n = 2*self._np
        _bin = numpy.zeros([n+1, n+1])
        a = numpy.zeros([n+1, n+1], dtype=numpy.complex)
        b = numpy.zeros(n, dtype=numpy.complex)
        dg = numpy.zeros(n+1, dtype=numpy.complex)
        dh1 = numpy.zeros(n, dtype=numpy.complex)
        dh2 = numpy.zeros(n, dtype=numpy.complex)
        dh3 = numpy.zeros(n, dtype=numpy.complex)
        fact = numpy.zeros(n+1)
        sig = self.k0*self._dr

        if ip == 1:
            nu, alp = 0, 0
        else:
            nu, alp = 1, -0.25

        # The factorials
        fact[0] = 1
        for i in range(1, n):
            fact[i] = (i+1)*fact[i-1]

        # The binomial coefficients
        for i in range(n+1):
            _bin[i, 0] = 1
            _bin[i, i] = 1
        for i in range(2, n+1):
            for j in range(1, i):
                _bin[i, j] = _bin[i-1, j-1] + _bin[i-1, j]

        # The accuracy constraints
        dg, dh1, dh2, dh3 = \
            self.deriv(n, sig, alp, dg, dh1, dh2, dh3, _bin, nu)
        for i in range(n):
            b[i] = dg[i+1]
        for i in range(n):
            if 2*i <= n-1:
                a[i, 2*i] = fact[i]
            for j in range(i+1):
                if 2*j+1 <= n-1:
                    a[i, 2*j+1] = -_bin[i+1, j+1]*fact[j]*dg[i-j]

        # The stability constraints

        if self._ns >= 1:
            z1 = -3 + 0j
            b[n-1] = -1
            for j in range(self._np):
                a[n-1, 2*j] = z1**(j+1)
                a[n-1, 2*j+1] = 0

        if self._ns >= 2:
            z1 = -1.5 + 0j
            b[n-2] = -1
            for j in range(self._np):
                a[n-2, 2*j] = z1**(j+1)
                a[n-2, 2*j+1] = 0

        a, b = self.gauss(n, a, b, self.pivot)

        dh1[0] = 1
        for j in range(self._np):
            dh1[j+1] = b[2*j]
        dh1, dh2 = self.fndrt(dh1, self._np, dh2, self.guerre)
        for j in range(self._np):
            self.pd1[j] = -1/dh2[j]

        dh1[0] = 1
        for j in range(self._np):
            dh1[j+1] = b[2*j+1]
        dh1, dh2 = self.fndrt(dh1, self._np, dh2, self.guerre)
        for j in range(self._np):
            self.pd2[j] = -1/dh2[j]

    @staticmethod
    def g(sig, x, alp, nu):

        '''The operator function'''

        g = (1 - nu*x)**2 * numpy.exp(alp*numpy.log(1 + x) +
                                      1j*sig*(-1 + numpy.sqrt(1 + x)))
        return g

    @staticmethod
    def deriv(n, sig, alp, dg, dh1, dh2, dh3, _bin, nu):

        '''The derivatives of the operator function at x=0'''

        dh1[0] = 0.5*1j*sig
        exp1 = -0.5
        dh2[0] = alp
        exp2 = -1
        dh3[0] = -2*nu
        exp3 = -1
        for i in range(1, n):
            dh1[i] = dh1[i-1]*exp1
            exp1 -= 1
            dh2[i] = dh2[i-1]*exp2
            exp2 -= 1
            dh3[i] = -nu*dh3[i-1]*exp3
            exp3 -= 1

        dg[0] = 1
        dg[1] = dh1[0] + dh2[0] + dh3[0]
        for i in range(1, n):
            dg[i+1] = dh1[i] + dh2[i] + dh3[i]
            for j in range(i):
                dg[i+1] += _bin[i, j]*(dh1[j] + dh2[j] + dh3[j])*dg[i-j]

        return dg, dh1, dh2, dh3

    @staticmethod
    def gauss(n, a, b, pivot):

        '''Gaussian elimination'''

        # Downward elimination
        for i in range(n):
            if i < n-1:
                a, b = pivot(n, i, a, b)
            a[i, i] = 1/a[i, i]
            b[i] *= a[i, i]
            if i < n-1:
                for j in range(i+1, n+1):
                    a[i, j] *= a[i, i]
                for k in range(i+1, n):
                    b[k] -= a[k, i]*b[i]
                    for j in range(i+1, n):
                        a[k, j] -= a[k, i]*a[i, j]

        # Back substitution
        for i in range(n-2, -1, -1):
            for j in range(i+1, n):
                b[i] -= a[i, j]*b[j]

        return a, b

    @staticmethod
    def pivot(n, i, a, b):

        '''Rows are interchanged for stability'''

        i0 = i
        amp0 = numpy.abs(a[i, i])
        for j in range(i+1, n):
            amp = numpy.abs(a[j, i])
            if amp > amp0:
                i0 = j
                amp0 = amp

        if i0 != i:
            b[i0], b[i] = b[i], b[i0]
            for j in range(i, n+1):
                a[i0, j], a[i, j] = a[i, j], a[i0, j]

        return a, b

    @staticmethod
    def fndrt(a, n, z, guerre):

        '''The root finding subroutine'''

        if n == 1:
            z[0] = -a[0]/a[1]
            return a, z

        if n != 2:
            for k in range(n-1, 1, -1):
                # Obtain an approximate root
                root = 0
                err = 1e-12
                a, root, err = guerre(a, k+1, root, err, 1000)
                # Refine the root by iterating five more times
                err = 0
                a, root, err = guerre(a, k+1, root, err, 5)
                z[k] = root
                # Divide out the factor (z-root).
                for i in range(k, -1, -1):
                    a[i] += root*a[i+1]
                for i in range(k+1):
                    a[i] = a[i+1]

        z[1] = 0.5*(-a[1] + numpy.sqrt(a[1]**2 - 4*a[0]*a[2]))/a[2]
        z[0] = 0.5*(-a[1] - numpy.sqrt(a[1]**2 - 4*a[0]*a[2]))/a[2]

        return a, z

    @staticmethod
    def guerre(a, n, z, err, nter):

        '''This subroutine finds a root of a polynomial of degree n > 2 by Laguerre's method'''

        az = numpy.zeros(n, dtype=numpy.complex)
        azz = numpy.zeros(n-1, dtype=numpy.complex)

        eps = 1e-20
        # The coefficients of p'(z) and p''(z)
        for i in range(n):
            az[i] = (i+1)*a[i+1]
        for i in range(n-1):
            azz[i] = (i+1)*az[i+1]

        _iter = 0
        jter = 0  # Missing from original code - assume this is correct
        dz = numpy.Inf

        while (numpy.abs(dz) > err) and (_iter < nter-1):
            p = a[n-1] + a[n]*z
            for i in range(n-2, -1, -1):
                p = a[i] + z*p
            if numpy.abs(p) < eps:
                return a, z, err

            pz = az[n-2] + az[n-1]*z
            for i in range(n-3, -1, -1):
                pz = az[i] + z*pz

            pzz = azz[n-3] + azz[n-2]*z
            for i in range(n-4, -1, -1):
                pzz = azz[i] + z*pzz

            # The Laguerre perturbation
            f = pz/p
            g = f**2 - pzz/p
            h = numpy.sqrt((n - 1)*(n*g - f**2))
            amp1 = numpy.abs(f + h)
            amp2 = numpy.abs(f - h)
            if amp1 > amp2:
                dz = -n/(f + h)
            else:
                dz = -n/(f - h)

            _iter += 1

            # Rotate by 90 degrees to avoid limit cycles

            jter += 1
            if jter == 9:
                jter = 0
                dz *= 1j
            z += dz

            if _iter == 100:
                raise ValueError('Laguerre method not converging. Try a different combination of DR and NP.')

        return a, z, err