"""
Description:
    Compare the "raw_pram" function

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

from pathlib import Path
import sys

import numpy 
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import pyram.pyram as rpr


freq=50.0
zs=50.0
zr=50.0
z_ss=numpy.array([0, 100.0, 400])
rp_ss=numpy.array([0, 25000.0])
cw=numpy.array([[1480, 1530.0],
                [1520, 1530.0],
                [1530, 1560.0]])
rbzb=numpy.array([[0, 200.0],
                  [40000.0, 400.0]])
z_sb=numpy.array([rbzb[:,-1]])
rp_sb=numpy.array(rbzb[:,0])
cb=numpy.array([[1700.0, 1700.0]])
print(cb.shape, z_sb.shape)
rhob=numpy.array([[1.5, 1.5]])
attn=numpy.array([[0.5, 0.5]])
rmax=50000.0
dr=500.0
dz=2.0

import time

print('running raw_pyram')
# run once to compile
vr, vz, tlg, tll, cpg, cpl, c0, proc_time = rpr.solve_field(freq, zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb, dr, dz)
# run 3 times to test speed
for i in range(3):
    now = time.time()
    vr, vz, tlg, tll, cpg, cpl, c0, proc_time = rpr.solve_field(freq, zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb, dr, dz)
    print('proc_time', time.time() - now)


print('-----------------------')
k0 = 2*numpy.pi*freq/c0
cpg = cpg.conj()  # pyram Fourier convention is S(omega) = int s(t) e^{i omega t} dt, my preference is e^{-i omega t} so i take conjugate
cpg *= numpy.exp(-1j*vr*k0) # this follows my preferred convention
cpg = -cpg / numpy.sqrt(vr * 8 * numpy.pi) * numpy.exp(-1j*numpy.pi/4) / numpy.pi # add cylindrical spreading and scalings for comparison with KRAKEN

raw_ram_tl = 20*numpy.log10(numpy.abs(numpy.squeeze(cpg))/numpy.max(numpy.abs(cpg)))


plt.figure()
plt.pcolormesh(vr, vz, raw_ram_tl, cmap='jet')
plt.colorbar()
plt.gca().invert_yaxis()


# run once to compile
# ram_mod = pr.PyRAM(freq, zs, zs, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb, **{'dr':dr, 'dz':dz})

# run 3 times to test speed
# print('running PyRAM')
# for i in range(3):
#     now = time.time()
#     ram_mod.run()
#     print('proc_time', time.time() - now)
# print('-----------------------')
# pout = ram_mod.cpg.conj()
# zout = ram_mod.vz
# rout = ram_mod.vr


# pout *= numpy.exp(-1j*rout*ram_mod.k0)
# pout = -pout / numpy.sqrt(rout * 8 * numpy.pi) * numpy.exp(-1j*numpy.pi/4) / numpy.pi

# ram_tl = 20*numpy.log10(numpy.abs(numpy.squeeze(pout))/numpy.max(numpy.abs(pout)))


# plt.figure()
# plt.pcolormesh(rout, zout, ram_tl, cmap='jet')
# plt.colorbar()
# plt.gca().invert_yaxis()

# plt.figure()
# plt.suptitle('Diff between raw and standard')
# plt.pcolormesh(rout, zout, numpy.abs(pout - cpg), cmap='jet')
# plt.colorbar()
# plt.gca().invert_yaxis()
plt.show()
