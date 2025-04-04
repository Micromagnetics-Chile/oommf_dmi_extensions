# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import oommfc as oc
import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.interpolate

# from typing import NamedTuple
from collections import namedtuple

from pathlib import Path
import dill

import matplotlib
matplotlib.style.use('default')

nm = 1e-9
import datetime

from pathlib import Path
# -

# Specify OOMMF path
oc.runner.runner = oc.oommf.TclOOMMFRunner("/home/david/software/oommf/oommf.tcl")

# +
# Material parameters
A = 11.1e-12    # exchange constant
M_s = 658e3     # saturation magnetization
K_u = 0.25e6    # uniaxial anis

D2 = 1.5e-3
D1_max = 2.0e-3
D1_min = 0.5e-3
n_D = 4
D1_values = np.linspace(D1_min, D1_max, n_D)

Bymax = 2.0
Bymin = 0.0
dBy = 5e-3
nB = int((Bymax - Bymin) / dBy)
By_values = np.linspace(Bymax, Bymin, nB + 1)
# -

n_xy = 500
mesh = df.Mesh(p1=(-500e-9, -500e-9, 0), p2=(500e-9, 500e-9, 1e-9), n=(n_xy, n_xy, 1))

print(D1_values)

# +
# name_table = '2D_sample_Cn_datatable.txt'

mag_arr = []
B_arr = []

for D1 in D1_values:

    # Define the system
    system = mm.System(name=f"sample_Cn")
    system.m = df.Field(mesh, nvdim=3, value=[0, 0, 1], norm=M_s)
    system.energy = (mm.Exchange(A=A) + 
                     mm.UniaxialAnisotropy(K=K_u, u=[0, 0, 1]) + 
                     mm.DMI(D1=D1, D2=D2, crystalclass="Cn_z", suffix='12ngbrs') +
                     mm.Demag() +
                     mm.Zeeman(H=(0., 0., 0.))
                     )
    md = oc.MinDriver()

    # SIM name
    name = f'calc_12ngbs_Cn_D1_{D1*1e4:02.0f}e-4_D2_20e-4'
    SAVEFIG = Path(f'{name}_figs')
    SAVEFIG.mkdir(exist_ok=True)
    SAVESP = Path(f'{name}_snaps')
    SAVESP.mkdir(exist_ok=True)

    F = open(f'table_{name}.txt', 'w')

    for i, By in enumerate(By_values):
        system.energy.zeeman.H = (0, By / mm.consts.mu0, 0)
        md.drive(system, verbose=0, stopping_mxHxm=0.1, n_threads=10)  # Def. toque is 0.01
    
        LINE = '{} '.format(datetime.datetime.now())
        LINE += f'{By} '
        means = system.m.mean()
        LINE += f'{means[0]} {means[1]} {means[2]} \n'

        F.write(LINE)
        F.flush()

        # Save snapshot
        np.save(SAVESP / "m_By_{:04.0f}_1e-4T.npy".format(By * 1e4), system.m.array)

        if i % 5 == 0:
            mfield = system.m.sel("z")  # xy plane slice
            mfield.norm = 1.
            f, ax = plt.subplots()
            mfield.z.mpl.scalar(cmap='magma', clim=[-1, 1], ax=ax, 
                                filename=SAVEFIG / "mz_By_{:04.0f}_1e-4T.png".format(By * 1e4)
                                )
            plt.close()

    F.close()
# -




