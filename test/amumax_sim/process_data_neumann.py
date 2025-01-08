import zarr
import dill
from collections import namedtuple
from pathlib import Path
import re
import math
import numpy as np


magDataAMX = namedtuple('magDataAMX', ['mx', 'my', 'mz', 'x', 'n_x', 'h'])
SAVEDIR = Path('amumax_1dchain_neumann_data')
SAVEDIR.mkdir(exist_ok=True)

# AmuMax version:
MX3_DIR = Path('./')
MX3_ZFILES = sorted(list(MX3_DIR.glob('*neumann*zarr')))

data_amx = {}

# Neumann BCs
for ZARR in MX3_ZFILES:
    Zdata = zarr.open(ZARR, mode='r')
    Lx = Zdata.attrs['Lx']
    DX = Zdata.attrs['dx']
    print('dx = ', DX)
    h = float(DX)
    nx = Zdata.attrs['Nx']
    # Shift x domain
    data_amx[nx] = magDataAMX(mx=Zdata['m'][0, 0, 0, :, 0],
                              my=Zdata['m'][0, 0, 0, :, 1],
                              mz=Zdata['m'][0, 0, 0, :, 2],
                              x=((h * 0.5 + (np.arange(nx) * h)) - Lx * 0.5) * 1e9,  # in nm
                              n_x=nx,
                              h=h
                              )

    with open(SAVEDIR / f'data_NX{nx}.pkl', 'wb') as FILE:
        dill.dump(data_amx[nx], FILE)

    print(data_amx[nx].x.shape, data_amx[nx].mz.shape)
