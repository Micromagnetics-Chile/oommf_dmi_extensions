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

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import skimage
import scipy.interpolate as si
import scipy.signal as ss

data = np.load('SIMS_output_D2_15eM4_dx2P5nm/sim_12ngbs_Cn_D1_20eM4_D2_15eM4_snaps/m_By_1000_1e-4T.npy')
Ms = 658e3

spins = data / Ms

mz_plane = spins[:, :, 0, 2].T

f, ax = plt.subplots()
ax.imshow(mz_plane, cmap='magma', origin='lower', vmin=-1, vmax=1)
plt.show()

mesh_X = np.load('sample_Cn_mesh_cells_X.npy')

# +
F = np.fft.fftshift(np.fft.fft2(mz_plane))
power_spectrum = np.abs(F)

dx = mesh_X[1] - mesh_X[0]  
dy = mesh_X[1] - mesh_X[0]
ny, nx = mz_plane.shape
kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
KX, KY = np.meshgrid(kx, ky)
# -

f, ax = plt.subplots()
ax.imshow(power_spectrum, cmap='magma', origin='lower', vmax=5e3)
plt.show()

peaks = skimage.feature.peak_local_max(power_spectrum, num_peaks=2)

f, ax = plt.subplots()
ax.contourf(KX * 1e-9, KY * 1e-9, power_spectrum, cmap='magma', origin='lower', vmax=5e3, levels=50)
ax.scatter(KX[peaks[0, 0], peaks[0, 1]] * 1e-9, KY[peaks[0, 0], peaks[0, 1]] * 1e-9, c='red', s=45, edgecolor='k')
ax.scatter(KX[peaks[1, 0], peaks[1, 1]] * 1e-9, KY[peaks[1, 0], peaks[1, 1]] * 1e-9, c='cyan', s=45, edgecolor='k')
ax.set_xlim(-0.04, 0.04)
ax.set_ylim(-0.04, 0.04)
ax.grid()
plt.show()

k1_peak = np.array([KX[peaks[0, 0], peaks[0, 1]], KY[peaks[0, 0], peaks[0, 1]]]) * 1e-8
# Peak in the positive quarter: (I)
k2_peak = np.array([KX[peaks[1, 0], peaks[1, 1]], KY[peaks[1, 0], peaks[1, 1]]]) * 1e-8
print(f'{k1_peak=}')
print(f'{k2_peak=}')
angle = np.atan2(k2_peak[1], k2_peak[0])
print(f'Sim angle: {np.degrees(angle):.3f}')

# +
D1, D2 = 0.0020, 0.0015
theory_angle = np.degrees(0.5 * np.pi - np.arctan2(D1, D2))

print(f'theory angle = {theory_angle:.3f}')
# -

# # Analysis of derivatives

# def compute_critical_field(datatable):
Ms = 658e3
dtable = np.loadtxt('SIMS_output_D2_15eM4_dx2nm/table_sim_12ngbs_Cn_D1_20eM4_D2_15eM4.txt', usecols=[2, 3, 4, 5])

# +
f, axs  = plt.subplots(ncols=2, figsize=(12, 4))
for i in range(2, 3):
    axs[0].plot(dtable[:, 0], dtable[:, i] / Ms, 'o', ms=2)
    
    interpF = si.make_interp_spline(dtable[:, 0][::-1], dtable[:, i][::-1] / Ms, k=3)
    x = np.linspace(0., 2.0, 200)
    axs[0].plot(x, interpF(x), '-', ms=2)

    DDinterpF = interpF.derivative(nu=1)
    # DDinterpF = DinterpF.derivative(nu=1)
    axs[1].plot(x, DDinterpF(x), 'o-')

# axs[1].set_xlim(0, 0.3)
axs[0].set_xlabel(r'$B_y$')
axs[0].set_ylabel(r'$m_y$')
axs[1].set_ylabel(r'$\mathrm{d} m_y \,/\, \mathrm{d} B_y$')
axs[1].set_xlabel(r'$B_y$')
ax.grid()
plt.show()

# +
f, ax  = plt.subplots()

gradX = np.gradient(dtable[:, 2][::-1] / Ms, dtable[:, 0][::-1])
gradXX = np.gradient(gradX, dtable[:, 0][::-1])
peak = ss.find_peaks(np.abs(gradXX), height=50)
print(peak)
print(dtable[:, 0][::-1][peak[0]], gradXX[peak[0]])
print(dtable[:, 0][::-1][peak[0]], gradX[peak[0]])

ax2 = ax.twinx()
ax2.plot(dtable[:, 0][::-1], np.abs(gradXX), '-', ms=2, c='C2')
ax.plot(dtable[:, 0][::-1], np.abs(gradX), '--', ms=2, c='C1')
ax.scatter(dtable[:, 0][::-1][peak[0]], gradX[peak[0]], c='k', zorder=5)
ax.grid()

ax.set_xlabel(r'$B_y$')
ax.set_ylabel(r'$\mathrm{d} m_y \,/\, \mathrm{d} B_y$')
ax2.set_ylabel(r'$\mathrm{d}^2 m_y \,/\, \mathrm{d} B_{y}^2$')

ax.set_xlim(0., 0.7)

plt.show()


# -

# # Summary of simulation outputs

def compute_angle(snapshot, mesh, D1, D2=0.0015, Ms=658e3):
    """
    """
    data = np.load(snapshot)
    spins = data / Ms
    mz_plane = spins[:, :, 0, 2].T
    # mesh_X = np.load('SIMS_output_D2_15eM4_dx2nm/sample_mesh_cells_X.npy')  # TODO: set as argument
    mesh_X = np.load(mesh)
    
    F = np.fft.fftshift(np.fft.fft2(mz_plane))
    power_spectrum = np.abs(F)

    # f, ax = plt.subplots()
    # ax.imshow(power_spectrum, cmap='magma', origin='lower', vmax=5e3)
    # plt.show()
    
    dx = mesh_X[1] - mesh_X[0]  
    dy = mesh_X[1] - mesh_X[0]
    ny, nx = mz_plane.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)

    peaks = skimage.feature.peak_local_max(power_spectrum, num_peaks=2)

    k1_peak = np.array([KX[peaks[0, 0], peaks[0, 1]], KY[peaks[0, 0], peaks[0, 1]]]) * 1e-8
    # Peak in the positive quarter: (I)
    k2_peak = np.array([KX[peaks[1, 0], peaks[1, 1]], KY[peaks[1, 0], peaks[1, 1]]]) * 1e-8
    # print(f'{k1_peak=}')
    # print(f'{k2_peak=}')

    kpeak = k1_peak if (k1_peak[0] >= 0 and k1_peak[1] >= 0) else k2_peak
    # print(kpeak)
    angle = np.atan2(kpeak[1], kpeak[0])
    angle = np.degrees(angle)

    # D1, D2 = 0.0020, 0.002
    theory_angle = np.degrees(np.arctan2(D1, D2))

    return angle, theory_angle


print("RESULTS: dx = 2 nm  12-neighbours FD")
print("-" * 40)
D2f = 15e-4
for D1 in [0, 5, 10, 15, 20, 25]:

    # Find critical field from data by looking at a very rough estimate of the 2nd derivative
    dtable = np.loadtxt(f'SIMS_output_D2_15eM4_dx2nm/table_sim_12ngbs_Cn_D1_{D1:02d}eM4_D2_15eM4.txt', usecols=[2, 3, 4, 5])
    # Take (my,B) data in ascending order
    M, B = dtable[:, 2][::-1] / Ms, dtable[:, 0][::-1]
    gradX = np.gradient(M, B)  # 1st deriv, dm/dB
    gradXX = np.gradient(gradX, dtable[:, 0][::-1])  # 2nd deriv
    peak = ss.find_peaks(np.abs(gradXX), height=100)
    peak = peak[0][0] - 1  # Take index of point just before the peak (data in jumps of 50 * 1e-4 T)

    CField = B[peak]
    snapIdx = int(np.around(CField * 1e4))
    # print(snapIdx)
    # print(CField)

    snap = f'SIMS_output_D2_15eM4_dx2nm/sim_12ngbs_Cn_D1_{D1:02d}eM4_D2_15eM4_snaps/m_By_{snapIdx:04d}_1e-4T.npy'
    mesh = f'SIMS_output_D2_15eM4_dx2nm/sample_mesh_cells_X.npy'
    D1f = D1 * 1e-4

    ang, theo = compute_angle(snap, mesh, D1f, D2=D2f, Ms=658e3)

    
    print(f'D1 = {D1f*1e3:>.2f}e-3  D2 = {D2f*1e3:>.2f}e-3  Bcrit = {CField*1e3:.0f} mT  Sim angle = {90 - ang:>7.4f}  Theor. angle = {theo:>7.4f}')

print("RESULTS: dx = 1nm  12-neighbours FD")
print("-" * 40)
D2f = 15e-4
for D1 in [0, 5]:

    snap = f'SIMS_output_D2_15eM4_dx1nm/sim_12ngbs_Cn_D1_{D1:02d}eM4_D2_15eM4_snaps/m_By_0000_1e-4T.npy'
    mesh = f'SIMS_output_D2_15eM4_dx1nm/sample_mesh_cells_X.npy'
    D1f = D1 * 1e-4

    ang, theo = compute_angle(snap, mesh, D1f, D2=D2f, Ms=658e3)

    
    print(f'D1 = {D1f*1e3:>.2f}e-3  D2 = {D2f*1e3:>.2f}e-3  Sim angle = {90 - ang:>7.4f}  Theor. angle = {theo:>7.4f}')

print("RESULTS: dx = 2.5 nm  12-neighbours FD")
print("-" * 40)
D2f = 15e-4
for D1 in [15, 20, 25]:

    # Find critical field from data by looking at a very rough estimate of the 2nd derivative
    dtable = np.loadtxt(f'SIMS_output_D2_15eM4_dx2P5nm/table_sim_12ngbs_Cn_D1_{D1:02d}eM4_D2_15eM4.txt', usecols=[2, 3, 4, 5])
    # Take (my,B) data in ascending order
    M, B = dtable[:, 2][::-1] / Ms, dtable[:, 0][::-1]
    gradX = np.gradient(M, B)  # 1st deriv, dm/dB
    gradXX = np.gradient(gradX, dtable[:, 0][::-1])  # 2nd deriv
    peak = ss.find_peaks(np.abs(gradXX), height=100)
    peak = peak[0][0] - 1  # Take index of point just before the peak (data in jumps of 50 * 1e-4 T)

    CField = B[peak]
    snapIdx = int(np.around(CField * 1e4))
    # print(snapIdx)
    # print(CField)

    snap = f'SIMS_output_D2_15eM4_dx2P5nm/sim_12ngbs_Cn_D1_{D1:02d}eM4_D2_15eM4_snaps/m_By_{snapIdx:04d}_1e-4T.npy'
    mesh = f'SIMS_output_D2_15eM4_dx2P5nm/sample_mesh_cells_X.npy'
    D1f = D1 * 1e-4

    ang, theo = compute_angle(snap, mesh, D1f, D2=D2f, Ms=658e3)

    
    print(f'D1 = {D1f*1e3:>.2f}e-3  D2 = {D2f*1e3:>.2f}e-3  Bcrit = {CField*1e3:.0f} mT  Sim angle = {90 - ang:>7.4f}  Theor. angle = {theo:>7.4f}')

print("RESULTS: dx = 2.5 nm  6-neighbours FreeBC  FD")
print("-" * 40)
D2f = 15e-4
for D1 in [0, 5, 10, 15, 20, 25]:

    # Find critical field from data by looking at a very rough estimate of the 2nd derivative
    dtable = np.loadtxt(f'SIMS_output_D2_15eM4_dx2P5nm_6ngbrsFree/table_sim_6ngbrsFree_Cn_D1_{D1:02d}eM4_D2_15eM4.txt',
                        usecols=[2, 3, 4, 5])
    # Take (my,B) data in ascending order
    M, B = dtable[:, 2][::-1] / Ms, dtable[:, 0][::-1]
    gradX = np.gradient(M, B)  # 1st deriv, dm/dB
    gradXX = np.gradient(gradX, dtable[:, 0][::-1])  # 2nd deriv
    peak = ss.find_peaks(np.abs(gradXX), height=100)
    peak = peak[0][0] - 1  # Take index of point just before the peak (data in jumps of 50 * 1e-4 T)

    CField = B[peak]
    snapIdx = int(np.around(CField * 1e4))
    # print(snapIdx)
    # print(CField)

    snap = f'SIMS_output_D2_15eM4_dx2P5nm_6ngbrsFree/sim_6ngbrsFree_Cn_D1_{D1:02d}eM4_D2_15eM4_snaps/m_By_{snapIdx:04d}_1e-4T.npy'
    mesh = f'SIMS_output_D2_15eM4_dx2P5nm_6ngbrsFree/sample_mesh_cells_X.npy'
    D1f = D1 * 1e-4

    ang, theo = compute_angle(snap, mesh, D1f, D2=D2f, Ms=658e3)

    
    print(f'D1 = {D1f*1e3:>.2f}e-3  D2 = {D2f*1e3:>.2f}e-3  Bcrit = {CField*1e3:.0f} mT  Sim angle = {90 - ang:>7.4f}  Theor. angle = {theo:>7.4f}')
    ang, theo = compute_angle(snap, mesh, D1f, D2=D2f, Ms=658e3)

D2f = 15e-4
for D1 in [15, 20, 25, 30]:

    snap = f'SIMS_output_D2_15eM4_circle_dx2nm/sim_12ngbs_Cn_D1_{D1:02d}eM4_D2_15eM4_snaps/m_By_0000_1e-4T.npy'
    mesh = f'SIMS_output_D2_15eM4_circle_dx2nm/sample_mesh_cells_X.npy'
    D1f = D1 * 1e-4

    ang, theo = compute_angle(snap, mesh, D1f, D2=D2f, Ms=658e3)

    
    print(f'D1 = {D1f*1e3:>.2f}e-3  D2 = {D2f*1e3:>.2f}e-3  Sim angle = {90 - ang:>7.4f}  Theor. angle = {theo:>7.4f}')

D2f = 15e-4
for D1 in [15, 20, 25]:

    # Find critical field from data by looking at a very rough estimate of the 2nd derivative
    dtable = np.loadtxt(f'SIMS_output_D2_15eM4_circle_dx2nm/table_sim_12ngbs_Cn_D1_{D1:02d}eM4_D2_15eM4.txt', usecols=[2, 3, 4, 5])
    # Take (my,B) data in ascending order
    M, B = dtable[:, 2][::-1] / Ms, dtable[:, 0][::-1]
    gradX = np.gradient(M, B)  # 1st deriv, dm/dB
    gradXX = np.gradient(gradX, dtable[:, 0][::-1])  # 2nd deriv
    peak = ss.find_peaks(np.abs(gradXX), height=100)
    peak = peak[0][0] - 1  # Take index of point just before the peak (data in jumps of 50 * 1e-4 T)

    CField = B[peak]
    snapIdx = int(np.around(CField * 1e4))
    # print(snapIdx)
    # print(CField)

    snap = f'SIMS_output_D2_15eM4_circle_dx2nm/sim_12ngbs_Cn_D1_{D1:02d}eM4_D2_15eM4_snaps/m_By_{snapIdx:04d}_1e-4T.npy'
    mesh = f'SIMS_output_D2_15eM4_circle_dx2nm/sample_mesh_cells_X.npy'
    D1f = D1 * 1e-4

    ang, theo = compute_angle(snap, mesh, D1f, D2=D2f, Ms=658e3)

    
    print(f'D1 = {D1f*1e3:>.2f}e-3  D2 = {D2f*1e3:>.2f}e-3  Bcrit = {CField*1e3:.0f} mT  Sim angle = {90 - ang:>7.4f}  Theor. angle = {theo:>7.4f}')
    print('-' * 10)


