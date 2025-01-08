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
# -

oc.runner.runner = oc.oommf.TclOOMMFRunner("/home/david/software/oommf/oommf.tcl")

# +
# Material parameters
A = 13e-12  # Jm^-1
D = 3e-3  # Jm^-2
M_s = 8.6e5  # Am^-1
K_u = 4e5  # Jm^-3

L = 50
# -

# Use units in the nm scale
nm = 1e-9
delta = (np.sqrt(A / K_u)) / nm
xi = (2 * A / D) / nm


# +
# Semi-analytic solution

# Set of differential equations for (u, v)
def ode_1d(theta, r, delta=1):
    return np.array([theta[1], np.cos(theta[0]) * np.sin(theta[0]) / delta ** 2])

# Define a function to solve the differential equation in the (0, l) range
# for any v(0) condition, u(0) is set to 1/xi
# The function returns the difference with respect to the desired solution at x=l
# The boundary conditions differ in sign for the interfacial and DMI cases
r_array = np.linspace(-L, L, 200)

def solve_ode(a, sign=1):  
    # Impose the condition over the angle rather than its derivative
    # solution, info = scipy.integrate.odeint(lambda t, r: ode_1d(t, r, delta=delta), 
    #                                         [np.arcsin(sign * delta / xi), a], r_array, full_output=True)
    solution = scipy.integrate.solve_ivp(lambda r, t: ode_1d(t, r, delta=delta),
                                         [-L, L],
                                         [np.arcsin(sign * delta / xi), a],
                                         t_eval=r_array,
                                         # full_output=True
                                         )
    # By inspection we found that the right extrema must be positive to match
    # the condition for the derivative
    return solution.y[0][-1] - np.arcsin(-sign * delta / xi)

# Find the root from the differences:
guess_int = scipy.optimize.brentq(lambda a: solve_ode(a, sign=-1), 0., 0.2)
print('Interfacial: ', guess_int)

# #  Solve the system with the right condition at theta(L)
# # for different x-axis discretization (to compare with the simulations)
# solution_int = {}

# xtheory = np.linspace(-L, L, 200)
# solution_theory = scipy.integrate.solve_ivp(
#         lambda r, t: ode_1d(t, r, delta=delta),
#         [-L, L],
#         [np.arcsin(-delta / xi), guess_int],
#         t_eval=xtheory, dense_output=True)

# theta_theory = solution_theory.sol(xtheory)[0]
# -

200 / 400

# Define different values for n_x and oommf suffix to loop over
n_x_values = [20 + 2 * i for i in range(40)] + [100, 150, 200, 250, 400, 500]
suffix_values = ["6ngbrs", "12ngbrs", "12ngbrs_RobinBC"]

n_x_values

# +
# class magData(NamedTuple):
#     x: np.ndarray
#     mz: 
#     fitFunc:

# Use collection to avoid looking at types
magData = namedtuple('magData', 'x mx my mz mz_interp mz_analytical theta_an_fun suffix n_x h')

# +
solutions = []

# Loop over different n_x and suffix values
for n_x in n_x_values:
    for suffix in suffix_values:
        mesh = df.Mesh(p1=(-50e-9, 0, 0), p2=(50e-9, 1e-9, 1e-9), n=(n_x, 1, 1))

        # Define the system
        system = mm.System(name=f"chain")
        system.m = df.Field(mesh, nvdim=3, value=[0, 0, 1], norm=M_s)
        system.energy = (
            mm.Exchange(A=A)
            + mm.UniaxialAnisotropy(K=K_u, u=[0, 0, 1])
            + mm.DMI(D=D, crystalclass="Cnv_z", suffix=suffix)
        )

        # Run the simulation
        mindriver = oc.MinDriver(stopping_mxHxm=0.1)
        mindriver.drive(system)

        x_nm = system.m.mesh.cells.x / nm

        # Obtain an analytical solution using the x coordinates
        # of the current data set (passed on the t_eval argument)
        # NOTE: If we use less or more points, the analytical sol should be the same
        #       We recompute here only to obtain the analytical sol array in the y[0] attribute
        solution_theory = scipy.integrate.solve_ivp(
            lambda r, t: ode_1d(t, r, delta=delta),
            [-L, L],
            [np.arcsin(-delta / xi), guess_int],
            t_eval=x_nm, dense_output=True)
        theta_an_sim = solution_theory.y[0]
        theta_an_fun = solution_theory.sol

        mz_interp = scipy.interpolate.interp1d(x_nm, system.m.orientation.z.array.squeeze(), 
                                               kind='cubic',
                                               bounds_error=False, fill_value='extrapolate')

        solutions.append(magData(x=x_nm,
                                 mx=system.m.orientation.x.array.squeeze(),
                                 my=system.m.orientation.y.array.squeeze(),
                                 mz=system.m.orientation.z.array.squeeze(),
                                 mz_interp=mz_interp,
                                 mz_analytical=np.cos(theta_an_sim),
                                 theta_an_fun=theta_an_fun,
                                 suffix=suffix,
                                 n_x=n_x,
                                 h=mesh.cell[0]
                                 )
                         )

# +
f, ax = plt.subplots()
for data in [s for s in solutions if s.n_x == 100]:
    ax.plot(data.x, 
            data.mz - data.mz_analytical, #  - np.cos(data.fitFunc(data.x / nm)),
            "o-",
            alpha=0.5, ms=4,
            label=f"OOMMF $m_x$ (n={data.n_x}, h={data.h/1e-9:.2g} nm, suffix={data.suffix})")

ax.set_xlim(-L, -L / 1.25)

plt.grid()
# -

suffix_values

np.array([d.h for d in sdata]) * 1e9

# +
f, ax = plt.subplots()

for s in suffix_values:
    sdata = [sol for sol in solutions if sol.suffix==s]
    sdata = sorted(sdata, key=lambda x: x.n_x)
    # print(sdata)
    # print(sdata.sort(key=lambda x: x.n_x))
    mzerr = [np.abs(d.mz_interp(-L) - np.cos(d.theta_an_fun(-L)[0])) for d in sdata]
    hs = np.array([d.h for d in sdata])
    ax.plot(hs / nm, mzerr,
            "o-",
            alpha=0.5, ms=4,
            label=f"OOMMF $m_z$ ({s})")

ax.legend()
ax.set_xlabel('h  nm')
ax.set_ylabel(r'log($|m_z - m_z^{\mathrm{an}}|$)  at x = -50 nm')
ax.set_yscale('log')
ax.set_xscale('log')
xts = [0.10 * i for i in range(1, 10)]
ax.set_xticks([x for x in xts])
ax.set_xticklabels(['{:.2f}'.format(h) for h in xts])
# ax.set_xticks([1., 4.])
plt.grid()

# +
f, ax = plt.subplots(figsize=(8, 8))

for s in suffix_values:
    sdata = [sol for sol in solutions if sol.suffix==s]
    sdata = sorted(sdata, key=lambda x: x.n_x)
    # print(sdata)
    # print(sdata.sort(key=lambda x: x.n_x))
    mzerr = [np.abs(d.mz_interp(-L) - np.cos(d.theta_an_fun(-L)[0])) for d in sdata]
    ns = np.array([d.n_x for d in sdata])
    z = np.polyfit(np.log(ns), np.log(mzerr), 1)
    print(s, z)
    ax.plot(ns, mzerr,
            "o-",
            alpha=0.5, ms=4,
            label=f"OOMMF $m_z$ ({s})")

ax.legend()
ax.set_xlabel('N')
ax.set_ylabel(r'log($|m_z - m_z^{\mathrm{an}}|$)  at x = -50 nm')
ax.set_yscale('log')
ax.set_xscale('log')
xts = [20 * i for i in range(1, 6)] + [200, 400]
ax.set_xticks([x for x in xts])
ax.set_xticklabels(['{:.0f}'.format(x) for x in xts])
# ax.set_xticks([1., 4.])
plt.grid()

plt.savefig('N_vs_err-mz_OOMMF_stencils_TEST.pdf', bbox_inches='tight')
# -
from pathlib import Path
import dill

AMX_DIR = Path('amumax_sim/amumax_1dchain_neumann_data/')

# +
amx_data = {}
amx_interp_data = {}

for PKL in AMX_DIR.glob('*.pkl'):
    with open(PKL, 'rb') as FILE:
        data = dill.load(FILE)
    
    amx_data[data.n_x] = data

    mz_interp = scipy.interpolate.interp1d(data.x, data.mz, kind='cubic',
                                           bounds_error=False, fill_value='extrapolate')
    amx_interp_data[data.n_x] = mz_interp
# -


xev = np.linspace(-L, L, 200)
solution_theory = scipy.integrate.solve_ivp(
            lambda r, t: ode_1d(t, r, delta=delta),
            [-L, L],
            [np.arcsin(-delta / xi), guess_int],
            t_eval=xev, dense_output=True)
theta_an_fun = solution_theory.sol

# +
L = 50
amx_n_vs_error = []
for k in sorted(amx_interp_data.keys()):
    intFun = amx_interp_data[k]
    
    mzerr = np.abs(intFun(-L) - np.cos(theta_an_fun(-L)[0]))
    amx_n_vs_error.append([k, mzerr])

amx_n_vs_error = np.array(amx_n_vs_error)
# -

import matplotlib
matplotlib.style.use('default')

# +
f, ax = plt.subplots(figsize=(6, 6))

for s in suffix_values:
    sdata = [sol for sol in solutions if sol.suffix==s]
    sdata = sorted(sdata, key=lambda x: x.n_x)
    # print(sdata)
    # print(sdata.sort(key=lambda x: x.n_x))
    mzerr = [np.abs(d.mz_interp(-L) - np.cos(d.theta_an_fun(-L)[0])) for d in sdata]
    ns = np.array([d.n_x for d in sdata])
    z = np.polyfit(np.log(ns), np.log(mzerr), 1)
    print(s, z)
    ax.plot(ns, mzerr,
            "o-",
            alpha=1.0, ms=4,
            label=f"OOMMF $m_z$ ({s})")

ax.plot(amx_n_vs_error[:, 0], amx_n_vs_error[:, 1], '--ok', label='MuMax3 (6ngbrs RobinBC)', ms=4)

# Scaling reference line
# xN2 = np.linspace(20, 400, 200)
# ax.plot(xN2, xN2**-2)

xN3 = np.linspace(20, 500, 200)
ax.plot(xN3, np.exp(3) * xN3**-3, '--', color='grey', lw=1, dashes=[6, 2])

ax.legend()
ax.set_xlabel('N')
ax.set_ylabel(r'log($|m_z - m_z^{\mathrm{an}}|$)  at x = -50 nm')
ax.set_yscale('log')
ax.set_xscale('log')
xts = [20 * i for i in range(1, 6)] + [200, 400]  # n values
ax.set_xticks([x for x in xts])
ax.set_xticklabels(['{:.0f}'.format(x) for x in xts])
ax.set_ylim(0.5e-6, 0.2)

ax.text(20, 0.5e-3, '$\propto N^{-3}$', color ='grey')

hs = [100 / n for n in xts]
ax2 = ax.secondary_xaxis('top')
ax2.set_xticks([x for x in xts])
ax2.set_xticklabels(['{:.1f}'.format(h) for h in hs])
ax2.set_xlabel('h  (nm)')

plt.grid()

plt.savefig('N_vs_err-mz_OOMMF_stencils_TEST.pdf', bbox_inches='tight')
# -

# # Masell model

# +
A = 13e-12
D = 3e-3
Ku = 0.4e6

Q = D / (2 * A)
kappa = Ku / (A * (Q**2))

def y0_constant(kappa, h=0):
    hk = 1. / np.sqrt(h + kappa)
    y0 = -hk * np.arccosh((h + kappa + np.sqrt((h + kappa)**2 - kappa)) / np.sqrt(h))
    return y0

def theta_JM(yp, kappa, h=0):
    y0p = y0_constant(kappa, h)
    hk = np.sqrt(h / (h + kappa))
    t = -np.pi + 2 * np.arctan(hk * np.sinh(np.sqrt(h + kappa) * (yp - y0p)))
    return t


# +
# xRef = np.linspace(0., 25., 501) * 1e-9
# thetaRef = theta_JM(xRef * Q, kappa, h=1e-15)
# -

theta_Boundary_MODEL = theta_JM(0. * Q, kappa, h=1e-20)

# +
# Compute mumax errors using M Model
L = 50
amx_n_vs_error_MM = []
for k in sorted(amx_interp_data.keys()):
    intFun = amx_interp_data[k]
    
    mzerr = np.abs(intFun(-L) - np.cos(theta_Boundary_MODEL))
    amx_n_vs_error_MM.append([k, mzerr])

amx_n_vs_error_MM = np.array(amx_n_vs_error_MM)

# +
f, ax = plt.subplots(figsize=(6, 6))

for s in suffix_values:
    sdata = [sol for sol in solutions if sol.suffix==s]
    sdata = sorted(sdata, key=lambda x: x.n_x)
    # print(sdata)
    # print(sdata.sort(key=lambda x: x.n_x))
    mzerr = [np.abs(d.mz_interp(-L) - np.cos(theta_Boundary_MODEL)) for d in sdata]
    ns = np.array([d.n_x for d in sdata])
    z = np.polyfit(np.log(ns), np.log(mzerr), 1)
    print(s, z)
    ax.plot(ns, mzerr,
            "o-",
            alpha=1.0, ms=4,
            label=f"OOMMF $m_z$ ({s})")

ax.plot(amx_n_vs_error_MM[:, 0], amx_n_vs_error_MM[:, 1], '--ok', label='MuMax3 (6ngbrs RobinBC)', ms=4)

# Scaling reference line
# xN2 = np.linspace(20, 400, 200)
# ax.plot(xN2, xN2**-2)

xN3 = np.linspace(20, 500, 200)
ax.plot(xN3, np.exp(3) * xN3**-3, '--', color='grey', lw=1, dashes=[6, 2])

ax.legend()
ax.set_xlabel('N')
ax.set_ylabel(r'log($|m_z - m_z^{\mathrm{an}}|$)  at x = -50 nm')
ax.set_yscale('log')
ax.set_xscale('log')
xts = [20 * i for i in range(1, 6)] + [200, 400]  # n values
ax.set_xticks([x for x in xts])
ax.set_xticklabels(['{:.0f}'.format(x) for x in xts])
ax.set_ylim(0.5e-6, 0.2)

ax.text(20, 0.5e-3, '$\propto N^{-3}$', color ='grey')

hs = [100 / n for n in xts]
ax2 = ax.secondary_xaxis('top')
ax2.set_xticks([x for x in xts])
ax2.set_xticklabels(['{:.1f}'.format(h) for h in hs])
ax2.set_xlabel('h  (nm)')

plt.grid()
# -


