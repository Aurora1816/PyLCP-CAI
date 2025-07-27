
#%%
from braggclass import Bragg
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pylcp.governingeq import governingeq
import pylcp
import numpy as np
import h5py
from pylcp.integration_tools import RandomOdeResult
import pathos
from pathos.pools import ProcessPool
from functools import partial
from pylcp.common import progressBar
from gclass import emovtion_atom
from gclass import process_atom
#%%
def power(base, exponent):
    return base ** exponent
square = partial(power, exponent=2)
cube = partial(power, exponent=3)
numbers = [1, 2, 3, 4, 5]
def plot_results(results_square, results_cube):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(numbers, results_square, label='Square')
    ax[1].plot(numbers, results_cube, label='Cube')
    ax[0].set_title('Square Function')
    ax[1].set_title('Cube Function')
    ax[0].legend()
    ax[1].legend()
    plt.show()
if __name__ == '__main__':
    with ProcessPool() as pool:
        results_square = pool.map(square, numbers)
        results_cube = pool.map(cube, numbers)
    plot_results(results_square, results_cube) #%% 
# Built-in parameters (fixed)
k = 2*np.pi/780E-9                      # Wave vector, unit: cm^{-1}
x0 = 1/k                                # Length unit conversion factor, converted unit: cm
# Configuration parameters (variables)
atom = pylcp.atom("87Rb")               # Atom type: 87Rb
sols_bragg = []
with h5py.File('upobe_sols3.h5', 'r') as f:
    for key in f.keys():
        group = f[key]
        t = np.array(group['t'])
        r = np.array(group['r'])
        v = np.array(group['v'])
        rho = np.array(group['rho'])       
        sol = RandomOdeResult(t=t/(atom.state[2].gammaHz), r=(r) * x0 +np.array([0,0,0])[:, np.newaxis], v=(v)* (atom.state[2].gammaHz / k)-np.array([0,0,0])[:, np.newaxis], rho=rho[:8, :8, :])
        sols_bragg.append(sol)
final_velocities = [sol.v[:, -1] for sol in sols_bragg]
mean_velocity = np.mean(final_velocities, axis=0)
for sol in sols_bragg:
    sol.v[:, -1] = sol.v[:, -1] + (mean_velocity - sol.v[:, -1]) * 0.95
# Convert v and r to SI units: m/s and m
# Shape annotation:
# sols4[natoms].r.shape=(3, 10002)
# sols4[natoms].v.shape=(3, 10002)
# sols4[natoms].rho.shape=(13, 13, 5001)
#%%
# Built-in parameters (fixed)
hbar = 1.0545718e-34                    # Planck constant, unit: m^2 kg / s
M = 1.443e-25                           # Mass of Rb87 F=2 atom, unit: kg
rs = 2.5                                 # Waist radius of Gaussian beam
omega_r = hbar * k**2 / (2 * M)         # Atomic vibration frequency in the optical potential well, unit: rad/s
tmin = 0.20/omega_r                     # Start time of Bragg Gaussian pulse
g = -np.array([0., 9.78, 0.])           # Gravitational acceleration, unit: m/s^2. Here, the Y-axis is the gravity axis.
gamma = atom.state[2].gammaHz           # Natural line width of the atom, unit: Hz
rho = 0.19/omega_r                      # Atomic state density
tau_bragg = 3.27*0.45/omega_r           # Pulse duration
Delta = 3e9                             # Detuning between the laser and the atomic resonance frequency, unit: Hz
v_r = hbar * k / M                      # Recoil velocity, unit: m/s
alpha = -1.6e7*9.78/2/np.pi             # Chirp
laserBeams1 = {}                        # Reset the configuration of the first Raman beam
n = 13                                  # Number of atomic states (fixed: 13)
initial_state = np.zeros(n, dtype=np.complex128)
initial_state[6] = 1 + 0.0j             # Initial atomic state distribution
# Configuration parameters (variables)
beamtype_bragg = 'guass'                # Beam type: Gaussian beam
wb_bragg = 2.4e5                        # Spot size, beam diameter, unit: cm. Range: [0, 10]
phi_i_bragg = 0                         # Beam polarization. Range: [0, 2*np.pi]
s_bragg = 280                           # Light intensity, unit: mW/cm^2. Range: [0, 200]
phase1_bragg = 0                        # Beam phase. Range: [0, 2*np.pi]    
kvec_bragg = np.array([0., 1., 0.])     # Beam wave vector. Range: 1, -1    
wscan1_bragg = 5469942.21580555         # Operating frequency of the first Raman beam, unit: Hz (obtained from the Raman spectrum)
delta1_bragg = 2*(6 + 0)*omega_r - wscan1_bragg*2*np.pi   # Calculate the detuning of the first Raman beam, unit: rad/s.
laserBeams1 = pylcp.laserBeams([
    {'kvec': kvec_bragg, 'pol': np.array([np.cos(phi_i_bragg/2), np.sin(phi_i_bragg/2), 0.]),
     'pol_coord': 'cartesian', 'delta': delta1_bragg, 's': s_bragg, 'phase': phase1_bragg, 'wb': wb_bragg},
    {'kvec': -kvec_bragg, 'pol': np.array([np.cos(phi_i_bragg/2), np.sin(phi_i_bragg/2), 0.]),
     'pol_coord': 'cartesian', 'delta': delta1_bragg, 's': s_bragg, 'phase': phase1_bragg, 'wb': wb_bragg},
], beam_type=pylcp.fields.gaussianBeam) # Configuration of the first Raman beam
Natoms_bragg = 8                        # Number of atoms. Range: [2, N]   
chunksize_bragg = 8                     # Number of cores used for computation. Range: [2, N] 
sols_i1_bragg = sols_bragg              # Introduce the initial state
progress_bar = True                     # Progress bar enabled: True or False
sols1_bragg = []                        # Save the solutions of the first pulse, sols5 
tmax1_bragg = 3.64e-06                  # Duration of the first Raman pulse, unit: s (obtained from the results)
time_span1_bragg = (0, 0 + tmax1_bragg) # Time range of the first Raman pulse, unit: s (obtained from the results)
kwargs1_bragg = {
    't_eval': np.linspace(0, 0 + tmax1_bragg, 5001),
    'progress_bar': True
}
Bragg_test1 = Bragg(hbar=hbar, M=M, v_r=v_r, laserBeams=laserBeams1, omega_r=omega_r, tmin=tmin, g=g, delta=delta1_bragg, alpha=alpha, k=k, phase=phase1_bragg, Delta=Delta, beamtype=beamtype_bragg, rho=rho, tau=tau_bragg)
#%%
# Interference module
if __name__ == '__main__':
    sol_range = np.arange(Natoms_bragg).reshape((int(Natoms_bragg/chunksize_bragg), chunksize_bragg))
    progress = progressBar()
    for jj in range(int(Natoms_bragg / chunksize_bragg)):
        with pathos.pools.ProcessPool(nodes=chunksize_bragg) as pool:
            arg_list = [(initial_state, sols_i1_bragg, time_span1_bragg, progress_bar, kwargs1_bragg, a) for a in sol_range[jj, :]]
            partial_function = partial(Bragg_test1.evolve_motion)
            # Use the process pool to map and compute the motion of each atom in parallel
            results = pool.map(partial_function, arg_list)
            sols1_bragg += results  # Add the results to sols5
        progress.update((jj + 1) / int(Natoms_bragg / chunksize_bragg))
# Shape annotation:
# sols5[natoms].r.shape=(3, 5001)
# sols5[natoms].v.shape=(3, 5001)
# sols5[natoms].N.shape=(13, 5001)
#%%
total_N_raman = np.sum([item_raman.N for item_raman in sols1_bragg], axis=0)/Natoms_bragg
plt.figure(dpi=600)
plt.plot(sols1_bragg[0].t, np.abs(sols1_bragg[0].y[12]), label=f'{6}*2$N\hbar $')
plt.title('Dynamical distribution')
plt.xlabel('Time (s)')
plt.ylabel('State distribution')
plt.legend()
plt.show()
#%%
# First Raman spectrum scan, for preliminary testing to obtain the frequency of the first Raman pulse (can be deleted)
rho_bragg1 = np.zeros(20)
j = 0  
w_scan1 = 5449942.215805551
w_scan2 = 5519942.215805551
for wscan in np.linspace(w_scan1, w_scan2, 20):
    delta1 = 0
    delta1 = 2*(6 + 0)*omega_r - wscan*2*np.pi                # Calculate the detuning, unit: rad/s.
    Bragg_test1 = Bragg(hbar=hbar, M=M, v_r=v_r, laserBeams=laserBeams1, omega_r=omega_r, tmin=tmin, g=g, delta=delta1, alpha=alpha, k=k, phase=phase1_bragg, Delta=3e9, beamtype=beamtype_bragg, rho=rho, tau=tau_bragg)
    sols_bragg = []
    jj = 0
    sol_range = np.arange(Natoms_bragg).reshape((int(Natoms_bragg/chunksize_bragg), chunksize_bragg))
    progress = progressBar()
    for jj in range(int(Natoms_bragg/chunksize_bragg)):
        with pathos.pools.ProcessPool(nodes=chunksize_bragg) as pool:
            arg_list = [(initial_state, sols_i1_bragg, time_span1_bragg, progress_bar, kwargs1_bragg, a) for a in sol_range[jj, :]]
            partial_function = partial(Bragg_test1.evolve_motion)
            sols_bragg += pool.map(partial_function, arg_list)
        progress.update((jj + 1) / int(Natoms_bragg / chunksize_bragg))       
    total_N_raman = np.sum([item_raman.N for item_raman in sols_bragg], axis=0)/Natoms_bragg
    rho_bragg1[j] = np.abs(total_N_raman[12, -1])
    j = j + 1
#%%
# Shape annotation:
# total_rho_raman.shape=(8, 8, 1001)
# First Raman spectrum scan, for preliminary testing to obtain the frequency of the first Raman pulse (can be deleted), plotting module
fig, ax = plt.subplots(1, 1, figsize=(35, 4.1), dpi=300)
ax.plot(np.linspace(w_scan1, w_scan2, 20)/1e6, rho_bragg1, linewidth=1.5, label=f'{6}*2$N\hbar $')
ax.legend(fontsize=7)
ax.set_xlabel('$Frequency (MHz)$')
ax.set_xticks(np.linspace(w_scan1, w_scan2, 40) / 1e6)
ax.set_ylabel('$\\rho_{ii}$')
#%% First motion evolution
sols_ig1_bragg = sols1_bragg            # Substitute the initial parameters of sols5
sols_g1_bragg = []                      # Save the solutions of the first motion evolution, sols_g1
tmax_g1_bragg = 0.05                    # Duration of the first motion evolution
t_g1_bragg = tmax_g1_bragg/4            # Time interval for setting the step size
atom_instance = emovtion_atom(g)
progress = progressBar()
sol_range = np.arange(Natoms_bragg).reshape((int(Natoms_bragg / chunksize_bragg), chunksize_bragg))
for jj in range(int(Natoms_bragg/chunksize_bragg)):
    with pathos.pools.ProcessPool(nodes=chunksize_bragg) as pool:
        arg_list = [(sols_ig1_bragg, g, tmax_g1_bragg, t_g1_bragg, idx) for idx in sol_range[jj, :]]
        partial_function = partial(process_atom.process_atom)
        sols_g1_bragg += pool.map(partial_function, arg_list)
    progress.update((jj + 1) / int(Natoms_bragg / chunksize_bragg))   

#%%
# Parameters for the second Raman beam
# Configuration parameters
phase2_bragg = 0                        # Phase of the second beam. Range: [0, 2*np.pi]    
wscan2_bragg = wscan1_bragg + alpha*(tmax_g1_bragg + tmax1_bragg)                # Operating frequency of the second Raman beam, unit: Hz (obtained from the Raman spectrum)
delta2_bragg = 2*(6 + 0)*omega_r - wscan2_bragg*2*np.pi   # Calculate the detuning of the second Raman beam, unit: rad/s.
laserBeams2 = {}                        # Reset the configuration of the second Raman beam
laserBeams2 = pylcp.laserBeams([
    {'kvec': kvec_bragg, 'pol': np.array([np.cos(phi_i_bragg/2), np.sin(phi_i_bragg/2), 0.]),
     'pol_coord': 'cartesian', 'delta': delta2_bragg, 's': s_bragg, 'phase': phase2_bragg, 'wb': wb_bragg},
    {'kvec': -kvec_bragg, 'pol': np.array([np.cos(phi_i_bragg/2), np.sin(phi_i_bragg/2), 0.]),
     'pol_coord': 'cartesian', 'delta': delta2_bragg, 's': s_bragg, 'phase': phase2_bragg, 'wb': wb_bragg},
], beam_type=pylcp.fields.gaussianBeam) # Configuration of the second Raman beam
sols_i2_bragg = sols_g1_bragg           # Introduce the initial state
sols2_bragg = []                        # Save the solutions of the second pulse, sols5 
tmax2_bragg = 7.28e-06                  # Duration of the second Raman pulse, unit: s (obtained from the results)
time_span2_bragg = (0, 0 + tmax2_bragg) # Time range of the second Raman pulse, unit: s (obtained from the results)
kwargs2_bragg = {
    't_eval': np.linspace(0, 0 + tmax2_bragg, 5001),
    'progress_bar': True
}
Bragg_test2 = Bragg(hbar=hbar, M=M, v_r=v_r, laserBeams=laserBeams2, omega_r=omega_r, tmin=tmin, g=g, delta=delta2_bragg, alpha=alpha, k=k, phase=phase2_bragg, Delta=Delta, beamtype=beamtype_bragg, rho=rho, tau=tau_bragg)
sol_range = np.arange(Natoms_bragg).reshape((int(Natoms_bragg/chunksize_bragg), chunksize_bragg))
progress = progressBar()
for jj in range(int(Natoms_bragg / chunksize_bragg)):
    with pathos.pools.ProcessPool(nodes=chunksize_bragg) as pool:
        arg_list = [(sols_i2_bragg, time_span2_bragg, progress_bar, kwargs2_bragg, a) for a in sol_range[jj, :]]
        partial_function = partial(Bragg_test2.evolve_motion1)
        # Use the process pool to map and compute the motion of each atom in parallel
        results = pool.map(partial_function, arg_list)
        sols2_bragg += results  # Add the results to sols5
    progress.update((jj + 1) / int(Natoms_bragg / chunksize_bragg))
# Shape annotation:
# sols6[natoms].r.shape=(3, 5001)
# sols6[natoms].v.shape=(3, 5001)
# sols6[natoms].N.shape=(13, 5001)
#%%
total_N_raman = np.sum([item_raman.N for item_raman in sols2_bragg], axis=0)/Natoms_bragg
plt.figure(dpi=600)
plt.plot(sols2_bragg[0].t, np.abs(total_N_raman[12]), label=f'{6}*2$N\hbar $')
plt.title('Dynamical distribution')
plt.xlabel('Time (s)')
plt.ylabel('State distribution')
plt.legend()
plt.show()
#%% Second motion evolution
sols_ig2_bragg = sols2_bragg            # Substitute the initial parameters
sols_g2_bragg = []                      # Save the solutions of the second motion evolution, sols_g1
tmax_g2_bragg = 0.05                    # Duration of the second motion evolution
t_g2_bragg = tmax_g2_bragg/4            # Time interval for setting the step size
atom_instance = emovtion_atom(g)
progress = progressBar()
sol_range = np.arange(Natoms_bragg).reshape((int(Natoms_bragg / chunksize_bragg), chunksize_bragg))
for jj in range(int(Natoms_bragg/chunksize_bragg)):
    with pathos.pools.ProcessPool(nodes=chunksize_bragg) as pool:
        arg_list = [(sols_ig2_bragg, g, tmax_g2_bragg, t_g2_bragg, idx) for idx in sol_range[jj, :]]
        partial_function = partial(process_atom.process_atom)
        sols_g2_bragg += pool.map(partial_function, arg_list)
    progress.update((jj + 1) / int(Natoms_bragg / chunksize_bragg))   

#%%
# Third Raman interference operation:
# Configuration parameters
phase3_bragg = 0                        # Phase of the third beam. Range: [0, 2*np.pi]    
wscan3_bragg = wscan2_bragg + alpha*(tmax_g2_bragg + tmax2_bragg)                # Operating frequency of the third Raman beam, unit: Hz (obtained from the Raman spectrum)
delta3_bragg = 2*(6 + 0)*omega_r - wscan3_bragg*2*np.pi   # Calculate the detuning of the third Raman beam, unit: rad/s.
laserBeams3 = {}                        # Reset the configuration of the third Raman beam
laserBeams3 = pylcp.laserBeams([
    {'kvec': kvec_bragg, 'pol': np.array([np.cos(phi_i_bragg/2), np.sin(phi_i_bragg/2), 0.]),
     'pol_coord': 'cartesian', 'delta': delta3_bragg, 's': s_bragg, 'phase': phase3_bragg, 'wb': wb_bragg},
    {'kvec': -kvec_bragg, 'pol': np.array([np.cos(phi_i_bragg/2), np.sin(phi_i_bragg/2), 0.]),
     'pol_coord': 'cartesian', 'delta': delta3_bragg, 's': s_bragg, 'phase': phase3_bragg, 'wb': wb_bragg},
], beam_type=pylcp.fields.gaussianBeam) # Configuration of the third Raman beam
sols_i3_bragg = sols_g2_bragg           # Introduce the initial state
sols3_bragg = []                        # Save the solutions of the third pulse, sols5 
tmax3_bragg = 3.64e-06                  # Duration of the third Raman pulse, unit: s (obtained from the results)
time_span3_bragg = (0, 0 + tmax3_bragg) # Time range of the third Raman pulse, unit: s (obtained from the results)
kwargs3_bragg = {
    't_eval': np.linspace(0, 0 + tmax3_bragg, 5001),
    'progress_bar': True
}
Bragg_test3 = Bragg(hbar=hbar, M=M, v_r=v_r, laserBeams=laserBeams3, omega_r=omega_r, tmin=tmin, g=g, delta=delta3_bragg, alpha=alpha, k=k, phase=phase3_bragg, Delta=Delta, beamtype=beamtype_bragg, rho=rho, tau=tau_bragg)
sol_range = np.arange(Natoms_bragg).reshape((int(Natoms_bragg/chunksize_bragg), chunksize_bragg))
progress = progressBar()
progress_bar = True
for jj in range(int(Natoms_bragg / chunksize_bragg)):
    with pathos.pools.ProcessPool(nodes=chunksize_bragg) as pool:
        arg_list = [(sols_i3_bragg, time_span3_bragg, progress_bar, kwargs3_bragg, a) for a in sol_range[jj, :]]
        partial_function = partial(Bragg_test3.evolve_motion1)
        # Use the process pool to map and compute the motion of each atom in parallel
        results = pool.map(partial_function, arg_list)
        sols3_bragg += results  # Add the results to sols5
    progress.update((jj + 1) / int(Natoms_bragg / chunksize_bragg))
# Shape annotation:
# sols6[natoms].r.shape=(3, 5001)
# sols6[natoms].v.shape=(3, 5001)
# sols6[natoms].N.shape=(13, 5001)
#%%
total_N_raman = np.sum([item_raman.N for item_raman in sols3_bragg], axis=0)/Natoms_bragg
plt.figure(dpi=600)
plt.plot(sols3_bragg[0].t, np.abs(total_N_raman[12])**2, label=f'{6}*2$N\hbar $')
plt.title('Dynamical distribution')
plt.xlabel('Time (s)')
plt.ylabel('State distribution')
plt.legend()
plt.show()
#%%
# Create a bragg_sols.h5 file to store the read sols0 (2D) data
t_list = [sol.t for sol in sols3_bragg]
r_list = [sol.r for sol in sols3_bragg]
v_list = [sol.v for sol in sols3_bragg]
N_list = [sol.N for sol in sols3_bragg]
with h5py.File('bragg_sols.h5', 'w') as f:
    for i, (t, r, v, N) in enumerate(zip(t_list, r_list, v_list, N_list)):
        group = f.create_group(f'sol_{i}')
        group.create_dataset('t', data=t)
        group.create_dataset('r', data=r)
        group.create_dataset('v', data=v)
        group.create_dataset('N', data=N)
#%%
# Read the sols0 data
sols021 = []
with h5py.File('bragg_sols.h5', 'r') as f:
    for key in f.keys():
        group = f[key]
        t = np.array(group['t'])
        r = np.array(group['r'])
        v = np.array(group['v'])
        N = np.array(group['N'])       
        sol = RandomOdeResult(t=t, r=r, v=v, N=N)
        sols_bragg.append(sol)         
# %% First fringe scan  
l = 0
rho_bragg_raman = np.zeros(40)
for phasei in np.linspace(0, 4*np.pi, 40):
    wscan = wscan2_bragg + alpha*(tmax_g2_bragg + tmax2_bragg)
    laserBeams = {}
    phase3 = 0
    delta3 = 2*(6 + 0)*omega_r - wscan*2*np.pi
    laserBeams = pylcp.laserBeams([
        {'kvec': kvec_bragg, 'pol': np.array([np.cos(phi_i_bragg/2), np.sin(phi_i_bragg/2), 0.]),
         'pol_coord': 'cartesian', 'delta': delta3, 's': s_bragg, 'phase': phase3, 'wb': wb_bragg},
        {'kvec': -kvec_bragg, 'pol': np.array([np.cos(phi_i_bragg/2), np.sin(phi_i_bragg/2), 0.]),
         'pol_coord': 'cartesian', 'delta': delta3, 's': s_bragg, 'phase': phase3, 'wb': wb_bragg},
    ], beam_type=pylcp.fields.gaussianBeam)
    Bragg_test3 = Bragg(hbar=hbar, M=M, v_r=v_r, laserBeams=laserBeams3, omega_r=omega_r, tmin=tmin, g=g, delta=delta3, alpha=alpha, k=k, phase=phasei, Delta=3e9, beamtype=beamtype_bragg, rho=rho, tau=tau_bragg)
    sols_bragg_raman = []
    jj = 0
    sol_range = np.arange(Natoms_bragg).reshape((int(Natoms_bragg/chunksize_bragg), chunksize_bragg))
    progress = progressBar()
    for jj in range(int(Natoms_bragg/chunksize_bragg)):
        with pathos.pools.ProcessPool(nodes=chunksize_bragg) as pool:
            arg_list = [(sols_i3_bragg, time_span3_bragg, progress_bar, kwargs3_bragg, a) for a in sol_range[jj, :]]
            partial_function = partial(Bragg_test3.evolve_motion1)
            sols_bragg_raman += pool.map(partial_function, arg_list)
        progress.update((jj + 1) / int(Natoms_bragg / chunksize_bragg))
    total_N_raman = np.sum([item_raman.N for item_raman in sols_bragg_raman], axis=0)/Natoms_bragg
    rho_bragg_raman[l] = np.real(total_N_raman[12, -1])
    l = l + 1
# Plotting module: Scanned interference fringes
fig, ax = plt.subplots(1, 1, figsize=(15, 4.1), dpi=300)
ax.plot(np.linspace(0, 4*np.pi, 40), rho_bragg_raman, linewidth=1.5, label=f'{6}*2$N\hbar $')
ax.legend(fontsize=7)
ax.set_xticks(np.linspace(0, 4*np.pi, 20))
ax.set_xlabel('$phase $')
ax.set_ylabel('$\\rho_{ii}$')  
#%% Third Raman spectrum scan
kg = 40
rho_bragg_chirp = np.zeros(kg)
j = 0
g_0 = np.dot(np.abs(Bragg_test3.laserBeams['g->e'].kvec()[0])* 1.6e7, g)
delta_v30 = np.max(np.abs(Bragg_test1.laserBeams['g->e'].kvec()[0]*sols_g2_bragg[0].v[:, 0] * 1.6e7))
delta_v3 = np.max(np.abs(Bragg_test1.laserBeams['g->e'].kvec()[0]*sols_g2_bragg[0].v[:, -1] * 1.6e7))
alpha1 = -24904965.495019782
alpha2 = -24904165.495019782  
T = 0.05  
phase3 = 0   
for alpha in np.linspace(alpha1, alpha2, kg):
    sols_chirp = []
    wscan3 = wscan2_bragg + alpha*(tmax_g2_bragg + tmax2_bragg)
    delta3 = 2*(6 + 0)*omega_r - wscan3*2*np.pi
    phase = g_0*((tmax_g2_bragg)**2) - alpha*2*np.pi*((tmax_g2_bragg)**2) + (((np.pi/4/(tmax3_bragg))*(1/delta_v3 - 1/delta_v30))/k/tmax_g2_bragg/tmax_g2_bragg*1e8)
    laserBeams3 = {}
    laserBeams3 = pylcp.laserBeams([
        {'kvec': kvec_bragg, 'pol': np.array([np.cos(phi_i_bragg/2), np.sin(phi_i_bragg/2), 0.]),
         'pol_coord': 'cartesian', 'delta': delta3, 's': s_bragg, 'phase': phase3, 'wb': wb_bragg},
        {'kvec': -kvec_bragg, 'pol': np.array([np.cos(phi_i_bragg/2), np.sin(phi_i_bragg/2), 0.]),
         'pol_coord': 'cartesian', 'delta': delta3, 's': s_bragg, 'phase': phase3, 'wb': wb_bragg},
    ], beam_type=pylcp.fields.gaussianBeam)
    Bragg_test_chirp = Bragg(hbar=hbar, M=M, v_r=v_r, laserBeams=laserBeams3, omega_r=omega_r, tmin=tmin, g=g, delta=delta3, alpha=alpha, k=k, phase=phase, Delta=3e9, beamtype=beamtype_bragg, rho=rho, tau=tau_bragg)
    jj = 0
    sol_range = np.arange(Natoms_bragg).reshape((int(Natoms_bragg/chunksize_bragg), chunksize_bragg))
    progress = progressBar()
    for jj in range(int(Natoms_bragg/chunksize_bragg)):
        with pathos.pools.ProcessPool(nodes=chunksize_bragg) as pool:
            arg_list = [(sols_i3_bragg, time_span3_bragg, progress_bar, kwargs3_bragg, a) for a in sol_range[jj, :]]
            partial_function = partial(Bragg_test_chirp.evolve_motion1)
            sols_chirp += pool.map(partial_function, arg_list)
        progress.update((jj + 1) / int(Natoms_bragg / chunksize_bragg))
    total_N_bragg = np.sum([item_raman.N for item_raman in sols_chirp], axis=0)/Natoms_bragg
    rho_bragg_chirp[j] = (np.real(total_N_bragg[12, -1]) + 0.48)
    j = j + 1  
# Plotting module: Scanned interference fringes
fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), dpi=300)
ax.plot(np.linspace(alpha1, alpha2, 40), rho_bragg_chirp, linewidth=2.5, label='$\\rho_{F=2}$')
ax.legend(fontsize=7)
ax.set_xlabel('$chirp $')
ax.set_ylabel('$rho$')

# %%
import pandas as pd
x_data1 = np.linspace(0, 4*np.pi, 40)
y_data1 = rho_bragg_raman
data1 = {
    'X': x_data1,
    'Y': y_data1
}
df1 = pd.DataFrame(data1)
df1.to_excel('T=50ms Bragg phase fringes 621.xlsx', index=False)
# %%
x_data2 = np.linspace(alpha1, alpha2, 40)
y_data2 = rho_bragg_chirp
data2 = {
    'X': x_data2,
    'Y': y_data2
}
df2 = pd.DataFrame(data2)
df2.to_excel('T=66ms Three-pulse alpha fringes 621-9.7 degrees.xlsx', index=False)

# %%
#%%
sols_filtered_bragg_F = [sol for sol in sols3_bragg[0:Natoms_bragg] if sol.t.shape[0] == 1001 and sol.F.shape[1] == 1001
                         and sol.r.shape[1] == 1001 and sol.r.shape[1] == 1001]
sols_filtered_bragg_fmag = [sol for sol in sols3_bragg[0:Natoms_bragg] if sol.t.shape[0] == 1001 and sol.fmag.shape[1] == 1001
                            and sol.r.shape[1] == 1001 and sol.r.shape[1] == 1001]
directions = ['x', 'y', 'z']
fig, axs = plt.subplots(3, 2, figsize=(9, 12))
i = 1
for j in range(3):
    intensity_values = []
    for k in range(1001):
        intensity = np.linalg.norm(sols_filtered_bragg_F[i].F[j, k]) #* 3.0e8 / 2 * (cts.hbar * k * gamma)
        intensity_values.append(intensity)
    t0 = 1 / gamma
    t_subset = sols_filtered_bragg_F[i].t * t0
    axs[j, 0].plot(t_subset, intensity_values, linewidth=2)  # Change the line width to 2
    direction = directions[j % 3]  # Take the remainder to ensure the correctness of the direction
    axs[j, 0].set_title('Light field force, Direction: {}'.format(direction))  # Set a detailed title
    axs[j, 0].set_xlabel('Time (s)')
    axs[j, 0].set_ylabel('Forces')
for j in range(3):
    intensity_values = []
    for k in range(1001):
        intensity = np.linalg.norm(sols_filtered_bragg_fmag[i].fmag[j, k]) #* 3.0e8 / 2 * (cts.hbar * k * gamma)
        intensity_values.append(intensity)
    t0 = 1 / gamma
    t_subset = sols_filtered_bragg_fmag[i].t * t0
    axs[j, 1].plot(t_subset, intensity_values, linewidth=2)  # Change the line width to 2
    direction = directions[j % 3]  # Take the remainder to ensure the correctness of the direction
    axs[j, 1].set_title('Magnetic force, Direction: {}'.format(direction))  # Set a detailed title
    axs[j, 1].set_xlabel('Time (s)')
    axs[j, 1].set_ylabel('Forces')
# Automatically adjust the subplot layout
plt.tight_layout()
# Add a general title
fig.suptitle('Intervention phase atoms subjected to light field forces', fontsize=16)
# Ensure that the general title does not overlap with the subplots
plt.subplots_adjust(top=0.9)
# Display the figure
plt.show()
# %%