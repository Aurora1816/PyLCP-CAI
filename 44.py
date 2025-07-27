
#%%
import numpy as np
import pylcp
import matplotlib.pyplot as plt
import scipy.constants as cts
from pylcp.common import progressBar
from pathos.pools import ProcessPool
from functools import partial
import pathos
import h5py
from pylcp.integration_tools import RandomOdeResult
from ramanclass import Raman_module
from gravity import emovtion_atom
from gravity import Process_atom
from scipy.spatial.transform import Rotation
import pandas as pd
#%% Use partial to check if parallel operations can be performed smoothly
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
    plot_results(results_square, results_cube) 
#%%
class generate_iniatom():
    def __init__(self, natoms, T, kb, x0, int_mass):
        self.kb = kb
        self.natoms = int(natoms)
        self.int_mass = int_mass
        v0 = np.sqrt(3 * kb * T / int_mass)
        scale_factor = v0 / np.sqrt(3) * 0.1  # Used to scale components so that the sum of squares equals v0
        vx1 =  1 * scale_factor
        vy1 = 1 * scale_factor
        vz1 =  1 * scale_factor
        vx = vx1 + np.random.randn(self.natoms) * 0.02 #* v0
        vy = vy1 * 100 + np.random.randn(self.natoms) * 0.02
        vz = vz1 + np.random.randn(self.natoms) * 0.02 #* v0
        self.v = np.array([vx, vy, vz]).T  # Transpose to ensure the shape is (natoms, 3)
        r0 = np.array([0.0, 0.0, 0.0]) / 100 / x0
        rx = r0[0] + np.random.randn(self.natoms) * 0.005 
        ry = r0[1] + np.random.randn(self.natoms) * 0.005 
        rz = r0[2] + np.random.randn(self.natoms) * 0.005 
        self.r = np.array([rx, ry, rz]).T  # Transpose to ensure the shape is (natoms, 3)
        probabilities = np.zeros(8)
        N1 = []
        rho1 = []
        def generate_one_hot_vector():
            probabilities = np.zeros(8)
            probabilities[0:3] = 0.02
            probabilities[4] = 0.3
            probabilities[5:7] = 0.8
            probabilities[7] = 0.1 
            if len(probabilities) != 8:
                raise ValueError("The length of the probability list should be 8.")
            rand_num = np.random.rand()
            selected_position = np.argmax(np.cumsum(probabilities) > rand_num)
            rho0 = np.zeros(8)
            rho0[selected_position] = 1
            return rho0
        for _ in range(self.natoms):
            rhoi = generate_one_hot_vector()
            rand_num = np.random.rand()
            selected_position = np.argmax(np.cumsum(probabilities) > rand_num)
            N0 = np.zeros(8)
            N0[selected_position] = 1
            rho0 = np.diag(rhoi)
            N1.append(N0)
            rho1.append(rho0)
        self.N = np.array(N1) 
        self.rho = np.array(rho1) 
# Input parameters (initial)
# Variables
atom = pylcp.atom("87Rb")               # Atom type: 87Rb 

# Fixed values
k = 2 * np.pi / 780E-7                      # Wave vector, unit: cm^{-1} 
x0 = 1 / k                                # Length unit conversion factor, converted unit: cm
gamma = atom.state[2].gammaHz             # Atomic natural linewidth, unit: Hz 
t0 = 1 / gamma                            # Time unit conversion factor, converted unit: s
kb = 1.3806503E-23
ini_mass = 86.9 * cts.value('atomic mass constant')

# Variables
natoms = 480
T = 30                                 # Atomic temperature 
# Assignment
Initial_data = generate_iniatom(natoms, T, kb, x0, ini_mass)
sols_r = Initial_data.r + np.array([0, 0.01, 0])
sols_v0 = Initial_data.v * 0.005
sols_N = Initial_data.N
sols_rho = Initial_data.rho
sols_t = np.zeros(Initial_data.natoms)
target_std = np.array([0.01124538, 0.01204617, 0.01154608])
target_mean = np.array([0.07315841, 3.92, 0.09146757])
current_std = np.std(sols_v0, axis=0)
current_mean = np.mean(sols_v0, axis=0)
mean_adjusted_sols_v = sols_v0 - current_mean + target_mean
mean_adjusted_std = np.std(mean_adjusted_sols_v, axis=0)
adjustment_factors = target_std / mean_adjusted_std
std_adjusted_sols_v = mean_adjusted_sols_v * adjustment_factors
sols_v = std_adjusted_sols_v - np.mean(std_adjusted_sols_v, axis=0) + target_mean
#%%
# Create an inti_solseqn.h5 file to store the read sols0 (2D) data
t_list = sols_t
r_list = sols_r
v_list = sols_v
rho_list = sols_rho
with h5py.File('inti_solseqn.h5', 'w') as f:
    for i, (t, r, v, rho) in enumerate(zip(t_list, r_list, v_list, rho_list)):
        group = f.create_group(f'sol_{i}')
        group.create_dataset('t', data=t)
        group.create_dataset('r', data=r)
        group.create_dataset('v', data=v)
        group.create_dataset('rho', data=rho)
#%%
# Read the sols0 data
sols4 = []
with h5py.File('inti_solseqn.h5', 'r') as f:
    for key in f.keys():
        group = f[key]
        t = np.array(group['t'])
        r = np.array(group['r'])
        v = np.array(group['v'])
        rho = np.array(group['rho'])       
        sol = RandomOdeResult(t=t, r=r, v=v, rho=rho)
        sols4.append(sol)


#%% First Raman interference parameters
# Input parameters (variable)
atom_rm4 = pylcp.atom("87Rb")               # Atom type: 87Rb
I_sat_rm4 = 1.6                            # Saturation light intensity, unit: mw/cm^2   
s_rm4 = 30 / I_sat_rm4                      # Light intensity, unit: mw/cm^2, value range: [0, 200]
det_rm4 = 400                        # Detuning, unit: Hz, value range: [0, 1000] 
phi_i_rm4 = np.pi                          # Beam polarization, value range: [0, 2*np.pi]
wb_rm4 = 2.4e-2                            # Spot size, beam diameter, unit: cm, value range: [0, 10]
thate_rm4 = 1.7                            # Beam ratio, unit: °, value range: [0, 360]
kvec_rm4 = np.array([0., 1., 0.])          # Wave vector direction    
mag_rm4 = np.array([1e-9, 9e-5, 1e-9])       # Magnetic field magnitude
alpha_rm4 = -1652781.8126853162#-4201898.491871586             # Chirp 
rotation_spec_rm4 = 'XYZ'                  # Rotation axis definition 
rotation_angles_rm4 = [0., 0., 1.50447382]# Rotation angles, defined in radians#1.4014993894
phase_rm1 = 0                          # Phase, unit: radians, value range: [0, 2*np.pi] 
t1_rm1 = 4.5e-6               # First Raman beam interaction time, unit: s 
w_scan_1 = 847445.9333279819   # First Raman beam interaction frequency, unit: Hz1389809.815825678
sols_i_rm4t1 = sols4                          # Substitute the initial parameters of sols4
g_rm4 = -np.array([0., 9.793386882176355, 0])  # Gravitational acceleration, unit: m/s^2. Here we take the Y-axis as the gravity axis
latitude = 30 # Latitude of the instrument's location
alpha_deg = 30.0  # Rotation angle around the x-axis, (unit: degrees)
beta_deg = 45.0  # Rotation angle around the y-axis, (unit: degrees)
gamma_deg = 60.0  # Rotation angle around the z-axis, (unit: degrees)
direction = 'z'# Earth's angular velocity vector in the original coordinate system (only has a component in the z-direction)
Natoms_rm4 = 48 # Number of atoms
chunksize_rm4 = 16                          # Number of cores 
# Built-in parameters (unchanged)
k_rm4 = 2 * np.pi / 780E-7                      # Wave vector, unit: cm^{-1}
x0_rm4 = 1 / k_rm4                           # Length unit conversion factor, converted unit: cm
gamma_rm4 = atom_rm4.state[2].gammaHz      # Atomic natural linewidth, unit: Hz
omega_z = (2 * np.pi / 86164.091) * np.cos(np.radians(latitude))
alpha_rad = np.deg2rad(alpha_deg)# Convert the angle from degrees to radians
beta_rad = np.deg2rad(beta_deg)
gamma_rad = np.deg2rad(gamma_deg)
if direction == 'x':
    omega = np.array([omega_z, 0, 0])
elif direction == 'y':
    omega = np.array([0, omega_z, 0])
elif direction == 'z':
    omega = np.array([0, 0, omega_z])
else:
    raise ValueError("Invalid direction. Please enter 'x', 'y', or 'z'.")
# Define the rotation matrix R_x(alpha) around the x-axis
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(alpha_rad), -np.sin(alpha_rad)],
    [0, np.sin(alpha_rad), np.cos(alpha_rad)]
])
# Define the rotation matrix R_y(beta) around the y-axis
R_y = np.array([
    [np.cos(beta_rad), 0, np.sin(beta_rad)],
    [0, 1, 0],
    [-np.sin(beta_rad), 0, np.cos(beta_rad)]
])
# Define the rotation matrix R_z(gamma) around the z-axis
R_z = np.array([
    [np.cos(gamma_rad), -np.sin(gamma_rad), 0],
    [np.sin(gamma_rad), np.cos(gamma_rad), 0],
    [0, 0, 1]
])
# Combine the rotation matrices, assuming the order is first around the z-axis, then the y-axis, and finally the x-axis
R_combined = np.dot(R_x, np.dot(R_y, R_z))
# Transform the angular velocity vector omega to the new coordinate system
omega_rotated_rm4 = np.dot(R_combined, omega)

rot_mat_rm4 = Rotation.from_euler(rotation_spec_rm4, rotation_angles_rm4).as_matrix()# Beam rotation setting parameters
rho0_rm = np.array([0, 1, 0, 0, 0, 0, 0, 0])     # Substitute the initial atomic state
sol_i_rm1 = sols4                        # Substitute the initial parameters of sols4, the atomic state at the start of Raman
sols5 = []                              # Save the solutions of the first pulse, sols5
po_1 = np.mean(np.vstack([item.r for item in sols4]), axis=0)# The action position of the first pulse
t0_rm1 = 0                     # The start time of the first Raman beam interaction, unit: s 
Raman_test_rm1 = Raman_module(rot_mat=rot_mat_rm4, po=po_1, omega=omega_rotated_rm4, atom=atom_rm4, k=k_rm4, phi_i=phi_i_rm4, wb=wb_rm4, s=s_rm4, phase=phase_rm1, thate=thate_rm4,
                        g=g_rm4, gamma=gamma_rm4, alpha=alpha_rm4, kvec1=kvec_rm4, mag=mag_rm4, include_mag_forces=True, det=det_rm4)#w_scan=w_scan_1, rot_mat=rot_mat_rm4,
g_0 = np.dot(np.abs(Raman_test_rm1.obe.laserBeams['g->e'].kvec()[0]), Raman_test_rm1.g)

last_columns = [item.v for item in sols4]
stacked_columns = np.vstack(last_columns)
average_values = np.mean(stacked_columns, axis=0)
delta_v = np.dot(np.abs(Raman_test_rm1.obe.laserBeams['g->e'].kvec()[0]), average_values  )
w_scan = delta_v / 2 / np.pi
print(w_scan, g_0 / 2 / np.pi)

#%% First Raman interference run
if __name__ == '__main__': 
    sol_range = np.arange(Natoms_rm4).reshape((int(Natoms_rm4 / chunksize_rm4), chunksize_rm4))
    progress = progressBar()
    for jj in range(int(Natoms_rm4 / chunksize_rm4)):
        with pathos.pools.ProcessPool(nodes=chunksize_rm4) as pool:
            arg_list = [(t0_rm1, t1_rm1, Raman_test_rm1, w_scan_1, sol_i_rm1, idx) for idx in sol_range[jj, :]]
            partial_function = partial(Raman_test_rm1.Scan_Raman_solutiontest)
            sols5 += pool.map(partial_function, arg_list)
        progress.update((jj + 1) / int(Natoms_rm4 / chunksize_rm4))
# Shape annotation:
# sols5[natoms].r.shape = (3, 1001)
# sols5[natoms].v.shape = (3, 1001)
# sols5[natoms].rho.shape = (8, 8, 1001)
# Create a sols0.h5 file to store the read sols0 (2D) data
#%%
soli = sols5
total_rho_raman = np.sum([item_raman.rho for item_raman in soli], axis=0) / Natoms_rm4
F2_rho_raman1 = np.ones(len(soli[0].t))
F1_rho_raman1 = np.ones(len(soli[0].t))
for j in range(len(soli[0].t)):
    F2_rho_raman1[j] = np.abs(np.sum(np.diag(total_rho_raman[:, :, j])[3:8])) 
    F1_rho_raman1[j] = np.abs(np.sum(np.diag(total_rho_raman[:, :, j])[0:3]))
fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=300)
ax.plot(soli[0].t, F2_rho_raman1, linewidth=1.5, label='$\\rho_{F=2}$')
ax.plot(soli[0].t, F1_rho_raman1, linewidth=1.5, label='$\\rho_{F=1}$')
ax.legend(fontsize=7)
ax.set_xlabel('$Time (s)$')
ax.set_ylabel('$\\rho_{ii}$')
#%% Plotting module
# Velocity and position
fig, ax = plt.subplots(3, 2, figsize=(15, 10))
for ii in range(3):
    for idx, soli in enumerate(sols5): 
        ax[ii, 0].plot(soli.t * t0, soli.v[ii] * (atom.state[2].gammaHz / k / 100), linewidth=1)
        ax[ii, 1].plot(soli.t * t0, soli.r[ii] * x0 / 100 , linewidth=1)
  
    ax[ii, 0].set_ylabel(f'$v_{{{"xyz"[ii]}}}$ (m/s)', fontsize=15)
    ax[ii, 1].set_ylabel(f'$r_{{{"xyz"[ii]}}}$ (m)', fontsize=15)  
    ax[ii, 0].tick_params(axis='both', labelsize=15)  # Adjust the font size of the tick labels
    ax[ii, 1].tick_params(axis='both', labelsize=15)  # Adjust the font size of the tick labels
for ax_i in ax[-1, :]:
    ax_i.set_xlabel('Time (s)', fontsize=15)
    ax_i.tick_params(axis='both', labelsize=15)  # Adjust the font size of the tick labels
fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.22)
plt.show()
#%%


# First Raman spectrum scan, used for preliminary testing to obtain the first Raman pulse frequency (can be deleted)
raman1_num = 20
F1_rho_raman1 = np.zeros(raman1_num) 
F2_rho_raman1 = np.zeros(raman1_num)
j = 0  
w_scan1 = 35436.185442632  
w_scan2 = 1135436.185442632  
for w_scan in np.linspace(w_scan1, w_scan2, raman1_num):
    sols_raman_raman = []
    jj = 0
    sol_range = np.arange(Natoms_rm4).reshape((int(Natoms_rm4 / chunksize_rm4), chunksize_rm4))
    progress = progressBar()
    for jj in range(int(Natoms_rm4 / chunksize_rm4)):
        with pathos.pools.ProcessPool(nodes=chunksize_rm4) as pool:
            arg_list = [(rho0_rm, t0_rm1, t1_rm1, Raman_test_rm1, w_scan, sol_i_rm1, idx) for idx in sol_range[jj, :]]
            partial_function = partial(Raman_test_rm1.Scan_Raman_solution)
            sols_raman_raman += pool.map(partial_function, arg_list)
        progress.update((jj + 1) / int(Natoms_rm4 / chunksize_rm4))
    total_rho_raman = np.sum([item_raman.rho for item_raman in sols_raman_raman], axis=0) / Natoms_rm4
    F2_rho_raman1[j] = np.abs(np.sum(np.diag(total_rho_raman[:, :, -1])[0:5])) 
    F1_rho_raman1[j] = np.abs(np.sum(np.diag(total_rho_raman[:, :, -1])[5:8]))
    j = j + 1
# Shape annotation:
# total_rho_raman.shape = (8, 8, 1001)
# First Raman spectrum scan, used for preliminary testing to obtain the first Raman pulse frequency (can be deleted), plotting module
fig, ax = plt.subplots(1, 1, figsize=(12, 4.1), dpi=300)
ax.plot(np.linspace(w_scan1, w_scan2, raman1_num) / 1e6, F1_rho_raman1, linewidth=1.5, label='$\\rho_{F=1}$')
ax.plot(np.linspace(w_scan1, w_scan2, raman1_num) / 1e6, F2_rho_raman1, linewidth=1.5, label='$\\rho_{F=2}$')
ax.legend(fontsize=7)
ax.set_xlabel('$Frequency (MHz)$')
ax.set_xticks(np.linspace(w_scan1, w_scan2, 10) / 1e6)
ax.set_ylabel('$\\rho_{ii}$')

#%% Second motion evolution
# Built-in parameters (unchanged)
sols_i_rm4t2 = sols5                         # Substitute the initial parameters of sols5
solsg2 = []                                 # Save the solutions of the first motion evolution, sols_g1 
# Input parameters (variable)
tmax_rm4t2 = 0.2                           # First motion evolution time, unit: S                       
step_rm4t2 = 4                       # Time interval used to determine the step size, unit: S
#%%
if __name__ == '__main__': 
    atom_instance = emovtion_atom(g_rm4, omega_rotated_rm4)
    progress = progressBar()
    sol_range = np.arange(Natoms_rm4).reshape((int(Natoms_rm4 / chunksize_rm4), chunksize_rm4))
    for jj in range(int(Natoms_rm4 / chunksize_rm4)):
        with pathos.pools.ProcessPool(nodes=chunksize_rm4) as pool:
            arg_list = [(sols_i_rm4t2, g_rm4, omega_rotated_rm4, tmax_rm4t2, step_rm4t2, idx) for idx in sol_range[jj, :]]
            partial_function =  partial(Process_atom.process_atom)
            solsg2 += pool.map(partial_function, arg_list)
        progress.update((jj + 1) / int(Natoms_rm4 / chunksize_rm4))
# solsg1[natoms].r.shape = (3, 3)
# solsg1[natoms].v.shape = (3, 3)
# solsg1[natoms].rho.shape = (8, 8, 3)
# Create a sols0.h5 file to store the read sols0 (2D) data
#%% Second Raman interference run
# Built-in parameters (unchanged)
sol_i_rm2 = solsg2                      # Substitute the initial parameters of solsg2, the atomic state at the start of Raman
sols6 = []                              # Save the solutions of the second pulse, sols6
w_scan_2 = w_scan_1 + alpha_rm4 * (tmax_rm4t2 + t1_rm1)
t0_rm2 = 0 
po_2 = np.mean(np.vstack([item.r[:, -1] for item in solsg2]), axis=0)
# Input parameters (variable)
t1_rm2 = 9e-6
phase_rm2 = 0
Raman_test_rm2 = Raman_module(rot_mat=rot_mat_rm4, po=po_2, omega=omega_rotated_rm4, atom=atom_rm4, k=k_rm4, phi_i=phi_i_rm4, wb=wb_rm4, s=s_rm4, phase=phase_rm2, thate=thate_rm4,
                    g=g_rm4, gamma=gamma_rm4, alpha=alpha_rm4, kvec1=kvec_rm4, mag=mag_rm4, include_mag_forces=True, det=det_rm4) 
#%%
if __name__ == '__main__': 
    sol_range = np.arange(Natoms_rm4).reshape((int(Natoms_rm4 / chunksize_rm4), chunksize_rm4))
    progress = progressBar()
    for jj in range(int(Natoms_rm4 / chunksize_rm4)):
        with pathos.pools.ProcessPool(nodes=chunksize_rm4) as pool:
            arg_list = [(t0_rm2, t1_rm2, Raman_test_rm2, w_scan_2, sol_i_rm2, idx) for idx in sol_range[jj, :]]
            partial_function = partial(Raman_test_rm2.Scan_Raman_solution)
            sols6 += pool.map(partial_function, arg_list)
        progress.update((jj + 1) / int(Natoms_rm4 / chunksize_rm4))
last_columns = [item.v[:, -1] for item in solsg2]
stacked_columns = np.vstack(last_columns)
average_values = np.mean(stacked_columns, axis=0)
delta_v = np.dot(np.abs(Raman_test_rm2.obe.laserBeams['g->e'].kvec()[0]), average_values  )
w_scan = delta_v / 2 / np.pi
print(w_scan)
# Shape annotation:
# sols6[natoms].r.shape = (3, 1001)
# sols6[natoms].v.shape = (3, 1001)
# sols6[natoms].rho.shape = (8, 8, 1001)
#%%
soli = sols6
total_rho_raman = np.sum([item_raman.rho for item_raman in soli], axis=0) / Natoms_rm4
F2_rho_raman1 = np.ones(len(soli[0].t))
F1_rho_raman1 = np.ones(len(soli[0].t))
for j in range(len(soli[0].t)):
    F2_rho_raman1[j] = np.abs(np.sum(np.diag(total_rho_raman[:, :, j])[3:8])) 
    F1_rho_raman1[j] = np.abs(np.sum(np.diag(total_rho_raman[:, :, j])[0:3]))
fig, ax = plt.subplots(1, 1, figsize=(12, 4.1), dpi=300)
ax.plot(soli[0].t, F2_rho_raman1, linewidth=1.5, label='$\\rho_{F=2}$')
ax.plot(soli[0].t, F1_rho_raman1, linewidth=1.5, label='$\\rho_{F=1}$')
ax.legend(fontsize=7)
ax.set_xlabel('$Time (s)$')
ax.set_ylabel('$\\rho_{ii}$')

#%% Third motion evolution
# Built-in parameters (unchanged)
sols_i_rm4t3 = sols6                         # Substitute the initial parameters of sols5
solsg3 = []                                 # Save the solutions of the first motion evolution, sols_g1 
# Input parameters (variable)
tmax_rm4t3 = 0.40                           # Second motion evolution time, unit: S                       
step_rm4t3 = 4                       # Time interval used to determine the step size, unit: S
#%%
if __name__ == '__main__': 
    atom_instance = emovtion_atom(g_rm4, omega_rotated_rm4)
    progress = progressBar()
    sol_range = np.arange(Natoms_rm4).reshape((int(Natoms_rm4 / chunksize_rm4), chunksize_rm4))
    for jj in range(int(Natoms_rm4 / chunksize_rm4)):
        with pathos.pools.ProcessPool(nodes=chunksize_rm4) as pool:
            arg_list = [(sols_i_rm4t3, g_rm4, omega_rotated_rm4, tmax_rm4t3, step_rm4t3, idx) for idx in sol_range[jj, :]]
            partial_function =  partial(Process_atom.process_atom)
            solsg3 += pool.map(partial_function, arg_list)
        progress.update((jj + 1) / int(Natoms_rm4 / chunksize_rm4))
# solsg1[natoms].r.shape = (3, 3)
# solsg1[natoms].v.shape = (3, 3)
# solsg1[natoms].rho.shape = (8, 8, 3)
# Shape annotation:
# solsg2[natoms].r.shape = (3, 3)
# solsg2[natoms].v.shape = (3, 3)
# solsg2[natoms].rho.shape = (8, 8, 3)
# Create a sols0.h5 file to store the read sols0 (2D) data
#%% Third Raman interference run
# Built-in parameters (unchanged)
sol_i_rm3 = solsg3                      # Substitute the initial parameters of solsg3, the atomic state at the start of Raman
sols7 = []                              # Save the solutions of the third pulse, sols7
w_scan_3 = w_scan_2 + alpha_rm4 * (tmax_rm4t3 + t1_rm2)
po_3 = np.mean(np.vstack([item.r[:, -1] for item in solsg3]), axis=0)
t0_rm3 = 0
# Input parameters (variable)
t1_rm3 = 9e-6
phase_rm3 = 0
Raman_test_rm3 = Raman_module(rot_mat=rot_mat_rm4, po=po_3, omega=omega_rotated_rm4, atom=atom_rm4, k=k_rm4, phi_i=phi_i_rm4, wb=wb_rm4, s=s_rm4, phase=phase_rm3, thate=thate_rm4,
                    g=g_rm4, gamma=gamma_rm4, alpha=alpha_rm4, kvec1=kvec_rm4, mag=mag_rm4, include_mag_forces=True, det=det_rm4)  #rot_mat==rot_mat_rm4,
#%%
if __name__ == '__main__': 
    sol_range = np.arange(Natoms_rm4).reshape((int(Natoms_rm4 / chunksize_rm4), chunksize_rm4))
    progress = progressBar()
    for jj in range(int(Natoms_rm4 / chunksize_rm4)):
        with pathos.pools.ProcessPool(nodes=chunksize_rm4) as pool:
            arg_list = [(t0_rm3, t1_rm3, Raman_test_rm3, w_scan_3, sol_i_rm3, idx) for idx in sol_range[jj, :]]
            partial_function = partial(Raman_test_rm3.Scan_Raman_solution)
            sols7 += pool.map(partial_function, arg_list)
        progress.update((jj + 1) / int(Natoms_rm4 / chunksize_rm4))
last_columns = [item.v[:, -1] for item in solsg3]
stacked_columns = np.vstack(last_columns)
average_values = np.mean(stacked_columns, axis=0)
delta_v = np.dot(np.abs(Raman_test_rm2.obe.laserBeams['g->e'].kvec()[0]), average_values  )
w_scan = delta_v / 2 / np.pi
print(w_scan)

# Shape annotation:
# sols7[natoms].r.shape = (3, 1001)
# sols7[natoms].v.shape = (3, 1001)
# sols7[natoms].rho.shape = (8, 8, 1001)
# Create a sols0.h5 file to store the read sols0 (2D) data
#%%
soli = sols7
total_rho_raman = np.sum([item_raman.rho for item_raman in soli], axis=0) / Natoms_rm4
F2_rho_raman1 = np.ones(len(soli[0].t))
F1_rho_raman1 = np.ones(len(soli[0].t))
for j in range(len(soli[0].t)):
    F2_rho_raman1[j] = np.abs(np.sum(np.diag(total_rho_raman[:, :, j])[0:5])) 
    
    F1_rho_raman1[j] = np.abs(np.sum(np.diag(total_rho_raman[:, :, j])[5:8]))
fig, ax = plt.subplots(1, 1, figsize=(12, 4.1), dpi=300)
ax.plot(soli[0].t, F2_rho_raman1, linewidth=1.5, label='$\\rho_{F=2}$')
ax.plot(soli[0].t, F1_rho_raman1, linewidth=1.5, label='$\\rho_{F=1}$')
ax.legend(fontsize=7)
ax.set_xlabel('$Time (s)$')
ax.set_ylabel('$\\rho_{ii}$')
#%% Fourth motion evolution
# Built-in parameters (unchanged)
sols_i_rm4t4 = sols7                         # Substitute the initial parameters of sols5
solsg4 = []                                 # Save the solutions of the first motion evolution, sols_g1
# Input parameters (variable)
tmax_rm4t4 = 0.2                            # Second motion evolution time, unit: S                       
step_rm4t4 = 4                       # Time interval used to determine the step size, unit: S
#%%
if __name__ == '__main__': 
    atom_instance = emovtion_atom(g_rm4, omega_rotated_rm4)
    progress = progressBar()
    sol_range = np.arange(Natoms_rm4).reshape((int(Natoms_rm4 / chunksize_rm4), chunksize_rm4))
    for jj in range(int(Natoms_rm4 / chunksize_rm4)):
        with pathos.pools.ProcessPool(nodes=chunksize_rm4) as pool:
            arg_list = [(sols_i_rm4t4, g_rm4, omega_rotated_rm4, tmax_rm4t4, step_rm4t3, idx) for idx in sol_range[jj, :]]
            partial_function =  partial(Process_atom.process_atom)
            solsg4 += pool.map(partial_function, arg_list)
        progress.update((jj + 1) / int(Natoms_rm4 / chunksize_rm4))
# solsg1[natoms].r.shape = (3, 3)
# solsg1[natoms].v.shape = (3, 3)
# solsg1[natoms].rho.shape = (8, 8, 3)
# Shape annotation:
# solsg2[natoms].r.shape = (3, 3)
# solsg2[natoms].v.shape = (3, 3)
# solsg2[natoms].rho.shape = (8, 8, 3)

#%% Fourth Raman interference run
# Built-in parameters (unchanged)
sol_i_rm4 = solsg4                      # Substitute the initial parameters of solsg3, the atomic state at the start of Raman
sols8 = []                              # Save the solutions of the third pulse, sols7
w_scan_4 = w_scan_3 + alpha_rm4 * (tmax_rm4t4 + t1_rm3)#9.18e5#327417.7027468643#w_scan_2 + alpha_rm4 * (tmax_rm4t3 + t1_rm2 - t0_rm2) * 0.1736 - 2#2752604.754200346 + 3.15e-5#2.909e6#w_scan_2 + alpha_rm4 * (tmax_rm4t3 + t1_rm2 - t0_rm2)
po_4 = np.mean(np.vstack([item.r[:, -1] for item in solsg4]), axis=0)
t0_rm4 = 0
# Input parameters (variable)
t1_rm4 = 4.5e-6
phase_rm4 = 0
Raman_test_rm4 = Raman_module(rot_mat=rot_mat_rm4, po=po_4, omega= omega_rotated_rm4, atom=atom_rm4, k=k_rm4, phi_i=phi_i_rm4, wb=wb_rm4, s=s_rm4, phase=phase_rm4, thate=thate_rm4,
                    g=g_rm4, gamma=gamma_rm4, alpha=alpha_rm4, kvec1=kvec_rm4, mag=mag_rm4, include_mag_forces=True, det=det_rm4)  #rot_mat==rot_mat_rm4,

#%%
if __name__ == '__main__': 
    sol_range = np.arange(Natoms_rm4).reshape((int(Natoms_rm4 / chunksize_rm4), chunksize_rm4))
    progress = progressBar()
    for jj in range(int(Natoms_rm4 / chunksize_rm4)):
        with pathos.pools.ProcessPool(nodes=chunksize_rm4) as pool:
            arg_list = [(t0_rm4, t1_rm4, Raman_test_rm4, w_scan_4, sol_i_rm4, idx) for idx in sol_range[jj, :]]
            partial_function = partial(Raman_test_rm4.last_Scan_Raman_solution)
            sols8 += pool.map(partial_function, arg_list)
        progress.update((jj + 1) / int(Natoms_rm4 / chunksize_rm4))
# Shape annotation:
# sols7[natoms].r.shape = (3, 1001)
# sols7[natoms].v.shape = (3, 1001)
# sols7[natoms].rho.shape = (8, 8, 1001)

#%%
soli = sols7
total_rho_raman = np.sum([item_raman.rho for item_raman in soli], axis=0) / Natoms_rm4
F2_rho_raman1 = np.ones(len(soli[0].t))
F1_rho_raman1 = np.ones(len(soli[0].t))
for j in range(len(soli[0].t)):
    F2_rho_raman1[j] = np.abs(np.sum(np.diag(total_rho_raman[:, :, j])[3:8])) 
    F1_rho_raman1[j] = np.abs(np.sum(np.diag(total_rho_raman[:, :, j])[0:3]))
fig, ax = plt.subplots(1, 1, figsize=(12, 4.1), dpi=300)
ax.plot(soli[0].t, F2_rho_raman1, linewidth=1.5, label='$\\rho_{F=1}$')
ax.plot(soli[0].t, F1_rho_raman1, linewidth=1.5, label='$\\rho_{F=2}$')
ax.legend(fontsize=7)
ax.set_xlabel('$Time (s)$')
ax.set_ylabel('$\\rho_{ii}$')
# %% First fringe scan  
l = 0
phase1 = 1.75 * np.pi
phase2 = 1.875 * np.pi
phase_num = 2
F2_rho = np.zeros(phase_num)
F1_rho = np.zeros(phase_num)
phasei = 0
for phase in np.linspace(phase1, phase2, phase_num):
    
    phasei = phase + g_0 * ((tmax_rm4t4) ** 2) - alpha_rm4 * 2 * np.pi * ((tmax_rm4t4) ** 2) - 4 * np.dot(1.6e7 * Raman_test_rm1.obe.laserBeams['g->e'].kvec()[0], (np.cross(omega_rotated_rm4, sol_i_rm3[0].v[:, -1])) * tmax_rm4t3 * tmax_rm4t3)
    Raman_test_all = Raman_module(rot_mat=rot_mat_rm4, po=po_4, omega=omega_rotated_rm4, atom=atom_rm4, k=k_rm4, phi_i=phi_i_rm4, wb=wb_rm4, s=s_rm4, phase=phasei, thate=thate_rm4,
                    g=g_rm4, gamma=gamma_rm4, alpha=alpha_rm4, kvec1=kvec_rm4, mag=mag_rm4, include_mag_forces=True, det=det_rm4) #
    sols_raman_raman = []
    jj = 0
    sol_range = np.arange(Natoms_rm4).reshape((int(Natoms_rm4 / chunksize_rm4), chunksize_rm4))
    progress = progressBar()
    for jj in range(int(Natoms_rm4 / chunksize_rm4)):
        with pathos.pools.ProcessPool(nodes=chunksize_rm4) as pool:
            arg_list = [(t0_rm4, t1_rm4, Raman_test_all, w_scan_4, sol_i_rm4, idx) for idx in sol_range[jj, :]]
            partial_function = partial(Raman_test_all.last_Scan_Raman_solution)
            sols_raman_raman += pool.map(partial_function, arg_list)
        progress.update((jj + 1) / int(Natoms_rm4 / chunksize_rm4))
    total_rho_raman = np.sum([item_raman.rho for item_raman in sols_raman_raman], axis=0)
    F2_rho[l] = np.abs(np.sum(np.diag(total_rho_raman[:, :, -1])[3:8]))
    F1_rho[l] = np.abs(np.sum(np.diag(total_rho_raman[:, :, -1])[0:3]))
    l = l + 1
#%%
# Plotting module for the scanned interference fringes
fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), dpi=300)
ax.plot(np.linspace(0, 4 * np.pi, phase_num), F2_rho / (F2_rho + F1_rho), linewidth=2.5, label='$\\rho_{F=2}$')
ax.legend(fontsize=7)
ax.set_xlabel('$phase $')
ax.set_ylabel('$\\rho_{ii}$')  
# %%
x_data1 = np.linspace(phase1, phase2, phase_num)
y_data1 = F2_rho / (F2_rho + F1_rho)
data1 = {
    'X': x_data1,
    'Y': y_data1
}
df1 = pd.DataFrame(data1)
df1.to_excel('Four-pulse phase fringes-8.xlsx', index=False)
# %%
