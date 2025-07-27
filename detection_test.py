#%% This is a test for multiple atoms
import numpy as np
import matplotlib.pyplot as plt
import pylcp
import scipy.constants as cts
from scipy.spatial.transform import Rotation
from detection_module import detection_module
from detection_module import detection_pumping
from detection_module import Blow_module
from detection_module import detection_tof_module
from detection_module import gaussiansignal
from pylcp.common import progressBar
import pathos
from functools import partial
from scipy.integrate import odeint
import math

#%% Parameters
# Input parameters
atom = pylcp.atom("87Rb")           # Type of atom: 87Rb 
I_sat = 1.6                         # Saturation light intensity, unit: mW/cm²
k = 2*np.pi/780E-7                  # Wave vector, unit: cm⁻¹   
x0 = 1/k                            # Length unit conversion factor, converted unit: cm            
gamma = atom.state[2].gammaHz       # Natural linewidth of the atom, unit: Hz 
t0 = 1/gamma                        # Time unit conversion factor, converted unit: s 
mass = 86.9*cts.value('atomic mass constant')*(x0*1e-2)**2/cts.hbar/t0  # Atomic mass: 86.9*cts.value('atomic mass constant')
muB = 1.399                         # Bohr magneton constant, unit: J/T  
# Input parameters
det = -0.5                          # Detuning, unit: Hz, value range: [-10,0]
s = 1                               # Light intensity, unit: mW/cm², value range: [0,10]
wb = 0.5                            # Width of the rectangular light spot, unit: cm, value range: [0,10]
hb = 0.3                            # Height of the rectangular light spot, unit: cm, value range: [0,10]
alpha = np.array([1e-4,1e-4,1e-4])  # Magnetic field parameter 
g = -np.array([0.,  9.8*t0**2/(x0*1e-2),0])     # Gravitational acceleration, unit: m/s². Here we take the Y-axis as the gravity axis

#%% First run of multi-atom detection, velocity initialization
if __name__ == '__main__':
    detection_test = detection_module(s=s, det=det, k=k, gamma=gamma, mass=mass, g=g, wb=wb, hb=hb)
    atom = pylcp.atom("87Rb")           # Type of atom: 87Rb 
    # Built-in parameters
    I_sat = 1.6                         # Saturation light intensity, unit: mW/cm²
    k = 2*np.pi/780E-7                  # Wave vector, unit: cm⁻¹   
    x0 = 1/k                            # Length unit conversion factor, converted unit: cm            
    gamma = atom.state[2].gammaHz       # Natural linewidth of the atom, unit: Hz 
    t0 = 1/gamma                        # Time unit conversion factor, converted unit: s 
    tmax = 0.0108/t0                    # Maximum evolution time, converted unit: s
    detection_test.update_tmax(tmax)    # Update the maximum evolution time in the class         
    roffset = np.array([0,1,0])/x0      # Initial position of the atom cloud center, unit: cm
    vscale = np.array([0.01, 0.01, 0.01])/(gamma*1e-2*x0) # Random scale of the atom cloud velocity, unit: m/s
    voffset = np.array([0,-2, 0.])/(gamma*1e-2*x0)  # Initial velocity of the atom cloud, unit: m/s
    rscale = np.array([0.05, 0.05, 0.05])/x0    # Random scale of the initial position of the atom cloud, unit: cm
    Natoms = 96                         # Number of atoms, value range: [2,N]  
    chunksize = 16                      # Number of cores for computation, value range: [2,N]
    sols_detection = []
    sol_range = np.arange(Natoms).reshape((int(Natoms/chunksize), chunksize))
    progress = progressBar()
    for jj in range(int(Natoms/chunksize)):
        with pathos.pools.ProcessPool(nodes=16) as pool:
            arg_list = [(detection_test.obe, roffset, vscale, voffset, rscale, idx) for idx in sol_range[jj,:]]
            partial_function = partial(detection_test.generate_random_solution)
            sols_detection += pool.map(partial_function, arg_list)
        progress.update((jj+1)/int(Natoms/chunksize))
    # Shape annotation:
    # sols_detection[1].r.shape = (3,5001)
    # sols_detection[1].v.shape = (3, 5001)
    # sols_detection[1].rho.shape = (15, 15, 5001)
    # sols_detection[1].t.shape = (5001,)
    # sols_detection[1].F.shape = (3,5001)

#%% Test for a large number of atoms. Some sol shapes are incorrect and need to be filtered out
sols_filtered = [sol for sol in sols_detection[0:Natoms] if sol.t.shape[0] == 5001 and sol.F.shape[1] == 5001
                 and sol.r.shape[1] == 5001 and sol.r.shape[1] == 5001]
print(len(sols_filtered))

#%%
Natoms = len(sols_filtered)
fig, ax = plt.subplots(3, 2, figsize=(6.25, 4 * 2.75), dpi=300)
i = 0
v = np.zeros((3, Natoms))
r = np.zeros((3, Natoms))
gamma = atom.state[2].gammaHz
alpha = (3/2)*cts.value('Bohr magneton in Hz/T')*1e-4*8*x0/gamma
for sol_all in sols_filtered:
    t, r[:, i], v[:, i] = sol_all.t / (atom.state[2].gammaHz), sol_all.r[:, -1] * 10*x0, sol_all.v[:, -1] * (k * 100) / (
                                      atom.state[2].gammaHz)
    i += 1
    for ii in range(3):
        ax[ii, 0].plot(sol_all.t / (atom.state[2].gammaHz) * 1e3, sol_all.v[ii] * (
                atom.state[2].gammaHz / k / 100), linewidth=0.25)
        ax[ii, 1].plot(sol_all.t / (atom.state[2].gammaHz) * 1e3, sol_all.r[ii] * x0*10, linewidth=0.35)
# The x-axis of the plot is t, unit: ms. The y-axis of the position is mm, and the y-axis of the velocity is m/s
for ax_i in ax[-1, :]:
    ax_i.set_xlabel('$t [ms]$')

for jj in range(2):
    for ax_i in ax[jj, :]:
        ax_i.set_xticklabels('')

for ax_i, lbl in zip(ax[:, 0], ['x', 'y', 'z']):
    ax_i.set_ylabel('$v_' + lbl + 'm/s$')

for ax_i, lbl in zip(ax[:, 1], ['x', 'y', 'z']):
    ax_i.set_ylabel('$\\alpha ' + lbl + '$/mm')

fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.32)

plt.show()

#%% Distribution of atomic forces. This is mainly to observe whether the atoms evolve successfully. The y-axis is in logarithmic scale, which has little impact on the results
import numpy as np
import matplotlib.pyplot as plt

# Create a 4x4 chart
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
# Iterate through the sols list and plot the data of each sol object
for i in range(4):
    for j in range(4):
        intensity_values = []
        for k in range(5001):
            intensity = np.linalg.norm(sols_filtered[i*4 + j].F[:,k]) * 3.0e8 / 2 * (cts.hbar * k * gamma)
            intensity_values.append(intensity)
        t0 = 1 / gamma
        t_subset = sols_filtered[i*4 + j].t * t0
        axs[i, j].plot(t_subset, intensity_values)
        axs[i, j].set_title('Plot {}'.format(i*4 + j + 1))
        axs[i, j].set_xlabel('Time (t)')
        axs[i, j].set_ylabel('Intensity')
        axs[i, j].set_yscale('log')
# Automatically adjust the layout of subplots
plt.tight_layout()
# Display the figure
plt.show()

#%% State distribution, subplots. This is also to see if the atoms evolve normally
import matplotlib.pyplot as plt

# Assume sols_detection is a file containing multiple sols. Here we use sol_list instead
sol_list = sols_filtered # sol_list contains multiple sols, and each sol has the same data structure

# Create a 4x4 subplot
fig, axs = plt.subplots(4, 4, figsize=(12, 12))

# Iterate through each sol and plot the data in the corresponding subplot
for i, sol in enumerate(sol_list):
    # Extract the data from the sol
    (t, r, v, rho) = (sol.t, sol.r, sol.v, sol.rho)
    # Plot the data in the corresponding subplot
    row = i // 4  # Calculate the row index of the current subplot
    col = i % 4   # Calculate the column index of the current subplot
    axs[row, col].plot(t*1e3, np.abs(sum(sum(rho[4:8,4:8,:]))), linewidth=0.5, label='$\\rho_{F=2}$')
    axs[row, col].plot(t*1e3, np.abs(sum(sum(rho[9:15,9:15,:]))), linewidth = 0.5, label='$\\rho_{F\'=3}$')
    axs[row, col].plot(t*1e3, np.abs(sum(sum(rho[0:3,0:3,:]))), linewidth=0.5, label='$\\rho_{F=1}$')
    axs[row, col].legend(fontsize=7)
    axs[row, col].set_xlabel('$t (ms)$')
    axs[row, col].set_ylabel('$\\rho_{ii}$')
# Adjust the spacing between subplots
plt.tight_layout()
plt.show()

#%%
decay_coefficient = 0.01 # Decay coefficient
theta_std = 0.1          # Standard deviation
gain_coefficient = 1e6   # Gain coefficient
all_intensities = []     # Store intensity values
total_voltage = np.array([]) # The same shape as sols[1].t.shape[0]
for i in range(sols_filtered[1].t.shape[0] ):
    # Get the atom positions and initial intensities at the current time step
    atoms_position, initial_intensity = detection_tof_module.position_and_intensity_singletime(sols_filtered , i, k, gamma, max_intensities=1e-16)
    all_intensities.append(np.sum(initial_intensity)) # Calculate the total initial intensity of all atoms at the current time step and store it
    collection_system_position = np.array([0, 0, 0.01]) # Position of the detection device
    efficiency = detection_tof_module.efficiency(atoms_position, collection_system_position) # Efficiency of the fluorescence collection system. Can be replaced with other functions
    voltage = detection_tof_module.simulate_voltage(atoms_position*x0, initial_intensity, decay_coefficient,
                                                        theta_std, gain_coefficient, efficiency=efficiency, s=0,
                                                        collection_system_position=collection_system_position, angle=120)
    total_voltage = np.append(total_voltage, voltage)
total_voltage = detection_tof_module.lowpass_filter(total_voltage, cutoff_frequency=0.019, order=4) # The shape is the same as t (5001,)
total_voltage = np.nan_to_num(total_voltage, nan=0) # Some boundary values are nan and need to be replaced with 0

#%%
t0 = 1/gamma
t_subset = sols_filtered[1].t*t0 # Time, unit: ms
# Plot the figure
plt.plot(t_subset, total_voltage) # The original image without Gaussian fitting
plt.xlabel('Time')
plt.ylabel('Total Voltage')
plt.title('Total Voltage vs Time')
plt.show()

#%% Gaussian fitting
import numpy as np
m = 1.4431606e-25         # Mass, kg
kB = 1.3806503e-23        # Boltzmann constant, J/K
g = -9.78                  # Gravitational acceleration, m/s²
sigma_pz = 1.5e-3         # Standard deviation of noise in the z direction, m
sigema_0 = 3.5e-3         # Initial standard deviation of noise, m
# Assume the create_fit function returns a dictionary containing parameters
total_voltage_with_background = total_voltage + np.random.random(total_voltage.shape) * 0.01 + 0.02 # Add local noise
t0 = 19.5e-3               # TOP, s. Can be modified manually, with little impact on the results
gaussian_signal = gaussiansignal()
# Plot the actual data
plt.scatter(t_subset, total_voltage_with_background, label='Actual Data')
integration_interval = (0.003, 0.007) # Integration interval of the signal
# Output the integration result
# If the curve fitting is not good, you need to change the initial value params; otherwise, it will not converge
integral_result, background_interval, params = gaussian_signal.integrate_gaussian_signal_with_auto_background(
    x=t_subset, y=total_voltage_with_background, params=[-1,1,-1,1], integration_interval=integration_interval, gain = 10)
t = t0 + params[0] 
# Calculate tt = t0 + params[0] 
g = 9.78
# Calculate v0
v0 = (0.2225 + 1/2 * g * t**2)/t
# Calculate T1
T1 = (m/kB) * ((((params[1])**2*0.5)*(v0-g*t)**2 - sigma_pz**2 - sigema_0**2)/t**2)
# Output the integration result and temperature value
plt.title(f'Atom_number: {integral_result:.4f}, Temperature (uk): {T1*1e6:.4f}')
# Plot the signal interval
x = t_subset
y = total_voltage_with_background
signal_x = x[(x >= integration_interval[0]) & (x <= integration_interval[1])]
signal_y = y[(x >= integration_interval[0]) & (x <= integration_interval[1])]
signal = signal_y
plt.fill_between(signal_x, signal, color='lightgreen', alpha=0.5, label='Signal Interval')
# Plot the background signal interval
background_interval = ( integration_interval[1], integration_interval[1] + (integration_interval[1] - integration_interval[0])/2)
background_x = x[(x >= background_interval[0]) & (x <= background_interval[1])]
background_y = y[(x >= background_interval[0]) & (x <= background_interval[1])]
background_signal = background_y
plt.fill_between(background_x, background_signal, color='lightcoral', alpha=0.5, label='Background Interval')
# Plot the fitting result
fitted_curve = gaussian_signal.generate_gaussian_signal(t_subset, *params)
plt.plot(t_subset, fitted_curve, label='Fitted Curve', linestyle='--', color='orange')

# Display the legend and axes
plt.legend()
plt.xlabel('Time(S)')
plt.ylabel('Amplitude')

# Display the figure
plt.show()

#%% Multiple atoms blow followed by detection. Use the sols_solution function in the Blow_module class. The detection results need to be passed in
# If the number of blown atoms is too small, you can consider increasing the laser intensity and the detection light height hb. Other parameters have been marked above
if __name__ == '__main__':
    g = -np.array([0.,  9.8*t0**2/(x0*1e-2),0])
    Blow_test = Blow_module(s=s, det=det, k=k, gamma=gamma, mass=mass, g=g, wb=wb, hb=0.8)
    obe = Blow_test.obe
    tmax = 0.0216/t0
    Blow_test.update_tmax(tmax)
    Blow_test.turn_position(position_turn=np.array([0,3/x0,0])) # Change the position, move the atom position up. The light position is still at z = 0, unit: cm. You can observe the position results of detection and change the parameter values
    Natoms = len(sols_filtered)
    #Natoms = len(sols_filtered)
    chunksize = 16
    sols_blow = []
    progress = progressBar()
    for jj in range(math.ceil(Natoms/chunksize)):
        with pathos.pools.ProcessPool(nodes=16) as pool:
            arg_list = [[Blow_test.obe, sols_filtered, idx] for idx in range(int(Natoms))]
            partial_function = partial(Blow_test.sol_solution)
            sols_blow += pool.map(partial_function, arg_list)
        progress.update((jj+1)/math.ceil(Natoms/chunksize))
# Shape annotation:
# sols_blow[1].r.shape = (3,5001)
# sols_blow[1].v.shape = (3, 5001)
# sols_blow[1].rho.shape = (15, 15, 5001)
# sols_blow[1].t.shape = (5001,)
# sols_blow[1].F.shape = (3,5001)

#%% Test for a large number of atoms. Some sol shapes are incorrect and need to be filtered out
sols_filtered_blow = [sol for sol in sols_blow[0:Natoms] if sol.t.shape[0] == 5001 and sol.F.shape[1] == 5001
                 and sol.r.shape[1] == 5001 and sol.r.shape[1] == 5001]
print(len(sols_filtered_blow))

#%% After blowing, you need to observe the positions of the atoms
Natoms = len(sols_filtered_blow)
fig, ax = plt.subplots(3, 2, figsize=(6.25, 4*2.75), dpi=300)
i = 0
rho = np.zeros((15, 15, Natoms))
v = np.zeros((3, Natoms))
r = np.zeros((3, Natoms))
for sol in sols_filtered_blow:
    for ii in range(3):
        ax[ii, 0].plot(sol.t/(atom.state[2].gammaHz)*1e3, sol.v[ii]*(gamma/k/100), linewidth=0.25)
        ax[ii, 1].plot(sol.t/(atom.state[2].gammaHz)*1e3, sol.r[ii]*x0*10, linewidth=0.35)
for ax_i in ax[-1, :]:
    ax_i.set_xlabel('$t [ms]$')
for jj in range(2):
    for ax_i in ax[jj, :]:
        ax_i.set_xticklabels('')
for ax_i, lbl in zip(ax[:, 0], ['x', 'y', 'z']):
    ax_i.set_ylabel('$v_' + lbl + 'm/s$')

for ax_i, lbl in zip(ax[:, 1], ['x', 'y', 'z']):
    ax_i.set_ylabel('$\\alpha ' + lbl + '$/mm')

fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.32)

#%% First, do a filtering job. If the blow light acts, the third dimension of the atom velocity will be particularly large. Observe the above plotted image.
sols_after_blow = []
for i in range(len(sols_filtered_blow)):
    if sols_filtered_blow[i].v[2,-1]*(gamma/k/100) < 1:
        sols_after_blow.append(sols_filtered_blow[i])
print(len(sols_after_blow))

#%%
if __name__ == '__main__':
    g = -np.array([0.,  9.8*t0**2/(x0*1e-2),0])
    pumping_test = detection_pumping(s1=1, s2 = 10, det1=-1, det2=-0.5, k=k, gamma=gamma, mass=mass, g=g, wb=0.5, hb=0.8)
    obe = pumping_test.obe
    tmax = 0.0216/t0
    pumping_test.update_tmax(tmax)
    pumping_test.turn_position(position_turn=np.array([0,5/x0,0])) # Change the position, move the atom position up. The light position is still at z = 0
    Natoms = len(sols_after_blow)
    chunksize = 16
    sols_pumping = []
    progress = progressBar()
    for jj in range(math.ceil(Natoms/chunksize)):
        with pathos.pools.ProcessPool(nodes=16) as pool:
            arg_list = [(pumping_test.obe, sols_after_blow, idx) for idx in range(len(sols_after_blow))]
            partial_function = partial(pumping_test.sol_solution)
            sols_pumping += pool.map(partial_function, arg_list)
        progress.update((jj+1)/math.ceil(Natoms/chunksize))
# Shape annotation:
# sols_pumping[1].r.shape = (3,5001)
# sols_pumping[1].v.shape = (3, 5001)
# sols_pumping[1].rho.shape = (18, 18, 5001)
# sols_pumping[1].t.shape = (5001,)
# sols_pumping[1].F.shape = (3,5001)

#%%
sols_filtered_pumping = [sol for sol in sols_pumping[0:Natoms] if sol.t.shape[0] == 5001]
Natoms = len(sols_filtered_pumping)
print(len(sols_filtered_pumping))

#%%
fig, ax = plt.subplots(3, 2, figsize=(6.25, 4 * 2.75), dpi=300)
i = 0
rho = np.zeros((18, 18, Natoms))
v = np.zeros((3, Natoms))
r = np.zeros((3, Natoms))
gamma = atom.state[2].gammaHz
alpha = (3/2)*cts.value('Bohr magneton in Hz/T')*1e-4*8*x0/gamma
for sol_all in sols_filtered_pumping:
    t, r[:, i], v[:, i] = sol_all.t / (atom.state[2].gammaHz), sol_all.r[:, -1] * 10*x0, sol_all.v[:, -1] * (k * 100) / (
                                      atom.state[2].gammaHz)
    i += 1
    for ii in range(3):
        ax[ii, 0].plot(sol_all.t / (atom.state[2].gammaHz) * 1e3, sol_all.v[ii] * (
                atom.state[2].gammaHz / k / 100), linewidth=0.25)
        ax[ii, 1].plot(sol_all.t / (atom.state[2].gammaHz) * 1e3, sol_all.r[ii] * x0*10, linewidth=0.35)

for ax_i in ax[-1, :]:
    ax_i.set_xlabel('$t [ms]$')

for jj in range(2):
    for ax_i in ax[jj, :]:
        ax_i.set_xticklabels('')

for ax_i, lbl in zip(ax[:, 0], ['x', 'y', 'z']):
    ax_i.set_ylabel('$v_' + lbl + 'm/s$')

for ax_i, lbl in zip(ax[:, 1], ['x', 'y', 'z']):
    ax_i.set_ylabel('$\\alpha ' + lbl + '$/mm')

fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.32)

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

# Create a 4x4 chart
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
# Iterate through the sols list and plot the data of each sol object
for i in range(4):
    for j in range(4):
        intensity_values = []
        for k in range(5001):
            intensity = np.linalg.norm(sols_filtered_pumping[i*4 + j].F[:,k]) * 3.0e8 / 2 * (cts.hbar * k * gamma)
            intensity_values.append(intensity)
        t0 = 1 / gamma
        t_subset = sols_filtered_pumping[i*4 + j].t * t0
        axs[i, j].plot(t_subset, intensity_values)
        axs[i, j].set_title('Plot {}'.format(i*4 + j + 1))
        axs[i, j].set_xlabel('Time (t)')
        axs[i, j].set_ylabel('Intensity')
        axs[i, j].set_yscale('log')
# Automatically adjust the layout of subplots
plt.tight_layout()
# Display the figure
plt.show()

#%%
import matplotlib.pyplot as plt

# Assume sols_detection is a file containing multiple sols. Here we use sol_list instead
sol_list = sols_filtered_pumping # sol_list contains multiple sols, and each sol has the same data structure

# Create a 4x4 subplot
fig, axs = plt.subplots(4, 4, figsize=(12, 12))

# Iterate through each sol and plot the data in the corresponding subplot
for i, sol in enumerate(sol_list):
    # Extract the data from the sol
    (t, r, v, rho) = (sol.t, sol.r, sol.v, sol.rho)
    # Plot the data in the corresponding subplot
    row = i // 4  # Calculate the row index of the current subplot
    col = i % 4   # Calculate the column index of the current subplot
    axs[row, col].plot(t*1e3, np.abs(sum(sum(rho[4:8,4:8,:]))), linewidth=0.5, label='$\\rho_{F=2}$')
    axs[row, col].plot(t*1e3, np.abs(sum(sum(rho[9:15,9:15,:]))), linewidth = 0.5, label='$\\rho_{F\'=3}$')
    axs[row, col].plot(t*1e3, np.abs(sum(sum(rho[0:3,0:3,:]))), linewidth=0.5, label='$\\rho_{F=1}$')
    axs[row, col].legend(fontsize=7)
    axs[row, col].set_xlabel('$t (ms)$')
    axs[row, col].set_ylabel('$\\rho_{ii}$')
# Adjust the spacing between subplots
plt.tight_layout()
plt.show()

#%%
decay_coefficient = 0.01
theta_std = 0.1
gain_coefficient = 1e6
all_intensities = []
total_voltage = np.array([]) # sols[1].t.shape[0]
for i in range(sols_filtered_pumping[1].t.shape[0] ):
    atoms_position, initial_intensity = detection_tof_module.position_and_intensity_singletime(sols_filtered , i, k, gamma, max_intensities=1e-16)
    all_intensities.append(np.sum(initial_intensity))
    collection_system_position = np.array([0, 0, 0.01]) # Position of the detection device
    efficiency = detection_tof_module.efficiency(atoms_position, collection_system_position) # Efficiency of the fluorescence collection system
    voltage = detection_tof_module.simulate_voltage(atoms_position*x0, initial_intensity, decay_coefficient,
                                                        theta_std, gain_coefficient, efficiency=efficiency, s=0,
                                                        collection_system_position=collection_system_position, angle=120)
    total_voltage = np.append(total_voltage, voltage)
total_voltage_pumping = detection_tof_module.lowpass_filter(total_voltage, cutoff_frequency=0.019, order=4)
total_voltage_pumping = np.nan_to_num(total_voltage_pumping, nan=0) # Some boundary values are nan and need to be replaced

#%%
t0 = 1/gamma
t_subset_pumping = sols_pumping[1].t*t0
# Plot the figure
plt.plot(t_subset_pumping, total_voltage_pumping)
plt.xlabel('Time')
plt.ylabel('Total Voltage')
plt.title('Total Voltage vs Time')
plt.show()

#%%
import numpy as np
m = 1.4431606e-25         # Mass, kg
kB = 1.3806503e-23        # Boltzmann constant, J/K
g = -9.78                  # Gravitational acceleration, m/s²
sigma_pz = 1.5e-3         # Standard deviation of noise in the z direction, m
sigema_0 = 3.5e-3         # Initial standard deviation of noise, m
# Assume the create_fit function returns a dictionary containing parameters
total_voltage_pumping_with_background = total_voltage_pumping + np.random.random(total_voltage_pumping.shape) * 0.01 + 0.02
t0 = 80e-3               # TOP, s
gaussian_signal = gaussiansignal()
# Plot the actual data
plt.scatter(t_subset_pumping, total_voltage_pumping_with_background, label='Actual Data')
integration_interval = (0.005, 0.015)
# Output the integration result. The parameters of params need to be adjusted. Sometimes the fitting may be abnormal.
integral_result_pumping, background_interval_pumping, params_pumping = gaussian_signal.integrate_gaussian_signal_with_auto_background(
    x=t_subset_pumping, y=total_voltage_pumping_with_background, params=[-1,1,-1,1], integration_interval=integration_interval, gain = 10)
t = t0 + params_pumping[0] 
# Calculate tt = t0 + params[0] 
g = 9.78
# Calculate v0
v0 = (0.2225 + 1/2 * g * t**2)/t
# Calculate T1
T1 = (m/kB) * ((((params_pumping[1])**2*0.5)*(v0-g*t)**2 - sigma_pz**2 - sigema_0**2)/t**2)
# Output the integration result and temperature value
plt.title(f'Atom_number: {integral_result_pumping:.4f}, Temperature (uk): {T1*1e6:.4f}')
x = t_subset_pumping
y = total_voltage_pumping_with_background
signal_x_pumping = x[(x >= integration_interval[0]) & (x <= integration_interval[1])]
signal_y_pumping = y[(x >= integration_interval[0]) & (x <= integration_interval[1])]
signal_pumping = signal_y_pumping
plt.fill_between(signal_x_pumping, signal_pumping, color='lightgreen', alpha=0.5, label='Signal Interval')
# Plot the background signal interval
background_interval_pumping = ( integration_interval[1], integration_interval[1] + (integration_interval[1] - integration_interval[0])/2)
background_x_pumping = x[(x >= background_interval_pumping[0]) & (x <= background_interval_pumping[1])]
background_y_pumping = y[(x >= background_interval_pumping[0]) & (x <= background_interval_pumping[1])]
background_signal_pumping = background_y_pumping
plt.fill_between(background_x_pumping, background_signal_pumping, color='lightcoral', alpha=0.5, label='Background Interval')
# Plot the fitting result
fitted_curve_pumping = gaussian_signal.generate_gaussian_signal(t_subset_pumping, *params_pumping)
plt.plot(t_subset_pumping, fitted_curve_pumping, label='Fitted Curve', linestyle='--', color='orange')

# Display the legend and axes
plt.legend()
plt.xlabel('Time(S)')
plt.ylabel('Amplitude')

# Display the figure
plt.show()

# %% Data reading
# Create an empty list to store the read data
import h5py
import numpy as np
from pylcp.integration_tools import RandomOdeResult
sols2 = []
# Open the HDF5 file
with h5py.File('detection_sols2.h5', 'r') as f:
    # Iterate through each group in the file
    for key in f.keys():
        group = f[key]
        # Extract the datasets from the group
        t = np.array(group['t'])
        r = np.array(group['r'])
        v = np.array(group['v'])
        rho = np.array(group['rho'])
        F = np.array(group['F'])
        # Create a RandomOdeResult object and add it to the sols list
        sol = RandomOdeResult(t=t, r=r, v=v, rho=rho, F = F)
        sols2.append(sol)