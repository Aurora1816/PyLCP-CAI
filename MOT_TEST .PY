<<<<<<< HEAD

=======
>>>>>>> origin/main
#%%
import numpy as np
import matplotlib.pyplot as plt
import pylcp
from pylcp.common import progressBar
import scipy.constants as cts
import pathos
from pylcp.integration_tools import RandomOdeResult
from functools import partial
from pathos.pools import ProcessPool
import h5py
from MOT_ALL import MOT2D_module
from MOT_ALL import MOT3D_module
from MOT_ALL import generate_atom
from UP_PGC import atomic_UP_process
from UP_PGC import atomic_PGC_process
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
    plot_results(results_square, results_cube) 
#%%
<<<<<<< HEAD
# Input parameters (initial)
# Variables
atom = pylcp.atom("87Rb")               # Type of atom: 87Rb 
# Fixed values
k = 2*np.pi/780E-7                      # Wave vector in cm^{-1} 
x0 = 1/k                                # Length unit conversion factor, resulting unit is cm
gamma=atom.state[2].gammaHz             # Natural linewidth of the atom in Hz 
t0 = 1/gamma                            # Time unit conversion factor, resulting unit is s
kb = 1.3806503E-23
int_mass = 86.9*cts.value('atomic mass constant')
# Variables
J = 1                                   # Atomic density  
A = 4                                   # Length in the definition of initial atomic motion 
B = 2                                   # Width in the definition of initial atomic motion 
C = 2                                   # Height in the definition of initial atomic motion 
T = 300                                 # Atomic temperature 
# Assignment
=======
#输入参数(初始)
#变量
atom = pylcp.atom("87Rb")               #原子的类型 87Rb 

#固定值
k = 2*np.pi/780E-7                      #波矢 单位cm^{-1} 
x0 = 1/k                                #长度单位换算因子 换算后单位是cm
gamma=atom.state[2].gammaHz             #原子自然线宽 单位Hz 
t0 = 1/gamma                            #时间单位换算因子 换算后单位是s
kb = 1.3806503E-23
int_mass = 86.9*cts.value('atomic mass constant')

#变量
J = 1                                   #原子的密度  
A = 4                                   #原子初始运动定义中的长度 
B = 2                                   #原子初始运动定义中的宽度 
C = 2                                   #原子初始运动定义中的高度 
T = 300                                 #原子温度 

#附值
>>>>>>> origin/main
Initial_data = generate_atom(J, A, B, C, T,kb,x0,t0,int_mass)
sols_r = Initial_data.r
sols_v = Initial_data.v*0.01
sols_N = Initial_data.N
sols_rho = Initial_data.rho
sols_t=np.zeros(Initial_data.natoms)
#%% 
<<<<<<< HEAD
# Create a sols0eqn.h5 file to store the read sols0 (2D) data
=======
# 创建一个sols0eqn.h5文件，用于存储读取的sols0(2D)数据
>>>>>>> origin/main
t_list = sols_t
r_list = sols_r
v_list = sols_v
N_list = sols_N
with h5py.File('inti_solseqn.h5', 'w') as f:
    for i, (t, r, v,N) in enumerate(zip(t_list, r_list, v_list, N_list)):
        group = f.create_group(f'sol_{i}')
        group.create_dataset('t', data=t)
        group.create_dataset('r', data=r)
        group.create_dataset('v', data=v)
        group.create_dataset('N', data=N)
#%%
<<<<<<< HEAD
# Read sols0 data
=======
#读取sols0的数据
>>>>>>> origin/main
sols_0 = []
with h5py.File('inti_solseqn.h5', 'r') as f:
    for key in f.keys():
        group = f[key]
        t = np.array(group['t'])
        r = np.array(group['r'])
        v = np.array(group['v'])
        N = np.array(group['N'])       
        sol = RandomOdeResult(t=t, r=r, v=v, N=N)
        sols_0.append(sol)
#%% 
<<<<<<< HEAD
# Create a sols0obe.h5 file to store the read sols0 (2D) data
=======
# 创建一个sols0obe.h5文件，用于存储读取的sols0(2D)数据
>>>>>>> origin/main
t_list = sols_t
r_list = sols_r
v_list = sols_v
rho_list = sols_rho
with h5py.File('inti_solsobe612.h5', 'w') as f:
    for i, (t, r, v,rho) in enumerate(zip(t_list, r_list, v_list, rho_list)):
        group = f.create_group(f'sol_{i}')
        group.create_dataset('t', data=t)
        group.create_dataset('r', data=r)
        group.create_dataset('v', data=v)
        group.create_dataset('rho', data=rho)
#%%
<<<<<<< HEAD
# Read sols0 data
=======
#读取sols0的数据
>>>>>>> origin/main
sols_0 = []
with h5py.File('inti_solsobe612.h5', 'r') as f:
    for key in f.keys():
        group = f[key]
        t = np.array(group['t'])
        r = np.array(group['r'])
        v = np.array(group['v'])
        rho = np.array(group['rho'])       
        sol = RandomOdeResult(t=t, r=r, v=v, rho=rho)
        sols_0.append(sol)
#%%
<<<<<<< HEAD
# Built-in parameters
I_sat=1.6                               # Saturation intensity in mw/cm^2  
# Input parameters (2D)
# Variables
atom = pylcp.atom("87Rb")               # Type of atom: 87Rb 
det_2D=-2                               # Detuning in Hz, range: [-100,0]
wb_2D=5/x0                              # Spot size, beam diameter, range: [0,10]
po_2D=np.array([0.,0.,-5.])/x0          # Optical trap center position, range: [-50,0] (here we use the Z-axis as the atomic pushing axis)
roffset_2D = (np.array([0.0, 0.0, -10.0])/x0)[:, np.newaxis]   # Initial atomic position offset, range: [-50,50]  
voffset_2D = np.array([0.0, 0.0, 0.0])  # Initial atomic velocity offset, range: [-5,5]
rscale_2D = np.array([0.2, 0.2, 0.2]) /x0  # Random addition to initial atomic position, range: [-1,1]
vscale_2D = np.array([0.1, 0.1, 0.1])   # Random addition to initial atomic velocity, range: [-1,1]
t0_2D=0
tmax_2D=0.03/t0                         # 2D MOT evolution time, range: [0,1] 
g_2D=-np.array([0.,9.8,0.])*t0**2/(x0*1e-2)  # Gravitational acceleration in m/s^2 (here we use the Y-axis as the gravity axis)
Natoms_2D =16                           # Number of atoms, range: [2,N]   
chunksize_2D = 4                        # Number of cores used for computation, range: [2,N] 
rotation_angles_2D=[0., 0., 0.]         # 2D light field rotation angles, range: [0,2*np.pi] 
sols_i_2D=sols_0                        # sols with initial parameters substituted
sols0 = []                              # sols to store solutions
Ige_2D=2/I_sat                          # Intensity of pump light in mW/cm², range: [0,1]
Ire_2D=15/I_sat                         # Intensity of cooling light in mW/cm², range: [0,16]
# Built-in parameters
I_sat=1.6                               # Saturation intensity in mw/cm^2  
alpha_2D =(3/2)*cts.value('Bohr magneton in Hz/T')*1e-4*8*x0/gamma*2  # Magnetic field parameter 
mass_2D = 86.9*cts.value('atomic mass constant')*(x0*1e-2)**2/cts.hbar/t0 # Atomic mass:86.9*cts.value('atomic mass constant')
#%%
# 2D MOT operation
if __name__ == '__main__':    
    # 2D MOT operation loop to solve sols, and store the results in sols3:
=======
#内置参数
I_sat=1.6                               #饱和光强 单位mw/cm^2  
#输入参数（2D）
#变量
atom = pylcp.atom("87Rb")               #原子的类型 87Rb 
det_2D=-2                               #失谐 单位Hz 取值范围：[-100,0]
wb_2D=5/x0                                  #光斑尺寸、光束直径     取值范围：[0,10]
po_2D=np.array([0.,0.,-5.])/x0             #光阱中心位置 取值范围：[-50,0] 在这里我们是以Z轴为原子推送轴 
roffset_2D = (np.array([0.0, 0.0, -10.0])/x0)[:, np.newaxis]   #原子初始位置补偿 取值范围：[-50,50]  
voffset_2D = np.array([0.0, 0.0, 0.0])     #原子初始速度补偿 取值范围：[-5,5]
rscale_2D = np.array([0.2, 0.2, 0.2]) /x0  #原子初始位置添加随机   取值范围：[-1,1]
vscale_2D = np.array([0.1, 0.1, 0.1])      #原子初始速度添加随机   取值范围：[-1,1]
t0_2D=0
tmax_2D=0.03/t0                         #2DMOT演化时间 取值范围：[0,1] 
g_2D=-np.array([0.,9.8,0.])*t0**2/(x0*1e-2)                #重力加速度 单位m/s^2 在这里我们是以Y轴为重力轴
Natoms_2D =16                             #原子个数  取值范围：[2,N]   
chunksize_2D = 4                          #用于运算的核的个数  取值范围：[2,N] 
rotation_angles_2D=[0., 0., 0.]         #2D光场旋转角度  取值范围：[0,2*np.pi] 
sols_i_2D=sols_0                            #代入初始参数的sols
sols0 = []                             #保存解的sols
Ige_2D=2/I_sat                                  #泵浦光的光强 单位mW/cm²  取值范围：[0,1]
Ire_2D=15/I_sat                                  #冷却光的光强 单位mW/cm²  取值范围：[0,16]
#内置参数
I_sat=1.6                               #饱和光强 单位mw/cm^2  
alpha_2D =(3/2)*cts.value('Bohr magneton in Hz/T')*1e-4*8*x0/gamma*2  #磁场参数 
mass_2D = 86.9*cts.value('atomic mass constant')*(x0*1e-2)**2/cts.hbar/t0 #原子质量:86.9*cts.value('atomic mass constant')
#%%
# 2D MOT运行
if __name__ == '__main__':    
    #2D MOT运行循环解sols，并将结果存储在sols3中：
>>>>>>> origin/main
    MOT2D_test=MOT2D_module(t0_2D,atom,alpha_2D,mass_2D,g_2D,det_2D,po_2D,rotation_angles_2D,wb_2D,Ige_2D,Ire_2D,roffset_2D,voffset_2D,rscale_2D,vscale_2D,tmax_2D,sols_i_2D)  
    sol_range = np.arange(Natoms_2D).reshape((int(Natoms_2D/chunksize_2D), chunksize_2D))
    progress = progressBar()
    for jj in range(int(Natoms_2D/chunksize_2D)):
        with pathos.pools.ProcessPool(nodes=2) as pool:
            arg_list = [(MOT2D_test.eqn,idx) for idx in sol_range[jj,:]]
            partial_function =  partial(MOT2D_test.generate_random_solution_2D_eqn)
            sols0 += pool.map(partial_function, arg_list)
        progress.update((jj+1)/int(Natoms_2D/chunksize_2D))  
     
<<<<<<< HEAD
#%% Plotting module
k = 2*np.pi/780E-7   
# Velocity and position
=======
#%%绘图模块
k = 2*np.pi/780E-7   
#速度位置
>>>>>>> origin/main
fig, ax = plt.subplots(3, 2, figsize=(15, 9))
for ii in range(3):
    for idx, soli in enumerate(sols0): 
        ax[ii, 0].plot(soli.t * t0, soli.v[ii] * (atom.state[2].gammaHz / k / 100), linewidth=1)
        ax[ii, 1].plot(soli.t * t0, soli.r[ii] * x0/100 , linewidth=1)
  
    ax[ii, 0].set_ylabel(f'$v_{{{"xyz"[ii]}}}$ (m/s)', fontsize=15)
    ax[ii, 1].set_ylabel(f'$r_{{{"xyz"[ii]}}}$ (m)', fontsize=15)  
<<<<<<< HEAD
    ax[ii, 0].tick_params(axis='both', labelsize=15)  # Adjust the font size of tick labels
    ax[ii, 1].tick_params(axis='both', labelsize=15)  # Adjust the font size of tick labels
for ax_i in ax[-1, :]:
    ax_i.set_xlabel('Time (s)', fontsize=15)
    ax_i.tick_params(axis='both', labelsize=15)  # Adjust the font size of tick labels
fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.22)
plt.show()
# State density
i=0
solsi=sols0
# Create subplot
fig, ax = plt.subplots(figsize=(6, 3))
# Calculate curves for each slice
=======
    ax[ii, 0].tick_params(axis='both', labelsize=15)  # 调整刻度标签字体大小
    ax[ii, 1].tick_params(axis='both', labelsize=15)  # 调整刻度标签字体大小
for ax_i in ax[-1, :]:
    ax_i.set_xlabel('Time (s)', fontsize=15)
    ax_i.tick_params(axis='both', labelsize=15)  # 调整刻度标签字体大小
fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.22)
plt.show()
#态密度
i=0
solsi=sols0
# 创建子图
fig, ax = plt.subplots(figsize=(6, 3))
# 计算每个切片的曲线
>>>>>>> origin/main
rho_F1 = np.sum(solsi[i].N[0:3, :], axis=0).real
rho_F2 = np.sum(solsi[i].N[3:8,: ], axis=0).real
rho_F0 = np.sum(solsi[i].N[8:9, :], axis=0).real
rho_F1_prime = np.sum(solsi[i].N[9:12,: ], axis=0).real
rho_F2_prime = np.sum(solsi[i].N[12:17, :], axis=0).real
rho_F3_prime = np.sum(solsi[i].N[17:23,: ], axis=0).real
<<<<<<< HEAD
# Plot curves in individual subplots with corresponding labels
=======
# 在单独的子图中绘制曲线，并添加相应的标签
>>>>>>> origin/main
ax.plot(solsi[i].t* 1e3, rho_F1, linewidth=0.5, label='$\\rho_{F=1}$')
ax.plot(solsi[i].t* 1e3, rho_F2, linewidth=0.5, label='$\\rho_{F=2}$')
ax.plot(solsi[i].t* 1e3, rho_F0, linewidth=0.5, label='$\\rho_{F\'\'=0}$')
ax.plot(solsi[i].t* 1e3, rho_F1_prime, linewidth=0.5, label='$\\rho_{F\'\'=1}$')
ax.plot(solsi[i].t* 1e3, rho_F2_prime, linewidth=0.5, label='$\\rho_{F\'\'=2}$')
ax.plot(solsi[i].t* 1e3, rho_F3_prime, linewidth=0.5, label='$\\rho_{F\'\'=3}$')
ax.set_xlabel('$t (ms)')
ax.yaxis.set_label_coords(1.08, 0.5)
ax.set_ylabel('$\\rho_{ii}$', rotation=0, labelpad=15)
ax.legend(fontsize=7, loc='upper right', ncol=2)
plt.tight_layout()
plt.show()
<<<<<<< HEAD
# Histogram
=======

#直方图
>>>>>>> origin/main
N=np.zeros([24,1001,Natoms_2D])
i=0
for sol in solsi:
    N[:,:,i]=sol.N
    i=i+1
diagonal_sums = np.sum(np.diagonal(N, axis1=0, axis2=1, offset=0), axis=0)
diagonal_sums*10000
fig = plt.figure(figsize=(10.25, 2*2.75), dpi=300)
<<<<<<< HEAD
plt.subplot(2, 2, 1)  # 1st subplot in 2 rows and 1 column
=======
plt.subplot(2, 2, 1)  # 2行1列的第1个子图
>>>>>>> origin/main
plt.bar(range(3), diagonal_sums[:3], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title('F=1')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_1 = ['mF=-1', 'mF=0', 'mF=1']
plt.xticks(range(3), mf_labels_1)
plt.subplot(2, 2, 2)  # 1st subplot in 2 rows and 1 column
=======
# 自定义横坐标标签
mf_labels_1 = ['mF=-1', 'mF=0', 'mF=1']
plt.xticks(range(3), mf_labels_1)

plt.subplot(2, 2, 2)  # 2行1列的第1个子图
>>>>>>> origin/main
plt.bar(range(5), diagonal_sums[3:8], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title('F=2')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_1 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_1)
# Create bar chart - last ten elements (upper level F'=1 and F'=3 states)
plt.subplot(2, 2, 3)  # 2nd subplot in 2 rows and 1 column
=======
# 自定义横坐标标签
mf_labels_1 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_1)
# 制作柱状图 - 后十个元素（上能级的 F'=1 和 F'=3 的态）
plt.subplot(2, 2, 3)  # 2行1列的第2个子图
>>>>>>> origin/main
plt.bar(range(5), diagonal_sums[12:17], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title(r'$F\' =2$')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_2 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_2)
plt.subplot(2, 2, 4)  # 2nd subplot in 2 rows and 1 column
=======
# 自定义横坐标标签
mf_labels_2 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_2)
plt.subplot(2, 2, 4)  # 2行1列的第2个子图
>>>>>>> origin/main
plt.bar(range(7), diagonal_sums[16:23], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title(r'$F\' =3$')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_2 = ['mF=-3','mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2','mF=3']
plt.xticks(range(7), mf_labels_2)
plt.tight_layout()  # Automatically adjust subplot layout to avoid overlap
plt.show()
#%%
# Input parameters
det_3D=-2                               # Detuning in Hz, range: [-50,0]
Ige_3D=2/I_sat                          # Intensity of pump light in mW/cm², range: [0,1]
Ire_3D=15/I_sat                         # Intensity of cooling light in mW/cm², range: [0,16]
wb_3D=2.5/x0                            # Spot size, beam diameter, range: [0,10]
tmax_3D=0.02/t0                         # 3D MOT evolution time, range: [0,1] 
Natoms_3D =2                            # Number of atoms, range: [2,N]   
chunksize_3D = 2                        # Number of cores used for computation, range: [2,N] 
rotation_angles_3D=[0., 0., 0.]         # 2D light field rotation angles, range: [0,2*np.pi] 
sols_i_3D=sols0                         # sols with initial parameters substituted
tmax_3D=0.0005/t0                       # 2D MOT evolution time, range: [0,1] 
sols1 = []                              # sols to store solutions
alpha_3D =(3/2)*cts.value('Bohr magneton in Hz/T')*1e-4*8*x0/gamma
sols_i_3=sols0
mass_3D=86.9*cts.value('atomic mass constant')*(x0*1e-2)**2/cts.hbar/t0 # Atomic mass:86.9*cts.value('atomic mass constant')
g_3D=-np.array([0.,9.8,0.])*t0**2/(x0*1e-2)  # Gravitational acceleration in m/s^2 (here we use the Y-axis as the gravity axis)
#%%
# 3D MOT operation
if __name__ == '__main__':
    # 3D MOT operation loop to solve sols, and store the results in sols4:
=======
# 自定义横坐标标签
mf_labels_2 = ['mF=-3','mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2','mF=3']
plt.xticks(range(7), mf_labels_2)
plt.tight_layout()  # 自动调整子图布局，以免重叠
plt.show()
#%%
# 输入参数
det_3D=-2                                  #失谐 单位Hz 取值范围：[-50,0]
Ige_3D=2/I_sat                                   #泵浦光的光强 单位mW/cm²  取值范围：[0,1]
Ire_3D=15/I_sat                                 #冷却光的光强 单位mW/cm²  取值范围：[0,16]
wb_3D=2.5/x0                                   #光斑尺寸、光束直径     取值范围：[0,10]
tmax_3D=0.02/t0                           #3DMOT演化时间 取值范围：[0,1] 
Natoms_3D =2                              #原子个数  取值范围：[2,N]   
chunksize_3D = 2                          #用于运算的核的个数  取值范围：[2,N] 
rotation_angles_3D=[0., 0., 0.]         #2D光场旋转角度  取值范围：[0,2*np.pi] 
sols_i_3D=sols0                            #代入初始参数的sols
tmax_3D=0.0005/t0                         #2DMOT演化时间 取值范围：[0,1] 
sols1 = []                              #保存解的sols
alpha_3D =(3/2)*cts.value('Bohr magneton in Hz/T')*1e-4*8*x0/gamma
sols_i_3=sols0
mass_3D=86.9*cts.value('atomic mass constant')*(x0*1e-2)**2/cts.hbar/t0 #原子质量:86.9*cts.value('atomic mass constant')
g_3D=-np.array([0.,9.8,0.])*t0**2/(x0*1e-2)                #重力加速度 单位m/s^2 在这里我们是以Y轴为重力轴
#%%
# 3D MOT运行
if __name__ == '__main__':
    #3D MOT运行循环解sols，并将结果存储在sols4中：
>>>>>>> origin/main
    MOT3D_test=MOT3D_module(alpha_3D,mass_3D,det_3D,Ige_3D, Ire_3D, atom,g_3D, rotation_angles_3D,wb_3D,sols_i_3,tmax_3D)
    sol_range = np.arange(Natoms_3D).reshape((int(Natoms_3D/chunksize_3D), chunksize_3D))
    progress = progressBar()
    for jj in range(int(Natoms_3D/chunksize_3D)):
        with pathos.pools.ProcessPool(nodes=2) as pool:
            arg_list = [(MOT3D_test.eqn,idx) for idx in sol_range[jj,:]]
            partial_function =  partial(MOT3D_test.generate_random_solution_3D_eqn)
            sols1 += pool.map(partial_function, arg_list)
        progress.update((jj+1)/int(Natoms_3D/chunksize_3D))
<<<<<<< HEAD
#%% Plotting module
k = 2*np.pi/780E-7   
# Velocity and position
=======

#%%绘图模块
k = 2*np.pi/780E-7   
#速度位置
>>>>>>> origin/main
fig, ax = plt.subplots(3, 2, figsize=(15, 9))
for ii in range(3):
    for idx, soli in enumerate(sols1): 
        ax[ii, 0].plot(soli.t * t0, soli.v[ii] * (atom.state[2].gammaHz / k / 100), linewidth=1)
        ax[ii, 1].plot(soli.t * t0, soli.r[ii] * x0/100 , linewidth=1)
  
    ax[ii, 0].set_ylabel(f'$v_{{{"xyz"[ii]}}}$ (m/s)', fontsize=15)
    ax[ii, 1].set_ylabel(f'$r_{{{"xyz"[ii]}}}$ (m)', fontsize=15)  
<<<<<<< HEAD
    ax[ii, 0].tick_params(axis='both', labelsize=15)  # Adjust the font size of tick labels
    ax[ii, 1].tick_params(axis='both', labelsize=15)  # Adjust the font size of tick labels
for ax_i in ax[-1, :]:
    ax_i.set_xlabel('Time (s)', fontsize=15)
    ax_i.tick_params(axis='both', labelsize=15)  # Adjust the font size of tick labels
fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.22)
plt.show()
#%%
# State density
i=0
solsi=sols1
# Create subplot
fig, ax = plt.subplots(figsize=(6, 3))
# Calculate curves for each slice
=======
    ax[ii, 0].tick_params(axis='both', labelsize=15)  # 调整刻度标签字体大小
    ax[ii, 1].tick_params(axis='both', labelsize=15)  # 调整刻度标签字体大小
for ax_i in ax[-1, :]:
    ax_i.set_xlabel('Time (s)', fontsize=15)
    ax_i.tick_params(axis='both', labelsize=15)  # 调整刻度标签字体大小
fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.22)
plt.show()
#%%
#态密度
i=0
solsi=sols1
# 创建子图
fig, ax = plt.subplots(figsize=(6, 3))
# 计算每个切片的曲线
>>>>>>> origin/main
rho_F1 = np.sum(solsi[i].N[0:3, :], axis=0).real
rho_F2 = np.sum(solsi[i].N[3:8,: ], axis=0).real
rho_F0 = np.sum(solsi[i].N[8:9, :], axis=0).real
rho_F1_prime = np.sum(solsi[i].N[9:12,: ], axis=0).real
rho_F2_prime = np.sum(solsi[i].N[12:17, :], axis=0).real
rho_F3_prime = np.sum(solsi[i].N[17:23,: ], axis=0).real
<<<<<<< HEAD
# Plot curves in individual subplots with corresponding labels
=======
# 在单独的子图中绘制曲线，并添加相应的标签
>>>>>>> origin/main
ax.plot(solsi[i].t* 1e3, rho_F1, linewidth=0.5, label='$\\rho_{F=1}$')
ax.plot(solsi[i].t* 1e3, rho_F2, linewidth=0.5, label='$\\rho_{F=2}$')
ax.plot(solsi[i].t* 1e3, rho_F0, linewidth=0.5, label='$\\rho_{F\'\'=0}$')
ax.plot(solsi[i].t* 1e3, rho_F1_prime, linewidth=0.5, label='$\\rho_{F\'\'=1}$')
ax.plot(solsi[i].t* 1e3, rho_F2_prime, linewidth=0.5, label='$\\rho_{F\'\'=2}$')
ax.plot(solsi[i].t* 1e3, rho_F3_prime, linewidth=0.5, label='$\\rho_{F\'\'=3}$')
ax.set_xlabel('$t (ms)')
ax.yaxis.set_label_coords(1.08, 0.5)
ax.set_ylabel('$\\rho_{ii}$', rotation=0, labelpad=15)
ax.legend(fontsize=7, loc='upper right', ncol=2)
plt.tight_layout()
plt.show()
<<<<<<< HEAD
# Histogram
=======

#直方图
>>>>>>> origin/main
N=np.zeros([24,1001,Natoms_3D])
i=0
for sol in solsi:
    N[:,:,i]=sol.N
    i=i+1
diagonal_sums = np.sum(np.diagonal(N, axis1=0, axis2=1, offset=0), axis=0)
diagonal_sums*10000
fig = plt.figure(figsize=(10.25, 2*2.75), dpi=300)
<<<<<<< HEAD
plt.subplot(2, 2, 1)  # 1st subplot in 2 rows and 1 column
=======
plt.subplot(2, 2, 1)  # 2行1列的第1个子图
>>>>>>> origin/main
plt.bar(range(3), diagonal_sums[:3], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title('F=1')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_1 = ['mF=-1', 'mF=0', 'mF=1']
plt.xticks(range(3), mf_labels_1)
plt.subplot(2, 2, 2)  # 1st subplot in 2 rows and 1 column
=======
# 自定义横坐标标签
mf_labels_1 = ['mF=-1', 'mF=0', 'mF=1']
plt.xticks(range(3), mf_labels_1)

plt.subplot(2, 2, 2)  # 2行1列的第1个子图
>>>>>>> origin/main
plt.bar(range(5), diagonal_sums[3:8], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title('F=2')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_1 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_1)
# Create bar chart - last ten elements (upper level F'=1 and F'=3 states)
plt.subplot(2, 2, 3)  # 2nd subplot in 2 rows and 1 column
=======
# 自定义横坐标标签
mf_labels_1 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_1)
# 制作柱状图 - 后十个元素（上能级的 F'=1 和 F'=3 的态）
plt.subplot(2, 2, 3)  # 2行1列的第2个子图
>>>>>>> origin/main
plt.bar(range(5), diagonal_sums[12:17], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title(r'$F\' =2$')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_2 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_2)
plt.subplot(2, 2, 4)  # 2nd subplot in 2 rows and 1 column
=======
# 自定义横坐标标签
mf_labels_2 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_2)
plt.subplot(2, 2, 4)  # 2行1列的第2个子图
>>>>>>> origin/main
plt.bar(range(7), diagonal_sums[16:23], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title(r'$F\' =3$')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_2 = ['mF=-3','mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2','mF=3']
plt.xticks(range(7), mf_labels_2)
plt.tight_layout()  # Automatically adjust subplot layout to avoid overlap
plt.show()
#%% 
# Input parameters
g_UP=np.array([0.,9.8,0.])*t0**2/(x0*1e-2)  # Gravitational acceleration in m/s^2 (here we use the Y-axis as the gravity axis)
alpha_UP =0                               # Magnetic field parameter, set to 0 because no magnetic field is applied during upward launch
mass_UP=86.9*cts.value('atomic mass constant')*(x0*1e-2)**2/cts.hbar/t0 # Atomic mass:86.9*cts.value('atomic mass constant')
delta_UP=-2                               # Detuning in Hz, range: [-20,0] 
shot_UP=-2                                # Detuning parameter, range: [-10,0]
s_UP=10                                   # Light intensity in mw/cm^2, range: [0,20]
wb_UP=2.5/x0                              # Spot size, beam diameter, range: [0,10]
phi_i_UP = np.pi/2                        # Beam polarization, range: [0,2*np.pi]
tmax_UP=0.02/t0                           # Evolution time, range: [0,1] 
Natoms_UP =2                              # Number of atoms, range: [2,N]   
chunksize_UP = 2                          # Number of cores used for computation, range: [2,N] 
rotation_angles_UP=[0, 0, np.pi/4]        # 2D light field rotation angles, range: [0,2*np.pi] 
# Built-in parameters
sols_i_UP=sols1                           # sols with initial parameters substituted
sols2 = []                                # sols to store solutions
#%%
# Upward launch module
if __name__== '__main__':
# Upward launch operation loop to solve sols, and store the results in sols3:
=======
# 自定义横坐标标签
mf_labels_2 = ['mF=-3','mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2','mF=3']
plt.xticks(range(7), mf_labels_2)
plt.tight_layout()  # 自动调整子图布局，以免重叠
plt.show()
#%% 
# 输入参数
g_UP=np.array([0.,9.8,0.])*t0**2/(x0*1e-2)        #重力加速度 单位m/s^2 在这里我们是以Y轴为重力轴
alpha_UP =0                                #磁场参数 ,上抛不作用磁场，所以置0
mass_UP=86.9*cts.value('atomic mass constant')*(x0*1e-2)**2/cts.hbar/t0 #原子质量:86.9*cts.value('atomic mass constant')
delta_UP=-2                              #失谐 单位Hz 取值范围：[-20,0] 
shot_UP=-2                                 #失谐参数   取值范围：[-10,0]
s_UP=10                                    #光强   单位mw/cm^2   取值范围：[0,20]
wb_UP=2.5/x0                                  #光斑尺寸、光束直径     取值范围：[0,10]
phi_i_UP = np.pi/2                         #光束偏振  取值范围：[0,2*np.pi]
tmax_UP=0.02/t0                              #演化时间 取值范围：[0,1] 
Natoms_UP =2                               #原子个数  取值范围：[2,N]   
chunksize_UP = 2                           #用于运算的核的个数  取值范围：[2,N] 
rotation_angles_UP=[0, 0, np.pi/4]         #2D光场旋转角度  取值范围：[0,2*np.pi] 
#内置参数
sols_i_UP=sols1                            #代入初始参数的sols
sols2 = []                              #保存解的sols
#%%
# 上抛模块
if __name__== '__main__':
#上抛运行循环解sols，并将结果存储在sols3中：
>>>>>>> origin/main
    UP_test=atomic_UP_process(g_UP,atom,alpha_UP,mass_UP,delta_UP,shot_UP,s_UP,wb_UP,rotation_angles_UP,phi_i_UP,tmax_UP,sols_i_UP)
    sol_range = np.arange(Natoms_UP).reshape((int(Natoms_UP/chunksize_UP), chunksize_UP))
    progress = progressBar()
    for jj in range(int(Natoms_UP/chunksize_UP)):
        with pathos.pools.ProcessPool(nodes=chunksize_UP) as pool:
            arg_list = [(UP_test.eqn,idx) for idx in sol_range[jj,:]]
            partial_function = partial(UP_test.generate_eqn_solution_UP)
            sols2 += pool.map(partial_function,arg_list)
            progress.update((jj+1)/int(Natoms_UP/chunksize_UP))
<<<<<<< HEAD
#%% Plotting module
# Velocity and position
=======

#%%绘图模块
#速度位置

>>>>>>> origin/main
fig, ax = plt.subplots(3, 2, figsize=(15, 9))
for ii in range(3):
    for idx, soli in enumerate(sols2): 
        ax[ii, 0].plot(soli.t * t0, soli.v[ii] * (atom.state[2].gammaHz / k / 100), linewidth=1)
        ax[ii, 1].plot(soli.t * t0, soli.r[ii] * x0/100 , linewidth=1)
  
    ax[ii, 0].set_ylabel(f'$v_{{{"xyz"[ii]}}}$ (m/s)', fontsize=15)
    ax[ii, 1].set_ylabel(f'$r_{{{"xyz"[ii]}}}$ (m)', fontsize=15)  
<<<<<<< HEAD
    ax[ii, 0].tick_params(axis='both', labelsize=15)  # Adjust the font size of tick labels
    ax[ii, 1].tick_params(axis='both', labelsize=15)  # Adjust the font size of tick labels
for ax_i in ax[-1, :]:
    ax_i.set_xlabel('Time (s)', fontsize=15)
    ax_i.tick_params(axis='both', labelsize=15)  # Adjust the font size of tick labels
fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.22)
plt.show()
#%%
# State density
i=0
solsi=sols2
# Create subplot
fig, ax = plt.subplots(figsize=(6, 3))
# Calculate curves for each slice
=======
    ax[ii, 0].tick_params(axis='both', labelsize=15)  # 调整刻度标签字体大小
    ax[ii, 1].tick_params(axis='both', labelsize=15)  # 调整刻度标签字体大小
for ax_i in ax[-1, :]:
    ax_i.set_xlabel('Time (s)', fontsize=15)
    ax_i.tick_params(axis='both', labelsize=15)  # 调整刻度标签字体大小
fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.22)
plt.show()
#%%
#态密度
i=0
solsi=sols2
# 创建子图
fig, ax = plt.subplots(figsize=(6, 3))
# 计算每个切片的曲线
>>>>>>> origin/main
rho_F1 = np.sum(solsi[i].N[0:3, :], axis=0).real
rho_F2 = np.sum(solsi[i].N[3:8,: ], axis=0).real
rho_F0 = np.sum(solsi[i].N[8:9, :], axis=0).real
rho_F1_prime = np.sum(solsi[i].N[9:12,: ], axis=0).real
rho_F2_prime = np.sum(solsi[i].N[12:17, :], axis=0).real
rho_F3_prime = np.sum(solsi[i].N[17:23,: ], axis=0).real
<<<<<<< HEAD
# Plot curves in individual subplots with corresponding labels
=======
# 在单独的子图中绘制曲线，并添加相应的标签
>>>>>>> origin/main
ax.plot(solsi[i].t* 1e3, rho_F1, linewidth=0.5, label='$\\rho_{F=1}$')
ax.plot(solsi[i].t* 1e3, rho_F2, linewidth=0.5, label='$\\rho_{F=2}$')
ax.plot(solsi[i].t* 1e3, rho_F0, linewidth=0.5, label='$\\rho_{F\'\'=0}$')
ax.plot(solsi[i].t* 1e3, rho_F1_prime, linewidth=0.5, label='$\\rho_{F\'\'=1}$')
ax.plot(solsi[i].t* 1e3, rho_F2_prime, linewidth=0.5, label='$\\rho_{F\'\'=2}$')
ax.plot(solsi[i].t* 1e3, rho_F3_prime, linewidth=0.5, label='$\\rho_{F\'\'=3}$')
ax.set_xlabel('$t (ms)')
ax.yaxis.set_label_coords(1.08, 0.5)
ax.set_ylabel('$\\rho_{ii}$', rotation=0, labelpad=15)
ax.legend(fontsize=7, loc='upper right', ncol=2)
plt.tight_layout()
plt.show()
<<<<<<< HEAD
# Histogram
=======

#直方图
>>>>>>> origin/main
N=np.zeros([24,1001,Natoms_2D])
i=0
for sol in solsi:
    N[:,:,i]=sol.N
    i=i+1
diagonal_sums = np.sum(np.diagonal(N, axis1=0, axis2=1, offset=0), axis=0)
diagonal_sums*10000
fig = plt.figure(figsize=(10.25, 2*2.75), dpi=300)
<<<<<<< HEAD
plt.subplot(2, 2, 1)  # 1st subplot in 2 rows and 1 column
=======
plt.subplot(2, 2, 1)  # 2行1列的第1个子图
>>>>>>> origin/main
plt.bar(range(3), diagonal_sums[:3], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title('F=1')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_1 = ['mF=-1', 'mF=0', 'mF=1']
plt.xticks(range(3), mf_labels_1)
plt.subplot(2, 2, 2)  # 1st subplot in 2 rows and 1 column
=======
# 自定义横坐标标签
mf_labels_1 = ['mF=-1', 'mF=0', 'mF=1']
plt.xticks(range(3), mf_labels_1)

plt.subplot(2, 2, 2)  # 2行1列的第1个子图
>>>>>>> origin/main
plt.bar(range(5), diagonal_sums[3:8], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title('F=2')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_1 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_1)
# Create bar chart - last ten elements (upper level F'=1 and F'=3 states)
plt.subplot(2, 2, 3)  # 2nd subplot in 2 rows and 1 column
=======
# 自定义横坐标标签
mf_labels_1 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_1)
# 制作柱状图 - 后十个元素（上能级的 F'=1 和 F'=3 的态）
plt.subplot(2, 2, 3)  # 2行1列的第2个子图
>>>>>>> origin/main
plt.bar(range(5), diagonal_sums[12:17], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title(r'$F\' =2$')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_2 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_2)
plt.subplot(2, 2, 4)  # 2nd subplot in 2 rows and 1 column
=======
# 自定义横坐标标签
mf_labels_2 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_2)
plt.subplot(2, 2, 4)  # 2行1列的第2个子图
>>>>>>> origin/main
plt.bar(range(7), diagonal_sums[16:23], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title(r'$F\' =3$')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_2 = ['mF=-3','mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2','mF=3']
plt.xticks(range(7), mf_labels_2)
plt.tight_layout()  # Automatically adjust subplot layout to avoid overlap
plt.show()
# Temperature plot
allr = np.concatenate([sol.r[:, :].T for i, sol in enumerate(solsi) if i not in [16, 23]]).T
allv = np.concatenate([sol.v[:, :].T for i, sol in enumerate(solsi) if i not in [16, 23]]).T
img, y_edges, z_edges = np.histogram2d(allr[1, ::100]/k, allr[2, ::100]/k, bins=[np.arange(-5., 5.01, 0.15), np.arange(-5., 5.01, 0.15)])
=======
# 自定义横坐标标签
mf_labels_2 = ['mF=-3','mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2','mF=3']
plt.xticks(range(7), mf_labels_2)
plt.tight_layout()  # 自动调整子图布局，以免重叠
plt.show()

#温度绘图
allr = np.concatenate([sol.r[:, :].T for i, sol in enumerate(solsi) if i not in [16, 23]]).T
allv = np.concatenate([sol.v[:, :].T for i, sol in enumerate(solsi) if i not in [16, 23]]).T
img, y_edges, z_edges = np.histogram2d(allr[1, ::100]/k, allr[2, ::100]/k, bins=[np.arange(-5., 5.01, 0.15), np.arange(-5., 5.01, 0.15)])

>>>>>>> origin/main
fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), dpi=300)
im = ax.imshow(img.T, origin='lower',  
               extent=(np.amin(y_edges), np.amax(y_edges),
                       np.amin(z_edges), np.amax(z_edges)),
               cmap='Blues',
               aspect='equal')
ax.set_xlabel('$y$ (mm)')
ax.set_ylabel('$z$ (mm)')
t_eval =np.linspace(0, tmax_UP,1001)
vs = np.nan*np.zeros((len(solsi), 3, len(t_eval)))
for vt, sol in zip(vs, solsi):
    vt[:, :sol.v.shape[1]] = sol.v
ejected = [np.bitwise_or( np.abs(sol.r[0, -1]*(1e4*x0))>200,np.abs(sol.r[1, -1]*(1e4*x0))>200 ) for sol in
           solsi]
print('Number of ejected atoms: %d' % np.sum(ejected))
sigma_v = np.nanstd(vs, axis=0)
sigma_v.shape
fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), dpi=300)
ax.plot(t[:1001]*1e3, 2*sigma_v.T[:,:1001]**2*UP_test.hamiltonian.mass*146)
result=2*sigma_v[1,-1]**2*UP_test.hamiltonian.mass*146
ax.set_ylabel('$T (\mu k)$')
ax.set_xlabel('$t (ms)$')
<<<<<<< HEAD
ax.set_title(f'Atomic Temperature\nResult: ${result} \mu k$')
plt.legend()
plt.show()
#%%
# PGC module
g_PGC=np.array([0.,9.8,0.])*t0**2/(x0*1e-2)  # Gravitational acceleration in m/s^2 (here we use the Y-axis as the gravity axis)
delta_PGC=-10                             # Detuning in Hz, range: [-50,0] 
shot_PGC=0                                # Detuning parameter, range: [-10,0]
s_PGC=2                                   # Light intensity in mw/cm^2, range: [0,20]
wb_PGC=2.5/x0                             # Spot size, beam diameter, range: [0,10]
phi_i_PGC = np.pi/2                        # Beam polarization, range: [0,2*np.pi]
tmax_PGC=0.01/t0                           # Evolution time, range: [0,1] 
Natoms_PGC =1                              # Number of atoms, range: [2,N]   
chunksize_PGC = 1                          # Number of cores used for computation, range: [2,N] 
rotation_angles_PGC=[0, 0, np.pi/4]        # 2D light field rotation angles, range: [0,2*np.pi] 
sols_i_PGC=sols2                           # sols with initial parameters substituted
sols3 = []                                 # sols to store solutions
# Built-in parameters
alpha_PGC =0                               # Magnetic field parameter, set to 0 because no magnetic field is applied during upward launch
mass_PGC=86.9*cts.value('atomic mass constant')*(x0*1e-2)**2/cts.hbar/t0 # Atomic mass:86.9*cts.value('atomic mass constant')
#%%
if __name__== '__main__':
    # PGC operation loop to solve sols, and store the results in sols4:      
=======
ax.set_title(f'Atomic Temputer\nResult: ${result} \mu k$')
plt.legend()
plt.show()

#%%
# PGC模块
g_PGC=np.array([0.,9.8,0.])*t0**2/(x0*1e-2) #重力加速度 单位m/s^2 在这里我们是以Y轴为重力轴
delta_PGC=-10                            #失谐 单位Hz 取值范围：[-50,0] 
shot_PGC=0                                  #失谐参数   取值范围：[-10,0]
s_PGC=2                                     #光强   单位mw/cm^2   取值范围：[0,20]
wb_PGC=2.5/x0                                  #光斑尺寸、光束直径     取值范围：[0,10]
phi_i_PGC = np.pi/2                         #光束偏振  取值范围：[0,2*np.pi]
tmax_PGC=0.01/t0                              #演化时间 取值范围：[0,1] 
Natoms_PGC =1                             #原子个数  取值范围：[2,N]   
chunksize_PGC = 1                          #用于运算的核的个数  取值范围：[2,N] 
rotation_angles_PGC=[0, 0, np.pi/4]         #2D光场旋转角度  取值范围：[0,2*np.pi] 
sols_i_PGC=sols2                            #代入初始参数的sols
sols3 = []                              #保存解的sols
#内置参数
alpha_PGC =0                                #磁场参数 ,上抛不作用磁场，所以置0
mass_PGC=86.9*cts.value('atomic mass constant')*(x0*1e-2)**2/cts.hbar/t0 #原子质量:86.9*cts.value('atomic mass constant')
#%%

if __name__== '__main__':
    #PGC运行循环解sols，并将结果存储在sols4中：      
>>>>>>> origin/main
    PGC_test=atomic_PGC_process(g_PGC,atom,alpha_PGC,delta_PGC,mass_PGC,shot_PGC,s_PGC,wb_PGC,rotation_angles_PGC,phi_i_PGC,tmax_PGC,sols_i_PGC)
    sol_range = np.arange(Natoms_PGC).reshape((int(Natoms_PGC/chunksize_PGC), chunksize_PGC))
    progress = progressBar()
    for jj in range(int(Natoms_PGC/chunksize_PGC)):
        with pathos.pools.ProcessPool(nodes=chunksize_PGC) as pool:
            arg_list = [(PGC_test.obe,idx) for idx in sol_range[jj,:]]
            partial_function = partial(PGC_test.generate_obe_solution_PGC)
            sols3 += pool.map(partial_function,arg_list)
            progress.update((jj+1)/int(Natoms_PGC/chunksize_PGC))
<<<<<<< HEAD
#%% Plotting module
# Velocity and position
=======
#%%绘图模块
#速度位置
>>>>>>> origin/main
soli=sols3
fig, ax = plt.subplots(3, 2, figsize=(15, 9))
for ii in range(3):
    ax[ii, 0].plot(soli.t * t0, soli.v[ii] * (atom.state[2].gammaHz / k / 100), linewidth=1)
    ax[ii, 1].plot(soli.t * t0, soli.r[ii] * x0/100 , linewidth=1)
  
    ax[ii, 0].set_ylabel(f'$v_{{{"xyz"[ii]}}}$ (m/s)', fontsize=15)
    ax[ii, 1].set_ylabel(f'$r_{{{"xyz"[ii]}}}$ (m)', fontsize=15)  
<<<<<<< HEAD
    ax[ii, 0].tick_params(axis='both', labelsize=15)  # Adjust the font size of tick labels
    ax[ii, 1].tick_params(axis='both', labelsize=15)  # Adjust the font size of tick labels
for ax_i in ax[-1, :]:
    ax_i.set_xlabel('Time (s)', fontsize=15)
    ax_i.tick_params(axis='both', labelsize=15)  # Adjust the font size of tick labels
fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.22)
plt.show()
# State density
i=0
solsi=sols3
# Create subplot
fig, ax = plt.subplots(figsize=(6, 3))
# Calculate curves for each slice
=======
    ax[ii, 0].tick_params(axis='both', labelsize=15)  # 调整刻度标签字体大小
    ax[ii, 1].tick_params(axis='both', labelsize=15)  # 调整刻度标签字体大小
for ax_i in ax[-1, :]:
    ax_i.set_xlabel('Time (s)', fontsize=15)
    ax_i.tick_params(axis='both', labelsize=15)  # 调整刻度标签字体大小
fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.22)
plt.show()

#态密度
i=0
solsi=sols3
# 创建子图
fig, ax = plt.subplots(figsize=(6, 3))
# 计算每个切片的曲线
>>>>>>> origin/main
rho_F1 = np.sum(solsi[i].N[0:3, :], axis=0).real
rho_F2 = np.sum(solsi[i].N[3:8,: ], axis=0).real
rho_F0 = np.sum(solsi[i].N[8:9, :], axis=0).real
rho_F1_prime = np.sum(solsi[i].N[9:12,: ], axis=0).real
rho_F2_prime = np.sum(solsi[i].N[12:17, :], axis=0).real
rho_F3_prime = np.sum(solsi[i].N[17:23,: ], axis=0).real
<<<<<<< HEAD
# Plot curves in individual subplots with corresponding labels
=======
# 在单独的子图中绘制曲线，并添加相应的标签
>>>>>>> origin/main
ax.plot(solsi[i].t* 1e3, rho_F1, linewidth=0.5, label='$\\rho_{F=1}$')
ax.plot(solsi[i].t* 1e3, rho_F2, linewidth=0.5, label='$\\rho_{F=2}$')
ax.plot(solsi[i].t* 1e3, rho_F0, linewidth=0.5, label='$\\rho_{F\'\'=0}$')
ax.plot(solsi[i].t* 1e3, rho_F1_prime, linewidth=0.5, label='$\\rho_{F\'\'=1}$')
ax.plot(solsi[i].t* 1e3, rho_F2_prime, linewidth=0.5, label='$\\rho_{F\'\'=2}$')
ax.plot(solsi[i].t* 1e3, rho_F3_prime, linewidth=0.5, label='$\\rho_{F\'\'=3}$')
ax.set_xlabel('$t (ms)')
ax.yaxis.set_label_coords(1.08, 0.5)
ax.set_ylabel('$\\rho_{ii}$', rotation=0, labelpad=15)
ax.legend(fontsize=7, loc='upper right', ncol=2)
plt.tight_layout()
plt.show()
<<<<<<< HEAD
# Histogram
=======

#直方图
>>>>>>> origin/main
N=np.zeros([24,1001,Natoms_2D])
i=0
for sol in solsi:
    N[:,:,i]=sol.N
    i=i+1
diagonal_sums = np.sum(np.diagonal(N, axis1=0, axis2=1, offset=0), axis=0)
diagonal_sums*10000
fig = plt.figure(figsize=(10.25, 2*2.75), dpi=300)
<<<<<<< HEAD
plt.subplot(2, 2, 1)  # 1st subplot in 2 rows and 1 column
=======
plt.subplot(2, 2, 1)  # 2行1列的第1个子图
>>>>>>> origin/main
plt.bar(range(3), diagonal_sums[:3], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title('F=1')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_1 = ['mF=-1', 'mF=0', 'mF=1']
plt.xticks(range(3), mf_labels_1)
plt.subplot(2, 2, 2)  # 1st subplot in 2 rows and 1 column
=======
# 自定义横坐标标签
mf_labels_1 = ['mF=-1', 'mF=0', 'mF=1']
plt.xticks(range(3), mf_labels_1)

plt.subplot(2, 2, 2)  # 2行1列的第1个子图
>>>>>>> origin/main
plt.bar(range(5), diagonal_sums[3:8], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title('F=2')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_1 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_1)
# Create bar chart - last ten elements (upper level F'=1 and F'=3 states)
plt.subplot(2, 2, 3)  # 2nd subplot in 2 rows and 1 column
=======
# 自定义横坐标标签
mf_labels_1 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_1)
# 制作柱状图 - 后十个元素（上能级的 F'=1 和 F'=3 的态）
plt.subplot(2, 2, 3)  # 2行1列的第2个子图
>>>>>>> origin/main
plt.bar(range(5), diagonal_sums[12:17], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title(r'$F\' =2$')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_2 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_2)
plt.subplot(2, 2, 4)  # 2nd subplot in 2 rows and 1 column
=======
# 自定义横坐标标签
mf_labels_2 = ['mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2']
plt.xticks(range(5), mf_labels_2)
plt.subplot(2, 2, 4)  # 2行1列的第2个子图
>>>>>>> origin/main
plt.bar(range(7), diagonal_sums[16:23], edgecolor='black')
plt.xlabel('mf State')
plt.ylabel('Diagonal Sum')
plt.title(r'$F\' =3$')
<<<<<<< HEAD
# Custom x-axis labels
mf_labels_2 = ['mF=-3','mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2','mF=3']
plt.xticks(range(7), mf_labels_2)
plt.tight_layout()  # Automatically adjust subplot layout to avoid overlap
plt.show()
# Temperature plot
allr = np.concatenate([sol.r[:, :].T for i, sol in enumerate(solsi) if i not in [16, 23]]).T
allv = np.concatenate([sol.v[:, :].T for i, sol in enumerate(solsi) if i not in [16, 23]]).T
img, y_edges, z_edges = np.histogram2d(allr[1, ::100]/k, allr[2, ::100]/k, bins=[np.arange(-5., 5.01, 0.15), np.arange(-5., 5.01, 0.15)])
=======
# 自定义横坐标标签
mf_labels_2 = ['mF=-3','mF=-2','mF=-1', 'mF=0', 'mF=1','mF=2','mF=3']
plt.xticks(range(7), mf_labels_2)
plt.tight_layout()  # 自动调整子图布局，以免重叠
plt.show()

#温度绘图
allr = np.concatenate([sol.r[:, :].T for i, sol in enumerate(solsi) if i not in [16, 23]]).T
allv = np.concatenate([sol.v[:, :].T for i, sol in enumerate(solsi) if i not in [16, 23]]).T
img, y_edges, z_edges = np.histogram2d(allr[1, ::100]/k, allr[2, ::100]/k, bins=[np.arange(-5., 5.01, 0.15), np.arange(-5., 5.01, 0.15)])

>>>>>>> origin/main
fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), dpi=300)
im = ax.imshow(img.T, origin='lower',  
               extent=(np.amin(y_edges), np.amax(y_edges),
                       np.amin(z_edges), np.amax(z_edges)),
               cmap='Blues',
               aspect='equal')
ax.set_xlabel('$y$ (mm)')
ax.set_ylabel('$z$ (mm)')
t_eval =np.linspace(0, tmax_PGC,1001)
vs = np.nan*np.zeros((len(solsi), 3, len(t_eval)))
for vt, sol in zip(vs, solsi):
    vt[:, :sol.v.shape[1]] = sol.v
ejected = [np.bitwise_or( np.abs(sol.r[0, -1]*(1e4*x0))>200,np.abs(sol.r[1, -1]*(1e4*x0))>200 ) for sol in
           solsi]
print('Number of ejected atoms: %d' % np.sum(ejected))
sigma_v = np.nanstd(vs, axis=0)
sigma_v.shape
fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), dpi=300)
ax.plot(t[:1001]*1e3, 2*sigma_v.T[:,:1001]**2*PGC_test.hamiltonian.mass*146)
result=2*sigma_v[1,-1]**2*PGC_test.hamiltonian.mass*146
ax.set_ylabel('$T (\mu k)$')
ax.set_xlabel('$t (ms)$')
<<<<<<< HEAD
ax.set_title(f'Atomic Temperature\nResult: ${result} \mu k$')
plt.legend()
plt.show()
=======
ax.set_title(f'Atomic Temputer\nResult: ${result} \mu k$')
plt.legend()
plt.show()
# %%
>>>>>>> origin/main
