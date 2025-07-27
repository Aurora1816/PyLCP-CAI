#%%
import numpy as np
import matplotlib.pyplot as plt
import pylcp
import scipy.constants as cts
from scipy.spatial.transform import Rotation
from pylcp.common import progressBar
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
from scipy.spatial.transform import Rotation
from pylcp.fields import laserBeam
import numpy as np
from scipy.integrate import solve_ivp
import pylcp
import scipy.constants as cts


class emovtion_atom:
    def __init__(self,g):
        self.g=g
    def g_evolution(self,t, y):
        # 重力加速度
        a = np.array([y[3], y[4], y[5]])
        # y[0:2] 表示坐标位置，y[3:5] 表示速度
        dydt = np.concatenate((a, self.g))
        return dydt
    def v_evolution(self,t, y):
        # 重力加速度
        # y[0:2] 表示坐标位置，y[3:5] 表示速度
        dydt = [y[3], y[4], y[5], 0, 0, 0]
        return dydt
    def transform_coordinates_3d_vector(self,R, translation=(0, 0, 0), rotation=(0, 0, 0),evolve_g=None, initial_velocity=(0, 0, 0), time_interval=(0, 1), max_step=0.1):
        # 将角度转换为弧度
        theta_x, theta_y, theta_z = np.radians(rotation)*np.random.randn(3)
        if all(angle == 0 for angle in rotation):
            # 如果旋转都为零，直接使用平移矩阵
            transformation_matrix = np.array([
                [1, 0, 0, translation[0]],
                [0, 1, 0, translation[1]],
                [0, 0, 1, translation[2]],
                [0, 0, 0, 1]
            ])
        else:
            # 将角度转换为弧度
            theta_x, theta_y, theta_z = np.radians(rotation)

            # 构建旋转矩阵
            rotation_matrix_x = np.array([
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]
            ])

            rotation_matrix_y = np.array([
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]
            ])

            rotation_matrix_z = np.array([
                [np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1]
            ])
            # 进行坐标变换
            rotation_matrix_combined = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))
            transformation_matrix = np.block([[rotation_matrix_combined, np.reshape(translation, (3, 1))], [0, 0, 0, 1]])
        # 构建齐次坐标向量
        homogeneous_coordinates = np.concatenate([R, [1]])

        # 在齐次坐标上添加额外的维度
        homogeneous_coordinates = homogeneous_coordinates.reshape((4, 1))
        # 进行坐标变换
        transformed_coordinates = np.dot(transformation_matrix, homogeneous_coordinates)

        evolution_time = None
        evolution_result = None

        if evolve_g:
            # 进行位置演化
            y0 = np.concatenate([transformed_coordinates[0:3].reshape(3), initial_velocity])  # 初始状态，包括坐标和速度
            t_span = (time_interval[0], time_interval[1])
            sol = solve_ivp(self.g_evolution, t_span, y0, t_eval=np.linspace(time_interval[0], time_interval[1], num=int((time_interval[1]-time_interval[0])/max_step)+1), max_step=max_step, method='RK45')
            transformed_coordinates[:3] = sol.y[0:3, -1][:, np.newaxis]  # 取演化过程中最后一个时间点的位置

            # 输出演化时间和结果
            evolution_time = sol.t
            evolution_result = sol.y
        
        else:
            y0 = np.concatenate([transformed_coordinates[0:3].reshape(3), initial_velocity])  # 初始状态，包括坐标和速度
            t_span = (time_interval[0], time_interval[1])
            sol = solve_ivp(self.v_evolution, t_span, y0, t_eval=np.linspace(time_interval[0], time_interval[1], num=int((time_interval[1]-time_interval[0])/max_step)+1), max_step=max_step, method='RK45')
            transformed_coordinates[:3] = sol.y[0:3, -1][:, np.newaxis]  # 取演化过程中最后一个时间点的位置

            # 输出演化时间和结果
            evolution_time = sol.t
            evolution_result = sol.y

        # 返回变换后的坐标和演化结果
        return transformed_coordinates[:3], evolution_time, evolution_result

class radom_create:
    def generate_one_hot_vector():
        probabilities = [0.01, 0.1, 0.05, 0.02, 0.1, 0.3, 0.3, 0.3]
        if len(probabilities) != 8:
            raise ValueError("概率列表的长度应为8。")
        rand_num = np.random.rand()
        selected_position = np.argmax(np.cumsum(probabilities) > rand_num)
        rho0 = np.zeros(8)
        rho0[selected_position] = 1
        return rho0
    def process_atom(args):
        sols_i,g,tmax,max_step,x0,gamma,additional_param= args
        args = ([0, tmax], )
        R = sols_i[additional_param].r[:,-1]* 1e-2 * x0
        initial_velocity = sols_i[additional_param].v[:,-1] *(gamma * 1e-2 * x0)
        Beam_location = -np.array([0.2, 0.3, 0.4])  # 平移矢量
        # 创建 'emovtion_atom' 类的实例
        atom_instance = emovtion_atom(g)
        # 进行坐标变换
        translated_R, sol_g_t, sol = atom_instance.transform_coordinates_3d_vector(
        R,
        translation=Beam_location,
        rotation=(0, 0, 0),
        evolve_g=True,
        initial_velocity=initial_velocity,
        time_interval=(0, tmax),
        max_step=max_step
        )
        # 从结果数组中提取位置和速度分量
        sol_r = sol[:3, :]  # 提取位置（前三行）
        sol_v = sol[-3:, :]  # 提取速度（后三行）
    #    if hasattr(sols_i[additional_param], 'rho'):    
        sol_rho0=np.real(sols_i[additional_param].rho[:,:,-1].ravel())
    #   else:
    #        original_array = sols_i[additional_param].N[:,-1]
    #        sol_rho0=np.real(np.diag(original_array).ravel())
        num_repeats= int(tmax / max_step) + 1
        sol_rho = np.tile(sol_rho0,(num_repeats,1))
        return sol_r, sol_v,sol_rho

class rectangularBeam(laserBeam):
    """
    Rectangular beam

    A beam with spatially constant k-vector and polarization, with a
    rectangular intensity modulation.

    Parameters
    ----------
    kvec : array_like with shape (3,) or callable
        The k-vector of the laser beam, specified as either a three-element
        list or numpy array.
    pol : int, float, array_like with shape (3,), or callable
        The polarization of the laser beam, specified as either an integer, float
        array_like with shape(3,).  If an integer or float,
        if `pol<0` the polarization will be left circular polarized relative to
        the k-vector of the light.  If `pol>0`, the polarization will be right
        circular polarized.  If array_like, polarization will be specified by the
        vector, whose basis is specified by `pol_coord`.
    s : float or callable
        The maximum intensity of the laser beam.
    wb : float
        The width of the rectangular beam.
    hb : float
        The height of the rectangular beam.
    **kwargs:
        Additional keyword arguments.
    """
    def __init__(self, kvec, pol, s,delta, wb, hb, **kwargs):
        if callable(kvec):
            raise TypeError('kvec cannot be a function for a rectangular beam.')

        if callable(pol):
            raise TypeError('Polarization cannot be a function for a rectangular beam.')
        
        super().__init__(kvec=kvec, pol=pol,delta = delta, **kwargs)
        
        self.con_kvec = kvec
        self.con_khat = kvec/np.linalg.norm(kvec)
        self.con_pol = self.pol(np.array([0., 0., 0.]), 0.) 
               
        # Save the parameters specific to the rectangular beam:
        self.s_max = s # maximum intensity
        self.wb = wb # width of the rectangular beam
        self.hb = hb # height of the rectangular beam

        # Use super class to define kvec(R, t), pol(R, t), and delta(t)


        # Other initialization steps
        self.define_rotation_matrix()

    def define_rotation_matrix(self):
        # Angles of rotation:
        th = np.arccos(self.con_khat[2])
        phi = np.arctan2(self.con_khat[1], self.con_khat[0])

        # Use scipy to define the rotation matrix
        self.rmat = Rotation.from_euler('ZY', [phi, th]).inv().as_matrix()

    def intensity(self, R=np.array([0., 0., 0.]), t=0.):
        # Rotate up to the z-axis where we can apply formulas:
        Rp = np.einsum('ij,j...->i...', self.rmat, R)
        
        # Compute intensity
        intensity_x = np.where(np.abs(Rp[0]) <= self.wb/2, 1, 0)
        intensity_y = np.where(np.abs(Rp[1]) <= self.hb/2, 1, 0)
        intensity = self.s_max * intensity_x * intensity_y
        
        return intensity


class detection_module():
    def __init__(self,s,det,k,gamma,mass,g,wb,hb):
        self.position_turn=np.zeros(3)
        x0=1/k
        self.t0=1/gamma
        self.tmax = 0.05/self.t0
        '''
        多进程的情况下，提示无法访问本地变量 'tmax'，因为它没有与值关联。根据你的代码，我认为这可能是由于多进程共享变量导致的问题。
        为了解决这个问题，你可以考虑使用 multiprocessing.Manager 创建一个共享变量。
        '''
        self.g = g
        #self.g=g*self.t0**2/(x0*1e-2)
        wb=wb/x0
        hb = hb/x0
        laserBeams = {}
        laserBeams['r->e'] =pylcp.laserBeams([
            {'kvec':np.array([0., 0., 1]), 'pol':1,
            'pol_coord':'spherical', 'delta':det, 's':s,'phase':0,'wb':wb,'hb':hb,},
            {'kvec':np.array([0., 0., -1]), 'pol':-1,
            'pol_coord':'spherical', 'delta':det, 's':s,'phase':np.pi/2,'wb':wb,'hb':hb,},
            ], beam_type=rectangularBeam)####dect
        
        Hg2, Bgq2 = pylcp.hamiltonians.singleF(F=2, gF=1/2, muB=1.33)
        He3, Beq3 = pylcp.hamiltonians.singleF(F=3, gF=2/3, muB=1.33)
        Hg1, Bgq1 = pylcp.hamiltonians.singleF(F=1, gF=-1/2, muB=1.33)
        dijq23 = pylcp.hamiltonians.dqij_two_bare_hyperfine(2,3)
        hamiltonian = pylcp.hamiltonian()
        hamiltonian.add_H_0_block('g', 0.*Hg1)
        hamiltonian.add_mu_q_block('g',  Bgq1, muB=1.33)
        hamiltonian.add_H_0_block('r', 0*Hg2)
        hamiltonian.add_mu_q_block('r',  Bgq2 , muB=1.33)
        hamiltonian.add_H_0_block('e', 0*He3)
        hamiltonian.add_mu_q_block('e',  Beq3, muB=1.33)
        hamiltonian.add_d_q_block('r','e',dijq23)
        hamiltonian.mass=mass
        self.hamiltonian = hamiltonian
        magField = pylcp.constantMagneticField(np.array([1e-8,1e-8,1e-8]))    
        self.obe = pylcp.obe(laserBeams, magField,hamiltonian,self.g,transform_into_re_im=True)
      
    def update_tmax(self, new_tmax_value):
        # Update the shared tmax value
        self.tmax = new_tmax_value   
    def turn_position(self,position_turn=np.zeros(3)):
        self.position_turn = position_turn 
   
    def generate_random_solution(self,arg_list):
        import numpy as np
        obe,roffset,vscale,voffset,rscale,idx=arg_list
        tmax = self.tmax
        kwargs = {'t_eval':np.linspace(0, tmax, 5001),
                'random_recoil':True,
                'progress_bar':False,
                'max_scatter_probability':0.5,
                'record_force':True}
        rho0= np.zeros(obe.hamiltonian.n,)
        radom_rho0=radom_create.generate_one_hot_vector()
        rho0[0:8]=radom_rho0+0.001
        obe.set_initial_position(rscale*np.random.randn(3) + roffset)
        obe.set_initial_velocity(vscale*np.random.randn(3) + voffset)
        obe.set_initial_rho(np.diag(rho0).ravel())
        obe.evolve_motion(t_span=([0,tmax]), **kwargs)
        return obe.sol
    
    def sol_solution(self,arg_list):
        import numpy as np
        obe,sols_i,idx=arg_list
        tmax = self.tmax
        kwargs = {'t_eval':np.linspace(0, tmax, 5001),
                'random_recoil':True,
                'progress_bar':False,
                'max_scatter_probability':0.5,
                'record_force':True}
        rho0= np.zeros(obe.hamiltonian.n,)
        if idx == -1:
            rho=np.diag(abs(sols_i.rho[:,:,-1]))
            obe.r0=sols_i.r[:,-1]+self.position_turn
            obe.v0=sols_i.v[:,-1]
        else:
            rho=np.diag(abs(sols_i[idx].rho[:,:,-1]))
            obe.r0=sols_i[idx].r[:,-1]+self.position_turn
            obe.v0=sols_i[idx].v[:,-1]

        rho0[0:8]=rho[0:8]
        obe.set_initial_rho(np.diag(rho0).ravel())
        obe.evolve_motion(t_span=([0,tmax]), **kwargs)
        return obe.sol
    
class Blow_module():
    def __init__(self,s,det,k,gamma,mass,g,wb,hb):
        self.position_turn=np.zeros(3)
        x0=1/k
        self.t0=1/gamma
        self.tmax = 0.05/self.t0
        '''
        多进程的情况下，提示无法访问本地变量 'tmax'，因为它没有与值关联。根据你的代码，我认为这可能是由于多进程共享变量导致的问题。
        为了解决这个问题，你可以考虑使用 multiprocessing.Manager 创建一个共享变量。
        '''
        self.g = g
        #self.g=g*self.t0**2/(x0*1e-2)
        wb=wb/x0
        hb = hb/x0
        laserBeams_Blow={}

        laserBeams_Blow['r->e'] =pylcp.laserBeams([
        {'kvec':np.array([0., 0., 1.]), 'pol':1,
        'pol_coord':'spherical', 'delta':det, 's':s,'phase':0,'wb':wb,'hb':hb},
        ], beam_type=rectangularBeam)
        Hg2, Bgq2 = pylcp.hamiltonians.singleF(F=2, gF=1/2, muB=1.33)
        He3, Beq3 = pylcp.hamiltonians.singleF(F=3, gF=2/3, muB=1.33)
        Hg1, Bgq1 = pylcp.hamiltonians.singleF(F=1, gF=-1/2, muB=1.33)
        dijq23 = pylcp.hamiltonians.dqij_two_bare_hyperfine(2,3)
        hamiltonian = pylcp.hamiltonian()
        hamiltonian.add_H_0_block('g', 0.*Hg1)
        hamiltonian.add_mu_q_block('g',  Bgq1, muB=1.33)
        hamiltonian.add_H_0_block('r', 0*Hg2)
        hamiltonian.add_mu_q_block('r',  Bgq2 , muB=1.33)
        hamiltonian.add_H_0_block('e', 0*He3)
        hamiltonian.add_mu_q_block('e',  Beq3, muB=1.33)
        hamiltonian.add_d_q_block('r','e',dijq23)
        hamiltonian.mass=mass
        self.hamiltonian = hamiltonian
        magField = pylcp.constantMagneticField(np.array([1e-8,1e-8,1e-8]))    
        self.obe = pylcp.obe(laserBeams_Blow, magField, self.hamiltonian,self.g,transform_into_re_im=True)
    
    def update_tmax(self, new_tmax_value):
        # Update the shared tmax value
        self.tmax = new_tmax_value    
   
    def turn_position(self,position_turn=np.zeros(3)):
        self.position_turn = position_turn
   
   
    def generate_random_solution(self,arg_list):
        import numpy as np
        rho0,obe,roffset,vscale,voffset,rscale,idx=arg_list
        tmax = self.tmax
        kwargs = {'t_eval':np.linspace(0, tmax, 5001),
                'random_recoil':True,
                'progress_bar':False,
                'max_scatter_probability':0.5,
                'record_force':True}
        rho0= np.zeros(obe.hamiltonian.n,)
        radom_rho0=radom_create.generate_one_hot_vector()
        rho0[0:8]=radom_rho0+0.001
        obe.set_initial_position(rscale*np.random.randn(3) + roffset)
        obe.set_initial_velocity(vscale*np.random.randn(3) + voffset)
        obe.set_initial_rho(np.diag(rho0).ravel())
        obe.evolve_motion(t_span=([0,tmax]), **kwargs)
        return obe.sol
    
    def sol_solution(self,arg_list):
        import numpy as np
        obe,sols_i,idx=arg_list
        tmax = self.tmax
        kwargs = {'t_eval':np.linspace(0, tmax, 5001),
                'random_recoil':True,
                'progress_bar':False,
                'max_scatter_probability':0.5,
                'record_force':True}
        rho0= np.zeros(obe.hamiltonian.n,)
        if idx == -1:
            rho=np.diag(abs(sols_i.rho[:,:,-1]))
            obe.r0=sols_i.r[:,-1]+self.position_turn
            obe.v0=sols_i.v[:,-1]
        else:
            rho=np.diag(abs(sols_i[idx].rho[:,:,-1]))
            obe.r0=sols_i[idx].r[:,-1]+self.position_turn
            obe.v0=sols_i[idx].v[:,-1]

        rho0[0:8]=rho[0:8]
        obe.set_initial_rho(np.diag(rho0).ravel())
        obe.evolve_motion(t_span=([0,tmax]), **kwargs)
        return obe.sol

        
class detection_pumping():
    def __init__(self,s1,s2,det1,det2,k,gamma,mass,g,wb,hb):
        self.position_turn=np.zeros(3)
        x0=1/k
        self.t0=1/gamma
        self.tmax = 0.05/self.t0
        '''
        多进程的情况下，提示无法访问本地变量 'tmax'，因为它没有与值关联。根据你的代码，我认为这可能是由于多进程共享变量导致的问题。
        为了解决这个问题，你可以考虑使用 multiprocessing.Manager 创建一个共享变量。
        '''
        self.g = g
        #self.g=g*self.t0**2/(x0*1e-2)
        wb=wb/x0
        hb = hb/x0
        laserBeams = {}

        laserBeams['r->e'] =pylcp.laserBeams([
            {'kvec':np.array([0., 0., 1.]), 'pol':1,
            'pol_coord':'spherical', 'delta':det2, 's':s2,'phase':0,'wb':wb,'hb':hb},
            {'kvec':np.array([0., 0., -1.]), 'pol':-1,
            'pol_coord':'spherical', 'delta':det2, 's':s2,'phase':np.pi/2,'wb':wb,'hb':hb},
            ], beam_type=rectangularBeam)####dect
        laserBeams['r->c'] =pylcp.laserBeams([
            {'kvec':np.array([0., 0., 1.]), 'pol':1,
            'pol_coord':'spherical', 'delta':det2, 's':s2,'phase':0,'wb':wb,'hb':hb},
            {'kvec':np.array([0., 0., -1.]), 'pol':-1,
            'pol_coord':'spherical', 'delta':det2, 's':s2,'phase':np.pi/2,'wb':wb,'hb':hb},
            ], beam_type=rectangularBeam)####dect
        laserBeams['g->c'] =pylcp.laserBeams([
            {'kvec':np.array([0., 0., 1.]), 'pol':1,
            'pol_coord':'spherical', 'delta':det1, 's':s1,'phase':0,'wb':wb,'hb':hb},
            {'kvec':np.array([0., 0., -1.]), 'pol':-1,
            'pol_coord':'spherical', 'delta':det1, 's':s1,'phase':np.pi/2,'wb':wb,'hb':hb},
            ], beam_type=rectangularBeam)####dect
    
        Hg2, Bgq2 = pylcp.hamiltonians.singleF(F=2, gF=1/2, muB=1.33)
        He3, Beq3 = pylcp.hamiltonians.singleF(F=3, gF=2/3, muB=1.33)
        He1, Beq1 = pylcp.hamiltonians.singleF(F=1, gF=2/3, muB=1.33)
        #相比于detection1新增的
        Hg1, Bgq1 = pylcp.hamiltonians.singleF(F=1, gF=-1/2, muB=1.33)
        dijq23 = pylcp.hamiltonians.dqij_two_bare_hyperfine(2,3)
        dijq11 = pylcp.hamiltonians.dqij_two_bare_hyperfine(1,1)
        dijq21 = pylcp.hamiltonians.dqij_two_bare_hyperfine(2,1)
        hamiltonian = pylcp.hamiltonian()
        hamiltonian.add_H_0_block('g', 0.*Hg1)
        hamiltonian.add_mu_q_block('g',  Bgq1, muB=1.33)
        hamiltonian.add_H_0_block('r', 0*Hg2)
        hamiltonian.add_mu_q_block('r',  Bgq2 , muB=1.33)
        hamiltonian.add_H_0_block('e', 0*He3)
        hamiltonian.add_mu_q_block('e',  Beq3, muB=1.33)
        hamiltonian.add_H_0_block('c', np.eye(3)*-1+He1)
        hamiltonian.add_mu_q_block('c',  Beq1, muB=1.33)
        hamiltonian.add_d_q_block('r','e',dijq23)
        hamiltonian.add_d_q_block('g','c',dijq11)
        hamiltonian.add_d_q_block('r','c',dijq21)
        hamiltonian.mass=mass
        self.hamiltonian = hamiltonian
        magField = pylcp.constantMagneticField(np.array([1e-8,1e-8,1e-8]))    
        self.obe = pylcp.obe(laserBeams, magField,hamiltonian,self.g,transform_into_re_im=True)
      
    def update_tmax(self, new_tmax_value):
        # Update the shared tmax value
        self.tmax = new_tmax_value   
    def turn_position(self,position_turn=np.zeros(3)):
        self.position_turn = position_turn  
   
    def generate_random_solution(self,arg_list):
        import numpy as np
        obe,roffset,vscale,voffset,rscale,idx=arg_list
        tmax = self.tmax
        kwargs = {'t_eval':np.linspace(0, tmax, 5001),
                'random_recoil':True,
                'progress_bar':False,
                'max_scatter_probability':0.5,
                'record_force':True}
        rho0= np.zeros(obe.hamiltonian.n,)
        radom_rho0=radom_create.generate_one_hot_vector()
        rho0[0:8]=radom_rho0+0.001
        obe.set_initial_position(rscale*np.random.randn(3) + roffset)
        obe.set_initial_velocity(vscale*np.random.randn(3) + voffset)
        obe.set_initial_rho(np.diag(rho0).ravel())
        obe.evolve_motion(t_span=([0,tmax]), **kwargs)
        return obe.sol 
    def sol_solution(self,arg_list):
        import numpy as np
        obe,sols_i,idx=arg_list
        tmax = self.tmax
        kwargs = {'t_eval':np.linspace(0, tmax, 5001),
                'random_recoil':True,
                'progress_bar':False,
                'max_scatter_probability':0.5,
                'record_force':True}
        rho0= np.zeros(obe.hamiltonian.n,)
        if idx == -1:
            rho=np.diag(abs(sols_i.rho[:,:,-1]))
            obe.r0=sols_i.r[:,-1]+self.position_turn
            obe.v0=sols_i.v[:,-1]
        else:
            rho=np.diag(abs(sols_i[idx].rho[:,:,-1]))
            obe.r0=sols_i[idx].r[:,-1]+self.position_turn
            obe.v0=sols_i[idx].v[:,-1]

        rho0[0:8]=rho[0:8]
        obe.set_initial_rho(np.diag(rho0).ravel())
        obe.evolve_motion(t_span=([0,tmax]), **kwargs)
        return obe.sol       

class detection_tof_module():
    def lowpass_filter(input_array, cutoff_frequency, order=4):
        """
        使用Butterworth低通滤波器对一维数组进行低通滤波。

        参数：
        - input_array: 输入的一维数组
        - cutoff_frequency: 截止频率，即保留的低频成分的最大频率
        - order: 滤波器阶数，默认为4

        返回：
        - 经过低通滤波后的一维数组
        """
        # 创建低通Butterworth滤波器系数
        b, a = butter(order, cutoff_frequency, btype='low', analog=False)
        # 应用低通滤波器
        filtered_array = lfilter(b, a, input_array)

        return filtered_array
    def efficiency(atoms_position,collection_system_position):
        # 计算每个原子到收集系统的距离
        distances_to_system = np.linalg.norm(atoms_position - collection_system_position, axis=1)

        # 设置荧光收集系统的接收效率，这里简化为与距离的倒数成正比
        efficiency = 1 / distances_to_system
        return efficiency
        
    
    def simulate_voltage(atoms_position, initial_intensity, decay_coefficient, theta_std, gain_coefficient,s,efficiency,
                             collection_system_position = np.array([0, 0, 0.01]),
                            angle=120, D=0.0012,landa1=0.99):
        """
        模拟探测器件电压输出的函数

        参数:
        - atoms_position: 二维NumPy数组，每一行表示一个原子的三维坐标
        - initial_intensity: 一维NumPy数组，每个元素表示对应原子的初始发光强度
        - decay_coefficient: 浮点数，表示发光强度衰减系数
        - theta_std: 浮点数，表示高斯发散函数的标准差，控制发散角度的大小
        - gain_coefficient: 浮点数，表示电压增益系数
        - angle: 探测器件 收集角度，单位为度，默认为 120 度
        - size: 探测器件 尺寸，单位为毫米，默认为 8 毫米
        - D: 透镜焦距
        - landa1: 窗片透射率
        返回:
        - total_voltage: 浮点数，表示模拟的探测器件总电压输出
        -s:本底光强
        """
        def intensity_decay_model(distance, initial_intensity, decay_coefficient,D,landa1,s):
            return initial_intensity * np.exp(-decay_coefficient)*landa1/(D**2)/ 4/np.pi/distance*2*0.23+s*0.000000001
        
        def gaussian_beam_profile(theta, theta_std):
            return np.exp(-(theta**2) / (2 * theta_std**2))
        
        # 计算每个原子到收集系统的距离
        distances_to_system = np.linalg.norm(atoms_position - collection_system_position, axis=1)
        

        # 生成随机发散角度
        theta = np.random.normal(loc=0, scale=theta_std, size=len(atoms_position))

        # 计算每个原子的发光强度，考虑发散角
        atoms_intensity = intensity_decay_model(distances_to_system, initial_intensity, decay_coefficient,D,landa1,s) * \
                        gaussian_beam_profile(theta, theta_std)

        # 计算每个原子在 探测器件 上的投影
        projection = atoms_intensity * efficiency
        
        # 计算探测器件方向的单位矢量
        direction = collection_system_position / np.linalg.norm(collection_system_position)

        # 计算每个原子和探测器件位置之间的角度
        angles = np.arccos(np.clip(np.dot(atoms_position, direction), -1.0, 1.0))

        # 根据探测器件收集角度，计算在收集范围内的原子
        in_range = np.abs(np.degrees(angles)) < angle / 2

        # 模拟将光强转换为电压，考虑所有原子
        total_voltage = np.sum(projection) * gain_coefficient
        #total_voltage=lowpass_filter(total_voltage , cutoff_frequency=0.1, order=4)
        return total_voltage
    
    def position_and_intensity_singletime(sols, i, k, gamma,max_intensities=1e-16):
        atoms_positions = []
        initial_intensities = []
        x0=1/k
        for sol in sols:
            intensity = np.linalg.norm(sol.F[:, i], axis=0) * 3.0e8 / 2 * (cts.hbar * k * gamma)
            if intensity >= max_intensities:
                intensity =0
            initial_intensities.append(intensity)
            atoms_positions.append(sol.r[:, i]* x0/100)
        
        # 转换为NumPy数组
        atoms_positions = np.array(atoms_positions)
        initial_intensities = np.array(initial_intensities)
        
        return atoms_positions, initial_intensities
class gaussiansignal():
    def generate_gaussian_signal(self, x, mean, std_dev, amplitude, background=0.0):
        signal = amplitude * np.exp(-((x - mean) / std_dev)**2 / 2) + background
        return signal

    def fit_gaussian_to_data(self, x, y, initial_params=None):
        # 定义高斯函数作为拟合模型
        def gaussian_function(x, mean, std_dev, amplitude, background):
            return self.generate_gaussian_signal(x, mean, std_dev, amplitude, background)
        # 使用curve_fit进行拟合
        popt, _ = curve_fit(gaussian_function, x, y, p0=initial_params,maxfev=10000)
        # 返回拟合参数
        return popt

    def integrate_gaussian_signal_with_auto_background(self, x, y, params=[0.005, 0.1, 0.01, 0], integration_interval=(0, 0.015), gain=10):
        # 拟合得到高斯信号的参数
        params = self.fit_gaussian_to_data(x, y, initial_params=params)
        # 高斯区域的积分
        integral_signal, _ = quad(lambda x: self.generate_gaussian_signal(x, *params), *integration_interval)
        # 本底区域的积分
        background_interval = (integration_interval[1], integration_interval[1] + (integration_interval[1] - integration_interval[0]) / 2)
        filtered_x = x[(x >= background_interval[0]) & (x <= background_interval[1])]
        filtered_y = y[(x >= background_interval[0]) & (x <= background_interval[1])]
        integral_background = np.trapz(filtered_y, filtered_x)

        integral_result = (integral_signal - 2 * integral_background) * gain

        return integral_result, background_interval, params
# %%
