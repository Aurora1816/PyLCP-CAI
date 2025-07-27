#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pylcp
import scipy.constants as cts
from scipy.spatial.transform import Rotation
from pylcp.common import progressBar
from functools import partial
from pylcp.integration_tools import RandomOdeResult
from itertools import repeat
from pathos.pools import ProcessPool
#%%         
class atomic_UP_3DMOTBeams(pylcp.fields.laserBeams):
    def __init__(self, pol=1,k=1, delta=0,phi_i=0,shot=0, rotation_angles=[0., 0., 0.],
                 rotation_spec='XYZ', beam_type=pylcp.fields.laserBeam, **kwargs):
        super().__init__()
        rot_mat = Rotation.from_euler(rotation_spec, rotation_angles).as_matrix()
        kvecs = [np.array([ 1.,  0.,  0.]), np.array([-1.,  0.,  0.]),
                 np.array([ 0.,  1.,  0.]), np.array([ 0., -1.,  0.]),
                 np.array([ 0.,  0.,  1.]), np.array([ 0.,  0., -1.])]
        pols = [-pol,-pol,-pol,-pol,pol,pol]
        deltas=[-shot+delta,shot+delta,-shot+delta,shot+delta,0+delta,0+delta]
        for kvec, pol,delta in zip(kvecs, pols,deltas):
            self.add_laser(beam_type(kvec=rot_mat @ (k*kvec), pol=pol, delta=delta,pol_coord='spherical',**kwargs))

                        
class atomic_UP_process:
    def __init__(self,g_UP,atom,alpha_UP,mass_UP,delta_UP,shot_UP,s_UP,wb_UP,rotation_angles_UP,phi_i_UP,tmax_UP,sols_i_UP):    
        #atom.state[2].gamma linewidth
        self.magField = pylcp.quadrupoleMagneticField(alpha_UP)
        self.tmax=tmax_UP
        self.sols_i=sols_i_UP
        # Hamiltonian for F=0->F=1
        H_g_D2, mu_q_g_D2 = pylcp.hamiltonians.hyperfine_coupled(
        atom.state[0].J, atom.I, atom.state[0].gJ, atom.gI,
        atom.state[0].Ahfs/atom.state[2].gammaHz, Bhfs=0, Chfs=0,
        muB=1)
        H_e_D2, mu_q_e_D2 = pylcp.hamiltonians.hyperfine_coupled(
        atom.state[2].J, atom.I, atom.state[2].gJ, atom.gI,
        Ahfs=atom.state[2].Ahfs/atom.state[2].gammaHz,
        Bhfs=atom.state[2].Bhfs/atom.state[2].gammaHz, Chfs=0,
        muB=1)
        dijq_D2 = pylcp.hamiltonians.dqij_two_hyperfine_manifolds(atom.state[0].J, atom.state[2].J, atom.I)
        E_e_D2 = np.unique(np.diagonal(H_e_D2))
        E_g_D2 = np.unique(np.diagonal(H_g_D2))

        self.hamiltonian = pylcp.hamiltonian(0.05*H_g_D2,H_e_D2, mu_q_g_D2, mu_q_e_D2, dijq_D2,mass=mass_UP)
        self.laserBeams =atomic_UP_3DMOTBeams(delta=E_e_D2[-1]-0.05*E_g_D2[1]+delta_UP,shot=shot_UP,rotation_angles=rotation_angles_UP,s=s_UP,phi_i =phi_i_UP,beam_type=pylcp.fields.gaussianBeam,wb=wb_UP )
        self.eqn =  pylcp.rateeq(self.laserBeams, self.magField, self.hamiltonian,g_UP)
        #self.obe = pylcp.obe(self.laserBeams, self.magField, self.hamiltonian,g_UP, transform_into_re_im=True)
        
  
    def generate_eqn_solution_UP(self,args):
        eqn,additional_param = args
        import numpy as np
        args = ([0, self.tmax], )
        kwargs = {'t_eval':np.linspace(0, self.tmax, 101),
                'random_recoil':True,
                'progress_bar':False,
                ' max_step':1,
                'record_force': True}
        eqn.v0=self.sols_i[additional_param].v[:,-1]
        eqn.r0=self.sols_i[additional_param].r[:,-1]
        if hasattr(self.sols_i[additional_param], 'rho'):
            original_array = self.sols_i[additional_param].rho[:, :, -1]
            eqn.N0=np.real(np.diag(original_array))#new_matrix.ravel()
        else:
            eqn.N0 = self.sols_i[additional_param].N[:,-1]
        eqn.evolve_motion(*args, **kwargs)
        return eqn.sol
"""
    def generate_obe_solution_UP(self,args):
        obe, additional_param = args
        import numpy as np
        args = ([0, self.tmax], )
        kwargs = {'t_eval':np.linspace(0, self.tmax, 1001),
                  'random_recoil':True,
                  'progress_bar':False,
                  ' max_step':1,
                  'record_force': True
                  }

        obe.v0=self.sols_i[additional_param].v[:,-1]
        obe.r0=self.sols_i[additional_param].r[:,-1]
        if hasattr(self.sols_i[additional_param], 'rho'):
            obe.rho0 = self.sols_i[additional_param].rho[:, :, -1]
        else:
            original_array = self.sols_i[additional_param].N[:,-1]
            obe.rho0=np.real(np.diag(original_array).ravel())
        obe.evolve_motion(*args, **kwargs)
        return obe.sol
"""        



    
class atomic_PGC_3DMOTBeams(pylcp.fields.laserBeams):

    def __init__(self, k=1, pol=1,delta=0,shot=0,phi_i=0, rotation_angles=[0., 0., 0.],
                 rotation_spec='XYZ', beam_type=pylcp.fields.laserBeam,pol_coord='spherical', **kwargs):
        super().__init__()
        rot_mat = Rotation.from_euler(rotation_spec, rotation_angles).as_matrix()
        kvecs = [np.array([ 1.,  0.,  0.]), np.array([-1.,  0.,  0.]),
                 np.array([ 0.,  1.,  0.]), np.array([ 0., -1.,  0.]),
                 np.array([ 0.,  0.,  1.]), np.array([ 0.,  0., -1.])]
        pols =  [-pol,pol,-pol,pol,pol,-pol]
        '''
        [-np.array([ 0,  0,  1.]), -np.array([0, np.cos(phi_i), np.sin(phi_i)]),
                -np.array([0., 0., 1.]), -np.array([np.cos(phi_i), 0, np.sin(phi_i)]), 
                +np.array([ 0,   1., 0]), +np.array([np.cos(phi_i), np.sin(phi_i),0 ])]
        '''
        deltas=[-2*shot+delta,2*shot+delta,-2*shot+delta,2*shot+delta,0+delta,0+delta]
        for kvec, pol,delta in zip(kvecs, pols,deltas):
            self.add_laser(beam_type(kvec=rot_mat @ (k*kvec), pol=pol, delta=delta,pol_coord=pol_coord,**kwargs))
            
class atomic_PGC_process:
    def __init__(self,g_PGC,atom,alpha_PGC,delta_PGC,mass_PGC,shot_PGC,s_PGC,wb_PGC,rotation_angles_PGC,phi_i_PGC,tmax_PGC,sols_i_PGC):      
        self.magField = pylcp.quadrupoleMagneticField(alpha_PGC)
        self.tmax=tmax_PGC
        self.sols_i=sols_i_PGC
        # Hamiltonian for F=0->F=1
        H_g_D2, mu_q_g_D2 = pylcp.hamiltonians.hyperfine_coupled(
        atom.state[0].J, atom.I, atom.state[0].gJ, atom.gI,
        atom.state[0].Ahfs/atom.state[2].gammaHz, Bhfs=0, Chfs=0,
        muB=1)
        H_e_D2, mu_q_e_D2 = pylcp.hamiltonians.hyperfine_coupled(
        atom.state[2].J, atom.I, atom.state[2].gJ, atom.gI,
        Ahfs=atom.state[2].Ahfs/atom.state[2].gammaHz,
        Bhfs=atom.state[2].Bhfs/atom.state[2].gammaHz, Chfs=0,
        muB=1)
        dijq_D2 = pylcp.hamiltonians.dqij_two_hyperfine_manifolds(atom.state[0].J, atom.state[2].J, atom.I)
        E_e_D2 = np.unique(np.diagonal(H_e_D2))
        E_g_D2 = np.unique(np.diagonal(H_g_D2))
        self.hamiltonian = pylcp.hamiltonian(0.05*H_g_D2,H_e_D2, mu_q_g_D2, mu_q_e_D2, dijq_D2,mass=mass_PGC)
        self.laserBeams =atomic_PGC_3DMOTBeams(delta=E_e_D2[-1]-0.05*E_g_D2[1]+delta_PGC,shot=shot_PGC,rotation_angles=rotation_angles_PGC,s=s_PGC,phi_i =phi_i_PGC,beam_type=pylcp.fields.gaussianBeam,wb=wb_PGC )
        self.eqn =  pylcp.rateeq(self.laserBeams, self.magField, self.hamiltonian,g_PGC)
        self.obe = pylcp.obe(self.laserBeams, self.magField, self.hamiltonian,g_PGC, transform_into_re_im=True)
        
    def generate_eqn_solution_PGC(self,args):
        eqn,additional_param = args
        import numpy as np
        args = ([0, self.tmax], )
        kwargs = {'t_eval':np.linspace(0, self.tmax, 101),
                  'random_recoil':True,
                  'progress_bar':True,
                  ' max_step':1,
                  'record_force': True}
        eqn.v0=self.sols_i[additional_param].v[:,-1]#-np.mean(np.array([sol.v[:,-1] for sol in self.sols_i]),axis=0)
        eqn.r0=self.sols_i[additional_param].r[:,-1]

        if hasattr(self.sols_i[additional_param], 'rho'):
            original_array =self.sols_i[additional_param].rho[:, :, -1]
            eqn.N0=np.real(np.diag(original_array))#new_matrix.ravel()
        else:
            eqn.N0=self.sols_i[additional_param].N[:,-1]
        eqn.evolve_motion(*args, **kwargs)
        return eqn.sol  
    
    def generate_obe_solution_PGC(self,args):
        obe,additional_param = args
        import numpy as np
        args = ([0, self.tmax], )
        kwargs = {'t_eval':np.linspace(0, self.tmax, 101),
                  'random_recoil':False,
                  'progress_bar':True,
                  ' max_step':1,
                  'record_force': True}
        obe.v0=self.sols_i[additional_param].v[:,-1]#-np.mean(np.array([sol.v[:,-1] for sol in self.sols_i]),axis=0)
        obe.r0=self.sols_i[additional_param].r[:,-1]
        if hasattr(self.sols_i[additional_param], 'rho'):
            obe.rho0 = self.sols_i[additional_param].rho[:, :, -1]
        else:
            original_array = self.sols_i[additional_param].N[:,-1]
            obe.rho0=np.real(np.diag(original_array).ravel())
        obe.evolve_motion(*args, **kwargs)
        return obe.sol 
    def generate_obe_solution_PGC1(self,args):
        obe= args
        import numpy as np
        args = ([0, self.tmax], )
        kwargs = {'t_eval':np.linspace(0, self.tmax, 101),
                  'random_recoil':False,
                  'progress_bar':True,
                  ' max_step':1,
                  'record_force': False}
        obe.v0=self.sols_i[0].v[:,-1]#-np.mean(np.array([sol.v[:,-1] for sol in self.sols_i]),axis=0)
        obe.r0=self.sols_i[0].r[:,-1]
        if hasattr(self.sols_i[0], 'rho'):
            obe.rho0 = self.sols_i[0].rho[:, :, -1]
        else:
            original_array = self.sols_i[0].N[:,-1]
            obe.rho0=np.real(np.diag(original_array).ravel())
        obe.evolve_motion(*args, **kwargs)
        return obe.sol 
# %%
