import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pylcp.governingeq import governingeq
import pylcp
from pylcp.fields import laserBeams as laserBeamsObject
# Function to calculate the Hamiltonian matrix
import numpy as np
import copy
from pylcp.common import progressBar

class Bragg(object):
<<<<<<< HEAD
    def __init__(self, hbar, M, v_r, laserBeams, omega_r, tmin, g, delta, alpha, k, phase, Delta=3e9, beamtype='guass', rho=None, tau=None):

        self.laserBeams = {}  # Laser beams are meant to be dictionary,
        #self.laserBeams['g->e'].delta(np.array([0.]))
        if isinstance(laserBeams, list):
            self.laserBeams['g->e'] = copy.copy(laserBeamsObject(laserBeams))  # Assume label is g->e
        elif isinstance(laserBeams, laserBeamsObject):
            self.laserBeams['g->e'] = copy.copy(laserBeams)  # Again, assume label is g->e
=======
    def __init__(self,hbar,M,v_r,laserBeams,omega_r,tmin,g,delta,alpha, k,phase,Delta =3e9,beamtype='guass',rho = None,tau=None):

        self.laserBeams = {} # Laser beams are meant to be dictionary,
        #self.laserBeams['g->e'].delta(np.array([0.]))
        if isinstance(laserBeams, list):
            self.laserBeams['g->e'] = copy.copy(laserBeamsObject(laserBeams)) # Assume label is g->e
        elif isinstance(laserBeams, laserBeamsObject):
            self.laserBeams['g->e'] = copy.copy(laserBeams) # Again, assume label is g->e
>>>>>>> origin/main
        elif isinstance(laserBeams, dict):
            for key in laserBeams.keys():
                if not isinstance(laserBeams[key], laserBeamsObject):
                    raise TypeError('Key %s in dictionary lasersBeams ' % key +
<<<<<<< HEAD
                                    'is not of type laserBeams.')
            self.laserBeams = copy.copy(laserBeams)  # Now, assume that everything is the same.
=======
                                     'is in not of type laserBeams.')
            self.laserBeams = copy.copy(laserBeams) # Now, assume that everything is the same.
>>>>>>> origin/main
        else:
            raise TypeError('laserBeams is not a valid type.')

        # Constants
<<<<<<< HEAD
        self.phase = phase
=======
        self.phase=phase
>>>>>>> origin/main
        self.tmin = tmin
        self.hbar = hbar  # Planck's constant (Js)
        self.k = k
        self.M = M  # Mass of Rb87 F=2 atom (kg)
        self.v_r = v_r  # Recoil velocity (m/s)
<<<<<<< HEAD
        self.g = g  # Acceleration due to gravity (m/s^2)
        self.delta = delta
        self.beamtype = beamtype
        self.alpha = alpha
        # Parameters
        self.omega_r = omega_r  # Angular frequency of the external magnetic field (rad/s)
        if rho is None:
            self.rho = 0.188 / np.linalg.norm(self.omega_r)
        else:
            self.rho = rho
        self.Delta = Delta

        if tau is None:
            self.tau = 0.36 / np.linalg.norm(self.omega_r)
        else:
            self.tau = tau

        pass

    def hamiltonian_matrix(self, t, n_cutoff, r, v0):
        if n_cutoff % 2 == 0:
            size = 2 * n_cutoff + 1
            H_tilde = np.zeros((size, size), dtype=np.complex128)
            By = np.array([0, 0, 1])
            delta_v = np.max(np.abs(self.laserBeams['g->e'].kvec()[0] * v0 * 1.6e7))
            g_0 = np.dot(np.abs(self.laserBeams['g->e'].kvec()[0]) * 1.6e7, self.g)
            if self.beamtype == 'guass':
                for key in self.laserBeams.keys():
                    Eq = self.laserBeams[key].total_electric_field(r, t)
                    phase = self.laserBeams[key].beam_vector[1].phase(t)
                    for ii, q in enumerate(np.arange(-1., 2., 1)):
                        for n in range(size):
                            if n + 1 < size:
                                #delta_v
                                #print(np.abs((-1.)**q*Eq[2-ii])**2*Bz[ii])
                                H_tilde[n, n + 1] += (np.abs((-1.)**q * Eq[2 - ii])**2 * By[ii] * self.omega_eff_guass(t) / 2) * np.exp(1j * 2 * ((self.delta + delta_v) * t + 0.5 * g_0 * (t**2) - 0.5 * self.alpha * 2 * np.pi * (t**2))) * np.exp(-1j * 4 * (2 * (n - n_cutoff + 1) - 1) * self.omega_r * t) * np.exp(1j * self.phase)
                                H_tilde[n + 1, n] += (np.abs((-1.)**q * Eq[2 - ii])**2 * By[ii] * self.omega_eff_guass(t) / 2) * np.exp(-1j * 2 * ((self.delta + delta_v) * t + 0.5 * g_0 * (t**2) - 0.5 * self.alpha * 2 * np.pi * (t**2))) * np.exp(1j * 4 * (2 * (n - n_cutoff) + 1) * self.omega_r * t) * np.exp(1j * phase)  #*np.exp(1j*(((np.pi/4/(t+1e-8))*(1/delta_v-1/delta_v1))/self.k/self.T/self.T*1e8))

            elif self.beamtype == 'squre':
                for key in self.laserBeams.keys():
                    Eq = self.laserBeams[key].total_electric_field(r, t)
                    phase = self.laserBeams[key].beam_vector[1].phase(t)
                    for ii, q in enumerate(np.arange(-1., 2., 1)):
                        for n in range(size):
                            if n + 1 < size:
                                #print(np.abs((-1.)**q*Eq[2-ii])**2*Bz[ii])
                                H_tilde[n, n + 1] += (np.abs((-1.)**q * Eq[2 - ii])**2 * By[ii] * self.omega_eff_squre(t) / 2) * np.exp(1j * 2 * ((self.delta + delta_v) * t + 0.5 * g_0 * (t**2) - 0.5 * self.alpha * 2 * np.pi * (t**2))) * np.exp(-1j * 4 * (2 * (n - n_cutoff + 1) - 1) * self.omega_r * t) * np.exp(1j * self.phase)  #*np.exp(1j*self.phase)*np.exp(1j*(((np.pi/4/(t+1e-8))*(1/delta_v-1/delta_v1))/self.k/self.T/self.T*1e8))
                                H_tilde[n + 1, n] += (np.abs((-1.)**q * Eq[2 - ii])**2 * By[ii] * self.omega_eff_squre(t) / 2) * np.exp(-1j * 2 * ((self.delta + delta_v) * t + 0.5 * g_0 * (t**2) - 0.5 * self.alpha * 2 * np.pi * (t**2))) * np.exp(1j * 4 * (2 * (n - n_cutoff) + 1) * self.omega_r * t) * np.exp(1j * phase)  #*np.exp(1j*self.phase)*np.exp(1j*(((np.pi/4/(t+1e-8))*(1/delta_v-1/delta_v1))/self.k/self.T/self.T*1e8))

        else:
            print(f"The number {n_cutoff} is not even.")
        return H_tilde

    # Delta function
    @staticmethod
    def delta_function(k):
        return 1 if k == 0 else 0

    def omega_eff_squre(self, t):
        if 0 < abs(t) < self.tau / 2:
            Isat = 1.6
            Gamma = 6.07e6

            Omega_bar = Gamma**2 / 4 / self.Delta

        else:
            Omega_bar = 0
        return Omega_bar  # Example, you may replace this with the actual expression

    def omega_eff_guass(self, t):
        Isat = 1.6
        Gamma = 6.07e6

        if 0 < abs(t) < self.tau / 2:
            Omega_bar = Gamma**2 / (4 * self.Delta) * np.exp(-np.pi * (t - self.tmin)**2 / (self.rho**2))
            #print(Omega_bar)
        else:
            Omega_bar = 0
        return Omega_bar  # Example, you may replace this with the actual expression

    # Effective Rabi frequency function

=======
        self. g = g # Acceleration due to gravity (m/s^2)
        self.delta=delta
        self.beamtype=beamtype
        self.alpha=alpha
        # Parameters
        self.omega_r = omega_r  # Angular frequency of the external magnetic field (rad/s)
        if rho ==   None:
            self.rho = 0.188/np.linalg.norm(self.omega_r)
        else:
            self.rho = rho
        self.Delta =Delta
    
        if tau ==   None:
            self.tau = 0.36/np.linalg.norm(self.omega_r)
        else:
            self.tau = tau
    
        pass
    
 
    def hamiltonian_matrix(self,t, n_cutoff,r,v0):
        if n_cutoff % 2 == 0:
            size = 2*n_cutoff + 1  
            H_tilde = np.zeros((size, size), dtype=np.complex128)
            By=np.array([0,0,1])
            delta_v = np.max(np.abs(self.laserBeams['g->e'].kvec()[0]*v0 * 1.6e7))
            g_0= np.dot(np.abs(self.laserBeams['g->e'].kvec()[0])* 1.6e7,self.g)
            if self.beamtype == 'guass':
                for key in self.laserBeams.keys():
                    Eq = self.laserBeams[key].total_electric_field(r, t)
                    phase=self.laserBeams[key].beam_vector[1].phase(t)
                    for ii, q in enumerate(np.arange(-1., 2., 1)):
                        for n in range(size):
                            if n+1 <  size:
                                #delta_v
                                #print(np.abs((-1.)**q*Eq[2-ii])**2*Bz[ii])
                                H_tilde[n, n+1]+= (np.abs((-1.)**q*Eq[2-ii])**2*By[ii]*self.omega_eff_guass((t)) /2) * np.exp(1j *2* ((self.delta + delta_v )* (t)+0.5*g_0*((t)**2)-0.5*self.alpha*2*np.pi*((t)**2))) * np.exp(-1j * 4*(2*(n-n_cutoff+1)-1)  *self.omega_r * (t))*np.exp(1j*self.phase)
                                H_tilde[n+ 1, n]+= (np.abs((-1.)**q*Eq[2-ii])**2*By[ii]*self.omega_eff_guass((t)) /2) * np.exp(-1j * 2*((self.delta + delta_v )* (t)+0.5*g_0*((t)**2)-0.5*self.alpha*2*np.pi*((t)**2))) * np.exp(1j * 4 *(2*(n-n_cutoff)+1) *self.omega_r * (t))*np.exp(1j*phase)#*np.exp(1j*(((np.pi/4/(t+1e-8))*(1/delta_v-1/delta_v1))/self.k/self.T/self.T*1e8))

            elif  self.beamtype == 'squre':
                for key in self.laserBeams.keys():
                    Eq = self.laserBeams[key].total_electric_field(r, t)
                    phase=self.laserBeams[key].beam_vector[1].phase(t)
                    for ii, q in enumerate(np.arange(-1., 2., 1)):
                        for n in range(size):
                            if n+1 <  size:
                                #print(np.abs((-1.)**q*Eq[2-ii])**2*Bz[ii])
                                H_tilde[n, n+1]+= (np.abs((-1.)**q*Eq[2-ii])**2*By[ii]*self.omega_eff_squre((t)) /2) * np.exp(1j *2* ((self.delta +delta_v )* (t)+0.5*g_0*((t)**2)-0.5*self.alpha*2*np.pi*((t)**2))) * np.exp(-1j * 4*(2*(n-n_cutoff+1)-1)  *self.omega_r * (t))*np.exp(1j*self.phase)#*np.exp(1j*self.phase)*np.exp(1j*(((np.pi/4/(t+1e-8))*(1/delta_v-1/delta_v1))/self.k/self.T/self.T*1e8))
                                H_tilde[n+ 1, n]+= (np.abs((-1.)**q*Eq[2-ii])**2*By[ii]*self.omega_eff_squre((t)) /2) * np.exp(-1j * 2*((self.delta+delta_v  )* (t)+0.5*g_0*((t)**2)-0.5*self.alpha*2*np.pi*((t)**2))) * np.exp(1j * 4 *(2*(n-n_cutoff)+1) *self.omega_r * (t))*np.exp(1j*phase)#*np.exp(1j*self.phase)*np.exp(1j*(((np.pi/4/(t+1e-8))*(1/delta_v-1/delta_v1))/self.k/self.T/self.T*1e8))
                
        else:
            print(f"The number {n} is not even.")   
        return H_tilde

    # Delta function
    def delta_function(k):
        return 1 if k == 0 else 0
    def omega_eff_squre(self,t):
        if 0< abs(t) <self.tau/ 2:
            Isat = 1.6
            Gamma = 6.07e6
        
            Omega_bar=Gamma**2/4/self.Delta
           
        else: 
            Omega_bar=0
        return Omega_bar  # Example, you may replace this with the actual expression
    def omega_eff_guass(self,t):
        Isat = 1.6
        Gamma = 6.07e6
 
        if 0< abs(t) <self.tau/ 2:
            Omega_bar=Gamma**2/(4*self.Delta)*np.exp(-np.pi*( t-self.tmin)**2 / (self.rho**2))
            #print(Omega_bar)
        else: 
            Omega_bar=0
        return Omega_bar  # Example, you may replace this with the actual expression
    # Effective Rabi frequency function
   
>>>>>>> origin/main
    def __reshape_sol(self):
        """
        Reshape the solution to have all the proper parts.
        """
<<<<<<< HEAD

        self.sol.N = self.sol.y[:-6]  #self.sol.y[:-6].reshape(self.hamiltonian.n, self.hamiltonian.n,self.sol.y[:-6].shape[1])
        self.sol.r = np.real(self.sol.y[-3:])
        self.sol.v = np.real(self.sol.y[-6:-3])

    def __calc_force(self, N):
        # This is an example, you need to calculate based on the actual Hamiltonian matrix and external field
        prob_density = np.abs(N)**2  # Momentum state probability density
        n_max = (len(N) - 1) // 2
        force = np.sum(prob_density * (np.arange(-n_max, n_max + 1) * self.hbar)) * 2 * np.sum(abs(self.laserBeams['g->e'].kvec()), 0)
        return force

    def evolve_motion(self, args):
        initial_state, sols_i, t_span, progress_bar, kwargs, a = args
=======
        
        self.sol.N =self.sol.y[:-6]#self.sol.y[:-6].reshape(self.hamiltonian.n, self.hamiltonian.n,self.sol.y[:-6].shape[1])
        self.sol.r = np.real(self.sol.y[-3:])
        self.sol.v = np.real(self.sol.y[-6:-3])
    def __calc_force(self, N):
        # 这里是一个示例，你需要根据实际的哈密顿矩阵和外场计算
        prob_density = np.abs(N)**2  # 动量态概率密度
        n_max = (len(N) - 1) // 2
        force = np.sum(prob_density * (np.arange(-n_max, n_max+1 ) * self.hbar)) *2* np.sum(abs(self.laserBeams['g->e'].kvec()),0)
        return force    

    def evolve_motion(self,args):
        initial_state,sols_i,t_span,progress_bar,kwargs,a=args
>>>>>>> origin/main

        if progress_bar:
            progress = progressBar()
        # Schrödinger and classical equations
<<<<<<< HEAD

        def dydt(t, state, alpha=0):
            if progress_bar and (t) <= t_span[1]:
                progress.update((t) / t_span[1])
            # Split the state density into velocity and position
            N = state[:-6]
            v0 = state[-6:-3]
            r = state[-3:]
            self.n = int((len(N) - 1) / 2)

            # Calculate the Hamiltonian matrix
            H_tilde = self.hamiltonian_matrix(t, self.n, r, v0)

            dN_dt = np.zeros(state[:-6].shape, dtype=np.complex128)
            # Calculate the differential equation of the state density

            dN_dt = -1j * H_tilde @ N
            # Calculate the differential equation of the velocity
            dv_dt = self.__calc_force(N) / self.M + self.g  # You need to implement this function

            # Calculate the differential equation of the position
            dr_dt = v0  # You need to implement this function

            # Combine the differential equations into an array
            dstate_dt = np.concatenate([dN_dt, dv_dt, dr_dt])

            return dstate_dt

        # Solve the system of equations
        self.r0 = sols_i[a].r[:, -1]
        self.v0 = sols_i[a].v[:, -1]
        self.N0 = initial_state
        self.sol = solve_ivp(dydt, t_span, np.concatenate([self.N0, self.v0, self.r0]),
=======
     
        def dydt(t, state,alpha=0):
            if progress_bar and (t)<=t_span[1]:
                progress.update((t)/t_span[1])
            # 拆分态密度为速度和位置
            N = state[:-6]
            v0 = state[-6:-3]
            r = state[-3:]
            self.n=int((len(N)-1)/2)
        
            # 计算 Hamiltonian 矩阵
            H_tilde = self.hamiltonian_matrix(t, self.n, r,v0)

            dN_dt=np.zeros(state[:-6].shape,dtype=np.complex128)
            # 计算态密度的微分方程
            
            dN_dt= -1j *H_tilde@ N
            # 计算速度的微分方程
            dv_dt =self.__calc_force(N)/self.M + self. g# 你需要实现这个函数
        
            # 计算位置的微分方程
            dr_dt =v0# 你需要实现这个函数

            # 将微分方程组合成一个数组
            dstate_dt = np.concatenate([dN_dt, dv_dt, dr_dt])

            return dstate_dt
        # Solve the system of equations
        self.r0= sols_i[a].r[:,-1]
        self.v0= sols_i[a].v[:,-1]
        self.N0=initial_state 
        self.sol= solve_ivp(dydt, t_span, np.concatenate([self.N0, self.v0, self.r0]),
>>>>>>> origin/main
                             **kwargs)
        if progress_bar:
            # Just in case the solve_ivp_random terminated due to an event.
            progress.update(1.)
        self.__reshape_sol()
        return self.sol
<<<<<<< HEAD

    def evolve_motion1(self, args):
        sols_i, t_span, progress_bar, kwargs, a = args
=======
    def evolve_motion1(self,args):
        sols_i,t_span,progress_bar,kwargs,a=args
>>>>>>> origin/main

        if progress_bar:
            progress = progressBar()
        # Schrödinger and classical equations
<<<<<<< HEAD

        def dydt(t, state, alpha=0):
            if progress_bar and (t) <= t_span[1]:
                progress.update((t) / t_span[1])
            # Split the state density into velocity and position
            N = state[:-6]
            v0 = state[-6:-3]
            r = state[-3:]
            self.n = int((len(N) - 1) / 2)

            # Calculate the Hamiltonian matrix
            H_tilde = self.hamiltonian_matrix(t, self.n, r, v0)

            dN_dt = np.zeros(state[:-6].shape, dtype=np.complex128)
            # Calculate the differential equation of the state density

            dN_dt = -1j * H_tilde @ N
            # Calculate the differential equation of the velocity
            dv_dt = self.__calc_force(N) / self.M + self.g  # You need to implement this function

            # Calculate the differential equation of the position
            dr_dt = v0  # You need to implement this function

            # Combine the differential equations into an array
            dstate_dt = np.concatenate([dN_dt, dv_dt, dr_dt])

            return dstate_dt

        self.N0 = sols_i[a].N[:, -1]
        self.v0 = sols_i[a].v[:, -1]
        self.r0 = sols_i[a].r[:, -1]
        # Solve the system of equations
        self.sol = solve_ivp(dydt, t_span, np.concatenate([self.N0, self.v0, self.r0]),
=======
     
        def dydt(t, state,alpha=0):
            if progress_bar and (t)<=t_span[1]:
                progress.update((t)/t_span[1])
            # 拆分态密度为速度和位置
            N = state[:-6]
            v0 = state[-6:-3]
            r = state[-3:]
            self.n=int((len(N)-1)/2)
      
            # 计算 Hamiltonian 矩阵
            H_tilde = self.hamiltonian_matrix(t, self.n, r,v0)

            dN_dt=np.zeros(state[:-6].shape,dtype=np.complex128)
            # 计算态密度的微分方程
            
            dN_dt= -1j *H_tilde@ N
            # 计算速度的微分方程
            dv_dt =self.__calc_force(N)/self.M + self. g# 你需要实现这个函数
        
            # 计算位置的微分方程
            dr_dt =v0# 你需要实现这个函数

            # 将微分方程组合成一个数组
            dstate_dt = np.concatenate([dN_dt, dv_dt, dr_dt])

            return dstate_dt
        self.N0= sols_i[a].N[:,-1]
        self.v0= sols_i[a].v[:,-1]
        self.r0= sols_i[a].r[:,-1]
        # Solve the system of equations
        self.sol= solve_ivp(dydt, t_span, np.concatenate([self.N0, self.v0, self.r0]),
>>>>>>> origin/main
                             **kwargs)
        if progress_bar:
            # Just in case the solve_ivp_random terminated due to an event.
            progress.update(1.)
        self.__reshape_sol()
        return self.sol
<<<<<<< HEAD

    def evolve_motion2(self, args):
        sols_i, t_span, progress_bar, kwargs = args
=======
        """
        self.laserBeams = {}
        self.laserBeams= pylcp.laserBeams([
        {'kvec':kvec, 'pol':np.array([np.cos(phi_i/2), np.sin(phi_i/2), 0.]),
            'pol_coord':'cartesian', 'delta':det, 's':s,'phase':0,'wb':wb,'rs':rs},
        {'kvec':-kvec, 'pol':np.array([np.cos(phi_i/2), np.sin(phi_i/2), 0.]),
            'pol_coord':'cartesian', 'delta':det, 's':s,'phase':np.pi/2,'wb':wb,'rs':rs},
        ], beam_type=pylcp.fields.clippedGaussianBeam)
        #self.laserBeams['g->e'].delta(np.array([0.]))
        self.laserBeams=laserBeams
        self.laserBeams = {} # Laser beams are meant to be dictionary,
        #self.laserBeams['g->e'].delta(np.array([0.]))
        if isinstance(laserBeams, list):
            self.laserBeams['g->e'] = copy.copy(laserBeamsObject(laserBeams)) # Assume label is g->e
        elif isinstance(laserBeams, laserBeamsObject):
            self.laserBeams['g->e'] = copy.copy(laserBeams) # Again, assume label is g->e
        elif isinstance(laserBeams, dict):
            for key in laserBeams.keys():
                if not isinstance(laserBeams[key], laserBeamsObject):
                    raise TypeError('Key %s in dictionary lasersBeams ' % key +
                                     'is in not of type laserBeams.')
            self.laserBeams = copy.copy(laserBeams) # Now, assume that everything is the same.
        else:
            raise TypeError('laserBeams is not a valid type.')
        self.delta=delta
        # Constants
        self.hbar = hbar  # Planck's constant (Js)
        self.k = k
        self.tmin=tmin
        self.M = M  # Mass of Rb87 F=2 atom (kg)
        self.v_r = v_r  # Recoil velocity (m/s)
        self. g = g # Acceleration due to gravity (m/s^2)
        self.beamtype=beamtype
        self.alpha=alpha
        """
    def evolve_motion2(self,args):
        sols_i,t_span,progress_bar,kwargs=args
>>>>>>> origin/main

        if progress_bar:
            progress = progressBar()
        # Schrödinger and classical equations
<<<<<<< HEAD

        def dydt(t, state, alpha=0):
            if progress_bar and (t) <= t_span[1]:
                progress.update((t) / t_span[1])
            # Split the state density into velocity and position
            N = state[:-6]
            v0 = state[-6:-3]
            r = state[-3:]
            self.n = int((len(N) - 1) / 2)

            # Calculate the Hamiltonian matrix
            H_tilde = self.hamiltonian_matrix(t, self.n, r, v0, t_span)

            dN_dt = np.zeros(state[:-6].shape, dtype=np.complex128)
            # Calculate the differential equation of the state density

            dN_dt = -1j * H_tilde @ N
            # Calculate the differential equation of the velocity
            dv_dt = self.__calc_force(N) / self.M + self.g  # You need to implement this function

            # Calculate the differential equation of the position
            dr_dt = v0  # You need to implement this function

            # Combine the differential equations into an array
            dstate_dt = np.concatenate([dN_dt, dv_dt, dr_dt])

            return dstate_dt

        self.N0 = sols_i.N[:, -1]
        self.v0 = sols_i.v[:, -1]
        self.r0 = sols_i.r[:, -1]
        # Solve the system of equations
        self.sol = solve_ivp(dydt, t_span, np.concatenate([self.N0, self.v0, self.r0]),
=======
     
        def dydt(t, state,alpha=0):
            if progress_bar and (t)<=t_span[1]:
                progress.update((t)/t_span[1])
            # 拆分态密度为速度和位置
            N = state[:-6]
            v0 = state[-6:-3]
            r = state[-3:]
            self.n=int((len(N)-1)/2)
      
            # 计算 Hamiltonian 矩阵
            H_tilde = self.hamiltonian_matrix(t,self.n, r,v0,t_span)

            dN_dt=np.zeros(state[:-6].shape,dtype=np.complex128)
            # 计算态密度的微分方程
            
            dN_dt= -1j *H_tilde@ N
            # 计算速度的微分方程
            dv_dt =self.__calc_force(N)/self.M + self. g# 你需要实现这个函数
        
            # 计算位置的微分方程
            dr_dt =v0# 你需要实现这个函数

            # 将微分方程组合成一个数组
            dstate_dt = np.concatenate([dN_dt, dv_dt, dr_dt])

            return dstate_dt
        self.N0= sols_i.N[:,-1]
        self.v0= sols_i.v[:,-1]
        self.r0= sols_i.r[:,-1]
        # Solve the system of equations
        self.sol= solve_ivp(dydt, t_span, np.concatenate([self.N0, self.v0, self.r0]),
>>>>>>> origin/main
                             **kwargs)
        if progress_bar:
            # Just in case the solve_ivp_random terminated due to an event.
            progress.update(1.)
        self.__reshape_sol()
        return self.sol