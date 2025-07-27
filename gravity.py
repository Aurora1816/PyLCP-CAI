
#%%
import numpy as np
from scipy.integrate import solve_ivp
#%%
class emovtion_atom:
    def __init__(self,g,omega):
        self.g=g
        self.omega = omega
    def g_evolution(self,t, y):
        # Gravitational acceleration
        drdt = np.array([y[3], y[4], y[5]])
        # y[0:2] represents the coordinate position, y[3:5] represents the velocity
        dvdt=self.g + np.cross(self.omega, drdt)
        dydt = np.concatenate((drdt, dvdt))
        return dydt
    def v_evolution(self,t, y):
        # Gravitational acceleration
        # y[0:2] represents the coordinate position, y[3:5] represents the velocity
        dydt = [y[3], y[4], y[5], 0, 0, 0]
        return dydt
    def __reshape_sol(self):
        self.sol.r = np.real(self.sol.y[-6:-3])
        self.sol.v = np.real(self.sol.y[-3:])     
        del self.sol.y
    def transform_coordinates_3d_vector(self,R, translation=(0, 0, 0), rotation=(0, 0, 0),evolve_g=None, initial_velocity=(0, 0, 0), initial_Time=0, time_interval=(0, 1), max_step=0.1):
        # Convert angles to radians
        theta_x, theta_y, theta_z = np.radians(rotation)*np.random.randn(3)
        if all(angle == 0 for angle in rotation):
            # If all rotations are zero, directly use the translation matrix
            transformation_matrix = np.array([
                [1, 0, 0, translation[0]],
                [0, 1, 0, translation[1]],
                [0, 0, 1, translation[2]],
                [0, 0, 0, 1]
            ])
        else:
            # Convert angles to radians
            theta_x, theta_y, theta_z = np.radians(rotation)

            # Construct the rotation matrix
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
            # Perform coordinate transformation
            rotation_matrix_combined = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))
            transformation_matrix = np.block([[rotation_matrix_combined, np.reshape(translation, (3, 1))], [0, 0, 0, 1]])

        # Construct the homogeneous coordinate vector
        homogeneous_coordinates = np.concatenate([R, [1]])

        # Add an extra dimension to the homogeneous coordinates
        homogeneous_coordinates = homogeneous_coordinates.reshape((4, 1))

        # Perform coordinate transformation
        transformed_coordinates = np.dot(transformation_matrix, homogeneous_coordinates)

        evolution_time = None
        evolution_result = None

        if evolve_g:
            y0 = np.concatenate([transformed_coordinates[0:3].reshape(3), initial_velocity])  # Initial state, including coordinates and velocity
            t_span = (initial_Time, time_interval[1])
            self.sol = solve_ivp(self.g_evolution, t_span, y0, t_eval=np.linspace(initial_Time, time_interval[1], num=int((time_interval[1]-initial_Time)/max_step)+1), max_step=max_step, method='RK45')
            transformed_coordinates[:3] = self.sol.y[0:3, -1][:, np.newaxis]  # Take the position at the last time point in the evolution process

        else:
            y0 = np.concatenate([transformed_coordinates[0:3].reshape(3), initial_velocity])  # Initial state, including coordinates and velocity
            t_span = (initial_Time, time_interval[1])
            self.sol = solve_ivp(self.v_evolution, t_span, y0, t_eval=np.linspace(initial_Time, time_interval[1], num=int((time_interval[1]-initial_Time)/max_step)+1), max_step=max_step, method='RK45')
            transformed_coordinates[:3] = self.sol.y[0:3, -1][:, np.newaxis]  # Take the position at the last time point in the evolution process

        self.__reshape_sol()
        return self.sol

class Process_atom:
    def process_atom(args):
        sols_i, g, omega,tmax, step, idx = args
        R = sols_i[idx].r[:, -1]
        initial_Time = sols_i[idx].t[-1]
        initial_velocity = sols_i[idx].v[:, -1]
        Beam_location = -np.array([0., 0., 0.])  # Translation vector
        # Create an instance of the 'emovtion_atom' class
        atom_instance = emovtion_atom(g,omega)
        # Perform coordinate transformation
        sol = atom_instance.transform_coordinates_3d_vector(
            R,
            translation=Beam_location,
            rotation=(0, 7.2722e-5*( initial_Time + tmax)*0.866, 0),
            evolve_g=True,
            initial_velocity=initial_velocity,
            initial_Time=initial_Time,
            time_interval=(initial_Time, initial_Time + tmax),
            max_step=tmax/step
        )
        if hasattr(sols_i[idx], 'rho'):
            sol_rho0 = np.real(sols_i[idx].rho[:, :, -1])
        else:
            original_array = sols_i[idx].N[:, -1]
            sol_rho0 = np.real(np.diag(original_array).ravel())
        num_repeats = int(step) + 1
        sol_rho = np.tile(sol_rho0[:, :, np.newaxis], (1, 1, num_repeats))
        sol.rho = sol_rho
        return sol
