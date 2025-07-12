import math
import numpy as np
from numpy import linalg as LA

class Car:
    """ Car Class to model dynamics and interactions """

    def __init__(self, z, u, z_other, ssd, s0, T, discount, horizon_interval, lane_boundaries, prediction_horizon, control_horizon):
        self.z = z
        self.u = u
        self.ssd = ssd
        self.s0 = s0
        self.T = T
        self.z_other = z_other
        self.discount = discount
        self.horizon_interval = horizon_interval
        self.lane_boundaries = lane_boundaries
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon

    # 4th order Runge-Kutta for updating states
    def rk4_step(self, z, u, t, interval):
        a, delta = u
        wb = 0.6 * z[5]  # Using length from z to compute wheelbase
        beta = math.atan(0.5 * math.tan(delta))

        def dynamics(z, u):
            _, x, y, v, theta, _, _, _ = z[:8]
            dx = v * math.cos(theta + beta)
            dy = v * math.sin(theta + beta)
            dv = a
            dtheta = (2 / wb) * v * math.sin(beta)
            dz = np.array([0, dx, dy, dv, dtheta, 0, 0, 0])
            return dz

       # Update steps
        k1 = dynamics(z, u)
        k2 = dynamics(z + 0.5 * interval * k1, u)
        k3 = dynamics(z + 0.5 * interval * k2, u)
        k4 = dynamics(z + interval * k3, u)
        z_next = z + (interval / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Update lane based on new y position
        z_next[7] = np.digitize(z_next[2], self.lane_boundaries)

        return z_next

    def get_players(self, z, z_i):
        """ z and z_i have structure: 'id', 'x', 'y', 'v', 'psi', 'width', 'height', 'lane' """

        if z_i.size == 0:
            return np.array([]) 

        # set boundary for the vehicles considered to be potential players
        x_bound = self.ssd # 250 m
        y_bound = 1 # 1 lane
        in_bound = z_i[(np.abs(z_i[:, 1] - z[1]) <= x_bound) & (np.abs(z_i[:, 7] - z[7]) <= y_bound)]

        if in_bound.size == 0:
            return np.array([])  # Return early if no vehicles are in bounds

        # Calculate relative position and velocity of other cars
        dx = in_bound[:, 1] - z[1]
        dlane = in_bound[:, 7] - z[7]

        opponents = np.array([])

        # current lane
        if any(dx[(dlane == 0) & (dx > 0)]):
            preceding_car = in_bound[(dlane == 0) & (dx > 0)][np.argmin(abs(dx[(dlane == 0) & (dx > 0)]))]
            opponents = np.append(opponents, preceding_car)
        if any(dx[(dlane == 0) & (dx < 0)]):
            following_car = in_bound[(dlane == 0) & (dx < 0)][np.argmin(abs(dx[(dlane == 0) & (dx < 0)]))]
            opponents = np.append(opponents, following_car)
        
        # left lane
        if any((dlane == 1) & ~((z[5]/2 < dx-in_bound[:, 5]/2) | (-z[5]/2 > dx+in_bound[:, 5]/2))):
            left_along = in_bound[(dlane == 1) & ~((z[5]/2 < dx-in_bound[:, 5]/2) | (-z[5]/2 > dx+in_bound[:, 5]/2))]
            opponents = np.append(opponents, left_along)
            left_ = in_bound[~np.isin(in_bound.tolist(), left_along.tolist()).all(axis=1)]
        else:
            left_ = in_bound

        left_dx = left_[:, 1] - z[1]
        left_dlane = left_[:, 7] - z[7]

        if any(left_dx[(left_dlane == 1) & (left_dx > 0)]):
            left_preceding = left_[(left_dlane == 1) & (left_dx > 0)][np.argmin(abs(left_dx[(left_dlane == 1) & (left_dx > 0)]))]
            opponents = np.append(opponents, left_preceding)
        if any(left_dx[(left_dlane == 1) & (left_dx < 0)]):
            left_following = left_[(left_dlane == 1) & (left_dx < 0)][np.argmin(abs(left_dx[(left_dlane == 1) & (left_dx < 0)]))]
            opponents = np.append(opponents, left_following)

        # right lane
        if any((dlane == -1) & ~((z[5]/2 < dx-in_bound[:, 5]/2) | (-z[5]/2 > dx+in_bound[:, 5]/2))):
            right_along = in_bound[(dlane == -1) & ~((z[5]/2 < dx-in_bound[:, 5]/2) | (-z[5]/2 > dx+in_bound[:, 5]/2))]
            opponents = np.append(opponents, right_along)
            right_ = in_bound[~np.isin(in_bound.tolist(), right_along.tolist()).all(axis=1)]
        else:
            right_ = in_bound

        right_dx = right_[:, 1] - z[1]
        right_dlane = right_[:, 7] - z[7]

        if any(right_dx[(right_dlane == -1) & (right_dx > 0)]):
            right_preceding = right_[(right_dlane == -1) & (right_dx > 0)][np.argmin(abs(right_dx[(right_dlane == -1) & (right_dx > 0)]))]
            opponents = np.append(opponents, right_preceding)
        if any(right_dx[(right_dlane == -1) & (right_dx < 0)]):
            right_following = right_[(right_dlane == -1) & (right_dx < 0)][np.argmin(abs(right_dx[(right_dlane == -1) & (right_dx < 0)]))]
            opponents = np.append(opponents, right_following)

        return opponents.reshape(int(len(opponents)/len(z)), len(z))

####################################################################################################################################################################################################################################

class Objective(Car):
    """ Objective Class to define the optimization problem """

    def __init__(self, z, u, z_other, ssd, s0, T, T_safe, l0, w0, alphas, discount, horizon_interval, lane_boundaries, prediction_horizon, control_horizon, truck):
        super().__init__(z, u, z_other, ssd, s0, T, discount, horizon_interval, lane_boundaries, prediction_horizon, control_horizon)
        self.l0 = l0
        self.w0 = w0
        self.T_safe = T_safe
        self.alphas = alphas
        self.truck = truck

        self.a_max, self.a_min = (1, -1.5) if self.truck == 1 else (3, -3.4)
        self.delta_max, self.delta_min = (math.pi/12, -math.pi/12) if self.truck == 1 else (math.pi/9, -math.pi/9)

    def TTC(self, z, z_i):
        """ Calculate Time to Collision (TTC) between two vehicles """

        if z_i.size == 0:
            return math.inf

        # Extract positions and velocities
        _, x_i, y_i, v_i, heading_i, l, w, _ = z
        _, x_j, y_j, v_j, heading_j, l_i, w_i, _ = z_i

        # Compute velocities in x and y directions
        v_i_x = v_i * math.cos(heading_i)
        v_i_y = v_i * math.sin(heading_i)
        v_j_x = v_j * math.cos(heading_j)
        v_j_y = v_j * math.sin(heading_j)

        # Compute relative velocities
        vij_x = v_i_x - v_j_x
        vij_y = v_i_y - v_j_y
        vij_norm = math.sqrt(vij_x**2 + vij_y**2) + 1e-6  # Avoid division by zero

        # Perpendicular to relative velocity
        vij_perp_x = -vij_y
        vij_perp_y = vij_x
        vij_perp_norm = math.sqrt(vij_perp_x**2 + vij_perp_y**2)

        # Distance in perpendicular direction
        Dis_perp = abs((x_i * vij_perp_x + y_i * vij_perp_y) / vij_perp_norm -
                    (x_j * vij_perp_x + y_j * vij_perp_y) / vij_perp_norm)

        # Calculate angle phi (direction of relative velocity)
        phi = math.atan2(vij_y, vij_x + 1e-6)

        # Projected widths for both vehicles in perpendicular direction
        sin_theta1 = math.sin(heading_i - phi) # sin(pi - x) = sin(x)
        cos_theta1 = -math.cos(heading_i - phi) # cos(pi - x) = -cos(x)
        w_proj_i_perp = math.sqrt((l)**2/2 * sin_theta1**2 + (w)**2/2 * cos_theta1**2)
        
        sin_theta2 = math.sin(heading_j - phi)
        cos_theta2 = -math.cos(heading_j - phi)
        w_proj_j_perp = math.sqrt((l_i)**2/2 * sin_theta2**2 + (w_i)**2/2 * cos_theta2**2)

        # Check for overlap and relative motion towards each other
        overlap = Dis_perp < (w_proj_i_perp + w_proj_j_perp)
        towards = ((x_j - x_i) * vij_x + (y_j - y_i) * vij_y) > 0

        # Distance in the direction of vij
        Dis_vij = abs((x_i * vij_x + y_i * vij_y) / vij_norm -
                    (x_j * vij_x + y_j * vij_y) / vij_norm)

        # Projected widths for both vehicles in vij direction
        sin_theta1_vij = math.cos(heading_i - phi) # sin(pi/2 - x) = cos(x)
        cos_theta1_vij = math.sin(heading_i - phi) # cos(pi/2 - x) = sin(x)
        w_proj_i_vij = math.sqrt((l)**2/2 * sin_theta1_vij**2 + (w)**2/2 * cos_theta1_vij**2)

        sin_theta2_vij = math.cos(heading_j - phi)
        cos_theta2_vij = math.sin(heading_j - phi)
        w_proj_j_vij = math.sqrt((l_i)**2/2 * sin_theta2_vij**2 + (w_i)**2/2 * cos_theta2_vij**2)

        dij = Dis_vij - (w_proj_i_vij + w_proj_j_vij)
        dij = dij if overlap and towards else math.inf

        return abs(dij) / vij_norm

    # Define the cost function
    def cost(self, z, z_i, u, t):
        """ z and z_i have structure: 'id', 'x', 'y', 'v', 'psi', 'width', 'height', 'lane' """

        # Calculate spacing cost
        def spacing_cost(z, z_i):
            if z_i.size == 0:
                return 0
            
            potential_FC = z_i[z_i[:,1]-z[1] > z_i[:,5]/2+z[5]/2]
            potential_FC = potential_FC[np.abs(potential_FC[:,2]-z[2]) < potential_FC[:,6]/2+z[6]/2+1]

            s = potential_FC[:,1] - z[1] - potential_FC[:,5]/2 - z[5]/2
            s_opt = self.s0 + z[3]*self.T_safe + (z[3]*(z[3]-potential_FC[:,3]))/(2*np.sqrt(self.a_max*-self.a_min))
            return max(np.append(s_opt - s, 0))

        ### Define costs
        # speed cost
        v_eq = self.T*self.a_min+np.sqrt(self.T**2*(self.a_min)**2 - 2*self.a_min*self.ssd)
        L1 = (z[3] - v_eq)**2
        # car following cost
        L2 = spacing_cost(z, z_i)
        # safety cost
        L3 = sum([np.exp(-self.TTC(z, z_i[i])) for i in range(len(z_i))])
        # acceleration cost
        L4 = u[0]**2/(-self.a_min * self.a_max)
        # steering cost
        L5 = u[1]**2/(-self.delta_min * self.delta_max)
        # lane centring cost
        L6 = np.prod([(z[2] - (self.lane_boundaries[c+1]+self.lane_boundaries[c])/2)**2 / ((self.lane_boundaries[c+1]-self.lane_boundaries[c])/2)**2 for c in range(len(self.lane_boundaries)-1)])
        return [L1, L2, L3, L4, L5, L6]

        
    # Define the optimization problem objective function
    def objective(self, u_sequence, z, z_i, u_i_sequence, t):
        total_cost = 0.0

        # Controls of ego vehicle
        u_sequence = u_sequence.reshape((self.control_horizon, 2))
        u_sequence = np.vstack((u_sequence, np.tile(u_sequence[-1:], (self.prediction_horizon-self.control_horizon, 1))))

        # Controls of other vehicles
        u_i_sequence = u_i_sequence.reshape(int(u_i_sequence.size/self.control_horizon/2), self.control_horizon, 2)
        u_i_sequence = [np.vstack((i, np.tile(i[-1:], (self.prediction_horizon-self.control_horizon, 1)))) for i in u_i_sequence]

        # Receding horizon cost
        for i in range(self.prediction_horizon):
            z_players = self.get_players(z[i], z_i[i])
            
            z[i + 1] = self.rk4_step(z[i], u_sequence[i], t + i, self.horizon_interval)
            if z_i.size != 0:
                z_i[i + 1] = [self.rk4_step(z_i[i][j], u_i_sequence[j][i], t + i, self.horizon_interval) for j in range(len(z_i[i]))]

            running_cost = self.cost(z[i], z_players, u_sequence[i], t + i)
            total_cost += self.discount**i * sum(self.alphas[j]*running_cost[j] for j in range(6))

        return total_cost