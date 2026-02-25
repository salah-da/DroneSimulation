import numpy as np

class INDIController:
    def __init__(self, drone):
        self.drone = drone
        
        # Rate control gains (INDI inner loop)
        self.K_rate = np.diag([8.0, 8.0, 3.0])
        
        # Attitude hold gains (outer loop)
        self.K_att = np.diag([2.0, 2.0, 1.0])  # Roll, pitch, yaw attitude gains
        
        self.tau_prev = np.zeros(3)
        self.w_prev = np.zeros(3)
        self.dt = 0.005
        
        self.A = np.array([
            [1, 1, 1, 1],
            [0, -drone.L, 0, drone.L],
            [drone.L, 0, -drone.L, 0],
            [drone.kQ/drone.kT, -drone.kQ/drone.kT, 
             drone.kQ/drone.kT, -drone.kQ/drone.kT]
        ])
        self.A_inv = np.linalg.inv(self.A)
        
        self.hover_omega = np.sqrt(drone.m * drone.g / (4 * drone.kT))
        self.hover_T = drone.m * drone.g / 4.0

    def quat_to_euler(self, q):
        """Extract roll, pitch, yaw from quaternion"""
        qw, qx, qy, qz = q
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        return roll, pitch, yaw

    def compute_rpm(self, roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd):
        """
        roll_cmd: desired roll angle (rad), 0 = level
        pitch_cmd: desired pitch angle (rad), 0 = level  
        yaw_rate_cmd: desired yaw rate (rad/s)
        thrust_cmd: total thrust (N)
        """
        
        # === OUTER LOOP: Attitude hold ===
        roll, pitch, yaw = self.quat_to_euler(self.drone.q)
        w = self.drone.w
        
       # FORCE SCALARS with float()
        roll = float(roll)
        pitch = float(pitch)
        w_x = float(w[0])  # Explicit scalar extraction
        w_y = float(w[1])
        w_z = float(w[2])
        
        # Attitude error → rate command
        roll_err = np.clip(float(roll_cmd) - roll, -0.5, 0.5)
        pitch_err = np.clip(float(pitch_cmd) - pitch, -0.5, 0.5)
        
        # Desired rates (ensure scalars)
        p_des = float(self.K_att[0,0] * roll_err - 0.3 * w_x)  # Use K_att[0,0] not K_att[0]
        q_des = float(self.K_att[1,1] * pitch_err - 0.3 * w_y)
        r_des = float(yaw_rate_cmd)
        
        w_des = np.array([p_des, q_des, r_des], dtype=float)
        
        # === INNER LOOP: INDI rate control ===
        
        # Measured angular acceleration
        if hasattr(self, 'w_prev'):
            w_dot_meas = (w - self.w_prev) / self.dt
            w_dot_meas = np.clip(w_dot_meas, -50, 50)
        else:
            w_dot_meas = np.zeros(3)
        
        # Desired angular acceleration (with limit)
        w_err = w_des - w
        w_err = np.clip(w_err, -5, 5)  # Limit rate error
        w_dot_des = self.K_rate @ w_err
        w_dot_des = np.clip(w_dot_des, -20, 20)
        
        # INDI torque increment
        I = self.drone.I
        delta_tau = I @ (w_dot_des - w_dot_meas)
        delta_tau = np.clip(delta_tau, -0.5, 0.5)
        
        tau = self.tau_prev + delta_tau
        tau = np.clip(tau, -2.0, 2.0)
        
        self.tau_prev = tau
        self.w_prev = w.copy()
        
        # === CONTROL ALLOCATION with thrust priority ===
        u = np.array([thrust_cmd, tau[0], tau[1], tau[2]])
        T = self.A_inv @ u
        
        # CRITICAL: Ensure minimum thrust (safety margin)
        T_min = 0.2 * self.hover_T  # Never drop below 20% hover per motor
        T_max = 3.0 * self.hover_T
        
        # If thrust saturation occurs, scale down torque
        if np.any(T < T_min) or np.any(T > T_max):
            # Try scaling torque down
            for scale in [1.0, 0.5, 0.2, 0.0]:
                u_scaled = np.array([thrust_cmd, scale*tau[0], scale*tau[1], scale*tau[2]])
                T_scaled = self.A_inv @ u_scaled
                if np.all(T_scaled >= T_min) and np.all(T_scaled <= T_max):
                    T = T_scaled
                    break
            else:
                # Emergency: equal thrust, ignore torque
                T = np.ones(4) * thrust_cmd / 4.0
        
        T = np.clip(T, T_min, T_max)
        
        # Convert to RPM
        Omega_cmd = np.sqrt(T / self.drone.kT)
        
        return Omega_cmd