import numpy as np


def omega_matrix(w):
    p, q, r = w
    return np.array([
        [0, -p, -q, -r],
        [p,  0,  r, -q],
        [q, -r,  0,  p],
        [r,  q, -p,  0]
    ])
def quat_to_matrix(q):
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ])


class DroneDynamics:

    def __init__(self):

        # Physical
        self.m = 1.0
        self.g = 9.81

        self.I = np.diag([0.02, 0.02, 0.04])
        self.I_inv = np.linalg.inv(self.I)

        self.L = 0.25

        # Motor constants
        self.kT = 1e-5
        self.kQ = 2e-6
        self.tau_m = 0.02

        # State
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.w = np.zeros(3)

        self.Omega = np.zeros(4)      # actual motor speeds
        self.Omega_cmd = np.zeros(4)  # commanded speeds
        self.rho = 1.225      # air density
        self.Cd = 1.0         # drag coefficient
        self.area = 0.05      # reference area
        self.wind = np.zeros(3)
        self.k_omega = 0.02
        self.R = 0.1
        self.Jr = 1e-5  # rotor inertia

        # Rotor spin directions (1 = CCW, -1 = CW)
        self.spin_dir = np.array([1, -1, 1, -1])


    def get_state_vector(self):
        return np.concatenate([
            self.position,
            self.velocity,
            self.q,
            self.w,
            self.Omega
        ])
    def set_state_vector(self, x):
    
        self.position = x[0:3]
        self.velocity = x[3:6]
        self.q = x[6:10]
        self.w = x[10:13]
        self.Omega = x[13:17]
        
    
    
    
    def derivatives(self):
    
        # ----- Motor lag -----
        dOmega = (self.Omega_cmd - self.Omega) / self.tau_m
    
        # ----- Thrust -----
        T = self.kT * self.Omega**2
        Q = self.kQ * self.Omega**2
    
        # Ground effect
        z = self.position[2]
        if 0 < z < 2 * self.R:
            factor = 1 / (1 - (self.R/(4*z))**2)
            T = T * factor
    
        total_thrust = np.sum(T)
    
        # ----- Rotation matrix -----
        R = quat_to_matrix(self.q)
    
        F_body = np.array([0, 0, total_thrust])
        F_world = R @ F_body
    
        # Wind
        v_rel = self.velocity - self.wind
        F_drag = -0.5 * self.rho * self.Cd * self.area * \
                 np.linalg.norm(v_rel) * v_rel
    
        Fg = np.array([0, 0, -self.m * self.g])
    
        dv = (F_world + F_drag + Fg) / self.m
    
        # ----- Torques -----
        tau_thrust = np.array([
            self.L * (T[3] - T[1]),
            self.L * (T[0] - T[2]),
            Q[0] - Q[1] + Q[2] - Q[3]
        ])
        H_rotor = np.array([0.0, 0.0, 0.0])
    
        for i in range(4):
            H_rotor += self.spin_dir[i] * self.Jr * self.Omega[i] * np.array([0,0,1])
        tau_gyro = np.cross(self.w, H_rotor)
         
    
    
        tau_damp = -self.k_omega * self.w
        tau = tau_thrust + tau_damp+ tau_gyro
    
        dw = self.I_inv @ (
            tau - np.cross(self.w, self.I @ self.w)
        )
    
        dq = 0.5 * omega_matrix(self.w) @ self.q
    
        return np.concatenate([
            self.velocity,
            dv,
            dq,
            dw,
            dOmega
        ])
    def step(self, dt):
    
        x = self.get_state_vector()
    
        # k1
        self.set_state_vector(x)
        k1 = self.derivatives()
    
        # k2
        self.set_state_vector(x + 0.5*dt*k1)
        k2 = self.derivatives()
    
        # k3
        self.set_state_vector(x + 0.5*dt*k2)
        k3 = self.derivatives()
    
        # k4
        self.set_state_vector(x + dt*k3)
        k4 = self.derivatives()
    
        x_next = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
        self.set_state_vector(x_next)
    
        # Normalize quaternion
        self.q /= np.linalg.norm(self.q)

