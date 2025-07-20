
import numpy as np

class KalmanFilter:
    """
    A simple Kalman filter for tracking 2D position and velocity.
    State vector x = [px, py, vx, vy].
    """
    def __init__(self, init_pos, dt=1.0, accel_variance=1.0, meas_variance=1.0):
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        q = accel_variance
        G = np.array([[dt**2/2, 0],
                      [0, dt**2/2],
                      [dt, 0],
                      [0, dt]])
        self.Q = G @ G.T * q
        self.R = np.eye(2) * meas_variance
        self.P = np.eye(4) * 500.
        self.x = np.array([init_pos[0], init_pos[1], 0., 0.], dtype=float)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, y):
        y = np.array(y)
        a = y - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ a
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x[:2]