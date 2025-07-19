import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, dt=1.0):
        # State: [x, y, vx, vy]
        self.state = initial_state
        self.dt = dt
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Process covariance
        self.Q = np.eye(4) * 0.01
        
        # Measurement matrix (only position observable)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement covariance
        self.R = np.eye(2) * 0.1
        
        # State covariance
        self.P = np.eye(4) * 1.0
        
    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]  # Return predicted position
        
    def update(self, measurement):
        # Measurement: [x, y]
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
    def get_position(self):
        return self.state[:2]
    
    def get_velocity(self):
        return self.state[2:]