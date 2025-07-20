
import numpy as np

class KalmanFilter:
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F  
        self.B = B  
        self.H = H  
        self.Q = Q  
        self.R = R  
        self.x = x0  
        self.P = P0  
        
    def predict(self, u=None):
        if u is None:
            u = np.zeros(self.B.shape[1])

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.x
        
    def update(self, z):
        z = np.array(z)
        
        # Compute innovation
        y = z - np.dot(self.H, self.x)
        
        # Compute innovation covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Compute Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state estimate
        self.x = self.x + np.dot(K, y)
        
        # Update covariance (simplified form)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        
        return self.x
    
    @property
    def position(self):
        return self.x[:2]