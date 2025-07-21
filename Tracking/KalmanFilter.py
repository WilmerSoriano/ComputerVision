
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
        
    def predict(self):
        # 1st. predict next state
        self.x = np.dot(self.F, self.x)
        # 2nd. predict covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q 

        return self.x
        
    def update(self, y):
        y = np.array(y) # 
        
        z = y - np.dot(self.H, self.x)
        
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, z)
        
        # Update covariance 
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        
        return self.x
    
    @property
    def position(self):
        return self.x[:2] # only used to find the esimated position