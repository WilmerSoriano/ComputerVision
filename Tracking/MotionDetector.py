import numpy as np

from skimage.color import rgb2gray
from skimage.morphology import dilation, square
from skimage.measure import label, regionprops

from KalmanFilter import KalmanFilter

"""
        activity - Frame hysteresis for determining active or inactive objects

        threshold - The motion threshold for filtering out noise

        dis - Distance threshold to determine if object candidate belongs to tracked object

        fskip - Number of frames to skip between detections

        N - Maximum number of objects to track
"""

class MotionDetector:
    def __init__(self, activity=5, threshold=0.05, dis=30, fskip=1, N=10, kf_params=None):
        self.activity = activity
        self.threshold = threshold
        self.dis = dis
        self.fskip = fskip
        self.N = N
        self.kf_params = kf_params or {'dt': 1.0, 'accel_var': 1.0, 'meas_var': 1.0}
        
        self.frame_buffer = []
        self.proposals = []
        self.tracked = []
        self.next_id = 0
        self.frame_count = 0

    def update(self, frame):
        gray = rgb2gray(frame)
        self.frame_buffer.append(gray)
        
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)
        if len(self.frame_buffer) < 3:
            return []
            
        self.frame_count += 1
        
        # Skip detection frames
        if self.frame_count % self.fskip != 0:
            for obj in self.tracked:
                obj['kf'].predict()
            return self._output_objects()
        
        # Compute motion mask
        f2, f1, f0 = self.frame_buffer
        diff1 = np.abs(f0 - f1)
        diff2 = np.abs(f1 - f2)
        motion = np.minimum(diff1, diff2)
        motion[motion < self.threshold] = 0
        motion[motion >= self.threshold] = 1
        
        # Process motion regions
        dil = dilation(motion, square(9))
        lbl = label(dil)
        regions = regionprops(lbl)
        
        # Create candidates with proper coordinate format
        candidates = []
        for r in regions:
            if r.area > 50:
                # Keep centroid as (row, col) as in original
                centroid = r.centroid
                candidates.append({'centroid': centroid, 'bbox': r.bbox})
        
        # Update tracking state
        self._update_proposals(candidates)
        self._confirm_proposals()
        self._update_tracked(candidates)
        self._prune_tracked()
        
        return self._output_objects()

    def _update_proposals(self, candidates):
        used = set()
        for cand in candidates:
            c = np.array(cand['centroid'])
            best_idx = None
            min_dist = float('inf')
            
            for i, p in enumerate(self.proposals):
                dist = np.linalg.norm(c - p['centroid'])
                if dist < min_dist and dist < self.dis:
                    min_dist = dist
                    best_idx = i
                    
            if best_idx is not None:
                self.proposals[best_idx]['centroid'] = c
                self.proposals[best_idx]['age'] += 1
                used.add(best_idx)
            else:
                self.proposals.append({'centroid': c, 'age': 1})
        
        # Remove old proposals
        self.proposals = [p for p in self.proposals if p['age'] < self.activity * 2]

    def _confirm_proposals(self):
        i = 0
        while i < len(self.proposals):
            p = self.proposals[i]
            if p['age'] >= self.activity and len(self.tracked) < self.N:
                kf = self._create_kalman_filter(p['centroid'])
                self.tracked.append({'id': self.next_id, 'kf': kf, 'missed': 0, 'history': [np.array(p['centroid'])]})
                self.next_id += 1
                del self.proposals[i]
            else:
                i += 1

    def _create_kalman_filter(self, centroid):
        """Create Kalman filter matching original implementation"""
        dt = self.kf_params.get('dt', 1.0)
        accel_var = self.kf_params.get('accel_var', 1.0)
        meas_var = self.kf_params.get('meas_var', 1.0)
        
        # State transition matrix
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Control matrix (not used)
        B = np.zeros((4, 1))
        
        # Measurement matrix
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance (match original calculation)
        G = np.array([
            [dt**2/2, 0],
            [0, dt**2/2],
            [dt, 0],
            [0, dt]
        ])
        Q = G @ G.T * accel_var
        
        # Measurement noise covariance
        R = np.eye(2) * meas_var
        
        # Initial state: [row, col, v_row, v_col] (as in original)
        x0 = np.array([centroid[0], centroid[1], 0., 0.], dtype=float)
        
        # Initial covariance (match original value)
        P0 = np.eye(4) * 500.
        
        return KalmanFilter(F, B, H, Q, R, x0, P0)

    def _update_tracked(self, candidates):
        for obj in self.tracked:
            # Predict object position
            obj['kf'].predict()
            predicted_pos = obj['kf'].position[:2]  # Use first two elements
            
            best_candidate = None
            min_dist = float('inf')
            
            # Find closest candidate
            for cand in candidates:
                dist = np.linalg.norm(predicted_pos - cand['centroid'])
                if dist < min_dist and dist < self.dis:
                    min_dist = dist
                    best_candidate = cand
                    
            # Update or mark as missed
            if best_candidate:
                obj['kf'].update(np.array(best_candidate['centroid']))
                obj['history'].append(np.array(best_candidate['centroid']))
                obj['missed'] = 0
            else:
                obj['missed'] += 1

    def _prune_tracked(self):
        self.tracked = [o for o in self.tracked if o['missed'] < self.activity]

    def _output_objects(self):
        return [{
            'id': o['id'], 
            'centroid': o['kf'].position[:2],  # Return only position
            'history': o['history']
        } for o in self.tracked]