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
    def __init__(self, activity=5, threshold=0.05, dis=30, fskip=1, N=10, kf_param=None):
        self.activity = activity
        self.threshold = threshold
        self.dis = dis
        self.fskip = fskip
        self.N = N

        self.kf_param = kf_param or {'frame_t': 1.0, 'accel': 1.0, 'meas': 1.0} # back up Kalman Filter parameter
        
        self.frame_buffer = []
        self.proposals = []
        self.tracked = []
        self.next_id = 0
        self.frame_count = 0

    def update(self, frame):
        gray = rgb2gray(frame)
        self.frame_buffer.append(gray)
        
        # Checks if the frame is within the range of 3 frames from start to end
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)
        if len(self.frame_buffer) < 3:
            return []
            
        self.frame_count += 1
        
        # Skip detection frames
        if self.frame_count % self.fskip != 0:
            for obj in self.tracked:
                obj['kf'].predict()
            return self.tracked_objects()
        
        # Calculation the difference of frames
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
        
        # Create possible candidates with proper coordinate 
        candidates = []
        for r in regions:
            if r.area > 50:
                # find the centroid (average position from row and column) 
                centroid = r.centroid
                candidates.append({'centroid': centroid, 'bbox': r.bbox})
        
        self.update_proposal(candidates)
        self.confirm_proposal()
        self.update_tracker(candidates)
        self.remove_tracker()
        
        return self.tracked_objects()

    def update_proposal(self, candidates):
        valid = set()

        for cand in candidates: # 1st loop, option each candidate centroid 
            c = np.array(cand['centroid'])
            best_id = None
            min_dist = float('inf')
            
            for id, p in enumerate(self.proposals): # 2nd loop, match the candidate to a possible proposal
                dist = np.linalg.norm(c - p['centroid'])
                if dist < min_dist and dist < self.dis:
                    min_dist = dist
                    best_id = id
                    
            if best_id is not None: # Finally update the best candidate proposal
                self.proposals[best_id]['centroid'] = c
                self.proposals[best_id]['age'] += 1
                valid.add(best_id)
            else:
                self.proposals.append({'centroid': c, 'age': 1})
        
        # Remove old proposals, not needed anymore.
        self.proposals = [p for p in self.proposals if p['age'] < self.activity * 2]

    def confirm_proposal(self):
        i = 0
        while i < len(self.proposals): # If the object is making noises for a while then accept the proposal, promote to track!
            p = self.proposals[i]

            if p['age'] >= self.activity and len(self.tracked) < self.N:

                kf = self.setup_KF(p['centroid'])
                self.tracked.append({'id': self.next_id, 'kf': kf, 'missed': 0, 'history': [np.array(p['centroid'])]})
                self.next_id += 1
                del self.proposals[i] # Remove the proposal from the list since they are now being tracked
            else:
                i += 1

    def setup_KF(self, centroid):
        frame_t = self.kf_param.get('frame_t', 1.0)
        accel = self.kf_param.get('accel', 1.0)
        meas = self.kf_param.get('meas', 1.0)
        
        F = np.array([[1, 0, frame_t, 0], [0, 1, 0, frame_t], [0, 0, 1, 0], [0, 0, 0, 1]])
        
        B = np.zeros((4, 1))
        
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        # FIX Remember we need position and velocity at each frame per time 
        G = np.array([[frame_t**2/2, 0],[0, frame_t**2/2],[frame_t, 0],[0, frame_t]])
        
        Q = np.linalg.multi_dot([G, G.T]) * accel 
        
        R = np.eye(2) * meas
        
        x0 = np.array([centroid[0], centroid[1], 0., 0.], dtype=float)
        
        P0 = np.eye(4) * 500.
        
        return KalmanFilter(F, B, H, Q, R, x0, P0)

    def update_tracker(self, candidates):
        for obj in self.tracked:
            # Predict object position
            obj['kf'].predict()
            predicted_pos = obj['kf'].x[:2]
            
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

    def remove_tracker(self):
        active_objects = []

        for obj in self.tracked:
            if obj['missed'] < self.activity:
                active_objects.append(obj)
        self.tracked = active_objects

    def tracked_objects(self): 
        active_object = []

        for obj in self.tracked:
            object_data = {'id': obj['id'],'centroid': obj['kf'].x[:2],'history': obj['history']}
            active_object.append(object_data)
        return active_object