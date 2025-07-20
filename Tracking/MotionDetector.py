import numpy as np
from skimage.color import rgb2gray
from skimage.morphology import dilation, square
from skimage.measure import label, regionprops

from KalmanFilter import KalmanFilter  # Import KalmanFilter from its own file

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
        self.proposals = []   # list of proposal dicts
        self.tracked = []     # list of tracked object dicts
        self.next_id = 0
        self.frame_count = 0

    def update(self, frame):
        """Process a frame and return tracked objects"""
        gray = rgb2gray(frame)
        self.frame_buffer.append(gray)
        
        # Maintain only last 3 frames in buffer
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)
        if len(self.frame_buffer) < 3:
            return []
            
        self.frame_count += 1
        
        # Skip detection frames according to fskip
        if self.frame_count % self.fskip != 0:
            # Predict existing tracks
            for obj in self.tracked:
                self._predict_object(obj)
            return self._output_objects()
        
        # Compute motion mask using three-frame difference
        f2, f1, f0 = self.frame_buffer
        diff1 = np.abs(f0 - f1)
        diff2 = np.abs(f1 - f2)
        motion = np.minimum(diff1, diff2)
        motion[motion < self.threshold] = 0
        motion[motion >= self.threshold] = 1
        
        # Dilate to connect nearby pixels
        dil = dilation(motion, square(9))
        lbl = label(dil)
        regions = regionprops(lbl)
        
        # Create candidate objects from regions
        candidates = [{'centroid': r.centroid, 'bbox': r.bbox}
                     for r in regions if r.area > 50]
        
        # Update tracking state
        self._update_proposals(candidates)
        self._confirm_proposals()
        self._update_tracked(candidates)
        self._prune_tracked()
        
        return self._output_objects()

    def _update_proposals(self, candidates):
        """Update unconfirmed object proposals"""
        used = set()
        for cand in candidates:
            c = np.array(cand['centroid'])
            best_idx = None
            min_dist = float('inf')
            
            # Find closest proposal
            for i, p in enumerate(self.proposals):
                dist = np.linalg.norm(c - p['centroid'])
                if dist < min_dist and dist < self.dis:
                    min_dist = dist
                    best_idx = i
                    
            # Update existing proposal or create new
            if best_idx is not None:
                self.proposals[best_idx]['centroid'] = c
                self.proposals[best_idx]['age'] += 1
                used.add(best_idx)
            else:
                self.proposals.append({'centroid': c, 'age': 1})
        
        # Remove old proposals
        self.proposals = [p for p in self.proposals 
                         if p['age'] < self.activity or p['age'] >= self.activity]

    def _confirm_proposals(self):
        """Convert mature proposals to tracked objects"""
        i = 0
        while i < len(self.proposals):
            p = self.proposals[i]
            if p['age'] >= self.activity and len(self.tracked) < self.N:
                # Create tracked object
                self.tracked.append({
                    'id': self.next_id,
                    'kf': KalmanFilter(
                        p['centroid'],
                        dt=self.kf_params.get('dt', 1.0),
                        accel_variance=self.kf_params.get('accel_var', 1.0),
                        meas_variance=self.kf_params.get('meas_var', 1.0)
                    ),
                    'missed': 0,
                    'history': [np.array(p['centroid'])]
                })
                self.next_id += 1
                del self.proposals[i]
            else:
                i += 1

    def _update_tracked(self, candidates):
        """Update existing tracked objects"""
        for obj in self.tracked:
            # Predict object position
            predicted_pos = obj['kf'].predict()
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
                obj['kf'].update(best_candidate['centroid'])
                obj['history'].append(np.array(best_candidate['centroid']))
                obj['missed'] = 0
            else:
                obj['missed'] += 1

    def _prune_tracked(self):
        """Remove inactive tracked objects"""
        self.tracked = [o for o in self.tracked if o['missed'] < self.activity]

    def _predict_object(self, obj):
        """Predict position for a tracked object (during skip frames)"""
        obj['kf'].predict()

    def _output_objects(self):
        """Format output for tracked objects"""
        return [{
            'id': o['id'], 
            'centroid': o['kf'].x[:2], 
            'history': o['history']
        } for o in self.tracked]