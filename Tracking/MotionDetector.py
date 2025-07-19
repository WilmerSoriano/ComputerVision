import numpy as np

from skimage.color import rgb2gray
from skimage.morphology import dilation
from skimage.measure import label, regionprops

import numpy
numpy.float = numpy.float64
numpy.int = numpy.int_

"""
    NOTE:
        activity - Frame hysteresis for determining active or inactive objects.

        threshold - The motion threshold for filtering out noise.

        dis - A distance threshold to determine if an object candidate belongs to an object currently being tracked.

        fskip - The number of frames to skip between detections. The tracker will still work well even if it is not updated every frame.

        N - The maximum number of objects to track.
"""

class MotionDetector:
    def __init__(self, activity=5, threshold=0.05, dis=50, fskip=0, N=10):
        self.activity = activity 
        self.threshold = threshold  
        self.dis = dis  
        self.fskip = fskip  
        self.N = N  
        
        # This tracks the current state of the frame.
        self.frame_counter = 0
        self.ppframe = None
        self.pframe = None
        self.kernel = np.ones((9, 9))
        self.min_blob_size = 50
        
    def update_frames(self, new_frame):
        self.ppframe = self.pframe
        self.pframe = rgb2gray(new_frame)

        return self.ppframe is not None and self.pframe is not None
        
    def detect_objects(self, current_frame):
        
        # Skip detection if it's not a detection frame
        if self.frame_counter % (self.fskip + 1) != 0:
            self.frame_counter += 1
            return []
        
        self.frame_counter += 1
        
        if not self.update_frames(current_frame):
            return []
            
        cframe = rgb2gray(current_frame)
        diff1 = np.abs(cframe - self.pframe)
        diff2 = np.abs(self.pframe - self.ppframe)
        motion_frame = np.minimum(diff1, diff2)
        
        # Thresholding and morphology
        thresh_frame = motion_frame > self.threshold
        dilated_frame = dilation(thresh_frame, self.kernel)
        
        # Region detection
        label_frame = label(dilated_frame)
        regions = regionprops(label_frame)
        
        # Filter small blobs
        objects = []
        for r in regions:
            if r.area >= self.min_blob_size:
                # Format: [centroid_x, centroid_y, bbox_minr, bbox_minc, bbox_maxr, bbox_maxc]
                centroid = r.centroid[::-1]  # Convert (y,x) to (x,y)
                bbox = r.bbox
                objects.append(np.array([
                    centroid[0], centroid[1], 
                    bbox[0], bbox[1], bbox[2], bbox[3]
                ]))
                
        return objects