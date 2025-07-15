import matplotlib.pyplot as plt
import numpy as np

from skimage.color import rgb2gray
from skimage.morphology import dilation
from skimage.measure import label, regionprops

"""
    activity - Frame hysteresis for determining active or inactive objects.

    threshold - The motion threshold for filtering out noise.

    dis - A distance threshold to determine if an object candidate belongs to an object currently being tracked.

    fskip - The number of frames to skip between detections. The tracker will still work well even if it is not updated every frame.

    N - The maximum number of objects to track.
"""


class MotionDetector:

    def __init__(self,a,T,d,s,N):
        self.a = a
        self.T = T
        self.d = d
        self.s = s
        self.N = N

    def test(self): 
    
if __name__ == "__main__":