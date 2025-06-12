import sys

import numpy as np


def RGB(img):
   V = np.max(img, axis=2)
   
def HVS():
   print("Hello")

def verify(hue, sat, val):
   if hue < 0 or hue > 360:
      raise ValueError("Hue")
   if sat < 0 or sat > 1:
      raise ValueError("Saturation")
   if val < 0 or val > 1:
      raise ValueError("Value")
   return 1 

if __name__ =="__main__":
   img = sys.argv[1]
   hue = int(sys.argv[2])
   sat = int(sys.argv[3])
   val = int(sys.argv[4])
   
   try:
      verify(hue, sat, val)
   except ValueError as e:
      print(f"Invalid argument for {e}")


