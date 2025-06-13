import sys

import numpy as np
from PIL import Image


# 3. Perform the Conversion and generate image
def rgb_to_hsv(img):
   rgb_img = img.convert('RGB')
   img_arry = np.array(rgb_img)
   # This make sure we are working with [0,1] for Saturation and Value
   img_arry = img_arry/255.0

   # Value
   V = np.max(img_arry, axis=2)
   # Chroma
   C = V - np.min(img_arry, axis=2)
   # Saturation
   # Numpy version of any condition if/else: ex: where(condition, X, Y)
   S = np.where(V != 0, C/V, 0.0)
   
   # Grab all RGB channel into 2D array
   r, g, b = img_arry[..., 0], img_arry[..., 1], img_arry[..., 2]

   # Hue piecewise function, and ignore divide by Zero warnings
   with np.errstate(divide='ignore', invalid='ignore'):
      H = np.where(C == 0, 0.0,
                  np.where(V == r, (np.divide((g-b), C)) % 6, 
                  np.where(V == g, (np.divide((b-r), C)) +2,
                  np.where(V == b, (np.divide((r-g), C)) +4, 0.0)))
                  )

   # Convert into degree
   H = (H * 60.0) % 360.0

   hsv = np.stack([H, S, V], axis=2)

   return hsv

def hvs_to_rgb(img, hue):
   
   # chrome = VxS
   C = val*sat



# 1. First verify user argument are valid
def verify(hue, sat, val):
   if hue < 0 or hue > 360:
      raise ValueError("Hue, must be between [0, 360]")
   if sat < 0 or sat > 1:
      raise ValueError("Saturation, must be between [0, 1.0]")
   if val < 0 or val > 1:
      raise ValueError("Value, must be between [0, 1.0]")
   return 1 
    
if __name__ =="__main__":
   img_path = sys.argv[1]
   hue = int(sys.argv[2])
   sat = float(sys.argv[3])
   val = float(sys.argv[4])
   
   try:
      verify(hue, sat, val)
   except ValueError as e:
      print(f"Invalid argument for {e}")

   # 2. Read the image from the path
   img = Image.open(img_path)

   hsv = rgb_to_hsv(img)
   hsv[..., 0] = (hsv[..., 0] + hue) % 360.0
   hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0.0, 1.0)
   hsv[..., 2] = np.clip(hsv[..., 2] * val, 0.0, 1.0)

   hvs_to_rgb(img,hsv)