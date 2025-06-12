import sys

import numpy as np
from PIL import Image


# 3. Perform the Conversion and generate image
def rgb_to_hsv(img, hue, sat, val):
   rgb_img = img.convert('RGB')
   r,g,b = rgb_img.getpixel((1,1))

   # Value
   V = np.max(img, axis=2)
   # Chroma
   C = V - np.min(img, axis=2)
   # Saturation
   S = 0
   if V.all:
      S = C/V
   
   # Hue
   H = 0
   if V.all == r:
      H = ((g-b)/C)%6
   elif V.all == g:
      H = ((b-r)/C)+2
   elif V.all == b:
      H = ((r-g)/C)+4 
   Hue = H*60

   hsv = np.stack([Hue, S, V], axis=1)
   
def hvs_to_rgb(img, hue, sat, val):
   
   # chrome = VxS
   C = val*sat



# 1. First verify user argument are valid
def verify(hue, sat, val):
   if hue < 0 or hue > 360:
      raise ValueError("Hue")
   if sat < 0 or sat > 1:
      raise ValueError("Saturation")
   if val < 0 or val > 1:
      raise ValueError("Value")
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

   # Display the image in a GUI, NOTE: only used for debugging
   #cv2.imshow('Image', img)
   #cv2.waitKey(0) 
   #cv2.destroyAllWindows()

   rgb_to_hsv(img, hue, sat, val)
   #hvs_to_rgb(img, hue, sat, val)