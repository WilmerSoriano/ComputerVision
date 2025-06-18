import sys

import numpy as np
from PIL import Image

"""
   Original Image is converted from RGB to HSV arrays/matrix
   (NOTE: image's in jpg, png, etc... expect to 
   read an RGB image to display)
   we then apply user input for HSV and convert 
   back to RGB so computer and user can see applied
   changes:
   RGB -> HSV -> apply userInput -> RGB
"""

# 3. Perform the Conversion to RGB-> HSV
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
   # Only perform the divioson for Index who do not have zeros.
   S = np.zeros_like(V)
   non_zero = V != 0
   S[non_zero] = C[non_zero] / V[non_zero]
   
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
 
# 5. Now convert image back to RGB. 
def hvs_to_rgb(hsv_img):
   # split HSV 
   H,S,V = hsv_img[..., 0], hsv_img[..., 1], hsv_img[..., 2]
   #Chrome
   C = V*S
   # Hue into 1 of 6 values
   Hn = H/60.0
   # X, is the 2nd largets component of color
   X = C * (1 - np.abs(Hn % (2 - 1)))

   # Declare all RGB array with zero float just like H just to be safe
   Rn = np.zeros_like(H)
   Gn = np.zeros_like(H)
   Bn = np.zeros_like(H)

   # Perform the piecewise function for the appropriate R,G, or B values only
   ones = (0 <= Hn) & (Hn < 1)
   Rn[ones], Gn[ones] = C[ones],X[ones]
   
   twos = (1 <= Hn) & (Hn < 2)
   Rn[twos], Gn[twos] = X[twos], C[twos]

   threes = (2 <= Hn) & (Hn < 3)
   Gn[threes], Bn[threes] = C[threes], X[threes]

   fours = (3 <= Hn) & (Hn < 4)
   Gn[fours], Bn[fours] = X[fours], C[fours]

   fives = (4 <= Hn) & (Hn < 5)
   Rn[fives], Bn[fives] = X[fives], C[fives]

   six = (5 <= Hn) & (Hn < 6)
   Rn[six], Bn[six] = C[six], X[six]

   # Final RGB value needed
   m = V - C

   # Now convert image back to RGB and all values back to range [0,255] for computer to read image; no more [0,1]
   R,G,B = ((Rn+m)*255,(Gn+m)*255, (Bn+m)*255)

   newRGB = np.stack([R,G,B], axis=2)
   # np.clip => Ensures all new RGB values are within 0 and 255, if less then 0, make it 0. Greater then 255, make it 255
   newRGB = np.clip(newRGB, 0, 255).astype(np.uint8)
   new_img = Image.fromarray(newRGB, 'RGB')

   new_img.save("HSV_output.jpg")
   print("New image completed, check it out!")

# 1. First verify user argument are valid
def verify(hue, sat, val):
   if hue < 0 or hue > 360:
      raise ValueError("Hue, must be between [0, 360]")
   if sat < 0 or sat > 1:
      raise ValueError("Saturation, must be between [0.0, 1.0]")
   if val < 0 or val > 1:
      raise ValueError("Value, must be between [0.0, 1.0]")
   return 1 
    
if __name__ =="__main__":
   img_path = sys.argv[1]
   hue = float(sys.argv[2])
   sat = float(sys.argv[3])
   val = float(sys.argv[4])
   
   try:
      verify(hue, sat, val)
   except ValueError as e:
      print(f"Invalid argument for {e}")

   # 2. Read the image from the path
   img = Image.open(img_path)

   # 4. Apply the user input for HSV
   hsv = rgb_to_hsv(img)

   hsv[..., 0] = (hsv[..., 0] + hue) % 360.0
   hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0.0, 1.0)
   hsv[..., 2] = np.clip(hsv[..., 2] * val, 0.0, 1.0)

   hvs_to_rgb(hsv)