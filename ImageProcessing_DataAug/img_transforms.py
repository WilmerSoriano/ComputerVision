import numpy as np
from PIL import Image

"""Random Cropping images"""
def random_crop(img_arry, S):
   # Optain the Size of an image
   width,height,_ = img_arry.shape

   # Check Size is valid value
   if not 0 < S or not S <= min(width,height):
      print("Inavlid Size")
      return 0

   
if __name__ == "__main__":

   # 1st. Random Cropping
   img = Image.open("original.jpg")
   img_arry = np.array(img)
   S = 0

   random_crop(img_arry, S)
