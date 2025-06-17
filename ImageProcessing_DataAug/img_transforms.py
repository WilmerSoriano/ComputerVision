import numpy as np
from numpy.lib import stride_tricks
from PIL import Image

"""Random Cropping images"""
def random_crop(img, S):
   # Optain the Size of an image
   width,height,_ = img.shape

   # Check Size is valid value
   if not 0 < S or not S <= min(width,height):
      print("Inavlid Size")
      return 0
   
   # Finding the width and height of Size. (Size = W*H)
   size_w = int(S/2)
   size_h = int(S-size_w)

   # Randomly choose a Center from between Size(w & h) and image(w & h) difference.
   rand_x = np.random.randint(size_w, width - size_h)
   rand_y = np.random.randint(size_w, height - size_h)
   
   # Finding the top to bottom and left to right corner of the crop (NOTE: both are subracted using the same variable to have a square like size)
   left_right = rand_x - size_w
   top_bottom = rand_y - size_w

   return img[top_bottom : top_bottom + S, left_right : left_right + S]

"""Patch Extraction"""
def extract_patch(img):
   # non-overlapping patches of size 8
   size = 8
   H, W,_ = img.shape
   shape = [H // size, W // size] + [size, size]

   # (row, col, patch_row, patch_col)
   strides = [size * s for s in img.strides] + list(img.strides)
   # extract patches
   patches = stride_tricks.as_strided(img, shape=shape, strides=strides)
   return patches

   
if __name__ == "__main__":

   img = Image.open("original.jpg")
   
   # 1st. Random Cropping
   img_arry = np.array(img)
   S = 200
   crop_img = random_crop(img_arry, S)
   
   new_img = Image.fromarray(crop_img)
   #new_img.save("crop_Img.jpg")

   # 2nd. Patch Extraction
   patch_img = extract_patch(img_arry)
      
   new_img = Image.fromarray(patch_img)
   new_img.save("patch_Img.jpg")
