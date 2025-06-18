import color_space_test as cst
import numpy as np
from numpy.lib import stride_tricks
from PIL import Image

"""Random Cropping images"""
def random_crop(img, S):
   # Optain the Size of an image
   width,height,_ = img.shape

   # Check Size is valid value
   if not 0 < S or not S < min(width,height):
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
def extract_patch(img, num_patches):
   size = num_patches
   H, W, _= img.shape
   shape = [H // size, W // size, size, size, 3]
   
   # (row, col, patch_row, patch_col)
   strides = [size * s for s in img.strides[:2]] + list(img.strides)
   # extract patches
   patches = stride_tricks.as_strided(img, shape=shape, strides=strides)

   return patches

"""Resizing"""
def resize_img(img, factor):
   # From original image, repeat each row and column and multiply by the factor such as (h*factor and w*factor)
   return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)

"""Color Jitter"""
def color_jitter(img, hue, saturation, value):
   # Randomly chose a value between given HSV and for Sat & Val limit between 0 and 1
   rand_hue = np.random.uniform(0, hue)
   rand_sat = np.random.uniform(0, saturation)
   rand_val = np.random.uniform(0, value)

   hsv = cst.rgb_to_hsv(img)
   
   hsv[..., 0] = (hsv[..., 0] + rand_hue) % 360.0
   hsv[..., 1] = np.clip(hsv[..., 1] * rand_sat, 0.0, 1.0)
   hsv[..., 2] = np.clip(hsv[..., 2] * rand_val, 0.0, 1.0)
   print(f"jitter HSV color:")

   return cst.hsv_to_rgb(hsv)
   
if __name__ == "__main__":

   img = Image.open("original.jpg")
   
   # 1st. Random Cropping
   print("============================")
   img_a = np.array(img)
   print("Original shape:", img_a.shape)
   S = 200
   crop_img = random_crop(img_a, S)
   print("Crop Image shape:", crop_img.shape)
   
   new_img = Image.fromarray(crop_img)
   new_img.save("crop_Img.jpg")

   # 2nd. Patch Extraction
   print("============================")
   img_b = np.array(img)
   print("Original shape:", img_b.shape)
   num_patches = 8
   patch_array = extract_patch(img_b, num_patches)
   print("Extracting Patch Image shape:", patch_array.shape)

   # 3rd. Resizing
   print("============================")
   img_c = np.array(img)
   print("Original shape:", img_c.shape)
   factor = 4
   re_array = resize_img(img_c, factor)
   print("Resized Image shape:", re_array.shape)
   new_img = Image.fromarray(re_array)
   new_img.save("resized_Img.jpg")

   # 4th. Color Jitter
   print("============================")
   print("Original RGB color:")
   hue = 100
   saturation = 1 
   value = 1
   color_jitter(img, hue, saturation, value)
