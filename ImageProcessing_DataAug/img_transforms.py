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
   
   # Finding the width and height of Size. (Size = W*H)
   size_w = int(S/2)
   size_h = int(S-size_w)

   # Randomly choose a Center from between Size(w & h) and image(w & h) difference.
   rand_x = np.random.randint(size_w, width - size_h)
   rand_y = np.random.randint(size_w, height - size_h)
   
   # Finding the top to bottom and left to right corner of the crop (NOTE: both are subracted using the same variable to have a square like size)
   left_right = rand_x - size_w
   top_bottom = rand_y - size_w

   return img_arry[top_bottom : top_bottom + S, left_right : left_right + S]


   
if __name__ == "__main__":

   # 1st. Random Cropping
   img = Image.open("original.jpg")
   img_arry = np.array(img)
   S = 200

   crop_img = random_crop(img_arry, S)
   new_img = Image.fromarray(crop_img)
   new_img.save("crop_Img.jpg")
