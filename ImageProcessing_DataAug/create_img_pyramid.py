import numpy as np
from PIL import Image


def img_pyramid(img, py_h):
   # Straight forword, take the original image size and 
   # divide by 2 for every py_h = number of time user want us to divide image repetative.
   H,W,_ = img.shape
   new_img = Image.fromarray(img)
   numImg = 2

   while py_h != 1:
      py_h = py_h-1
      H = H//2
      W = W//2
      resized_image = new_img.resize((W, H))  # (width, height)
      resized_image.save(f"img_{numImg}x.png")
      numImg = numImg*2

img = Image.open("original.jpg")
img_array = np.array(img)
py_h = 4
img_pyramid(img_array, py_h)
