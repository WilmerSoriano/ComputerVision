import numpy as np
from PIL import Image


def img_pyramid(img, py_h):
   print("Hello")





img = Image.open("original.jpg")
img_array = np.array(img)
py_h = 4
img_pyramid(img_array, py_h)
