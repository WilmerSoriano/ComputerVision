import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
from matplotlib.patches import ConnectionPatch
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import SIFT
from skimage.transform import resize

"""
    Keypoint Matching using SIFT
    This script demonstrates how to perform keypoint matching between two images using SIFT.
    It detects keypoints in both images, computes their descriptors, and matches them.
    The matched keypoints are then displayed on the images.
"""

def display_keypoints(matches, dst_img, src_img):
   dst = matches[0]
   src = matches[1]

   fig = plt.figure(figsize=(8, 4))
   ax1 = fig.add_subplot(121)
   ax2 = fig.add_subplot(122)
   ax1.imshow(dst_img, cmap='gray')
   ax2.imshow(src_img, cmap='gray')

   for i in range(src.shape[0]):
      coordB = [dst[i, 1], dst[i, 0]]
      coordA = [src[i, 1], src[i, 0]]
      con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data",axesA=ax2, axesB=ax1, color="red")
      ax2.add_artist(con)
      ax1.plot(dst[i, 1], dst[i, 0], 'ro')
      ax2.plot(src[i, 1], src[i, 0], 'ro')

def keypoint_match(descriptors1, descriptors2):
   matches = []
   for i, desc1 in enumerate(descriptors1):
       best_match = None
       best_distance = float('inf')
       for j, desc2 in enumerate(descriptors2):
           distance = np.linalg.norm(desc1 - desc2)
           if distance < best_distance:
               best_distance = distance
               best_match = j
       if best_match is not None:
           matches.append((i, best_match))
   return matches

def sift_keypoints(img):
   detect = SIFT()
   detect.detect_and_extract(img)
   return detect.keypoints, detect.descriptors

# Setting up the image to gray for SIFT processing
def setup_Image(img):
   img_rgb = rgba2rgb(img)
   img_gray = rgb2gray(img_rgb)
   return img_gray

if __name__ == "__main__":
    # Convert images to array
    dst_img = np.asarray(Image.open('destination.png'))
    src_img = np.asarray(Image.open('source.png'))

    # Convert images to grayscale
    dst_gray = setup_Image(dst_img)
    src_gray = setup_Image(src_img)

    keypoints1, descriptors1 = sift_keypoints(dst_gray)
    keypoints2, descriptors2 = sift_keypoints(src_gray)

    matches = keypoint_match(descriptors1, descriptors2)

    display_keypoints(matches, gray1, gray2)
    plt.show()