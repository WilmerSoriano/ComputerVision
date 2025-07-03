import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import SIFT
from matplotlib.patches import ConnectionPatch
from skimage.color import rgb2gray, rgba2rgb

def display_keypoints(matches, dst_img, src_img):
   dst = keypoints1[matches[:, 0]]
   src = keypoints2[matches[:, 1]]

   fig = plt.figure(figsize=(8, 4))
   ax1 = fig.add_subplot(121)
   ax2 = fig.add_subplot(122)
   ax1.imshow(dst_img, cmap='gray')
   ax2.imshow(src_img, cmap='gray')

   for i in range(src.shape[0]):
      coordB = [dst[i, 1], dst[i, 0]]
      coordA = [src[i, 1], src[i, 0]]
      con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data",
                           axesA=ax2, axesB=ax1, color="red")
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

def sift_keypoints(image):
   sift = SIFT()
   sift.detect_and_extract(image)
   return sift.keypoints, sift.descriptors

# Setting up the image for SIFT processing
def setup_Image(flat):
   img = flat.reshape(32, 32, 3)
   img = img.transpose(1, 0, 2) 
   img = rgb2gray(img)
   return img

if __name__ == "__main__":
   # Load the image from CIFAR-10 dataset
   data = np.load("cifar10.npz", allow_pickle=True)
   X_train = data["X_train"].astype(np.uint8)

   # Pick two examples
   flat1 = X_train[0]
   flat2 = X_train[1]

   # Convert images to grayscale
   gray1 = setup_Image(flat1)
   gray2 = setup_Image(flat2)
   
   # Detect keypoints and descriptors
   keypoints1, descriptors1 = sift_keypoints(gray1)
   keypoints2, descriptors2 = sift_keypoints(gray2)

   # Match keypoints
   matches = keypoint_match(descriptors1, descriptors2)

   display_keypoints(matches, gray1, gray2)
   plt.show()