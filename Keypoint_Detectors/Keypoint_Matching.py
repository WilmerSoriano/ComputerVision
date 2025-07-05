import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
from matplotlib.patches import ConnectionPatch
from scipy.spatial.distance import cdist
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import SIFT

"""
    Keypoint Matching using SIFT
    This script demonstrates how to perform keypoint matching between two images using SIFT.
    It detects keypoints in both images, computes their descriptors, and matches them.
    The matched keypoints are then displayed on the images.
"""

def display_keypoints(matches, keypoint1, keypoint2, dst_img, src_img):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(dst_img, cmap='gray')
    ax2.imshow(src_img, cmap='gray')

    # NOTE: updated the way matches handles keypoints
    for i,j in matches:
        coordB = [keypoint1[i, 1], keypoint1[i, 0]]
        coordA = [keypoint2[j, 1], keypoint2[j, 0]]
        con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data",axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)
        ax1.plot(keypoint1[i, 1], keypoint1[i, 0], 'ro')
        ax2.plot(keypoint2[j, 1], keypoint2[j, 0], 'ro')

# Function to match keypoints based on their descriptors
# NOTE: The ratio test is important to determine the distance between keypoints.(distance < 0.75 * distance)
def keypoint_match(descriptors_dst, descriptors_src):
    ratio = 0.75
    matches = []
    # Before we perform the ratio test, we need the minimum euclidean distance from nearest neighbors for best candidate match for each keypoint.
    best_distance = cdist(descriptors1, descriptors2, 'euclidean')

    # Now we can perform the ratio test for each descriptor in descriptors1
    for i in range(best_distance.shape[0]):
        j = np.argsort(best_distance[i])
        """ 
            Check if the best match is significantly better than the second best match,
            If the distance of the best match is less than 0.75 times the distance of the 
            second best match, we consider it a match.
        """
        if best_distance[i, j[0]] < ratio * best_distance[i, j[1]]:
            matches.append((i, j[0]))  

    return matches

def sift_keypoints(img):
    detect = SIFT()
    detect.detect_and_extract(img)
    return detect.keypoints, detect.descriptors

# Setting up the image to gray for SIFT processing
def setup_Image(img):
    if img.shape[2] == 4:
        img = rgba2rgb(img)
    img_gray = rgb2gray(img)
    return img_gray

if __name__ == "__main__":
    # Convert images to array
    dst_img = np.asarray(Image.open('destination.jpg'))
    src_img = np.asarray(Image.open('source.jpg'))

    # Convert images to grayscale
    dst_gray = setup_Image(dst_img)
    src_gray = setup_Image(src_img)

    keypoints1, descriptors1 = sift_keypoints(dst_gray)
    keypoints2, descriptors2 = sift_keypoints(src_gray)

    # Match keypoints between the two images (main part of the code)
    matches = keypoint_match(descriptors1, descriptors2)

    display_keypoints(matches,keypoints1,keypoints2, dst_gray, src_gray)
    plt.show()