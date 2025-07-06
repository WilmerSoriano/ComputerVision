import numpy as np
from skimage import color
from skimage.color import rgb2gray
from skimage.feature import SIFT, hog
from sklearn.cluster import KMeans
from tqdm import tqdm  # Used as progress bar


# TODO: Create feature processing functions for SIFT and HOG
def sift_features(train_img, train_label):
    sift = SIFT()
    features = []
    y_features = []

def hog_features(images):
    features = []  
    for img in tqdm(images, desc="HOG feature extraction"):
        rgb = img.reshape(3,32,32).transpose(1,2,0)
        gray_img = color.rgb2gray(rgb)

        hog_features = hog(gray_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        features.append(hog_features)
    return np.array(features)

if __name__ == "__main__":
    # Load the pre-split data
    data = np.load("cifar10.npz", allow_pickle=True)

    # TODO: Extract features from the training data
    X_train = data["X_train"].astype(np.uint8)
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(X_train[0])

    # TODO: Extract features from the testing data
    sift_features_train = sift_features(X_train, y_train)
    sift_features_test = sift_features(X_test, y_test)

    hog_features_train = hog_features(X_train)
    hog_features_test = hog_features(X_test)

    # TODO: Save the extracted features to a file
    np.savez("HOG_cifar10_features.npz",
    X_train=hog_features_train,
    X_test=hog_features_test,
    y_train=y_train,
    y_test=y_test)

    np.savez("SIFT_cifar10_features.npz",
    X_train=sift_features_train,
    X_test=sift_features_test,
    y_train=y_train,
    y_test=y_test)
