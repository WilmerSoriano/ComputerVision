import numpy as np
from skimage.color import rgb2gray
from skimage.feature import SIFT, hog
from sklearn.cluster import KMeans
from skimage.transform import resize
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

# TODO: Create feature processing functions for SIFT and HOG
def create_histograms(features, kmeans, vocab_size=100):
    img_histograms = []

    for feature in tqdm(features, desc="Building histograms"):
        if len(feature) > 0:
            clusters = kmeans.predict(feature)
            histogram = np.bincount(clusters, minlength=vocab_size) # updated: use np.bincount because it is more efficient (SVM before 0.10 , after 0.30)
            img_histograms.append(histogram)
        else:
            img_histograms.append(np.zeros(vocab_size)) # <= Handle empty features
    return np.array(img_histograms)

def extract_descriptors(images):
    sift = SIFT()
    descriptors_list = []
    
    for img in tqdm(images, desc="Extracting SIFT descriptors"):
        rgb = img.reshape(3, 32, 32).transpose(1, 2, 0)
        rgb_large = resize(rgb, (64, 64))
        img_gray = rgb2gray(rgb_large)

        try:
            sift.detect_and_extract(img_gray)
            if sift.descriptors is not None and len(sift.descriptors) > 0: # <= This helps avoid errors when no keypoints are found
                descriptors_list.append(sift.descriptors)
            else:
                descriptors_list.append(np.array([]))
        except:
            descriptors_list.append(np.array([]))
    return descriptors_list

def sift_features(X_train, X_test):
    train_descriptors = extract_descriptors(X_train)
    test_descriptors = extract_descriptors(X_test)

    # ======== Build vocabulary ========
    print("Building vocabulary from training descriptors... may take a while...6 minutes")

    non_empty_train = [d for d in train_descriptors if len(d) > 0] # <= Filter out empty descriptors
    if len(non_empty_train) == 0:
        raise ValueError("No SIFT descriptors found in training set")
    
    train_descriptors_concat = np.concatenate(non_empty_train)

    vocab_size = 300  # Changed to 300, encountered issues with 100
    kmeans = KMeans(n_clusters=vocab_size, random_state=42, n_init='auto') # <= Use n_init='auto' for better performance
    kmeans.fit(train_descriptors_concat)
    
    # ======== Create histograms ========
    X_train_hist = create_histograms(train_descriptors, kmeans, vocab_size)
    X_test_hist = create_histograms(test_descriptors, kmeans, vocab_size)

    return X_train_hist, X_test_hist

def hog_features(images):
    features = []

    for img in tqdm(images, desc="Extracting HOG features"):
        # Handles reshaping and resizing of the CIFAR-10 image data
        rgb = img.reshape(3, 32, 32).transpose(1, 2, 0)
        rgb_large = resize(rgb, (64, 64))
        img_gray = rgb2gray(rgb_large)

        # Update: added channel_axis=None to handle grayscale images correctly
        hog_feat = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=None, visualize=False)
        features.append(hog_feat)
    
    return np.array(features)

if __name__ == "__main__":
    # Load the pre-split data
    data = np.load("cifar10.npz", allow_pickle=True)

    # TODO: Extract features from the training data
    X_train = data["X_train"].astype(np.uint8)
    y_train = data["y_train"].astype(np.int32)  
    X_test = data["X_test"].astype(np.uint8)
    y_test = data["y_test"].astype(np.int32)

    print(X_train[0]) 

    # TODO: Extract features from the testing data
    sift_train_hist, sift_test_hist = sift_features(X_train, X_test)

    combined_hist = np.vstack((sift_train_hist, sift_test_hist)) # combine train and test histograms
    
    # Adjusting the Frequency Vector
    tfidf = TfidfTransformer()
    combined_tfidf = tfidf.fit_transform(combined_hist).toarray()
    X_train_tfidf = combined_tfidf[:len(X_train)]
    X_test_tfidf = combined_tfidf[len(X_train):]

    hog_features_train = hog_features(X_train)
    hog_features_test = hog_features(X_test)

    # TODO: Save the extracted features to a file
    hog_data = {
        "X_train": hog_features_train,
        "X_test": hog_features_test,
        "y_train": y_train,
        "y_test": y_test
    }
    np.savez("HOG_cifar10.npz", **hog_data)

    # Save SIFT features
    sift_data = {
        "X_train": X_train_tfidf,
        "X_test": X_test_tfidf,
        "y_train": y_train,
        "y_test": y_test
    }
    np.savez("SIFT_cifar10.npz", **sift_data)
    