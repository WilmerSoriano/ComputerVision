import numpy as np
from skimage import color
from skimage.color import rgb2gray
from skimage.feature import SIFT, hog
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm  # Used as progress bar


# TODO: Create feature processing functions for SIFT and HOG
def create_histograms(features, kmeans):
    vocab_size = 100
    img_histograms = []

    for feature in tqdm(features, desc="Building histograms"):
        if len(feature) > 0:
            clusters = kmeans.predict(feature)
            histogram = np.bincount(clusters, minlength=vocab_size)
            img_histograms.append(histogram)
        else:
            img_histograms.append(np.zeros(vocab_size)) # <= Update: Handle images with no features
    return np.array(img_histograms)

def extract_descriptors(images):
    sift = SIFT()
    descriptors_list = []
    
    for img in tqdm(images, desc="Extracting SIFT descriptors"):
        rgb = img.reshape(3, 32, 32).transpose(1, 2, 0)
        img_gray = rgb2gray(rgb)
        
        try:
            sift.detect_and_extract(img_gray)
            if sift.descriptors is not None and len(sift.descriptors) > 0:
                descriptors_list.append(sift.descriptors)
            else:
                descriptors_list.append(np.array([]))  # <= Update: Empty array for no features
        except:
            descriptors_list.append(np.array([]))
    
    return descriptors_list

def sift_features(X_train, X_test):
    # =========== Extract SIFT descriptors ============
    train_descriptors = extract_descriptors(X_train)
    test_descriptors = extract_descriptors(X_test)

    # =========== Build vocabulary ============
    train_descriptors_concat = np.concatenate([d for d in train_descriptors if len(d) > 0]) # <= Update: Filter out empty descriptors
    vocab_size = 100
    kmeans = KMeans(n_clusters=vocab_size, random_state=42)
    kmeans.fit(train_descriptors_concat)
    
    # =========== Create histograms ===========
    X_train_hist = create_histograms(train_descriptors, kmeans)
    X_test_hist = create_histograms(test_descriptors, kmeans)

    return X_train_hist, X_test_hist

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
    sift_train, sift_test = sift_features(X_train, X_test)
    transformer = TfidfTransformer()
    transformer.fit(sift_train)
    X_train_tfidf = transformer.transform(sift_train).toarray()
    X_test_tfidf  = transformer.transform(sift_test).toarray()

    hog_features_train = hog_features(X_train)
    hog_features_test = hog_features(X_test)

    # TODO: Save the extracted features to a file
    np.savez("HOG_cifar10_features.npz",
    X_train=hog_features_train,
    X_test=hog_features_test,
    y_train=y_train,
    y_test=y_test)

    np.savez("SIFT_cifar10_features.npz",
    X_train=sift_train,
    X_test=sift_test,
    y_train=y_train,
    y_test=y_test)
