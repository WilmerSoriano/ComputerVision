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
    image_histograms = []

    for feature in tqdm(features, desc="Building histograms"):
        clusters = kmeans.predict(feature) if feature.size else []
        histogram, _ = np.histogram(clusters, bins=vocab_size, range=(0, vocab_size))
        image_histograms.append(histogram)

    # Another padding issue, if no valid histograms were created, return a zero array
    if len(image_histograms) == 0:
        return np.zeros((len(features), 100), dtype=int)
    return np.stack(image_histograms, axis=0)

def descriptors(images):
    sift = SIFT()
    descriptors_list = []
    valid_id = []
    
    # I modified the loop to also return the valid indices per image.
    for id, img in enumerate(tqdm(images, desc="Extracting SIFT descriptors")):
        rgb = img.reshape(3, 32, 32).transpose(1, 2, 0)
        img_gray = rgb2gray(rgb)
        
        try:
            sift.detect_and_extract(img_gray)
            if sift.descriptors is not None and len(sift.descriptors) > 0:
                descriptors_list.append(sift.descriptors)
                valid_id.append(id)
        except:
            pass
            
    return descriptors_list, valid_id

# NOTE: both train and test should be in one Kmeans.
def sift_features(X_train, X_test):
    # ==== Extracting Visual Features ====
    train_feature, train_id = descriptors(X_train)
    test_features, test_id = descriptors(X_test)

    # ==== Building a Vocabulary ====
    sift_features_np = np.concatenate(train_feature)
    kmeans = KMeans(n_clusters=100, random_state=42)
    kmeans.fit(sift_features_np)

    # ==== Creating Histograms (for both train and test using the same Kmeans) ====
    X_train_hist = create_histograms(train_feature, kmeans) 
    X_test_hist = create_histograms(test_features, kmeans)

    # Fixing padding issues
    full_X_train = np.zeros((len(X_train), 100), dtype=int)
    full_X_test  = np.zeros((len(X_test), 100), dtype=int)

    full_X_train[train_id] = X_train_hist
    full_X_test[test_id]   = X_test_hist

    # ==== Adjusting Frequency Vectors ====
    tfidf = TfidfTransformer()
    tfidf.fit(full_X_train)
    X_train_tfidf = tfidf.transform(full_X_train)
    X_test_tfidf = tfidf.transform(full_X_test)
    
    return X_train_tfidf.toarray(), X_test_tfidf.toarray()

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
