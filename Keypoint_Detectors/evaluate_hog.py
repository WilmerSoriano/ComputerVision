# Train an SVM classifier
import numpy as np
from sklearn.svm import LinearSVC

# ================Data extraction====================
data = np.load("HOG_cifar10_features.npz", allow_pickle=True)
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

# Create an SVM model
svm = LinearSVC(random_state=42)

# Train the model
svm.fit(X_train, y_train)

# Evaluate the model
accuracy = svm.score(X_test, y_test)
print(f'SVM accuracy: {accuracy:.2f}')