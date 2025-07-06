# Train an SVM classifier
from sklearn.svm import LinearSVC
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_histograms_tfidf, np.array(y_features, dtype=int), test_size=0.2, random_state=42)
# Create an SVM model
svm = LinearSVC(random_state=42)

# Train the model
svm.fit(X_train, y_train)

# Evaluate the model
accuracy = svm.score(X_test, y_test)
print(f'SVM accuracy: {accuracy:.2f}')