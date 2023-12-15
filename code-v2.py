import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, silhouette_score
from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
import seaborn as sns


script_dir = os.path.dirname(os.path.realpath(__file__))
# Update with the correct relative paths
test_kmean = os.path.join(script_dir, 'test.csv')
test_kmean2 = os.path.join(script_dir, 'test2.csv')
train_file = os.path.join(script_dir, 'train-classes', 'train.csv')
test_file = os.path.join(script_dir, 'test-classes', 'test.csv')
image_folder = os.path.join(script_dir, 'train-classes')
image_folder_test = os.path.join(script_dir, 'test-classes')


# load info from csv files
train_data = pd.read_csv(train_file, delimiter=';')
test_data = pd.read_csv(test_file, delimiter=';')

# Load images
train_images = [os.path.join(image_folder, filename) for filename in train_data['Filename']]
test_images = [os.path.join(image_folder_test, filename) for filename in test_data['Filename']]


# Function to load images
# Function to load and resize images
# Function to make hog for photes insted of gray scale
def load_images(image_paths, target_size=(100, 100), use_hog=True):
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, target_size)  # Resize to a common size


        if use_hog:
            # Compute HOG features
            hog_features, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
            # Concatenate flattened image and HOG features
            features = np.concatenate([img.flatten(), hog_features])
        else:
            features = img.flatten()
        # cv2.imshow("Resized Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # hog = cv2.HOGDescriptor()
        # hoog = hog.compute(img)
       # images.append(img.flatten())
        images.append(features)
    return np.array(images)

# Load training and test images
X_train = load_images(train_images, use_hog=False)
X_test = load_images(test_images, use_hog=False)
y_train = train_data['ClassId']
y_test = test_data['ClassId']

# Train the logistic regression model-----------------
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train, y_train)
# Evaluate the model
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("for Logistic Regression model")
print(f"Accuracy: {accuracy}")
# Plot confusion matrix
conf_matrix = confusion_matrix(test_data['ClassId'], y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')
# kmeans model----------------------------
X_train = load_images(train_images, use_hog=True)
X_test = load_images(test_images, use_hog=True)
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=4, n_init=30)
cluster_assignments = kmeans.fit_predict(X_train)
# Predict the clusters for the test data
test_cluster_assignments = kmeans.predict(X_test)
# Add cluster assignments to the test set
test_data['KMeans_Cluster'] = test_cluster_assignments
print("for kmeans model")
print("Number of Clusters:", kmeans.n_clusters)
#edited by vim
# test_data.to_csv(test_kmean2, index=False)
# read from kmean  csv file of
test_kmenans_file = pd.read_csv(test_kmean, delimiter=',')
test_kmenans_file2 = pd.read_csv(test_kmean2, delimiter=',')
accuracy = accuracy_score(test_kmenans_file['KMeans_Cluster'], test_kmenans_file['ClassId'])
accuracy2 = accuracy_score(test_kmenans_file2['KMeans_Cluster'], test_kmenans_file2['ClassId'])
print(f"Accuracy from file with changed cluster names by hand: {accuracy}")
print(f"Accuracy with hog from file with changed cluster names by hand: {accuracy2}")
# Silhouette analysis for training data
silhouette_train = silhouette_score(X_train, cluster_assignments)
print(f"Silhouette Score for Training Data: {silhouette_train}")
# Silhouette analysis for test data
silhouette_test = silhouette_score(X_test, test_cluster_assignments)
print(f"Silhouette Score for Test Data: {silhouette_test}")


# Apply PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
# Visualize original data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='viridis', legend='full')
plt.title('Original Data Distribution')
# Visualize KMeans clusters
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=cluster_assignments, palette='viridis', legend='full')
plt.title('KMeans Clusters')
plt.show()
