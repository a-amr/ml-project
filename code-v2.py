import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import cv2

# Load train data
train_file = '/home/ababa/Downloads/ml-project/code-test/v1/train-classes/train.csv'
# Update with the correct path
train_data = pd.read_csv(train_file, delimiter=';')

# Load test data
test_file = '/home/ababa/Downloads/ml-project/code-test/v1/test-classes/test.csv'
# Update with the correct path
test_data = pd.read_csv(test_file, delimiter=';')

# Load images
image_folder = '/home/ababa/Downloads/ml-project/code-test/v1/train-classes/'
image_folder_test = '/home/ababa/Downloads/ml-project/code-test/v1/test-classes/'
train_images = [os.path.join(image_folder, filename) for filename in train_data['Filename']]
test_images = [os.path.join(image_folder_test, filename) for filename in test_data['Filename']]


# Function to load images
# Function to load and resize images
def load_images(image_paths, target_size=(100, 100)):
    images = []
    for path in image_paths:
        # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(path)
        img = cv2.resize(img, target_size)  # Resize to a common size
        #cv2.imshow("Resized Image", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #hog = cv2.HOGDescriptor()
        # --------------------------------------------
        #hoog = hog.compute(img)
        images.append(img.flatten())
    return np.array(images)

# Load training and test images
X_train = load_images(train_images)
X_test = load_images(test_images)
y_train = train_data['ClassId']
y_test = test_data['ClassId']

# Train the logistic regression model
logreg = LogisticRegression(random_state=42,max_iter=1000)
logreg.fit(X_train, y_train)
# Evaluate the model
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
# kmeans model
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_assignments = kmeans.fit_predict(X_train)
# Predict the clusters for the test data
test_cluster_assignments = kmeans.predict(X_test)
# Add cluster assignments to the test set
test_data['KMeans_Cluster'] = test_cluster_assignments


print("Number of Clusters:", kmeans.n_clusters)
test_kmean = '/home/ababa/Downloads/ml-project/code-test/v1/test.csv'



# test_data.to_csv(test_kmean, index=False)
# read from kmean  csv file
test_kmenans_file = pd.read_csv(test_kmean, delimiter=',')
accuracy3 = accuracy_score(test_kmenans_file['KMeans_Cluster'],test_kmenans_file['ClassId'])
print(f"Accuracy from file with changed cluster names: {accuracy3}")


# Plot confusion matrix
conf_matrix = confusion_matrix(test_data['ClassId'], y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

