import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Read the data from the CSV file
filename = 'feature_extraction_data.csv'
data = pd.read_csv(filename)

# Preprocess the data and split it into training and test sets
X = data[['Formant 1 (Hz)', 'Format 2 (Hz)', 'Format 3 (Hz)']]
y = data['Class Number']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=data['Gender'])

# Choose the value of 'k'
k = 5

# Implement the k-Nearest Neighbors algorithm
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predict the phoneme classes for the test data
y_pred = knn.predict(X_test)

# Evaluate the classifier using a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)