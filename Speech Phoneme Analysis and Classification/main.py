# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.decomposition import PCA

# Read the data from the CSV file
filename = 'feature_extraction_data.csv'
data = pd.read_csv(filename)

# Perform PCA to select the most informative features
pca = PCA(n_components=2)
X = pca.fit_transform(data[['Formant 1 (Hz)', 'Formant 2 (Hz)', 'Formant 3 (Hz)']])
y = data['Class Number']

# Range of k values to try
k_values = [3, 5, 7, 9, 11]

# Number of experiments
num_experiments = 5

# Distance metric to use
#distance_metric = 'manhattan'

# Initialize variables to store the best k and its average F1 score
best_k = None
best_average_f1_score = -1

# Loop through the different k values
for k in k_values:
    # Initialize a list to store the F1 scores of each experiment
    f1_scores = []
    print(f"----- k = {k} -----\n")
    
    # Loop through the number of experiments
    for i in range(num_experiments):
        # Split the data into training and test sets, ensuring different sets in each experiment
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i)

        # Implement the k-Nearest Neighbors algorithm with the current k value
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Predict the phoneme classes for the test data
        y_pred = knn.predict(X_test)

        # Evaluate the classifier using a confusion matrix and F1 score
        conf_matrix = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Print the results of the current experiment
        print(f"Experiment {i+1}:")
        print("Confusion Matrix:")
        print(conf_matrix)
        print(f"F1 Score: {f1}\n")
        
        # Append the F1 score to the list for calculating the average later
        f1_scores.append(f1)

    # Calculate the average F1 score over all experiments for the current k value
    average_f1_score = np.mean(f1_scores)
    print(f"Average F1 Score for k = {k}: {average_f1_score}\n")
    print("-" * 125 + "\n")
    
    # Update the best k value and its average F1 score if necessary
    if average_f1_score > best_average_f1_score:
        best_k = k
        best_average_f1_score = average_f1_score

# Print the best k value and its average F1 score
print(f"Best performing k value: {best_k}")
print(f"Average F1 Score for best k: {best_average_f1_score}")
print("\n" + "-" * 125 + "\n")