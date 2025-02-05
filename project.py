# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
file_path = "E:/Academic/data_analysis/Social_Network_Ads.csv"
try:
    dataset = pd.read_csv(file_path)
    print("The imported data is:\n", dataset.head())
except FileNotFoundError:
    print("The specified file path is invalid. Please check the path.")
    exit()

# Insert some blank data and incorrect data types
dataset.loc[0:2, 'Age'] = np.nan  # Insert some blank data
if 'Gender' not in dataset.columns:
    dataset['Gender'] = [25, "Male", "Female", 30]  # Insert incorrect data types

# Show the dataset with inserted anomalies
print("Dataset with inserted blank and incorrect data:\n", dataset.head())

# Select features and target
features = dataset.iloc[:, [2, 3]].values
target = dataset.iloc[:, 4].values

# Handle missing data using SimpleImputer
imputer = SimpleImputer(strategy='mean')
features[:, 0:1] = imputer.fit_transform(features[:, 0:1])
print("Processed features after handling missing data:\n", features[:5])

# Split the dataset into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=0)

# Apply MinMax scaling
scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Show scaled training and testing datasets
print("Scaled training data:\n", features_train[:5])
print("Scaled testing data:\n", features_test[:5])

# Train SVM model with linear kernel
svm_classifier = SVC(kernel='linear', random_state=0)
svm_classifier.fit(features_train, target_train)

# Make predictions
svm_predictions = svm_classifier.predict(features_test)

# Evaluate model performance
svm_confusion_matrix = confusion_matrix(target_test, svm_predictions)
svm_accuracy = accuracy_score(target_test, svm_predictions)

print("SVM Confusion Matrix:")
print(svm_confusion_matrix)
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print("SVM Classification Report:")
print(classification_report(target_test, svm_predictions))

# Train Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=0)
rf_classifier.fit(features_train, target_train)

# Make predictions
rf_predictions = rf_classifier.predict(features_test)

# Evaluate model performance
rf_confusion_matrix = confusion_matrix(target_test, rf_predictions)
rf_accuracy = accuracy_score(target_test, rf_predictions)

print("Random Forest Confusion Matrix:")
print(rf_confusion_matrix)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print("Random Forest Classification Report:")
print(classification_report(target_test, rf_predictions))

# Visualize SVM decision boundary
x_min, x_max = features_train[:, 0].min() - 1, features_train[:, 0].max() + 1
y_min, y_max = features_train[:, 1].min() - 1, features_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                      np.arange(y_min, y_max, 0.01))
zz = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
zz = zz.reshape(xx.shape)

plt.figure(figsize=(12, 7))
plt.contour(xx, yy, zz, colors='black', linewidths=0.5)
plt.scatter(features_train[:, 0], features_train[:, 1], c=target_train, cmap='plasma', edgecolor='white', s=50)
plt.title('Enhanced SVM Decision Boundary Visualization Without Background')
plt.xlabel('Feature 1 (Scaled Age)')
plt.ylabel('Feature 2 (Scaled Estimated Salary)')
plt.show()