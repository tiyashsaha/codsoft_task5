import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset (ensure you replace this with the correct file path)
data = pd.read_csv("creditcard.csv")

# Check for null values
print(data.isnull().sum())

# Basic Data Exploration
print(data.head())
print(data.describe())

# Visualize the distribution of classes (fraud vs. non-fraud)
sns.countplot(x='Class', data=data)
plt.title("Class Distribution")
plt.show()

# Feature Engineering: 
# Separate the features (X) and the target (y)
X = data.drop('Class', axis=1)  # Drop the target variable (Class)
y = data['Class']  # Target variable (Class) where 0 is non-fraud, 1 is fraud

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
