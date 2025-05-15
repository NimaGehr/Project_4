# Project_4
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Load data diabates
data = pd.read_csv('../04-project/data/diabetes.csv')

# data preprocessing and visualisation
# Display the first few rows of the dataset
print(data.head())

# Display the shape of the dataset
print(data.shape)

# Check for missing values
print(data.isnull().sum())

# Check for duplicates
print(data.duplicated().sum())

# Check for data types
print(data.dtypes)

# Check for unique values in each column
print(data.nunique())

# Replace zeros with NaN where 0 is not physiologically meaningful
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols] = data[cols].replace(0, np.nan)

# Fill missing values with the median of each column
data.fillna(data.median(), inplace=True)

# Data path
data_path = '/Users/milanhavranek/Downloads/04-project/data/diabetes.csv'
output_folder = '/Users/milanhavranek/Downloads/04-project/plots'

# create a folder for the plots
os.makedirs(output_folder, exist_ok=True)

# Check for outliers
def check_outliers(data):
    for column in data.columns:
        if data[column].dtype != 'object':
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=data[column])
            plt.title(f'Boxplot of {column}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'boxplot_{column}.png'))
            plt.close()
check_outliers(data)

# Check for correlation
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'correlation_matrix.png'))
plt.close()


# Split the data into features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC() 
    }
results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f'\n{name} Accuracy: {acc:.2f}')
    print(classification_report(y_test, y_pred))


# Visualize the results
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'model_performance_comparison.png'))
plt.close()



