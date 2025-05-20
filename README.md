# Project_4 – mit Voting & Stacking (Meta-Klassifikatoren)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC

# Load data
data = pd.read_csv('../04-project/data/diabetes.csv')

# Überblick
print(data.head())
print(data.shape)
print(data.isnull().sum())
print(data.duplicated().sum())
print(data.dtypes)
print(data.nunique())

# Null-Werte als NaN ersetzen
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols] = data[cols].replace(0, np.nan)
data.fillna(data.median(), inplace=True)

# Pfad für Plots
output_folder = '../04-project/plots'
os.makedirs(output_folder, exist_ok=True)

# Boxplots
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

# Korrelationsmatrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'correlation_matrix.png'))
plt.close()

# Features und Ziel
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Skalierung
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5-fache Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Basis-Modelle
logreg_l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
knn = KNeighborsClassifier()
rf = RandomForestClassifier()
svm = SVC(probability=True)

# Meta-Modelle
voting = VotingClassifier(estimators=[
    ('knn', knn),
    ('rf', rf),
    ('svm', svm)
], voting='soft')

stacking = StackingClassifier(estimators=[
    ('knn', knn),
    ('rf', rf),
    ('svm', svm)
], final_estimator=logreg_l1)

# Modelle definieren
models = {
    'L1 Logistic Regression': logreg_l1,
    'K-Nearest Neighbors': knn,
    'Random Forest': rf,
    'Support Vector Machine': svm,
    'Voting Classifier': voting,
    'Stacking Classifier': stacking
}

# Bewertungsmetriken
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

# Ergebnisse sammeln
results = {}

for name, model in models.items():
    print(f'\n🔍 {name}')
    scores = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring)
    avg_scores = {metric: np.mean(scores[f'test_{metric}']) for metric in scoring}
    results[name] = avg_scores
    for metric, value in avg_scores.items():
        print(f'{metric.capitalize():<10}: {value:.3f}')

# In DataFrame für Plot umwandeln
results_df = pd.DataFrame(results).T

# Balkendiagramm für Accuracy
plt.figure(figsize=(10, 5))
sns.barplot(x=results_df.index, y=results_df['accuracy'])
plt.ylabel('Accuracy (5-Fold CV)')
plt.title('Model Performance Comparison (inkl. Meta-Klassifikatoren)')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'model_cv_meta_comparison.png'))
plt.close()
