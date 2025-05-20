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


# Selecting the best features for visualization (based on correlation)    SUPPORT VECTOR MACHINE

# Calculate correlations with the target variable
correlations = data.corr(numeric_only=True)['Outcome'].abs().sort_values(ascending=False)

# Select the two most correlated features (excluding Outcome itself)
top2_features = correlations.index[1:3].tolist()  # [0] is 'Outcome'

print("\n🔎 Correlation analysis – strongest predictors for Outcome:")
for feature in top2_features:
    print(f"{feature}: {correlations[feature]:.3f}")

# Result: Glucose and BMI have the highest correlation with the presence of diabetes.
# Therefore, we use these two variables to visualize the decision boundary in 2D space.

# 🔍 Visualizing SVM decision boundary (Glucose & BMI only)
print("\n📈 SVM Decision Boundary (Glucose vs. BMI) is being plotted...")

# Select two features
svm_features = top2_features  # ['Glucose', 'BMI']
X_svm = data[svm_features].values
y_svm = data['Outcome'].values

# Scale the features
scaler_2d = StandardScaler()
X_svm_scaled = scaler_2d.fit_transform(X_svm)

# Train linear SVM
svm_clf = SVC(kernel='linear', max_iter=1000)
svm_clf.fit(X_svm_scaled, y_svm)

# Create grid for decision surface
x_min, x_max = X_svm_scaled[:, 0].min() - 1, X_svm_scaled[:, 0].max() + 1
y_min, y_max = X_svm_scaled[:, 1].min() - 1, X_svm_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = svm_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z > 0, alpha=0.3)
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors='k', linestyles=['--', '-', '--'])
plt.scatter(X_svm_scaled[:, 0], X_svm_scaled[:, 1], c=y_svm, cmap='coolwarm', edgecolors='k')
plt.xlabel(f'{svm_features[0]} (scaled)')
plt.ylabel(f'{svm_features[1]} (scaled)')
plt.title(f'SVM Decision Boundary – {svm_features[0]} vs. {svm_features[1]}')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'svm_decision_boundary_glucose_bmi.png'))
plt.close()
