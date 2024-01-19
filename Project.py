import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
from scipy.stats import loguniform
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier
import seaborn as sns
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Perceptron
from scipy import stats


#reading Data
df = pd.read_csv('water_potability.csv')

#sns.pairplot(df)
#plt.show()
# Correlation heatmap to visualize relationships between features
"""
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Distribution of a numerical variable using a histogram
sns.histplot(df['numerical_column'], kde=True)
plt.title('Distribution of Numerical Column')
plt.show()

# Count plot for a categorical variable
sns.countplot(x='categorical_column', data=df)
plt.title('Count Plot for Categorical Column')
plt.show()
"""
def remove_outliers_zscore(data, threshold=3):
    z_scores = stats.zscore(data)
    outliers = (abs(z_scores) > threshold).any(axis=1)
    return data[~outliers]

outliers_before = (abs(stats.zscore(df)) > 3).any(axis=1)
num_outliers_before = outliers_before.sum()

df_no_outliers = remove_outliers_zscore(df)

# Calculate the number of outliers after removing
outliers_after = (abs(stats.zscore(df_no_outliers)) > 3).any(axis=1)
num_outliers_after = outliers_after.sum()

print(f"Number of outliers before removing: {num_outliers_before}")
print(f"Number of outliers after removing: {num_outliers_after}")

# Calculate the total number of null or empty values in each row
df['null_count'] = df.isnull().sum(axis=1) + df.eq('').sum(axis=1)

rows_to_remove = df.sort_values(by='null_count', ascending=False).head().index

# Remove rows with the most null or empty values
df = df.drop(index=rows_to_remove)

# Drop the 'null_count' column if you no longer need it
df = df.drop(columns='null_count')

rows_with_null = df.isnull().any(axis=1).sum()
print("Number of rows that have null values (before filling them):",rows_with_null)

#Replacing empty cells.
a = df["ph"].mean()
df["ph"].fillna(a, inplace = True)

b = df["Hardness"].mean()
df["Hardness"].fillna(b, inplace = True)

c = df["Solids"].mean()
df["Solids"].fillna(c, inplace = True)

e = df["Chloramines"].mean()
df["Chloramines"].fillna(e, inplace = True)

f = df["Sulfate"].mean()
df["Sulfate"].fillna(f, inplace = True)

g = df["Conductivity"].mean()
df["Conductivity"].fillna(g, inplace = True)

h = df["Organic_carbon"].mean()
df["Organic_carbon"].fillna(h, inplace = True)

j = df["Trihalomethanes"].mean()
df["Trihalomethanes"].fillna(j, inplace = True)

i = df["Turbidity"].mean()
df["Turbidity"].fillna(i, inplace = True)

rows_with_null = df.isnull().any(axis=1).sum()
print("Number of rows that have null values (after filling):",rows_with_null)
print("\n")

positive_class = df[df['Potability'] == 1]
negative_class = df[df['Potability'] == 0]

print(f"Positive class percentage: {len(positive_class) / len(df) * 100}")
print(f"Negative class percentage: {len(negative_class) / len(df) * 100}")

#for distribution of the features
"""
for column in df.columns:
    sns.histplot(df[column], kde=False)
    plt.title(f'Histogram of {column}')
    plt.show()
"""

#rounding to 2 decimal places
df = df.round(2)

df['Potability'] = df['Potability'].astype(int)
print("\n")
#print(df)

X = df.drop(columns='Potability')  # Features
y = df['Potability']  # Target variable

#scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k_folds = 7
kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)



# First Algorithm - Logistic Regression
param_dist = {'C': loguniform(0.001, 100)}

model = LogisticRegression(random_state=42)

random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=kf, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

best_C = random_search.best_params_['C']

cv_results = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
mean_cv_accuracy = cv_results.mean()

best_model = LogisticRegression(C=best_C, random_state=42, max_iter=5000, solver='liblinear')
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Best hyperparameters: {best_C}")
print(f"Accuracy(LogisticRegression): {accuracy:.2f}")
print(f"  F1 Score: {f1:.2f}")
print(f"  Precision: {precision:.2f}")
print(f"  Recall: {recall:.2f}")
print("\n")


# Second Algorithm - SVM
param_dist_svm = {
    'C': loguniform(0.01, 50),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

svm_model = SVC(random_state=42)

random_search_svm = RandomizedSearchCV(svm_model, param_distributions=param_dist_svm, n_iter=20, cv=kf, scoring='accuracy', random_state=42)
random_search_svm.fit(X_train, y_train)

best_params_svm = random_search_svm.best_params_

best_svm_model = SVC(**best_params_svm, random_state=42)
best_svm_model.fit(X_train, y_train)

y_pred_svm = best_svm_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)

print(f"Best hyperparameters: {best_params_svm}")
print(f"Accuracy(SVM): {accuracy_svm:.2f}")
print(f"  F1 Score: {f1_svm:.2f}")
print(f"  Precision: {precision_svm:.2f}")
print(f"  Recall: {recall_svm:.2f}")
print("\n")

# Third Algorithm - Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Define a parameter grid for hyperparameter tuning
param_dist_decision_tree = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Use RandomizedSearchCV for hyperparameter tuning
random_search_decision_tree = RandomizedSearchCV(
    decision_tree_model, param_distributions=param_dist_decision_tree,
    n_iter=10, cv=5, scoring='accuracy', random_state=42
)
random_search_decision_tree.fit(X_train, y_train)

# Get the best hyperparameters
best_params_decision_tree = random_search_decision_tree.best_params_

# Cross-validation using the best hyperparameters
cv_results_decision_tree = cross_val_score(decision_tree_model, X_train, y_train, cv=kf, scoring='accuracy')
mean_cv_accuracy_decision_tree = cv_results_decision_tree.mean()

# Use the best hyperparameters to train the final model
best_decision_tree_model = DecisionTreeClassifier(**best_params_decision_tree, random_state=42)
best_decision_tree_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_decision_tree = best_decision_tree_model.predict(X_test)

# Evaluate the model's performance
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
f1_decision_tree = f1_score(y_test, y_pred_decision_tree)
precision_decision_tree = precision_score(y_test, y_pred_decision_tree)
recall_decision_tree = recall_score(y_test, y_pred_decision_tree)

print(f"Best hyperparameters: {best_params_decision_tree}")
print(f"Accuracy (DecisionTreeClassifier): {accuracy_decision_tree:.2f}")
print(f"  F1 Score: {f1_decision_tree:.2f}")
print(f"  Precision: {precision_decision_tree:.2f}")
print(f"  Recall: {recall_decision_tree:.2f}")
print("\n")

# Fourth Algorithm - KNeighborsClassifier
knn_model = KNeighborsClassifier()

# Define a parameter grid for hyperparameter tuning
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]  # 1 for Manhattan distance (L1), 2 for Euclidean distance (L2)
}

# Use GridSearchCV for hyperparameter tuning
grid_search_knn = GridSearchCV(
    knn_model, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search_knn.fit(X_train, y_train)

# Get the best hyperparameters
best_params_knn = grid_search_knn.best_params_

# Use the best hyperparameters to train the final model
best_knn_model = KNeighborsClassifier(**best_params_knn)
best_knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_knn = best_knn_model.predict(X_test)

# Evaluate the model's performance
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)

print(f"Best hyperparameters: {best_params_knn}")
print(f"Accuracy (KNeighborsClassifier): {accuracy_knn:.2f}")
print(f"  F1 Score: {f1_knn:.2f}")
print(f"  Precision: {precision_knn:.2f}")
print(f"  Recall: {recall_knn:.2f}")
print("\n")

# Fifth algorithm XGboost

# Create DMatrix objects for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set hyperparameters
params = {
    'objective': 'binary:logistic',  
    'max_depth': 3,
    'learning_rate': 0.1,
    'eval_metric': 'logloss',
}

# Train the XGBoost model
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)

# Use XGBoost's cv method for k-fold cross-validation
cv_results_xgboost = xgb.cv(params, dtrain, num_boost_round=num_rounds, nfold=k_folds, metrics='error', seed=42)

# Cross-validation using XGBoost
mean_cv_accuracy_xgboost = 1 - cv_results_xgboost['test-error-mean'].iloc[-1]


# Make predictions on the test set
y_pred = model.predict(dtest)

# Convert probabilities to binary predictions
predictions = [round(value) for value in y_pred]

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
f1_xgboost = f1_score(y_test, predictions)
precision_xgboost = precision_score(y_test, predictions)
recall_xgboost = recall_score(y_test, predictions)

print(f"Accuracy (Xgboost): {accuracy:.2f}")
print(f"  F1 Score: {f1_xgboost:.2f}")
print(f"  Precision: {precision_xgboost:.2f}")
print(f"  Recall: {recall_xgboost:.2f}")

# Sixth algorithm RandomForest
random_forest_model = RandomForestClassifier(random_state=42)

# Define a parameter grid for hyperparameter tuning
param_dist_random_forest = {
    'n_estimators': [50, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['log2', 'sqrt', None],
}
random_search_random_forest = RandomizedSearchCV(
    random_forest_model, param_distributions=param_dist_random_forest,
    n_iter=10, cv=5, scoring='accuracy', random_state=42
)
random_search_random_forest.fit(X_train, y_train)

# Get the best hyperparameters
best_params_random_forest = random_search_random_forest.best_params_

# Use the best hyperparameters to train the final model
best_random_forest_model = RandomForestClassifier(**best_params_random_forest, random_state=42)
best_random_forest_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_random_forest = best_random_forest_model.predict(X_test)

# Evaluate the model's performance
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)
f1_random_forest = f1_score(y_test, y_pred_random_forest)
precision_random_forest = precision_score(y_test, y_pred_random_forest)
recall_random_forest = recall_score(y_test, y_pred_random_forest)


print("\n")
print(f"Best hyperparameters: {best_params_random_forest}")
print(f"Accuracy (RandomForestClassifier): {accuracy_random_forest:.2f}")
print(f"  F1 Score: {f1_random_forest:.2f}")
print(f"  Precision: {precision_random_forest:.2f}")
print(f"  Recall: {recall_random_forest:.2f}")
print("\n")

feature_importance = best_random_forest_model.feature_importances_
#print(len(X))
#print(len(feature_importance))
# Visualize feature importance
plt.bar(range(len(feature_importance)), feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance from Random Forest')
plt.xticks(rotation=45)
plt.show()

#Ensemble Learning
voting_classifier = VotingClassifier(
    estimators=[
        ('logistic', best_model),
        ('svm', best_svm_model),
        ('decision_tree', best_decision_tree_model),
        ('knn', best_knn_model),
        ('random_forest', best_random_forest_model)
    ],
    voting='hard'
)
# Train the Voting Classifier on the entire training set
voting_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_voting = voting_classifier.predict(X_test)

# Evaluate the Voting Classifier
accuracy_voting = accuracy_score(y_test, y_pred_voting)
f1_voting = f1_score(y_test, y_pred_voting)
precision_voting = precision_score(y_test, y_pred_voting)
recall_voting = recall_score(y_test, y_pred_voting)

print("Ensemble Learning Results:")
print(f"Accuracy (Voting Classifier): {accuracy_voting:.2f}")
print(f"F1 Score (Voting Classifier): {f1_voting}")


#stacked learning 
# Define base models
base_models = [
    ('logistic', best_model),
    ('svm', best_svm_model),
    ('decision_tree', best_decision_tree_model),
    ('knn', best_knn_model),
    ('random_forest', best_random_forest_model)
]
#class_weight={'logistic': 1, 'svm': 2, 'random_forest': 1}
# Define meta-model
meta_model = SVC(random_state= 42)

# Create the stacked model

stacked_classifier = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    stack_method='auto'
)

# Train the stacked model on the entire training set
stacked_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_stacked = stacked_classifier.predict(X_test)

# Evaluate the stacked model
accuracy_stacked = accuracy_score(y_test, y_pred_stacked)
f1_stacked = f1_score(y_test, y_pred_stacked)
precision_stacked = precision_score(y_test, y_pred_stacked)
recall_stacked = recall_score(y_test, y_pred_stacked)

print("\n")
print("Stacked Model Results:")
print(f"Accuracy (Stacked Model): {accuracy_stacked:.2f}")
print(f"F1 Score (Stacked Model): {f1_stacked:.2f}")
print(f"Precision (Stacked Model): {precision_stacked:.2f}")
print(f"Recall (Stacked Model): {recall_stacked:.2f}")


