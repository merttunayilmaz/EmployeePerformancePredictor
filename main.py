import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load

# Configure pandas settings for data display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load the dataset and take an initial look
employee_df = pd.read_csv('Data/employee.csv')
print(employee_df.describe())

np.random.seed(42)  # Rastgeleliği sabitle

# Fill missing salary information with random values
employee_df['Maas'] = np.random.randint(30000, 100000, len(employee_df))

# Separate the data into independent variables (X) and the target variable (y)
X = employee_df.drop('Performans Notu', axis=1)
y = employee_df['Performans Notu']

# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train various models and evaluate their performance
model_performances = {}

# Logistic Regression model
log_reg = LogisticRegression().fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
model_performances['Logistic Regression'] = accuracy_score(y_test, log_reg_pred)

# Decision Tree model
dec_tree = DecisionTreeClassifier().fit(X_train, y_train)
dec_tree_pred = dec_tree.predict(X_test)
model_performances['Decision Tree'] = accuracy_score(y_test, dec_tree_pred)

# KNN model
knn = KNeighborsClassifier().fit(X_train, y_train)
knn_pred = knn.predict(X_test)
model_performances['KNN'] = accuracy_score(y_test, knn_pred)

# Compile performance results into a DataFrame and display
performance_df = pd.DataFrame(list(model_performances.items()), columns=['Model', 'Accuracy'])
print(performance_df.sort_values(by='Accuracy', ascending=False))

# Export the best model
dump(dec_tree, 'decision_tree_model.joblib')

# Load and test the exported model
loaded_model = load('decision_tree_model.joblib')
loaded_model_predictions = loaded_model.predict(X_test)
