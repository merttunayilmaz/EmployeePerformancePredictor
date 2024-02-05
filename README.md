# Employee Performance Predictor

## Project Overview
This project aims to predict employee performance based on various features like age, department, education level, annual leave usage, weekly working hours, satisfaction level, and an artificially generated salary. The dataset is processed and fed into three different machine learning models: Logistic Regression, Decision Tree, and K-Nearest Neighbors (KNN), to evaluate their performance in predicting the performance rating of employees.

## Setup
To run this project, you need to have Python installed on your machine along with the following libraries:
- pandas
- numpy
- scikit-learn
- joblib

You can install these dependencies using pip: pip install pandas numpy scikit-learn joblib


## Running the Code
1. Place your employee dataset in the `Data` directory and name it `employee.csv`.
2. Run the script using your preferred Python environment or IDE. The script will:
   - Load the dataset and provide a statistical summary.
   - Fill missing salary data with random values.
   - Scale the feature data using Min-Max normalization.
   - Split the dataset into training and testing sets.
   - Train Logistic Regression, Decision Tree, and KNN models.
   - Evaluate the models' accuracy and display the results.
   - Export the Decision Tree model as a joblib file.

## Data
The dataset should be in a CSV format with columns for age, department, education level, annual leave usage, weekly working hours, satisfaction level, and performance rating. The script will add a `Maas` (salary) column with random values.

## Models and Performance
The script evaluates three models:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors

The performance of these models is assessed based on accuracy, and the results are displayed in the console. Due to the randomness introduced by the salary generation and data splitting, results may vary between runs. To ensure reproducibility, a random seed is set at the beginning of the script.

## Exporting the Model
The Decision Tree model, identified for its performance or suitability, is exported as `decision_tree_model.joblib`, which can be loaded and used for further predictions.

## Contributing
Contributions to this project are welcome. Please ensure to follow the coding standards and submit pull requests for any enhancements.

