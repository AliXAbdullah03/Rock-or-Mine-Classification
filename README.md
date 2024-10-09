# Rock-or-Mine-Classification
This project uses Logistic Regression to classify sonar data, distinguishing between rocks and mines. It reads data from a CSV, separates features and labels, splits the dataset into training and testing sets, trains the model, evaluates accuracy, and predicts outcomes based on input sonar readings.
Sonar Data Classification Using Logistic Regression
Overview
This project implements a classification model using Logistic Regression to predict sonar data outcomes, specifically to distinguish between rocks and mines based on sonar readings. The dataset consists of various features that represent sonar signal characteristics, with the target variable indicating whether the signal corresponds to a rock or a mine.

Features
**Data Preparation:** The code reads sonar data from a CSV file, separates the features and labels, and groups the data by the target variable to analyze its distribution.

**Data Splitting:** It splits the dataset into training and testing subsets using a stratified split to maintain the same distribution of classes in both sets.

**Model Training:** The Logistic Regression model is trained on the training dataset.

**Model Evaluation:** The code evaluates the model's accuracy on both the training and testing datasets, providing insights into its performance.

**Prediction:** A function is included to make predictions based on new input data, reshaping it as necessary for the model.
Code Walkthrough

**Import Libraries:** The code imports necessary libraries such as NumPy, Pandas, and scikit-learn for data manipulation and model training.

**Load Dataset:** The sonar data is loaded from a CSV file with no header.

**Exploratory Data Analysis:** The code includes commented-out sections to explore data distribution and relationships.

**Feature and Label Separation:** The features (sonar readings) and labels (rock or mine) are separated for modeling.

**Train-Test Split:** The dataset is split into training and testing sets with a test size of 10%.

**Model Training:** A Logistic Regression model is instantiated and trained on the training dataset.

**Model Evaluation:** The model's accuracy is evaluated on both training and testing datasets.

**Prediction:** The model can predict the class for a given set of sonar readings.

# Usage
To use this code, ensure you have the sonar data.csv file in the same directory as the script. You can run the script to train the model and make predictions based on the provided input data. Modify the input_data tuple to test different sonar readings.

# Dependencies
Python 3.x
NumPy
Pandas
scikit-learn
# License
This project is licensed under the MIT License.


