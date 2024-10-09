import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #The Code will split data into two by itself
from sklearn.linear_model import LogisticRegression #Importing the Regression Model
from sklearn.metrics import accuracy_score #Identifying the Accuracy Score

sonar_data = pd.read_csv('sonar data.csv', header=None)
#Total Data
#print(sonar_data[60].value_counts())
#Data Grouped By mean
#print(sonar_data.groupby(60).mean())
#Seperating Data and Labels
X=sonar_data.drop(columns=60, axis=1)
Y=sonar_data[60]
#Spliting Train and Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, stratify=Y, random_state=1)
#print(X.shape,X_train.shape, X_test.shape)
#print(Y.shape,Y_train.shape, Y_test.shape)
#Training Model
model = LogisticRegression()
model.fit(X_train, Y_train)
#Model Evaluation 
#testing on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
#print("Result on Training Data: ")
#print(training_data_accuracy)   
#testing on Test Data
X_test_prediction=model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
#print("Result on Testing Data: ")
#print(testing_data_accuracy)   
#Making Predictive System
input_data = (0.0261, 0.0266, 0.0223, 0.0749, 0.1364, 0.1513, 0.1316, 0.1654, 0.1864, 0.2013,
0.2890, 0.3650, 0.3510, 0.3495, 0.4325, 0.5398, 0.6237, 0.6876, 0.7329, 0.8107,
0.8396, 0.8632, 0.8747, 0.9607, 0.9716, 0.9121, 0.8576, 0.8798, 0.7720, 0.5711,
0.4264, 0.2860, 0.3114, 0.2066, 0.1165, 0.0185, 0.1302, 0.2480, 0.1637, 0.1103,
0.2144, 0.2033, 0.1887, 0.1370, 0.1376, 0.0307, 0.0373, 0.0606, 0.0399, 0.0169,
0.0135, 0.0222, 0.0175, 0.0127, 0.0022, 0.0124, 0.0054, 0.0021, 0.0028, 0.0023)
#changing input data into numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the np array as we are predicting for on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(prediction)