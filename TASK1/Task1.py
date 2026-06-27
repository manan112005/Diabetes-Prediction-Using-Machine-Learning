import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data Collection and Analysis
# PIMA Diabetes Dataset

# Load the dataset
diabetes_data = pd.read_csv('TASK1/diabetes.csv')
print(diabetes_data.head())
print(diabetes_data.shape)
print(diabetes_data.describe())
print(diabetes_data['Outcome'].value_counts())
print(diabetes_data.groupby('Outcome').mean())

# Separate the data and labels
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']
print(X)
print(Y)

# Data standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)

X=standardized_data
Y=diabetes_data['Outcome']
print(X)
print(Y)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Train the model 
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)


# Model evaluation
# Accuracy score on train data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) 
print('Accuracy score of training data : ', training_data_accuracy)

# Accuracy score on test data
X_test_prediction = classifier.predict(X_test)  
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of test data : ', test_data_accuracy)

# Making a predictive system
input_data = (3,173,78,39,185,33.8,0.97,31)


# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0]== 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')   