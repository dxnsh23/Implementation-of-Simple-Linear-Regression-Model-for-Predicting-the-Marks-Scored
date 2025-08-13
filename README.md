Name : S Dinesh Raghavendara


Reg no : 212224040078


# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
 
*/
```

## Output:
Head values
<img width="1090" alt="exp2-1" src="https://github.com/user-attachments/assets/09b3ad21-340b-4bc6-97a8-b8fb7be91c65" />
Tail values
<img width="1090" alt="exp2-2" src="https://github.com/user-attachments/assets/16e97bf5-d82d-491c-86d0-b3f6a4d39dc8" />
Compare dataset
<img width="1090" alt="exp2-3" src="https://github.com/user-attachments/assets/6c85ccf1-4a96-4c9e-92e3-93824e1bce3f" />
Prediction values
<img width="1090" alt="exp2-4" src="https://github.com/user-attachments/assets/e6279d3e-d070-4ed9-8d13-1fb7b76cb957" />
Training set
<img width="1090" alt="exp2-5" src="https://github.com/user-attachments/assets/a8adb3dc-073d-40a0-ae4e-236d7583899a" />
Testing set
<img width="1090" alt="exp2-6" src="https://github.com/user-attachments/assets/ee73bcb0-9b4b-4ef4-9dca-85601c7464e0" />
MSE,MAE and RMSE
<img width="1090" alt="exp2-7" src="https://github.com/user-attachments/assets/06c7b141-4d8d-4ea0-83af-e7c65194aa51" />













## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
