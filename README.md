# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given
   datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:SHABREENA VINCENT 
RegisterNumber:  212222230141
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
Dataset:

![image](https://github.com/rajalakshmi8248/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860827/e522e106-0a1a-47b3-be66-1c1ce1b7deed)

Head values:
 
![image](https://github.com/rajalakshmi8248/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860827/447b086f-f37f-4e67-8b11-9dd1feb66f74)

Tail values:

![image](https://github.com/rajalakshmi8248/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860827/76d40c4b-12eb-46ae-a84c-baa0dc5e2f07)

X and Y values:

![image](https://github.com/rajalakshmi8248/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860827/737fc7eb-1931-458e-ad88-2dcab0b3b5cc)

Predication values of X and Y:

![image](https://github.com/rajalakshmi8248/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860827/b0b25d32-73b2-43b6-9369-0ebbb8ebb3b4)

MSE,MAE and RMSE:

![image](https://github.com/rajalakshmi8248/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860827/7e9f72f9-1766-4ef8-b1f9-b249b85bcebd)

Training Set:

![image](https://github.com/rajalakshmi8248/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860827/95ad35ad-9e05-42c7-a70c-35f02b22e12e)

Testing Set:

![image](https://github.com/rajalakshmi8248/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860827/8543f186-7155-413d-a60a-ffe20c0bf1a0)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
