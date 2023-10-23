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


Program to implement the simple linear regression model for predicting the marks scored.

Developed by: 212222240044

RegisterNumber:  KANISHKAR M


```python
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
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```
## Output:

### Input Data

![image](https://github.com/PSriVarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/114944059/387e03fa-748e-444c-842a-625d7043d96f)

### Head and Tail of the Data set 

![image](https://github.com/PSriVarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/114944059/5d78824f-2cc0-4aaf-9a94-8b32de49100f)

### X Values and Y Values 

![image](https://github.com/PSriVarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/114944059/66e1afef-09da-4db4-a9aa-d9c02fd7e254)

### Prediction Values 

![image](https://github.com/PSriVarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/114944059/6412580e-f138-476e-adb4-47aa36b95f32)

### Training Set 

![image](https://github.com/PSriVarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/114944059/33f85129-d75f-4dbf-bb05-4efeab624031)

### Test Set 

![image](https://github.com/PSriVarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/114944059/eaa07de8-cfdc-4a8e-af6a-be627ac41844)

### Values of MSE , MAE and RMSE

![image](https://github.com/PSriVarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/114944059/3f687733-3423-4794-a5dc-6ad13e2f7898)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
