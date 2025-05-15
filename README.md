# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Loading & Preparation: The California Housing dataset is loaded and converted into a DataFrame, where "AveOccup" and "HousingPrice" are chosen as the dependent variables (targets), and the rest (excluding "AveOccup") are used as features (independent variables).

2.Data Splitting: The data is split into training and testing sets using an 80-20 split.

3.Feature Scaling: Both the feature data (X) and target variables (Y) are standardized using StandardScaler to improve the performance of the gradient-based learning model.

4.Model Training: A stochastic gradient descent (SGD) regressor is wrapped in a MultiOutputRegressor to support multi-target prediction, and the model is trained on the scaled training data.

5.Prediction & Evaluation: Predictions are made on the test data, then inverse-transformed to return to original scale. The performance is evaluated using Mean Squared Error (MSE), and a sample of predicted values is displayed.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Rishab p doshi 
RegisterNumber:  212224240134
*/
```
```
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
```
```
dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
```
```
df['HousingPrice'] = dataset.target
print(df.head())
```
```
X = df.drop(columns=['AveOccup', 'HousingPrice'])  # Independent variables
X.info()
```
```
Y = df[['AveOccup', 'HousingPrice']]  # Dependent variables (Multi-output)
Y.info()
```
```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
```
```
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)
```
```
Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
```
```
print("Mean Squared Error:", mse)
```
```
print("\nPredictions:\n", Y_pred[:5])
```

## Output:
CONTENT OF DATAFILE

![image](https://github.com/user-attachments/assets/d607dc98-28a4-4c09-960a-c4ae00c98042)

X-INFO

![image](https://github.com/user-attachments/assets/87df8a83-b316-4c3f-8b4e-6aed7d34ed51)

Y-INFO

![image](https://github.com/user-attachments/assets/2e0165e0-c9ca-4f1b-a135-b080850b7998)

MSE VALUES

![image](https://github.com/user-attachments/assets/e726939c-bb9d-41ca-bc8f-5fa3e916ef4f)

PREDICTED VALUES

![image](https://github.com/user-attachments/assets/1cb57370-d411-4ff6-b028-7259c6af508f)









## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
