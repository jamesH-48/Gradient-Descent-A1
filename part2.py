import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
url = "https://raw.githubusercontent.com/jamesH-48/Gradient-Descent-A1/master/Concrete_Data.csv"
data = pd.read_csv(url, header=None)
values = data.values
fig1 = plt.figure()
print(np.max(values,axis=0))

# Train-Test Split Dataset
X = values[:, [0, 1, 2, 3, 4, 5, 6, 7]]
Y = values[:, 8]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=7)
# Create Linear Regression Object
regr = LinearRegression()
# Train the model using the training datasets
regr.fit(X_train,Y_train)
# Make predictions using the testing dataset
Y_pred = regr.predict(X_test)
print('Coefficients: \n', regr.coef_)
print(mean_squared_error(Y_pred,Y_test))
print(r2_score(Y_pred,Y_test))

# Print Plot of Outputs
plt.plot(Y_test, color='black', markersize=5)
plt.plot(Y_pred, color='magenta', markersize=5)
plt.show()