import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

#  Pre-Processing ~ Remove Outliers from dataset
# Over each column (except open/closed)
def removeOutliers(data):
    removeList = []
    for i in range(data.values.shape[1] - 1):
        # Calculate Mean & Standard Deviation
        Data_Mean, Data_STD = np.mean(data.values[:,i]), np.std(data.values[:,i])
        # Define Outlier Boundary by the standard deviation
        bound = Data_STD * 4
        lower, upper = Data_Mean - bound, Data_Mean + bound
        # Remove Outliers that are below the lower bound
        below = [j for j in range(data.values.shape[0]) if data.values[j,i] < lower]
        if below:
            for j in range(len(below)):
                removeList.append(below[j]) # Append Row to Remove to Remove List
        # Remove Outliers that are above the upper bound
        above = [j for j in range(data.values.shape[0]) if data.values[j,i] > upper]
        if above:
            for j in range(len(above)):
                removeList.append(above[j]) # Append Row to Remove to Remove List
    # Remove Duplicates if they exist
    removeList = list(set(removeList))
    # Sort Remove List in increasing order
    removeList.sort()
    print("length: ", len(removeList))
    # Remove Rows for Data Frame
    for i in range(len(removeList)):
        data = data.drop(removeList[i])
    return data

# Attributes:
# Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate
# Fine Aggregate, Age, Concrete Compressive Strength
# 8 input variables, 1 output variable
# Retrieve Data from GitHub Repository
url = "https://raw.githubusercontent.com/jamesH-48/Gradient-Descent-A1/master/Concrete_Data.csv"
data = pd.read_csv(url, header=None)
data = data.rename(columns={0:"Cement",1:"Blast Furnace Slag",2:"Fly Ash",3:"Water",4:"Superplasticizer",5:"Coarse Aggregate",6:"Fine Aggregate",7:"Age",8:"Concrete Compressive Strength"})
values = data.values

# Add Column Intercept of 1s to data frame
# data.insert(0, 'Intercept', 1)
# So far hasn't changed values

'''
Pre-Processing 
'''
# Pre-Processing ~ Remove Outliers
data = removeOutliers(data)

'''
Graphic Display
'''
# Compute pairwise correlation of columns
corr = data.corr()
# Display Heatmap of Correlations
sns.set()
axi1 = sns.heatmap(corr, cmap="BuPu", annot=True)
#axi2 = sns.pairplot(data)

'''
Graphic Display
'''
# Plot Data ~ each column has its own subplot
fig1 = plt.figure()
fig1.suptitle('Input Attributes', fontsize=16)
for i in range(values.shape[1]):
    plt.subplot(values.shape[1], 1, i + 1)
    plt.plot(values[:, i])

'''
Pre-Processing 
'''
# Pre-Processing ~ Train-Test Split Dataset
# Create X data frame that contains the inputs
Xdf = data[["Cement","Blast Furnace Slag","Fly Ash","Water","Superplasticizer","Coarse Aggregate","Fine Aggregate","Age"]]
# Convert X data frame to numpy array (just known as X)
X = Xdf.to_numpy()
# Create Y data frame that contains the output
Ydf = data[["Concrete Compressive Strength"]]
# Convert Y data frame to numpy array (just known as Y)
Y = Ydf.to_numpy()
# Split into 4 datasets for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

# Proof of Shape
print("X_train trans: ", np.transpose(X_train).shape)
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("Y_train: ", Y_train.shape)
print("Y_test: ", Y_test.shape)

# Create Linear Regression Object
regr = LinearRegression()
# Train the model using the training datasets
regr.fit(X_train,Y_train)
# Make predictions using the testing dataset
Y_pred1 = regr.predict(X_train)
Y_pred2 = regr.predict(X_test)
print('Coefficients: \n', regr.coef_)
print("MSE Train: ", mean_squared_error(Y_pred1,Y_train, squared=True))
print("R^2 Train: ", r2_score(Y_pred1,Y_train))
print("MSE Test: ", mean_squared_error(Y_pred2,Y_test, squared=True))
print("R^2 Test: ", r2_score(Y_pred2,Y_test))

# Print Plot of Outputs
figure1, ax = plt.subplots()
figure2, ax2 = plt.subplots()
ax.plot(Y_train, color='red', markersize=5)
ax.plot(Y_pred1, color='cyan', markersize=5)
ax2.plot(Y_test, color='black', markersize=5)
ax2.plot(Y_pred2, color='magenta', markersize=5)
plt.show()
