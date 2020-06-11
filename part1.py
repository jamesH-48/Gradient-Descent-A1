import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn import preprocessing

def gradientdescent(x, y, weights, LR, iterations):
    # Graph MSE
    MSEgraph = np.zeros((iterations,1))
    for k in range(iterations):
        # Initialize Hypothesis
        H = np.dot(x, weights)
        #print("H",H.shape)
        # Define Error
        # E = H - Y
        E = np.subtract(H, y)
        #print("E", E.shape)
        # Define Mean Squared Error
        MSE = (1 / (2 * (int(len(y))))) * np.dot(np.transpose(E), E)
        #print("MSE", MSE.shape)
        MSEgraph[k] = MSE
        # print("MSE ", MSE)
        # Define Gradient -> MSE derivative to weight
        gradient = (1 / (int(len(y)))) * np.dot(np.transpose(x), E)
        #print("gradient", gradient.shape)

        # Revise Weights
        # New Weight = Old Weight - Learning Rate * Gradient
        weights = np.subtract(weights, LR * gradient)
    # Plot MSE
    print(MSEgraph)
    print("Final Weights: \n", weights)
    fig2, ax = plt.subplots()
    ax.plot(MSEgraph)
    ax.set_title("Mean Squared Error")
    ax.set_xlabel("No. of Iterations")
    return weights

# Attributes:
# Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate
# Fine Aggregate, Age, Concrete Compressive Strength
# 8 input variables, 1 output variable
# Retrieve Data from GitHub Repository
url = "https://raw.githubusercontent.com/jamesH-48/Gradient-Descent-A1/master/Concrete_Data.csv"
data = pd.read_csv(url, header=None)
data = data.rename(columns={0:"Cement",1:"Blast Furnace Slag",2:"Fly Ash",3:"Water",4:"Superplasticizer",5:"Coarse Aggregate",6:"Fine Aggregate",7:"Age",8:"Concrete Compressive Strength"})

# Add Column Intercept of 1s to data frame
#data.insert(0, 'Intercept', 1)
# So far hasn't changed value

values = data.values

'''
#Going to test this to see what leads to best results
'''
# Compute pairwise correlation of columns
corr = data.corr()
# Display Heatmap of Correlations
sns.set()
axi1 = sns.heatmap(corr, cmap="BuPu", annot=True)
#axi2 = sns.pairplot(data)

# Plot Data ~ each column has its own subplot
fig1 = plt.figure()
fig1.suptitle('Input Attributes', fontsize=16)
for i in range(values.shape[1]):
    plt.subplot(values.shape[1], 1, i + 1)
    plt.plot(values[:, i])

#print(data.values[0])
#data.to_csv('new.csv')

#  Pre-Processing ~ Remove Outliers from dataset
# Over each column (except open/closed)
removeList = []
for i in range(values.shape[1] - 1):
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

#data.to_csv('new.csv')

# Train-Test Split Dataset
# Create X data frame that contains the inputs
Xdf = data[["Cement","Blast Furnace Slag","Fly Ash","Water","Superplasticizer","Coarse Aggregate","Fine Aggregate","Age"]]
# Convert X data frame to numpy array (just known as X)
X = Xdf.to_numpy()
# Create Y data frame that contains the output
Ydf = data[["Concrete Compressive Strength"]]
# Convert Y data frame to numpy array (just known as Y)
Y = Ydf.to_numpy()


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)


# Proof of Shape
print("X_train trans: ", np.transpose(X_train).shape)
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("Y_train: ", Y_train.shape)
print("Y_test: ", Y_test.shape)

# Initialize Weights
#Weights = np.random.uniform(0.0, .2, size=8)
Weights = np.array([[0],[0],[0],[0],[0],[0],[0],[0]])
print("Weights: ", Weights)
print(Weights.shape)

# Initialize Learning Rate
LR = .000001
# Set Iterations
iterations = 10000

'''
#Gradient Descent
'''
FinalWeights = gradientdescent(X_train, Y_train, Weights, LR, iterations)
print(FinalWeights.shape)
print(FinalWeights)
# Weights Bar Graph

# Apply Model to Test Data Set
# Get X Values from Test Data x Weights Found
# Compare to X Values with actual output values from test data set
Y_pred1 = np.dot(X_train, FinalWeights)
Y_pred2 = np.dot(X_test, FinalWeights)

print(mean_squared_error(Y_pred1,Y_train))
print(r2_score(Y_pred1,Y_train))
print('\n')
print(mean_squared_error(Y_pred2,Y_test))
print(r2_score(Y_pred2,Y_test))
# Print Plot of Outputs
figure1, ax = plt.subplots()
figure2, ax2 = plt.subplots()
ax.plot(Y_train, color='red', markersize=5)
ax.plot(Y_pred1, color='cyan', markersize=5)
ax2.plot(Y_test, color='black', markersize=5)
ax2.plot(Y_pred2, color='magenta', markersize=5)
plt.show()
