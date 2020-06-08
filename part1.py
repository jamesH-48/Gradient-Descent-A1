import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def gradientdescent(x, y, weights, LR, iterations):
    # Graph MSE
    MSEgraph = np.zeros((iterations,1))
    for k in range(iterations):
        if(iterations == 90000):
            LR = .0000001
        # Initialize Hypothesis
        H = np.dot(x, weights)
        # Define Error
        # E = H - Y
        E = np.subtract(H, y)
        # Define Mean Squared Error
        MSE = (1 / (2 * (int(len(y))))) * np.dot(np.transpose(E), E)
        MSEgraph[k] = MSE
        # print("MSE ", MSE)
        # Define Gradient -> MSE derivative to weight
        gradient = (1 / (int(len(y)))) * np.dot(E, x)

        # Revise Weights
        # New Weight = Old Weight - Learning Rate * Gradient
        weights = np.subtract(weights, LR * gradient)
    # Plot MSE
    print(MSEgraph)
    print("Final Weights: ", weights)
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
values = data.values

'''
Going to test this to see what leads to best results
'''
#  Pre-Processing ~ Remove Outliers from dataset
# Over each column (except open/closed)
'''for i in range(values.shape[1] - 1):
	# Calculate Mean & Standard Deviation
	Data_Mean, Data_STD = np.mean(values[:,i]), np.std(values[:,i])
	# Define Outlier Boundary by the standard deviation
	bound = Data_STD * 4
	lower, upper = Data_Mean - bound, Data_Mean + bound
	# Remove Outliers that are below the lower bound
	below = [j for j in range(values.shape[0]) if values[j,i] < lower]
	values = np.delete(values, below, 0)	# delete row if outlier
	# Remove Outliers that are above the upper bound
	above = [j for j in range(values.shape[0]) if values[j,i] > upper]
	values = np.delete(values, above, 0)	# delete row if outlier'''

# Plot Data ~ each column has its own subplot
fig1 = plt.figure()
fig1.suptitle('Input Attributes', fontsize=16)
for i in range(values.shape[1]):
    plt.subplot(values.shape[1], 1, i + 1)
    plt.plot(values[:, i])
# Show Plot Graph of each attribute value vs number of attributes per column
# plt.show()

# Split Data based on Inputs vs Outputs for X and y
Input = values[:, [0, 1, 2, 3, 4, 5, 6, 7]]
Output = values[:, 8]

'''
Split Test
A = np.array([[1,2],[1,2],[1,2],[3,4]])
B = A[:int(len(A)*.5),:]
C = A[int(len(A)*.5):,:]
print(B,"\n",C)
'''

# Split Data based on Training/Test -> 80/20
# X - attributes
InputTrain = Input[:int(len(Input) * .8), :]
InputTest = Input[int(len(Input) * .8):, :]
# Y
OutputTrain = Output[:int(len(Output) * .8)]
OutputTest = Output[int(len(Output) * .8):]
# Initialize Weights
Weights = np.random.uniform(0.0, .3, size=8)
# Initialize Learning Rate
LR = .000001
# Set Iterations
iterations = 100000

'''
Gradient Descent
'''
FinalWeights = gradientdescent(InputTrain, OutputTrain, Weights, LR, iterations)
#plt.show()

# Apply Model to Test Data Set
# Get X Values from Test Data x Weights Found
# Compare to X Values with actual output values from test data set
Y_pred1 = np.dot(InputTrain, FinalWeights)
Y_pred2 = np.dot(InputTest, FinalWeights)
'''correlation_matrix = np.corrcoef(X_values, OutputTest)
correlation_xy = correlation_matrix[0, 1]
r_squared = correlation_xy ** 2
print(r_squared)'''

print(mean_squared_error(Y_pred1,OutputTrain))
print(r2_score(Y_pred1,OutputTrain))
print('\n')
print(mean_squared_error(Y_pred2,OutputTest))
print(r2_score(Y_pred2,OutputTest))
# Print Plot of Outputs
figure1, ax = plt.subplots()
figure2, ax2 = plt.subplots()
ax.plot(OutputTest, color='black', markersize=5)
ax.plot(Y_pred2, color='magenta', markersize=5)
ax2.plot(OutputTrain, color='red', markersize=5)
ax2.plot(Y_pred1, color='cyan', markersize=5)
plt.show()
