import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def GradientDescent(x, y, weights, LR, iterations):
	for k in range(iterations):
		# Initialize Hypothesis
		H = np.dot(x, weights)
		# Define Error
		# E = H - Y
		E = np.subtract(H, y)
		# Define Mean Squared Error
		MSE = (1 / (2 * (int(len(H))))) * np.dot(np.transpose(E), E)
		print("MSE ", MSE)
		# Define Gradient -> MSE derivative to weight
		gradient = (1 / ((int(len(H))))) * np.dot(E, x)

		# Revise Weights
		# New Weight = Old Weight - Learning Rate * Gradient
		weights = np.subtract(weights, LR * gradient)
		print(weights)
	return weights

# Attributes:
# Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate
# Fine Aggregate, Age, Concrete Compressive Strength
# 8 input variables, 1 output variable
# Retrieve Data from GitHub Repository
url = "https://raw.githubusercontent.com/jamesH-48/Gradient-Descent-A1/master/Concrete_Data.csv"
data = pd.read_csv(url,header=None)
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
plt.figure()
for i in range(values.shape[1]):
	plt.subplot(values.shape[1], 1, i+1)
	plt.plot(values[:, i])
# Show Plot Graph of each attribute value vs number of attributes per column
# plt.show()

# Split Data based on Inputs vs Outputs for X and y
Input = values[:,[0,1,2,3,4,5,6,7]]
Output = values[:,8]

'''
Split Test
A = np.array([[1,2],[1,2],[1,2],[3,4]])
B = A[:int(len(A)*.5),:]
C = A[int(len(A)*.5):,:]
print(B,"\n",C)
'''

# Split Data based on Training/Test -> 80/20
# X - attributes
InputTrain = Input[:int(len(Input)*.8),:]
InputTest = Input[int(len(Input)*.8):,:]
# Y
OutputTrain = Output[:int(len(Output)*.8)]
OutputTest = Output[int(len(Output)*.8):]
# Initialize Weights
Weights = np.random.uniform(0.0, 1.0, size = 8)
# Initialize Learning Rate
LR = .000001
# Set Iterations
iterations = 10000

'''
Gradient Descent
'''
# GD in main so that we can plot to graph
MSEgraph = []
for k in range(iterations):
	# Initialize Hypothesis
	H = np.dot(InputTrain, Weights)
	# Define Error
	# E = H - Y
	E = np.subtract(H, OutputTrain)
	# Define Mean Squared Error
	MSE = (1 / (2 * (int(len(H))))) * np.dot(np.transpose(E), E)
	MSEgraph.append(MSE)
	#print("MSE ", MSE)
	# Define Gradient -> MSE derivative to weight
	gradient = (1 / ((int(len(H))))) * np.dot(E, InputTrain)

	# Revise Weights
	# New Weight = Old Weight - Learning Rate * Gradient
	Weights = np.subtract(Weights, LR * gradient)
	#print(Weights)

# Plot MSE
print(MSEgraph)
print("Final Weights: ", Weights)
fig2, ax = plt.subplots()
ax.plot(MSEgraph)
ax.set_title("Mean Squared Error")
ax.set_xlabel("No. of Iterations")
plt.show()

# Apply Model to Test Data Set
# Get X Values from Test Data x Weights Found
# Compare to X Values with actual output values from test data set
X_values = np.dot(InputTest, Weights)
correlation_matrix = np.corrcoef(X_values, OutputTest)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print(r_squared)
