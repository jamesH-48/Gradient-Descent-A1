# Assignment 1 ~ part1
# James Hooper ~ NETID: jah171230
# Hritik Panchasara ~ NETID: hhp160130
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def gradientdescent(x, y, weights, LR, iterations):
    # Graph MSE
    MSEgraph = np.zeros((iterations,1))
    for k in range(iterations):
        # Initialize Hypothesis
        H = np.dot(x, weights)
        # Define Error
        # E = H - Y
        E = np.subtract(H, y)
        # Define Mean Squared Error
        MSE = (1 / (2 * (int(len(y))))) * np.dot(np.transpose(E), E)
        MSEgraph[k] = MSE
        # Define Gradient -> MSE derivative to weight
        gradient = (1 / (int(len(y)))) * np.dot(np.transpose(x), E)
        # Revise Weights
        # New Weight = Old Weight - Learning Rate * Gradient
        weights = np.subtract(weights, LR * gradient)
    return weights, MSEgraph

def removeOutliers(data, deviations):
    removeList = []
    for i in range(data.values.shape[1] - 1):
        # Calculate Mean & Standard Deviation
        Data_Mean, Data_STD = np.mean(data.values[:,i]), np.std(data.values[:,i])
        # Define Outlier Boundary by the standard deviation
        bound = Data_STD * deviations
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
    # Remove Rows for Data Frame
    for i in range(len(removeList)):
        data = data.drop(removeList[i])
    return data

def main(state):
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
    deviations = 4
    data = removeOutliers(data, deviations)

    '''
    Graphic Display ~ Attribute Correlation Heatmap
    '''
    # Compute pairwise correlation of columns
    corr = data.corr()
    # Display Heatmap of Correlations
    axHeat = plt.axes()
    axi1 = sns.heatmap(corr, ax = axHeat, cmap="BuPu", annot=True)
    axHeat.set_title('Heatmap of Attribute Correlation', fontsize = 24)

    '''
    Graphic Display ~ Attribute Plots (inputs & output)
    '''
    labelsAttPlot = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate',
              'Fine Aggregate', 'Age','Concrete Compressive Strength']
    # Plot Data ~ each column has its own subplot
    fig1, axs = plt.subplots(9)
    fig1.suptitle('Input/Output Attribute Values', fontsize=16)
    for i in range(values.shape[1]):
        axs[i].plot(values[:, i])
        axs[i].set_ylabel(labelsAttPlot[i])
    axs[8].set_xlabel("No. of Values")

    '''
    Pre-Processing ~ Train Test Split
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
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1, random_state=state)

    '''
    Intialization
    '''
    # Initialize Weights
    Weights = np.array([[0],[0],[0],[0],[0],[0],[0],[0]])
    # Initialize Learning Rate
    LR = .000001
    # Set Iterations
    iterations = 10000

    '''
    Gradient Descent
    '''
    FinalWeights, MSEgraph = gradientdescent(X_train, Y_train, Weights, LR, iterations)

    '''
    Final Values Print ~ Mean Squared Error & R^2
    '''
    # Apply Model found Weights to Test Data Set
    # Get Y prediction Values from Test Data x Weights Found
    # Compare Y prediction Values with actual output values from test data set
    Y_pred1 = np.dot(X_train, FinalWeights)
    Y_pred2 = np.dot(X_test, FinalWeights)
    # Parameters Used
    print("Parameters Used:")
    print("State: ", state)
    print("Standard Deviations for Outlier Removal: ", deviations)
    print("Learning Rate: ", LR)
    print("Iterations: ", iterations)
    # Coefficients
    coef = []           # Initialize
    for i in range(FinalWeights.shape[0]):  # For Print & Bar Graph
        coef.append(FinalWeights[i][0])
    print('Coefficients: \n', coef)
    # Train Accuracy
    print("Train Accuracy:")
    print("Mean Squared Error: ", mean_squared_error(Y_pred1,Y_train))
    print("R^2 Value: ", r2_score(Y_pred1,Y_train))
    # Test Accuracy
    print("Test Accuracy:")
    print("Mean Squared Error: ", mean_squared_error(Y_pred2,Y_test))
    print("R^2 Value: ", r2_score(Y_pred2,Y_test))

    '''
    Graphic Display ~ Train Accuracy & Test Accuracy Plots
    '''
    # Print Plot of Outputs
    figure1, ax = plt.subplots()
    figure2, ax2 = plt.subplots()
    ax.plot(Y_train, color='red', markersize=5, label="Actual")
    ax.plot(Y_pred1, color='cyan', markersize=5, label="Prediction")
    ax.set_title('Y Train Dataset')
    ax.set_xlabel('No. of Values')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    ax2.plot(Y_test, color='black', markersize=5, label="Actual")
    ax2.plot(Y_pred2, color='magenta', markersize=5, label="Prediction")
    ax2.set_title('Y Test Dataset')
    ax2.set_xlabel('No. of Values')
    ax2.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

    '''
    Graphic Display ~ Mean Squared Error
    '''
    figureMSE, axMSE = plt.subplots()
    axMSE.plot(MSEgraph)
    axMSE.set_title("Mean Squared Error")
    axMSE.set_xlabel("No. of Iterations")

    '''
    Graphic Display ~ Coefficient Bar Graph
    '''
    # Weights Bar Graph
    labels = ['Cement','Blast Furnace Slag','Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate','Age']
    x = np.arange(len(labels))       # Location of Labels
    width = .5                       # Width of the bars
    figureW, axW = plt.subplots()
    bars = axW.bar(x,coef,width,color='orange') # Coef is from Weight Print
    axW.set_ylabel('Weight')
    axW.set_title('Coefficients')
    axW.set_xticks(x)
    axW.set_xticklabels(labels)

    plt.show()

if __name__ == '__main__':
    print("Part 1: Gradient Descent")
    # State is the order of data that is randomized in train-test-split
    # The state can be seen as the seed for repeatable datasets
    state = 4
    main(state)
