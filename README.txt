Assignment 1 
James Hooper ~ NETID: jah171230
Hritik Panchasara ~ NETID: hhp160130

- For this assignment we used PyCharm to create/edit/run the code. 
- In PyCharm the code should run by simply pressing the Run dropdown, then clicking run making sure you are running either part1 or part2 accordingly.
- The dataset is public on a Github account that one of us own. 
- The def main(state) function houses the values for the parameters such as deviations, learning rate, and iterations. Part1 has all three that can be altered as wanted. Part2 only has the deviations value. The deviations value affects the def removeOutliers(data, deviations) function. Keep in mind a deviations value of 2 will effectively remove all rows and a value of 6 will remove none. This is described in the log.
- Running the code there are different states that are fed to the train_test_split sklearn function. This can be seen as a seed for the same dataset orders when feeding these datasets through the model. We must have the seeds to properly compare the two models in question: Gradient Descent & Sklearn Linear Regression.
- Part1 has the def gradientdescent(x, y, weights, LR, iterations) function that is the model we are utilizing. THe operations within this code are equivalent to the step by step calculations shown in the pdf.
- In if __name__ == '__main__' , select the state you wish to run and then press run as explained previously. This is the simplest way to build/run the code.

Libraries Used for Part1:
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

Libraries Used for Part2:
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

Just in case. To import libraries/packages in PyCharm.
- Go to File.
- Press Settings.
- Press Project drop down.
- Press Project Interpreter.
- Press the plus sign on the top right box, should be to the right of where it says "Latest Version".
- Search and Install packages as needed.
- For this assignment the packages are: matplotlib, pandas, numpy, seabonr, and scikit-learn.
