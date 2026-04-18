# Perform and plot linear regression for all three iris species: Iris_setosa, Iris_virginica, Iris_versicolor


import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Import metadata
dataframe = pd.read_csv("iris.csv")

# Subset & plot Iris_setosa

x = dataframe.petal_length_cm
y = dataframe.sepal_length_cm  
regression = stats.linregress(x, y)
slope = regression.slope
intercept = regression.intercept
plt.scatter(x, y, label = 'Data')  
plt.plot(x, slope * x + intercept, color = "orange", label = 'Fitted line')
plt.xlabel("Petal length (cm)")
plt.ylabel("Sepal length (cm)")
plt.legend()
plt.savefig("setosa_petal_v_sepal_length_regress.png")

# Subset & plot Iris_virginica

x = dataframe.petal_length_cm
y = dataframe.sepal_length_cm  
regression = stats.linregress(x, y)
slope = regression.slope
intercept = regression.intercept
plt.scatter(x, y, label = 'Data')  
plt.plot(x, slope * x + intercept, color = "orange", label = 'Fitted line')
plt.xlabel("Petal length (cm)")
plt.ylabel("Sepal length (cm)")
plt.legend()
plt.savefig("virginica_petal_v_sepal_length_regress.png")

# Subset & plot Iris_versicolor

x = dataframe.petal_length_cm
y = dataframe.sepal_length_cm  
regression = stats.linregress(x, y)
slope = regression.slope
intercept = regression.intercept
plt.scatter(x, y, label = 'Data')  
plt.plot(x, slope * x + intercept, color = "orange", label = 'Fitted line')
plt.xlabel("Petal length (cm)")
plt.ylabel("Sepal length (cm)")
plt.legend()
plt.savefig("versicolor_petal_v_sepal_length_regress.png")

