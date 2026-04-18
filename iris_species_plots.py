# Perform and plot linear regression for all three iris species: Iris_setosa, Iris_virginica, Iris_versicolor


import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Import metadata
dataframe = pd.read_csv("iris.csv")

# Subset & plot Iris_setosa
setosa = dataframe[dataframe.species == "Iris_setosa"]
print(setosa)

x = setosa.petal_length_cm
y = setosa.sepal_length_cm  
setosa_regression = stats.linregress(x, y)
setosa_slope = setosa_regression.slope
setosa_intercept = setosa_regression.intercept
plt.scatter(x, y, label = 'Setosa Data')  
plt.plot(x, setosa_slope * x + setosa_intercept, color = "orange", label = 'Fitted line')
plt.xlabel("Setosa Petal length (cm)")
plt.ylabel("Setosa Sepal length (cm)")
plt.legend()
plt.savefig("setosa_petal_v_sepal_length_regress.png")

# Subset & plot Iris_virginica
virginica = dataframe[dataframe.species == "Iris_virginica"]
print(virginica)

x = virginica.petal_length_cm
y = virginica.sepal_length_cm  
virginica_regression = stats.linregress(x, y)
virginica_slope = virginica_regression.slope
virginica_intercept = virginica_regression.intercept
plt.scatter(x, y, label = 'Virginica Data')  
plt.plot(x, virginica_slope * x + virginica_intercept, color = "orange", label = 'Fitted line')
plt.xlabel("Virginica Petal length (cm)")
plt.ylabel("Virginica Sepal length (cm)")
plt.legend()
plt.savefig("virginica_petal_v_sepal_length_regress.png")

# Subset & plot Iris_versicolor
versicolor = dataframe[dataframe.species == "Iris_versicolor"]
print(versicolor)

x = versicolor.petal_length_cm
y = versicolor.sepal_length_cm  
versicolor_regression = stats.linregress(x, y)
versicolor_slope = versicolor_regression.slope
versicolor_intercept = versicolor_regression.intercept
plt.scatter(x, y, label = 'Versicolor Data')  
plt.plot(x, versicolor_slope * x + versicolor_intercept, color = "orange", label = 'Fitted line')
plt.xlabel("Versicolor Petal length (cm)")
plt.ylabel("Versicolor Sepal length (cm)")
plt.legend()
plt.savefig("versicolor_petal_v_sepal_length_regress.png")

