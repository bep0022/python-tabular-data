#! /usr/bin/env python3

"""
Perform and plot linear regression for all three iris species: Iris_setosa, Iris_virginica, Iris_versicolor"

This script reads iris data from a CSV file, computes linear regression
between petal length and sepal length for each species, and saves a plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == '__main__':
    # Import metadata
    dataframe = pd.read_csv("iris.csv")

    # Subset & plot Iris_setosa
    setosa = dataframe[dataframe.species == "Iris_setosa"]
    print(setosa)

    setosa_x = setosa.petal_length_cm
    setosa_y = setosa.sepal_length_cm  
    setosa_regression = stats.linregress(setosa_x, setosa_y)
    setosa_slope = setosa_regression.slope
    setosa_intercept = setosa_regression.intercept
    plt.scatter(setosa_x, setosa_y, label = 'Setosa Data')  
    plt.plot(setosa_x, setosa_slope * setosa_x + setosa_intercept, color = "orange", label = 'Fitted line')
    plt.xlabel("Setosa Petal length (cm)")
    plt.ylabel("Setosa Sepal length (cm)")
    plt.legend()

    # Subset & plot Iris_virginica
    virginica = dataframe[dataframe.species == "Iris_virginica"]
    print(virginica)

    virginica_x = virginica.petal_length_cm
    virginica_y = virginica.sepal_length_cm  
    virginica_regression = stats.linregress(virginica_x, virginica_y)
    virginica_slope = virginica_regression.slope
    virginica_intercept = virginica_regression.intercept
    plt.scatter(virginica_x, virginica_y, label = 'Virginica Data')  
    plt.plot(virginica_x, virginica_slope * virginica_x + virginica_intercept, color = "orange", label = 'Fitted line')
    plt.xlabel("Virginica Petal length (cm)")
    plt.ylabel("Virginica Sepal length (cm)")
    plt.legend()

    # Subset & plot Iris_versicolor
    versicolor = dataframe[dataframe.species == "Iris_versicolor"]
    print(versicolor)

    versicolor_x = versicolor.petal_length_cm
    versicolor_y = versicolor.sepal_length_cm  
    versicolor_regression = stats.linregress(versicolor_x, versicolor_y)
    versicolor_slope = versicolor_regression.slope
    versicolor_intercept = versicolor_regression.intercept
    plt.scatter(versicolor_x, versicolor_y, label = 'Versicolor Data')  
    plt.plot(versicolor_x, versicolor_slope * versicolor_x + versicolor_intercept, color = "orange", label = 'Fitted line')
    plt.xlabel("Versicolor Petal length (cm)")
    plt.ylabel("Versicolor Sepal length (cm)")
    plt.legend()
    plt.savefig("species_petal_v_sepal_length_regress.png")
