# -*- coding: utf-8 -*-
"""
Created on Sat May 13 19:01:20 2023

@author: franlin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from time import sleep
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# cargar el dataset de data 
data = pd.read_csv("..\datasets\coches2.csv")

# Nice data visualization
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

data2 = data[['year', 'fuel', 'price','kms']]

#clasificar los tipos de combustible en ints: donde 1-Diésel, 2-Gasolina,3-Hibrido
#4-Eléctrico, 5-Hibrido enchufable,6-Gas licuado (GLP), 7-Gas natural (CNG)
data2.loc[data2['fuel'] == 'Diésel', 'fuel'] = '1'
data2.loc[data2['fuel'] == 'Gasolina', 'fuel'] = '2'
data2.loc[data2['fuel'] == 'Híbrido', 'fuel'] = '3'
data2.loc[data2['fuel'] == 'Eléctrico', 'fuel'] = '4'
data2.loc[data2['fuel'] == 'Híbrido enchufable', 'fuel'] = '5'
data2.loc[data2['fuel'] == 'Gas licuado (GLP)', 'fuel'] = '6'
data2.loc[data2['fuel'] == 'Gas natural (CNG)', 'fuel'] = '7'

##Eliminamos las muestras que son NaN, si lo eliminamos perdemos muchas muestras
#data = data.dropna()
##Solucion poner los Nan a 0
data2['fuel'] = data2['fuel'].fillna(0)
data2['price'] = data2['price'].fillna(0)
data2['year'] = data2['year'].fillna(0)
data2['kms'] = data2['kms'].fillna(0)
#Convertir las varibles de interes a los tipos de datos que correspondan
data2['fuel'] = data2['fuel'].astype(np.int64)
data2['price'] = data2['price'].astype(np.int64)
data2['year'] = data2['year'].astype(np.int64)
data2['kms'] = data2['kms'].astype(np.int64)
# Attribute information
print(data2.info())

filtered_data = data2

####################FUEL
#Before we create the model we transform dataframe to array
dataX =filtered_data[["fuel"]]
X_train = np.array(dataX)
y_train = filtered_data['price'].values

# Creation of model as Lineal Regression with 2 variables
regr = linear_model.LinearRegression()
# Fit
regr.fit(X_train, y_train)
# Prediction
y_pred = regr.predict(X_train)

# Now we predict with a new value:
# How many share I will get if I create an article of .... words?
mensaje = input("Enter the fuel type:")
mensaje = np.array(mensaje, dtype=int)
y_pred_new = regr.predict([[mensaje]])
#print("the price predicted is:",int(y_pred_new))
####################YEAR
#Before we create the model we transform dataframe to array
dataX2 =filtered_data[["year"]]
X_train2 = np.array(dataX2)
y_train2 = filtered_data['price'].values

# Creation of model as Lineal Regression with 2 variables
regr2 = linear_model.LinearRegression()
# Fit
regr2.fit(X_train2, y_train2)
# Prediction
y_pred2 = regr.predict(X_train2)

# Now we predict with a new value:
# How many share I will get if I create an article of .... words?
mensaje2 = input("Enter the car year:")
mensaje2 = np.array(mensaje2, dtype=int)
y_pred_new2 = regr2.predict([[mensaje2]])
#print("the price predicted is:",int(y_pred_new2))
##################KMS
#Before we create the model we transform dataframe to array
dataX3 =filtered_data[["kms"]]
X_train3 = np.array(dataX3)
y_train3 = filtered_data['price'].values

# Creation of model as Lineal Regression with 2 variables
regr3 = linear_model.LinearRegression()
# Fit
regr3.fit(X_train3, y_train3)
# Prediction
y_pred3 = regr3.predict(X_train3)

# Now we predict with a new value:
# How many share I will get if I create an article of .... words?
mensaje3 = input("Enter the car kms(put all values):")
mensaje3 = np.array(mensaje3, dtype=int)
y_pred_new3 = regr3.predict([[mensaje3]])
#print("the price predicted is:",int(y_pred_new3))


#Antes de predecir el precio vamos a aplicar una devaluación por año
total_price = (y_pred_new+y_pred_new2+y_pred_new3)/3
print("the price predicted is:",int(total_price))
precio_inicial = int(total_price) # Precio inicial del coche 
año_coche = int(mensaje2) # Año del coche que queremos valorar

# Tasa de depreciación para cada año desde 2023
devaluacion_por_año = {1:0.001,2:0.16,3:0.33,
                       4:0.44,5:0.53,6:0.61,7:0.66,8:0.72,9:0.76,10:0.81,
                       11:0.83,12:0.87}

# Calcular el descuento para cada año
años_desde_2023 = 2023 - año_coche
print("the years is:",int(años_desde_2023))
descuento_total = 0
if años_desde_2023 in devaluacion_por_año:
        descuento_total = devaluacion_por_año[años_desde_2023]
else:
        descuento_total = 0.91

# Aplicar el descuento al precio inicial
precio_descontado = precio_inicial * (1 - descuento_total)

print("the real price predicted is:",int(precio_descontado))



