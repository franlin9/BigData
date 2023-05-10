# -*- coding: utf-8 -*-
"""
Created on Sat May  2 11:58:05 2023

@author: Franlin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# cargar el dataset de coches 
coches = pd.read_csv("..\datasets\coches.csv")

# eliminar la columna que tenga no aporta inforacion relevante
coches.drop(['normalized-losses'], axis=1, inplace=True)
#eliminar las filas que no tienen datos o sea que son '?' y no aporta nada
coches.drop(coches[(coches['price'] == '?')].index, inplace=True)
coches.drop(coches[(coches['peak-rpm'] == '?')].index, inplace=True)
coches.drop(coches[(coches['horsepower'] == '?')].index, inplace=True)
coches.drop(coches[(coches['stroke'] == '?')].index, inplace=True)
coches.drop(coches[(coches['bore'] == '?')].index, inplace=True)
coches.drop(coches[(coches['num-of-doors'] == '?')].index, inplace=True)

# convertir variables a lo q le corresponde
coches['price'] = coches['price'].astype('float64')
coches['bore'] = coches['bore'].astype('float64')
coches['horsepower'] = coches['horsepower'].astype('int64')
coches['peak-rpm'] = coches['peak-rpm'].astype('int64')
coches['stroke'] = coches['stroke'].astype('float64')

# Attribute information
print(coches.info())

#ver si hay muestras duplicadas
coches.loc[coches.duplicated()]

# Attributes distributions
print(coches.describe(include='all'))

##### Attribute to Attribute relationship

# Correlation: correlation coefficient matrix
# coches_corr = coches.corr()
# print(coches_corr)

############# Plotear las muestras Visualizacion de los precios
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Diagrama de distribución de precios de automóviles')
sns.distplot(coches.price)

plt.subplot(1,2,2)
plt.title('Diferencial de precios de automóviles')
sns.boxplot(y=coches.price)

plt.show()

############## VISUALIZACION DE OTROS DATOS RELEVANTES

coches['make'] = coches['make'].astype('category')
plt.figure(figsize=(25, 6))
plt.subplot(1,3,1)
plt1 = coches['make'].value_counts().plot(kind = 'bar')
plt.title('Histograma de compañias')
plt1.set(xlabel = 'Compañia de coche', ylabel='frecuencia de la compañia')

plt.subplot(1,3,2)
plt1 = coches['fuel-type'].value_counts().plot(kind = 'bar')
plt.title('Histograma del tipo de combustible')
plt1.set(xlabel = 'Tipo de combustible', ylabel='Frecuencia del tipo de combustible')

plt.subplot(1,3,3)
plt1 = coches['body-style'].value_counts().plot(kind = 'bar')
plt.title('Histograma de tipo de coche')
plt1.set(xlabel = 'Tipo de coche', ylabel='Frecuencia de tipo de coche')

plt.show()

######## Correlation: correlation coefficient matrix
coches_corr = coches.corr()
print(coches_corr)
# Correlation coefficient: feature vs feature
coches_corr_values = coches_corr['price'][:-1]
coches_corr_values = coches_corr_values.sort_values(ascending=False)
print(coches_corr_values)

#########Visualizacion de Atributos con mas correlacion
df = pd.DataFrame(coches.groupby(['engine-type'])['price'].mean().sort_values(ascending = False))
df.plot.bar(figsize=(8,6))
plt.title('Tipo de motor vs Precio medio')
plt.show()


df = pd.DataFrame(coches.groupby(['make'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Fabricante vs Precio medio')
plt.show()

df = pd.DataFrame(coches.groupby(['fuel-type'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Tipo de combustible vs Precio medio')
plt.show()

df = pd.DataFrame(coches.groupby(['body-style'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Tipo de coche vs  Precio medio')
plt.show()

#### Listado de las variables significantes
coches_lr = coches[['price', 'fuel-type', 'aspiration','body-style', 'drive-wheels','wheel-base',
                  'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'bore','horsepower', 
                     'length','width']]
coches_lr.head()

sns.pairplot(coches_lr)
plt.show()