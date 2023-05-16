# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:09:10 2023

@author: Franlin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# cargar el dataset de data 
data = pd.read_csv("..\datasets\coches2.csv")

# =============================================================================
#                   Visualización de gráficas relevantes
# =============================================================================


############# Plotear las muestras Visualizacion de los precios
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Diagrama de distribución de precios de automóviles')
sns.distplot(data.price)

plt.subplot(1,2,2)
plt.title('Diferencial de precios de automóviles')
sns.boxplot(y=data.price)

plt.show()

# Las siguientes graficas muestran la cantidad de coches por compañia y por tipo
# de combustible
plt.figure(figsize=(35, 12))
plt.subplot(1,2,1)
plt1 = data['make'].value_counts().plot(kind = 'bar')
plt.title('Histograma de compañias')
plt1.set(xlabel = 'Compañia de coche', ylabel='frecuencia de la compañia')

plt.subplot(1,2,2)
plt1 = data['fuel'].value_counts().plot(kind = 'bar')
plt.title('Histograma del tipo de combustible')
plt1.set(xlabel = 'Tipo de combustible', ylabel='Frecuencia del tipo de combustible')

plt.show()

plt.figure(figsize=(35, 12))
plt.subplot(1,1,1)
plt1 = data['province'].value_counts().plot(kind = 'bar')
plt.title('Histograma de compañias')
plt1.set(xlabel = 'Compañia de coche por ciudad', ylabel='frecuencia de la compañia')
plt.show()

############# Correlation: correlation coefficient matrix
data_corr = data.corr()
print(data_corr)
# Correlation coefficient: feature vs feature
data_corr_values = data_corr['price'][:-1]
data_corr_values = data_corr_values.sort_values(ascending=False)
print(data_corr_values)

#########Visualizacion de Atributos con mas correlacion
df = pd.DataFrame(data.groupby(['shift'])['price'].mean().sort_values(ascending = False))
df.plot.bar(figsize=(8,6))
plt.title('Tipo de caja vs Precio medio')
plt.show()

df = pd.DataFrame(data.groupby(['make'])['price'].mean().sort_values(ascending = False))
df.plot.bar(figsize=(18,6))
plt.title('Fabricante vs Precio medio')
plt.show()

df = pd.DataFrame(data.groupby(['fuel'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Tipo de combustible vs Precio medio')
plt.show()

df = pd.DataFrame(data.groupby(['year'])['price'].mean().sort_values(ascending = False))
df.plot.bar(figsize=(18,6))
plt.title('Año de coche vs  Precio medio')
plt.show()

#### Listado de las variables significantes
data_lr =data[['price', 'fuel', 'shift','make', 'year']]
data_lr.head()

sns.pairplot(data_lr)
plt.show()

# =============================================================================
#                   Procesamiento de Datos
# =============================================================================

# eliminar la columna que tenga no aporta inforacion relevante
data.drop(['url'], axis=1, inplace=True)
data.drop(['company'], axis=1, inplace=True)

#Problema:cuando pasas a code te asigna numeros aleatorios y eso no mola
#data['fuel'] = data['fuel'].astype('category')
#data['fuel'] = data['fuel'].cat.codes

#clasificar los tipos de combustible en ints: donde 1-Diésel, 2-Gasolina,3-Hibrido
#4-Eléctrico, 5-Hibrido enchufable,6-Gas licuado (GLP), 7-Gas natural (CNG)
data.loc[data['fuel'] == 'Diésel', 'fuel'] = '1'
data.loc[data['fuel'] == 'Gasolina', 'fuel'] = '2'
data.loc[data['fuel'] == 'Híbrido', 'fuel'] = '3'
data.loc[data['fuel'] == 'Eléctrico', 'fuel'] = '4'
data.loc[data['fuel'] == 'Híbrido enchufable', 'fuel'] = '5'
data.loc[data['fuel'] == 'Gas licuado (GLP)', 'fuel'] = '6'
data.loc[data['fuel'] == 'Gas natural (CNG)', 'fuel'] = '7'

#ver si hay muestras duplicadas
data.loc[data.duplicated()]

##Eliminamos las muestras que son NaN, si lo eliminamos perdemos muchas muestras
#data = data.dropna()
##Solucion poner los Nan a 0
data['fuel'] = data['fuel'].fillna(0)
data['price'] = data['price'].fillna(0)
data['year'] = data['year'].fillna(0)

#Convertir las varibles de interes a los tipos de datos que correspondan
data['fuel'] = data['fuel'].astype(np.int64)
data['price'] = data['price'].astype(np.int64)
data['year'] = data['year'].astype(np.int64)
data['province'] = data['province'].astype(str)
# Attribute information
print(data.info())

# =============================================================================
#          Predicción del precio por Regresion Lineal (FUCNION PRINCIPAL)
# =============================================================================
def predecir_precio():    
    msg = ''
    data2 = data[['year', 'fuel', 'price','kms']]
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
    mensaje = input("Tipo de combustible (1 = Diésel , 2 = Gasolina, 3 = Hibrido, 4 = Eléctrico, 5 = Hibrido enchufable, 6 = Gas licuado (GLP), 7 = Gas natural (CNG)): ")
    if mensaje == 0:
        msg += '\nNecesitamos saber el tipo de combustible!'
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
    mensaje2 = input("Año del coche: ")
    if mensaje2 == 0:
        msg += '\nNecesitamos saber el año del coche!'
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
    mensaje3 = input("Kms del coche: ")
    if mensaje3 == 0:
        msg += '\nNecesitamos saber los kms del coche!'
    mensaje3 = np.array(mensaje3, dtype=int)
    y_pred_new3 = regr3.predict([[mensaje3]])
    #print("the price predicted is:",int(y_pred_new3))


    #Antes de predecir el precio vamos a aplicar una devaluación por año
    total_price = (y_pred_new+y_pred_new2+y_pred_new3)/3
    #print("the price predicted is:",int(total_price))
    precio_inicial = int(total_price) # Precio inicial del coche
    año_coche = int(mensaje2) # Año del coche que queremos valorar

    # Tasa de depreciación para cada año desde 2023, lo ponemos en un diccionario
    devaluacion_por_año = {1:0.001,2:0.16,3:0.33,
                           4:0.44,5:0.53,6:0.61,7:0.66,8:0.72,9:0.76,10:0.81,
                           11:0.83,12:0.87}

    # Calcular el descuento para cada año
    años_desde_2023 = 2023 - año_coche
    #print("the years is:",int(años_desde_2023))
    descuento_total = 0
    if años_desde_2023 in devaluacion_por_año:
            descuento_total = devaluacion_por_año[años_desde_2023]
    else:
            descuento_total = 0.91

    # Aplicar el descuento al precio inicial
    precio_descontado = precio_inicial * (1 - descuento_total)
    #print("El precio real total de su coche es de: ",int(precio_descontado))
    
    if mensaje != 0 and mensaje2 != 0 and mensaje3 != 0:
        precio = round(precio_descontado,2)
        if pd.isna(precio_descontado) == True:
            msg = "\nLo sentimos no podemos predecir su precio!"
        else:
            msg = f"\nEL precio de su coche es de aprox {precio}€\n"
    return msg
    #print(msg)
    #return int(precio_descontado)
# =============================================================================
#                   DEFINIMOS LAS FUNCIONES A UTILIZAR
# =============================================================================

#ponemos a una variable los campor que seran de utilidad

data = data[['make','model', 'year', 'fuel', 'price','kms','province']]

coches = data.copy()

# Attributes distributions
print(coches.describe(include='all'))

sample = coches.sample(axis = 0)
makeRand = sample['make'].values[0]


def mostrar_coches_por_presupuesto():

    coches = data.copy()
    
    print("Inserte el precio mas bajo:")
    low_price = int(input())
    
    print("Inserte el precio mas alta:")
    high_price = int(input())
    
    n = 0
    
    if low_price == 0:    
        if high_price == 0:
            ascending = False
            n = 1
        else: 
            ascending = True
            coches = coches[(coches['price'] <= high_price)]
    else:  
        if high_price == 0:
            ascending = False
            coches = coches[(coches['price'] >= low_price)]     
        else:  
            ascending = True
            coches = coches[(coches['price'] >= low_price) &
                                    (coches['price'] <= high_price)]

    if n == 0:
        coches = coches.sort_values('price', ascending=ascending)
        return coches
    elif n == 1:
        return 0

#El valor predeterminado de 0.8 significa que los autos recomendados serán el 
# 80% superior de los autos en función de su precio.
def obtener_coche_por_ciudad(percentile=0.8):
    
    coches = data.copy()

    print("Ingrese la ciudad (por ejemplo, Barcelona):")
    city = str(input())
    
    print("Marca preferida:")  
    make = str(input())
    
    print("Precio mínimo:")  
    low_price = int(input())
    
    print("Precio máximo:") 
    high_price = int(input())
    
    coches = coches[(coches['make'] == make.upper()) &
        (coches['price'] >= low_price) &
        (coches['price'] <= high_price) & (coches['province'] == city)]

    m = coches['price'].quantile(percentile)

    q_coches = coches.copy().loc[coches['price'] <= m]

    q_coches = q_coches.sort_values('price', ascending=False)
    
    return q_coches

#El valor predeterminado de 0.8 significa que los autos recomendados serán el 
# 80% superior de los autos en función de su precio.
def obtener_recomendacion(percentile=0.8):
    coches = data.copy()

    while True:
        make = str(input("Marca preferida: "))
        if make.upper() in coches['make'].unique():
            break
        else:
            print("Marca no válida. Intente nuevamente.")

    while True:
        try:
            low_price = int(input("Precio mínimo: "))
            high_price = int(input("Precio máximo: "))
            if low_price >= 0 and high_price >= low_price:
                break
            else:
                print("Precios no válidos. Intente nuevamente.")
        except ValueError:
            print("Precios no válidos. Intente nuevamente.")

    while True:
        try:
            low_year = int(input("Año mínimo: "))
            high_year = int(input("Año máximo: "))
            if low_year >= 0 and high_year >= low_year:
                break
            else:
                print("Años no válidos. Intente nuevamente.")
        except ValueError:
            print("Años no válidos. Intente nuevamente.")

    while True:
        try:
            fuel = int(input("Tipo de combustible (1 = Diésel, 2 = Gasolina, 3 = Hibrido, 4 = Eléctrico, 5 = Hibrido enchufable, 6 = Gas licuado (GLP), 7 = Gas natural (CNG)): "))
            if fuel in [1, 2, 3, 4, 5, 6, 7]:
                break
            else:
                print("Tipo de combustible no válido. Intente nuevamente.")
        except ValueError:
            print("Tipo de combustible no válido. Intente nuevamente.")

    coches = coches[(coches['make'] == make.upper()) &
                    (coches['price'] >= low_price) &
                    (coches['price'] <= high_price) &
                    (coches['year'] >= low_year) &
                    (coches['year'] <= high_year) &
                    (coches['fuel'] == fuel)]

    m = coches['price'].quantile(percentile)

    q_coches = coches.copy().loc[coches['price'] <= m]

    q_coches = q_coches.sort_values('price', ascending=False)

    return q_coches

# =============================================================================
#                   INTERACCION POR CONSOLA CON EL USUARIO
# =============================================================================

while True:
    print("\n¿Qué desea realizar?\n")
    print("(1) Predecir precio de mi coche")
    print("(2) Mostrar recomendaciones de un coche")
    print("(3) Recomendaciones por mi presupuesto")
    print("(4) Coches en tu ciudad ")
    print("(5) Salir")
    
    option = input()
    
    if option == '1':
        print(predecir_precio())
    elif option == '2':
        coche = obtener_recomendacion(percentile=0.8)
        if coche.empty:
            print("\nLo siento, no hay ningún coche parecido a lo que quieres :C")
        else:
            print(coche)
    elif option == '3':
        coche = mostrar_coches_por_presupuesto()
        if type(coche) is int:
            print('\n¡No hay coches con precio de 0€!')
        elif coche.empty:
            print('\nLo siento, no encontramos ningún coche con esos precios.')
        else:
            print(f'{coche}')
    elif option == '4':
        coche = obtener_coche_por_ciudad(percentile=0.8)
        if coche.empty:
            print("\nLo siento, no hay ningún coche parecido a lo que quieres :C")
        else:
            print(coche)
            
    elif option == '5':
        print("\nCerrando la APP...")
        break
    else:
        print("\nOpción no válida. Vuelve a intentarlo.")

