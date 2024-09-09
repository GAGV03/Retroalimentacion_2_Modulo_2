#Autor: Gustavo Alejandro Gutiérrez Valdes 

import numpy as np
import pandas as pd

#se importan los recursos necesarios del framework de sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #type: ignore
from sklearn.metrics import mean_squared_error #type: ignore
from sklearn.metrics import mean_absolute_percentage_error #type: ignore
from sklearn.ensemble import RandomForestClassifier #type: ignore
from sklearn.model_selection import GridSearchCV

#Se lee el dataset de estudiantes que compondrá la etapa de training y validation
df = pd.read_csv('dataset_students.csv')

#Son las columnas que se utilizarán para entrenar el modelo
features = df.drop(columns='Admision')

#Es la columna objetivo
objective = df['Admision']

# Dividir en entrenamiento y validación (70% train, 30% validation)
X_train, X_val, y_train, y_val = train_test_split(features, objective, test_size=0.3, random_state=35)

# Se realiza la escalación de los datos de entrenamiento y validación utilizando MinMaxScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_valid_scaled = scaler.transform(X_val)

# Definir los parámetros a ajustar en el entrenamiento
param_grid = {
    'n_estimators': [10, 50,100, 150],
    'max_depth': [None, 5, 10,20],
    'min_samples_split': [2, 5, 10]
}

# Crear un objeto GridSearchCV. Esto servirá para encontrar los mejores hiperparámetros para el entrenamiento
grid_search = GridSearchCV(estimator=RandomForestClassifier(), 
                          param_grid=param_grid, 
                          cv=5, 
                          n_jobs=-1)

#Se aplican el objeto de GridSearchCV a los datos de entrenamiento
grid_search.fit(x_train_scaled, y_train)

# Se obtiene el mejor modelo entrenado
best_model = grid_search.best_estimator_

# Evaluar el modelo en el conjunto de validación
y_val_pred = best_model.predict(x_valid_scaled)

#Se obtiene el error cuadrático medio y la raíz del error cuadrático medio para entender su desempeño
mse = mean_squared_error(y_val, y_val_pred)
print("*"*50)
print("MSE del modelo entrenado:", mse)
rmse = np.sqrt(mse)
print("\nRaíz del MSE:", rmse)

#Se obtiene el valor de la precisión del modelo en la etapa de validación
accuracy_val = accuracy_score(y_val, y_val_pred)
print(f"\nPrecisión en validación: {accuracy_val:.2f}")

# Se obtiene el porcentaje de error en la etapa de validación
porcentaje_error = (1-accuracy_val) * 100

#Se muestra el porcentaje de error en la etapa de validación
print(f"\nPorcentaje de error para set de validación: {porcentaje_error:.2f}%")
print("*"*50)

#Se obtiene la matriz de confusión del modelo
print("\nMatriz de confusión:")
print(confusion_matrix(y_val, y_val_pred))
print("*"*50)

#Se obtiene el reporte de clasificación del modelo (Métricas como F1Score, Precision, Recall)
print("\nReporte de clasificación:")
print(classification_report(y_val, y_val_pred))
print("*"*50)

#Se lee el dataset de la etapa de test
dataset_test = pd.read_csv('test.csv')

#Se convierte en array para poder hacer un despligue más amigable de las predicciones
entradas = np.array(dataset_test)

#Se escalan los datos de testing para poder hacer predicciones
x_test_scaled = scaler.transform(dataset_test)

# Se hacen las predicciones correspondientes con el modelo entrenado
predicciones = best_model.predict(x_test_scaled)

#Se recorren tanto las entradas como las predicciones para mostrarlas al usuario en la consola
for x,y in zip (entradas,predicciones):
    estado = "Admitido" if y == 1 else "No admitido"
    print(f"Entrada (PuntajeExamen,PromedioAcumulado): {x} -> Predicción: {estado}")
print("*"*50)






