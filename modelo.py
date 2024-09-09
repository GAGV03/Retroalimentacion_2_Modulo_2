#Autor: Gustavo Alejandro Gutiérrez Valdes 

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('dataset_students.csv')
features = df.drop(columns='Admision')
objective = df['Admision']

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(features, objective, test_size=0.3, random_state=42)

# Escalar los datos usando StandardScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_valid_scaled = scaler.transform(X_val)

# Definir los parámetros a ajustar
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Crear un objeto GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                          param_grid=param_grid, 
                          cv=5, 
                          n_jobs=-1)

# Entrenar el modelo con el conjunto de entrenamiento
randomForestModel = RandomForestClassifier(n_estimators=100, random_state=42)
grid_search.fit(x_train_scaled, y_train)

# Obtener el mejor modelo
best_model = grid_search.best_estimator_

# Evaluar el modelo en el conjunto de validación
y_val_pred = best_model.predict(x_valid_scaled)

accuracy_val = accuracy_score(y_val, y_val_pred)
print(f'Precisión en validación: {accuracy_val:.2f}')

print("Matriz de confusión:")
print(confusion_matrix(y_val, y_val_pred))

print("\nReporte de clasificación:")
print(classification_report(y_val, y_val_pred))

dataset_test = pd.read_csv('test.csv')
x_test_scaled = scaler.transform(dataset_test)

# Hacer predicciones con el modelo entrenado
predicciones = best_model.predict(dataset_test)
print(predicciones)




