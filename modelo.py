#Autor: Gustavo Alejandro Gutiérrez Valdes 

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('dataset_students.csv')
objective = df['Admision']
features = df.drop(columns='Admision')

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(features, objective, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_valid_scaled = scaler.fit_transform(X_val)

# Entrenar el modelo con el conjunto de entrenamiento
randomForestModel = RandomForestClassifier(n_estimators=100, random_state=42)
randomForestModel.fit(x_train_scaled, y_train)

# Evaluar el modelo en el conjunto de validación
y_val_pred = randomForestModel.predict(x_valid_scaled)
accuracy_val = accuracy_score(y_val, y_val_pred)
print(f'Precisión en validación: {accuracy_val:.2f}')
print("Matriz de confusión:")
print(confusion_matrix(y_val, y_val_pred))

print("\nReporte de clasificación:")
print(classification_report(y_val, y_val_pred))

dataset_test = pd.read_csv('test.csv')

# Evaluar el modelo en el conjunto de prueba
y_test_pred = randomForestModel.predict(dataset_test)
print(y_test_pred)




