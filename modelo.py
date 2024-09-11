#Autor: Gustavo Alejandro Gutiérrez Valdes - A01747869

import numpy as np
import pandas as pd

#Se importan los recursos necesarios del framework de sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #type: ignore
from sklearn.metrics import mean_squared_error #type: ignore
from sklearn.ensemble import RandomForestClassifier #type: ignore
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve

#Se lee el dataset de estudiantes que compondrá la etapa de training y validation
df = pd.read_csv('dataset_students.csv')

#Son las columnas que se utilizarán para entrenar el modelo (quitando la columna objetivo)
features = df.drop(columns='Admision')

#Se guarda la columna objetivo para hacer comparaciones posteriormente
objective = df['Admision']

# Dividir en entrenamiento y validación (70% train, 30% validation)
X_train, X_val, y_train, y_val = train_test_split(features, objective, test_size=0.3, random_state=35)

# Se realiza la escalación de los datos de entrenamiento y validación utilizando MinMaxScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_valid_scaled = scaler.transform(X_val)

# Se definen los diferentes conjuntos de hiperparámetros que se probarán entre si para encontrar la mejor combinación en el 
# siguiente paso. Se eligieron la cantidad de arboles a generar, la profundiad máxima de cada uno y la cantidad  de muestras 
# con la que los nodos se dividirán, ya que estos hiperparámetros controlan directamente la capacidad predictiva al 
# reducir la varianza y hacer las predicciones mas robustas. De igual forma, se mantiene dominio acerca del subajuste o 
# sobreajuste del modelo y finalmente se mitiga el impacto del ruido, dandole al modelo una mayor capacidad de generalización.
param_grid = {
    'n_estimators': [10, 50,100, 150],
    'max_depth': [None, 5, 10,20],
    'min_samples_split': [2, 5, 10]
}

# Crear un objeto GridSearchCV. Este objeto se encargará de probar todas las combinaciones posibles de hiperparámetros y elegir 
# la que mejor funcione con el modelo elegido, se probará con validación cruzada de 5 folds y se utilizarán todos los núcleos 
# disponibles para el trabajo
grid_search = GridSearchCV(estimator=RandomForestClassifier(), 
                          param_grid=param_grid, 
                          cv=5, 
                          n_jobs=-1)

#Se aplican el objeto de GridSearchCV a los datos de entrenamiento
grid_search.fit(x_train_scaled, y_train)

# Se obtiene el mejor modelo dentro de las combinaciones disponibles
best_model = grid_search.best_estimator_

y_train_pred = best_model.predict(x_train_scaled)

accuracy_train = accuracy_score(y_train, y_train_pred)

# Evaluar el modelo en el conjunto de validación
y_val_pred = best_model.predict(x_valid_scaled)

#Se obtiene el error cuadrático medio y la raíz del error cuadrático medio para entender su desempeño con el conjunto de validación
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

# Graficar las precisiones de entrenamiento y validación
plt.figure(figsize=(6, 4))
plt.bar(['Entrenamiento', 'Validación'], [accuracy_train, accuracy_val], color=['blue', 'green'])
plt.title('Precisión en Entrenamiento y Validación')
plt.ylabel('Precisión')
plt.ylim(0, 1)
plt.show()

# Obtener la curva de aprendizaje
train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_model,
    X=x_train_scaled,
    y=y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

# Promediar las puntuaciones de precisión
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

# Graficar la curva de aprendizaje
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Precisión en Entrenamiento', color='blue', marker='o')
plt.plot(train_sizes, val_mean, label='Precisión en Validación', color='green', marker='o')
plt.title('Curva de Aprendizaje')
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('Precisión')
plt.legend(loc='best')
plt.grid(True)
plt.show()


