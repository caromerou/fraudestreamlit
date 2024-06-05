import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st

# Leer el archivo CSV
df = pd.read_csv('/kaggle/input/fraude/PS_20174392719_1491204439457_log.csv')

# Codificar variables categóricas
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Escalar características numéricas
scaler = StandardScaler()
df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']] = scaler.fit_transform(
    df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
)

# Definir las características y la variable objetivo
X = df.drop(['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
y = df['isFraud']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
clf = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo
clf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = clf.predict(X_test)

# Mostrar el reporte de clasificación
st.write("Reporte de Clasificación:")
st.write(classification_report(y_test, y_pred))

# Mostrar la matriz de confusión
st.write("Matriz de Confusión:")
st.write(confusion_matrix(y_test, y_pred))

# Crear el modelo Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Entrenar el modelo Gradient Boosting
gb_model.fit(X_train, y_train)

# Hacer predicciones con Gradient Boosting
y_pred_gb = gb_model.predict(X_test)

# Evaluar el modelo Gradient Boosting
st.write("Accuracy GB:", accuracy_score(y_test, y_pred_gb))
st.write("Confusion Matrix GB:\n", confusion_matrix(y_test, y_pred_gb))
st.write("Classification Report GB:\n", classification_report(y_test, y_pred_gb))

# Crear el modelo Support Vector Machine
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')

# Entrenar el modelo Support Vector Machine
svm_model.fit(X_train, y_train)

# Hacer predicciones con Support Vector Machine
y_pred_svm = svm_model.predict(X_test)

# Evaluar el modelo Support Vector Machine
st.write("Accuracy SVM:", accuracy_score(y_test, y_pred_svm))
st.write("Confusion Matrix SVM:\n", confusion_matrix(y_test, y_pred_svm))
st.write("Classification Report SVM:\n", classification_report(y_test, y_pred_svm))


import pandas as pd
import numpy as np
import subprocess

# Instalar scikit-learn
subprocess.run(["pip", "install", "scikit-learn"])

df = pd.read_csv('/kaggle/input/fraude/PS_20174392719_1491204439457_log.csv')

# Otras operaciones en el DataFrame df...

# Definir las características y la variable objetivo
X = df.drop(['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
y = df['isFraud']

# Dividir los datos utilizando pandas split_data
X_train, X_test = pd.split_data(X, test_size=0.2, random_state=42)
y_train, y_test = pd.split_data(y, test_size=0.2, random_state=42)

# Crear el modelo DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo
clf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = clf.predict(X_test)

# Mostrar el reporte de clasificación
print(classification_report(y_test, y_pred))

# Mostrar la matriz de confusión
print(confusion_matrix(y_test, y_pred))

# Otros modelos y evaluaciones...



