import pandas as pd
import numpy as np
import streamlit as st



from google.colab import files
files.upload()  # Sube el archivo JSON de la API de Kaggle



# Cambiar los permisos del archivo
!chmod 600 ~/.kaggle/kaggle.json

# Descargar el conjunto de datos de Kaggle
!kaggle datasets download -d ealaxi/paysim1

# Descomprimir el archivo descargado
!unzip paysim1.zip


# Leer el archivo CSV
#df = pd.read_csv('kaggle datasets download -d ealaxi/paysim1')

# Codificar variables categóricas usando pandas
df['type_encoded'] = df['type'].map({'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4})

# Escalar características numéricas con StandardScaler de scikit-learn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['amount_scaled', 'oldbalanceOrg_scaled', 'newbalanceOrig_scaled', 
    'oldbalanceDest_scaled', 'newbalanceDest_scaled']] = scaler.fit_transform(
    df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
)

# Definir las características y la variable objetivo
X = df.drop(['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
y = df['isFraud']

# Dividir los datos utilizando train_test_split de scikit-learn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.ensemble import GradientBoostingClassifier
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
from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')

# Entrenar el modelo Support Vector Machine
svm_model.fit(X_train, y_train)

# Hacer predicciones con Support Vector Machine
y_pred_svm = svm_model.predict(X_test)

# Evaluar el modelo Support Vector Machine
st.write("Accuracy SVM:", accuracy_score(y_test, y_pred_svm))
st.write("Confusion Matrix SVM:\n", confusion_matrix(y_test, y_pred_svm))
st.write("Classification Report SVM:\n", classification_report(y_test, y_pred_svm))


