import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    # Mostrar primeras filas del DataFrame
    st.write(df.head())

    # Verificar si la columna 'type' existe y no tiene valores nulos
    if 'type' in df.columns and not df['type'].isnull().any():
        # Codificar variables categóricas
        df['type_encoded'] = df['type'].map({'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4})

        # Verificar si el mapeo fue exitoso
        if df['type_encoded'].isnull().any():
            st.write("Error: Existen valores en 'type' que no se pudieron mapear.")
        else:
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

            # Crear el modelo Decision Tree
            clf = DecisionTreeClassifier(random_state=42)

            # Entrenar el modelo
            clf.fit(X_train, y_train)

            # Predecir en el conjunto de prueba
            y_pred = clf.predict(X_test)

            # Mostrar el reporte de clasificación
            st.write("Reporte de Clasificación (Decision Tree):")
            st.write(classification_report(y_test, y_pred))

            # Mostrar la matriz de confusión
            st.write("Matriz de Confusión (Decision Tree):")
            st.write(confusion_matrix(y_test, y_pred))

            # Crear el modelo Gradient Boosting
            gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

            # Entrenar el modelo Gradient Boosting
            gb_model.fit(X_train, y_train)

            # Hacer predicciones con Gradient Boosting
            y_pred_gb = gb_model.predict(X_test)

            # Evaluar el modelo Gradient Boosting
            st.write("Accuracy (Gradient Boosting):", accuracy_score(y_test, y_pred_gb))
            st.write("Confusion Matrix (Gradient Boosting):")
            st.write(confusion_matrix(y_test, y_pred_gb))
            st.write("Classification Report (Gradient Boosting):")
            st.write(classification_report(y_test, y_pred_gb))

            # Crear el modelo Support Vector Machine
            svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')

            # Entrenar el modelo Support Vector Machine
            svm_model.fit(X_train, y_train)

            # Hacer predicciones con Support Vector Machine
            y_pred_svm = svm_model.predict(X_test)

            # Evaluar el modelo Support Vector Machine
            st.write("Accuracy (Support Vector Machine):", accuracy_score(y_test, y_pred_svm))
            st.write("Confusion Matrix (Support Vector Machine):")
            st.write(confusion_matrix(y_test, y_pred_svm))
            st.write("Classification Report (Support Vector Machine):")
            st.write(classification_report(y_test, y_pred_svm))
    else:
        st.write("Error: La columna 'type' no existe o contiene valores nulos.")
else:
    st.write("Por favor, sube un archivo CSV.")
