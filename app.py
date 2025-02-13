import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import requests
from io import BytesIO

# Cargar el modelo y el escalador
modelo_knn = joblib.load('modelo_knn.bin')
escalador = joblib.load('escalador.bin')

# Función para realizar la predicción
def predecir(edad, colesterol):
    # Crear el dataframe con los datos ingresados
    datos = pd.DataFrame({'edad': [edad], 'colesterol': [colesterol]})
    
    # Normalizar los datos
    datos_normalizados = escalador.transform(datos)
    
    # Realizar la predicción
    prediccion = modelo_knn.predict(datos_normalizados)
    return prediccion[0]

# Título y autor
st.title("Asistente Cardiaco")
st.markdown("Autor: Alfredo Díaz")

# Instrucción de uso
st.markdown("""
### Instrucciones:
1. Ingresa tu edad y nivel de colesterol usando los sliders en el primer tab.
2. El modelo realizará una predicción sobre la posibilidad de tener problemas cardíacos.
""")

# Crear los tabs
tab1, tab2 = st.tabs(["Ingreso de Datos", "Predicción"])

with tab1:
    # Ingreso de datos usando sliders
    edad = st.slider("Edad", min_value=18, max_value=80, value=30)
    colesterol = st.slider("Colesterol", min_value=100, max_value=600, value=200)
    
    st.write(f"Edad seleccionada: {edad}")
    st.write(f"Nivel de colesterol seleccionado: {colesterol}")

with tab2:
    # Realizar predicción cuando se den los datos
    if st.button("Predecir"):
        resultado = predecir(edad, colesterol)
        
        if resultado == 0:
            st.write("No tienes problemas cardíacos.")
            image_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxIQ..."
            st.image(image_url, caption="No tienes problemas cardíacos")
        else:
            st.write("Tienes problemas cardíacos.")
            image_url = "https://cloudfront-us-east-1.images.arcpublishing.com/infobae/WRI4UH2CFFG3PFSLDLXBXW4YV4.jpg"
            st.image(image_url, caption="Tienes problemas cardíacos")

