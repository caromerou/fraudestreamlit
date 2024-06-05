


%%writefile modelo.py

import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image


#Funcion para nuestra animacion
def load_lottieurl(url):
  r = requests.get(url)
  if r.status_code != 200:
    return None
  return r.json()

lottie_coding = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_0yfsb3a1.json")
imagen_video = Image.open("/content/maxresdefault (2).jpg") #Pon la ruta de tu imagen

with st.container():
  st.subheader("Hola bienvenido a mi sitio web")
  st.title("Introduccion a la ciencia de datos")
  st.write("Bienvenido a mi canal")
  st.write("[Mas informacion >](https:///www.youtube.com")

with st.container():
  st.write("---")
  left_column, right_column = st.columns(2)
  with left_column:
    st.header("Mi objetivo")
    st.write(
      """
        Texto aquí
      """
    )
    st.write("[Youtube >](https://youtube.com)")
  with right_column:
    st_lottie(lottie_coding, height=300, key="coding")

with st.container():
  st.write("--")
  st.header("Mis videos")
  image_column, text_column = st.columns((1, 2))
  with image_column:
    st.image(imagen_video)
  with text_column:
    st.write(
      """
      texto aquí
      """
    )
    st.markdown("[Ver video...](http:///youtube.mx)")

!streamlit run app.py & npx localtunnel --port 8501
  
