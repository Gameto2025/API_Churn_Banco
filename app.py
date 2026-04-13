import streamlit as st
import joblib
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Alura Bank - Churn Predictor", page_icon="🏦")

# Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load("modelo_Banco_Alura_churn.pkl")

model = load_model()

st.title("🏦 Dashboard de Predicción: Banco Alura")
st.markdown("Identifica clientes en riesgo de abandono mediante Machine Learning.")

st.divider()

# Formulario de entrada simplificado
st.subheader("📋 Datos del Cliente")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Edad del cliente", 18, 90, 40)
    products = st.selectbox("Número de productos", options=[1, 2, 3, 4])

with col2:
    inactivo = st.selectbox("¿Es un cliente inactivo (40-70)?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    
    # Mapeo de países: Francia = 0, Alemania = 1, España = 2
    pais_seleccionado = st.selectbox("País", ["Francia", "Alemania", "España"])
    diccionario_paises = {"Francia": 0, "Alemania": 1, "España": 2}
    c_risk = diccionario_paises[pais_seleccionado]

# Botón de acción
if st.button("Analizar Cliente"):
    # IMPORTANTE: Mantenemos 'Products_Risk_Flag' con valor 0 para no romper la estructura que el modelo espera
    data = pd.DataFrame([[age, products, inactivo, 0, c_risk]], 
                        columns=['Age', 'NumOfProducts', 'Inactivo_40_70', 'Products_Risk_Flag', 'Country_Risk_Flag'])
    
    # Predicción de probabilidad
    prob = model.predict_proba(data)[0, 1]
    
    st.subheader("🚀 Resultado del Análisis")
    
    # Umbral de decisión (Threshold) de 0.58
    if prob >= 0.58:
        st.error(f"**RIESGO ALTO: {prob:.2%} de probabilidad de abandono**")
        st.write(f"Sugerencia: El cliente de {pais_seleccionado} requiere atención prioritaria.")
    else:
        st.success(f"**CLIENTE SEGURO: {prob:.2%} de riesgo**")
        st.write(f"El perfil del cliente en {pais_seleccionado} es estable.")