import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Alura Bank - Churn Predictor", page_icon="🏦")

@st.cache_resource
def load_model():
    return joblib.load("modelo_Banco_Alura_churn.pkl")

model = load_model()

st.title("🏦 Dashboard de Predicción: Banco Alura")
st.markdown("Identifica clientes en riesgo de abandono mediante Machine Learning.")
st.divider()

st.subheader("📋 Datos del Cliente")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Edad del cliente", 18, 90, 40)
    products = st.selectbox("Número de productos", options=[1, 2, 3, 4])

with col2:
    inactivo = st.selectbox("¿Es un cliente inactivo (40-70)?", options=[0, 1],
                            format_func=lambda x: "Sí" if x == 1 else "No")
    pais_seleccionado = st.selectbox("País", ["Francia", "Alemania", "España"])
    diccionario_paises = {"Francia": 0, "Alemania": 1, "España": 2}
    c_risk = diccionario_paises[pais_seleccionado]

# Inicializar historial si no existe (debe estar fuera del botón para persistir)
if "historial" not in st.session_state:
    st.session_state.historial = []

if st.button("Analizar Cliente"):
    data = pd.DataFrame([[age, products, inactivo, 0, c_risk]],
                        columns=['Age', 'NumOfProducts', 'Inactivo_40_70',
                                 'Products_Risk_Flag', 'Country_Risk_Flag'])

    prob = model.predict_proba(data)[0, 1]
    pct = round(prob * 100, 2)

    st.subheader("🚀 Resultado del Análisis")

    if prob >= 0.58:
        color_aguja = "#e24b4a"
        estado = "RIESGO ALTO"
        mensaje = f"El cliente de {pais_seleccionado} requiere atención prioritaria."
    elif prob >= 0.40:
        color_aguja = "#f5a623"
        estado = "RIESGO MEDIO"
        mensaje = f"El cliente de {pais_seleccionado} debe ser monitoreado."
    else:
        color_aguja = "#3fc47a"
        estado = "CLIENTE SEGURO"
        mensaje = f"El perfil del cliente en {pais_seleccionado} es estable."

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color_aguja, "thickness": 0.25},
            "steps": [
                {"range": [0, 40],  "color": "#d4f5e2"},
                {"range": [40, 58], "color": "#fde8bc"},
                {"range": [58, 100],"color": "#fcd4d4"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.75,
                "value": 58
            }
        },
        title={"text": f"{estado}", "font": {"size": 18}}
    ))

    fig.update_layout(height=300, margin=dict(t=60, b=10, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)
    st.info(mensaje)

    # --- Guardar en historial ---
    # (Dentro del botón para que solo agregue cuando se haga clic)
    st.session_state.historial.append({
        "riesgo": pct,
        "alto": prob >= 0.58,
        "edad": age,
        "pais": pais_seleccionado,
        "productos": products,
        "activo": "No" if inactivo == 1 else "Sí"
    })

# --- Bloque de Visualización (FUERA del botón) ---
if len(st.session_state.historial) > 0:
    st.divider()
    st.subheader("📊 Resumen de la sesión")

    historial = st.session_state.historial
    total = len(historial)
    riesgo_alto = sum(1 for h in historial if h["alto"])
    riesgo_seguro = total - riesgo_alto
    promedio = sum(h["riesgo"] for h in historial) / total

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total analizados", total)
    col2.metric("Riesgo alto", riesgo_alto)
    col3.metric("Clientes seguros", riesgo_seguro)
    col4.metric("Riesgo promedio", f"{promedio:.1f}%")

    # --- Tabla historial últimos 10 clientes ---
    st.subheader("🗂️ Últimos clientes analizados")

    df_historial = pd.DataFrame(st.session_state.historial[-10:][::-1])
    
    # Formatear la tabla
    df_historial["Estado"] = df_historial["alto"].apply(
        lambda x: "🔴 Riesgo alto" if x else "🟢 Seguro"
    )

    df_historial = df_historial.rename(columns={
        "riesgo":     "% Riesgo",
        "edad":        "Edad",
        "pais":        "País",
        "productos":  "Productos",
        "activo":      "Activo"
    })[["Edad", "País", "Productos", "Activo", "% Riesgo", "Estado"]]

    st.dataframe(df_historial, use_container_width=True)
