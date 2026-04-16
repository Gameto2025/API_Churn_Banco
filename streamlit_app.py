import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# 1. Configuración de página
st.set_page_config(page_title="Bank - Churn Predictor", page_icon="🏦", layout="wide")

# --- CSS PERSONALIZADO ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    [data-testid="stVerticalBlock"] > div:has(div.stSlider) {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    div.stButton > button:first-child {
        background-color: #004a99;
        color: white;
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        font-weight: bold;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("modelo_Banco_Alura_churn.pkl")

model = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
    st.title("Panel de Control")
    if st.button("Limpiar Historial"):
        st.session_state.historial = []
        st.rerun()

# --- HEADER ---
st.title("🏦 Churn Insight Banking")
st.markdown("*Inteligencia Artificial aplicada a la retención de clientes.*")
st.divider()

# --- SECCIÓN DE ENTRADA ---
st.subheader("📋 Datos del Cliente")

if "historial" not in st.session_state:
    st.session_state.historial = []

with st.container():
    # Input de ID
    client_id = st.text_input("ID del Cliente", placeholder="Ej: CLI-2024-001")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Edad del cliente", 18, 90, 40)
        num_productos = st.selectbox("Número de productos", options=[1, 2, 3, 4])
    with col2:
        inactivo = st.selectbox("¿Estado de actividad?", options=[0, 1],
                                format_func=lambda x: "Inactivo" if x == 1 else "Activo")
        pais_seleccionado = st.selectbox("País de residencia", ["Francia", "Alemania", "España"])
        diccionario_paises = {"Francia": 0, "Alemania": 1, "España": 2}
        c_risk = diccionario_paises[pais_seleccionado]

    analyze_btn = st.button("🔍 Analizar Cliente")

# --- LÓGICA DE PREDICCIÓN Y VALIDACIÓN ---
if analyze_btn:
    # 1. Verificar si el campo está vacío
    if not client_id:
        st.error("⚠️ Error: El ID del cliente no puede estar vacío.")
    
    # 2. Verificar si el ID ya existe en el historial (NUEVA VALIDACIÓN)
    elif any(item['ID Cliente'] == client_id for item in st.session_state.historial):
        st.error(f"❌ Error: El ID '{client_id}' ya ha sido analizado en esta sesión. Use un ID diferente.")
    
    else:
        # Si pasa las validaciones, procedemos con el modelo
        data = pd.DataFrame([[age, num_productos, inactivo, 0, c_risk]],
                            columns=['Age', 'NumOfProducts', 'Inactivo_40_70',
                                     'Products_Risk_Flag', 'Country_Risk_Flag'])

        prob = model.predict_proba(data)[0, 1]
        pct = round(prob * 100, 2)

        if prob >= 0.58:
            color_hex, color_bg, estado_texto = "#e24b4a", "#fcd4d4", "RIESGO ALTO"
            recomendacion = "Atención prioritaria: Oferta de retención inmediata."
            etiqueta_tabla = "🔴 Riesgo Alto"
        elif prob >= 0.40:
            color_hex, color_bg, estado_texto = "#f5a623", "#fde8bc", "RIESGO MEDIO"
            recomendacion = "Seguimiento: Contactar para encuesta de satisfacción."
            etiqueta_tabla = "🟡 Riesgo Medio"
        else:
            color_hex, color_bg, estado_texto = "#3fc47a", "#d4f5e2", "CLIENTE SEGURO"
            recomendacion = "Fidelizado: Mantener servicios actuales."
            etiqueta_tabla = "🟢 Seguro"

        st.divider()
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=pct,
                number={"suffix": "%", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color_hex},
                    "steps": [
                        {"range": [0, 40], "color": "#d4f5e2"},
                        {"range": [40, 58], "color": "#fde8bc"},
                        {"range": [58, 100], "color": "#fcd4d4"},
                    ]
                }
            ))
            fig.update_layout(height=250, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with res_col2:
            st.markdown(f"""
                <div style="background-color:{color_bg}; padding:20px; border-radius:15px; border-left: 8px solid {color_hex};">
                    <h3 style="color:#333; margin:0;">{estado_texto}</h3>
                    <p style="color:#444; font-size:1.1em;">{recomendacion}</p>
                </div>
                """, unsafe_allow_html=True)

        # --- GUARDAR EN HISTORIAL ---
        st.session_state.historial.append({
            "ID Cliente": client_id,
            "Edad": age,
            "País": pais_seleccionado,
            "Productos contratados": num_productos,
            "Activo": "No" if inactivo == 1 else "Sí",
            "% Riesgo": pct,
            "Estado": etiqueta_tabla,
            "Plan de Acción": recomendacion
        })

# --- VISUALIZACIÓN DE HISTORIAL ---
if len(st.session_state.historial) > 0:
    st.divider()
    st.subheader("📊 Resumen de la sesión")

    with st.expander("🗂️ Ver detalle de últimos analizados", expanded=True):
        df_display = pd.DataFrame(st.session_state.historial[-10:][::-1])
        st.dataframe(df_display, use_container_width=True, hide_index=True)
