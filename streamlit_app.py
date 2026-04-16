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

# --- FUNCIÓN PROCESADORA DE RIESGO ---
def procesar_datos(df_input):
    X = df_input[['Age', 'NumOfProducts', 'Inactivo_40_70', 'Products_Risk_Flag', 'Country_Risk_Flag']]
    probs = model.predict_proba(X)[:, 1]
    
    resultados = []
    for i, prob in enumerate(probs):
        pct = round(prob * 100, 2)
        if prob >= 0.58:
            estado, rec, color = "🔴 Riesgo Alto", "Atención prioritaria: Oferta de retención inmediata.", "#e24b4a"
        elif prob >= 0.40:
            estado, rec, color = "🟡 Riesgo Medio", "Seguimiento: Contactar para encuesta de satisfacción.", "#f5a623"
        else:
            estado, rec, color = "🟢 Seguro", "Fidelizado: Mantener servicios actuales.", "#3fc47a"
            
        resultados.append({
            "ID Cliente": df_input.iloc[i].get("ID Cliente", f"Batch-{i}"),
            "Edad": df_input.iloc[i]["Age"],
            "País": df_input.iloc[i].get("Pais_Nombre", "N/A"),
            "Productos contratados": df_input.iloc[i]["NumOfProducts"],
            "Activo": "Sí" if df_input.iloc[i]["Inactivo_40_70"] == 0 else "No",
            "% Riesgo": pct,
            "Estado": estado,
            "Plan de Acción": rec,
            "color_hex": color 
        })
    return resultados

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
    st.title("Panel de Control")
    st.subheader("📁 Carga Masiva")
    uploaded_file = st.file_uploader("Subir archivo CSV", type=["csv"])
    if st.button("Limpiar Historial"):
        st.session_state.historial = []
        st.rerun()

# --- HEADER ---
st.title("🏦 Churn Insight Banking")
st.markdown("*Inteligencia Artificial aplicada a la retención de clientes.*")
st.divider()

if "historial" not in st.session_state:
    st.session_state.historial = []

# --- LÓGICA CSV ---
if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')
        if st.button("🚀 Procesar Archivo CSV"):
            nuevos = procesar_datos(df_upload)
            st.session_state.historial.extend(nuevos)
            st.success(f"✅ {len(nuevos)} clientes agregados.")
    except Exception as e:
        st.error(f"Error al procesar CSV: {e}")

# --- ANÁLISIS INDIVIDUAL ---
st.subheader("📋 Análisis Individual")
with st.container():
    client_id = st.text_input("ID del Cliente", placeholder="Ej: CLI-001")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Edad", 18, 90, 40)
        num_p = st.selectbox("Productos", [1, 2, 3, 4])
    with col2:
        inactivo = st.selectbox("Actividad", [0, 1], format_func=lambda x: "Inactivo" if x==1 else "Activo")
        pais = st.selectbox("País", ["Francia", "Alemania", "España"])
        c_risk = {"Francia": 0, "Alemania": 1, "España": 2}[pais]

    analyze_btn = st.button("🔍 Analizar Cliente")

if analyze_btn:
    if not client_id:
        st.error("Ingrese un ID.")
    elif any(item.get('ID Cliente') == client_id for item in st.session_state.historial):
        st.error("ID duplicado.")
    else:
        df_m = pd.DataFrame([{'ID Cliente': client_id, 'Age': age, 'NumOfProducts': num_p, 
                             'Inactivo_40_70': inactivo, 'Products_Risk_Flag': 0, 
                             'Country_Risk_Flag': c_risk, 'Pais_Nombre': pais}])
        res = procesar_datos(df_m)[0]
        st.session_state.historial.append(res)
        
        # Gráfico
        st.divider()
        g1, g2 = st.columns([2, 1])
        with g1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=res['% Riesgo'],
                number={"suffix": "%", "font": {"size": 40}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": res['color_hex']},
                       "steps": [{"range": [0, 40], "color": "#d4f5e2"},
                                 {"range": [40, 58], "color": "#fde8bc"},
                                 {"range": [58, 100], "color": "#fcd4d4"}]}))
            fig.update_layout(height=250, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            st.markdown(f"""<div style="background-color:#f8f9fa; padding:20px; border-radius:15px; border-left: 8px solid {res['color_hex']};">
                <h3>{res['Estado']}</h3><p>{res['Plan de Acción']}</p></div>""", unsafe_allow_html=True)

# --- TABLA RESUMEN ---
if st.session_state.historial:
    st.divider()
    st.subheader("📊 Resumen de la sesión")
    df_h = pd.DataFrame(st.session_state.historial[::-1])
    
    # SOLUCIÓN AL ERROR: Usamos errors='ignore' para que no falle si la columna no existe
    df_visible = df_h.drop(columns=['color_hex'], errors='ignore')
    
    st.dataframe(df_visible, use_container_width=True, hide_index=True)
    
    # Botón de descarga
    csv_data = df_visible.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar Reporte", csv_data, "reporte_churn.csv", "text/csv")
