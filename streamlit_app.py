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
    """Aplica el modelo y genera las etiquetas de negocio"""
    # Preparar datos para el modelo (usando los nombres de columnas originales)
    X = df_input[['Age', 'NumOfProducts', 'Inactivo_40_70', 'Products_Risk_Flag', 'Country_Risk_Flag']]
    
    probs = model.predict_proba(X)[:, 1]
    
    resultados = []
    for i, prob in enumerate(probs):
        pct = round(prob * 100, 2)
        if prob >= 0.58:
            estado, rec = "🔴 Riesgo Alto", "Atención prioritaria: Oferta de retención inmediata."
        elif prob >= 0.40:
            estado, rec = "🟡 Riesgo Medio", "Seguimiento: Contactar para encuesta de satisfacción."
        else:
            estado, rec = "🟢 Seguro", "Fidelizado: Mantener servicios actuales."
            
        resultados.append({
            "ID Cliente": df_input.iloc[i].get("ID Cliente", f"Batch-{i}"),
            "Edad": df_input.iloc[i]["Age"],
            "País": df_input.iloc[i].get("Pais_Nombre", "N/A"),
            "Productos contratados": df_input.iloc[i]["NumOfProducts"],
            "Activo": "Sí" if df_input.iloc[i]["Inactivo_40_70"] == 0 else "No",
            "% Riesgo": pct,
            "Estado": estado,
            "Plan de Acción": rec
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

# --- LÓGICA DE CARGA POR ARCHIVO ---
if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)
        # Verificamos columnas mínimas necesarias
        required = ['Age', 'NumOfProducts', 'Inactivo_40_70', 'Products_Risk_Flag', 'Country_Risk_Flag']
        if all(col in df_upload.columns for col in required):
            if st.button("🚀 Procesar Archivo CSV"):
                nuevos_registros = procesar_datos(df_upload)
                st.session_state.historial.extend(nuevos_registros)
                st.success(f"✅ Se han procesado {len(nuevos_registros)} clientes con éxito.")
        else:
            st.error(f"El CSV debe contener las columnas: {required}")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")

# --- SECCIÓN DE ENTRADA MANUAL ---
st.subheader("📋 Análisis Individual")
with st.container():
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

    analyze_btn = st.button("🔍 Analizar Cliente Individual")

if analyze_btn:
    if not client_id:
        st.error("⚠️ El ID del cliente es obligatorio.")
    elif any(item['ID Cliente'] == client_id for item in st.session_state.historial):
        st.error(f"❌ El ID '{client_id}' ya existe en el historial.")
    else:
        # Crear DataFrame de un solo registro para la función
        df_manual = pd.DataFrame([{
            'ID Cliente': client_id,
            'Age': age,
            'NumOfProducts': num_productos,
            'Inactivo_40_70': inactivo,
            'Products_Risk_Flag': 0,
            'Country_Risk_Flag': c_risk,
            'Pais_Nombre': pais_seleccionado
        }])
        
        resultado = procesar_datos(df_manual)[0]
        st.session_state.historial.append(resultado)
        
        # Mostrar métricas del análisis individual
        st.metric("Probabilidad de Churn", f"{resultado['% Riesgo']}%")

# --- VISUALIZACIÓN DE HISTORIAL ---
if len(st.session_state.historial) > 0:
    st.divider()
    st.subheader("📊 Resumen de la sesión")
    
    with st.expander("🗂️ Ver detalle de últimos analizados", expanded=True):
        df_display = pd.DataFrame(st.session_state.historial[::-1])
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Opción para descargar los resultados de la sesión
        csv_download = df_display.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Resultados (CSV)", csv_download, "analisis_churn.csv", "text/csv")
