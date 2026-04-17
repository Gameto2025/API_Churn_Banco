import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import io
from fpdf import FPDF
import time

# --- FUNCIÓN GENERADORA DE PDF ---
def generar_pdf(df, fig_pais, fig_pie):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Reporte Churn Insight Bank", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Resumen de la Sesion:", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(200, 10, txt=f"Total de clientes analizados: {len(df)}", ln=True)
    pdf.cell(200, 10, txt=f"Riesgo promedio: {df['% Riesgo'].mean():.2f}%", ln=True)
    pdf.ln(5)

    try:
        time.sleep(0.5)
        img_pais_bytes = fig_pais.to_image(format="png", engine="kaleido")
        img_pie_bytes = fig_pie.to_image(format="png", engine="kaleido")
        
        pdf.image(io.BytesIO(img_pais_bytes), x=10, y=50, w=90)
        pdf.image(io.BytesIO(img_pie_bytes), x=110, y=50, w=90)
        pdf.ln(65)
    except Exception as e:
        pdf.set_font("Arial", 'I', 10)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(200, 10, txt="Nota: Los graficos no pudieron incluirse en este reporte.", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

    pdf.set_font("Arial", 'B', 10)
    pdf.cell(35, 10, "ID Cliente", 1)
    pdf.cell(25, 10, "% Riesgo", 1, 0, 'C')
    pdf.cell(35, 10, "Estado", 1)
    pdf.cell(95, 10, "Plan de Accion", 1)
    pdf.ln()

    pdf.set_font("Arial", '', 9)
    for i, row in df.iterrows():
        id_c = str(row['ID Cliente'])
        riesgo = f"{float(row['% Riesgo']):.2f}%"
        estado = str(row['Estado']).replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "")
        plan = str(row['Plan de Acción'])
        
        def limpiar_texto(t):
            return t.replace('ñ', 'n').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')

        pdf.cell(35, 10, id_c, 1)
        pdf.cell(25, 10, riesgo, 1, 0, 'C')
        pdf.cell(35, 10, limpiar_texto(estado), 1)
        pdf.cell(95, 10, limpiar_texto(plan)[:65], 1)
        pdf.ln()
    
    return pdf.output(dest='S').encode('latin-1', errors='replace')

# 1. Configuración de página
st.set_page_config(page_title="Bank - Churn Predictor", page_icon="🏦", layout="wide")

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
        background-color: #004a99; color: white; width: 100%;
        border-radius: 10px; height: 3.5em; font-weight: bold; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("modelo_Banco_Alura_churn.pkl")

model = load_model()

def procesar_datos(df_input):
    df_input.columns = df_input.columns.str.replace(r'\r|\n', '', regex=True).str.strip()
    required = ['Age', 'NumOfProducts', 'Inactivo_40_70', 'Products_Risk_Flag', 'Country_Risk_Flag']
    
    missing = [c for c in required if c not in df_input.columns]
    if missing:
        st.error(f"Faltan columnas en el archivo: {missing}")
        return []

    X = df_input[required]
    probs = model.predict_proba(X)[:, 1]
    
    resultados = []
    for i, prob in enumerate(probs):
        pct = round(prob * 100, 2)
        if prob >= 0.58:
            estado, rec, color = "🔴 Riesgo Alto", "Atención prioritaria: Oferta de retención inmediata.", "#e24b4a"
        elif prob >= 0.40:
            estado, rec, color = "🟡 Riesgo Medio", "Seguimiento: Llamar y realizar encuesta de satisfacción.", "#f5a623"
        else:
            estado, rec, color = "🟢 Seguro", "Fidelizado: Mantener servicios actuales.", "#3fc47a"
            
        resultados.append({
            "ID Cliente": df_input.iloc[i].get("ID Cliente", df_input.iloc[i].get("ID", f"Batch-{i}")),
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

if "historial" not in st.session_state:
    st.session_state.historial = []

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png")
    st.title("Panel de control")
    
    st.subheader("📂 Carga Masiva")
    uploaded_file = st.file_uploader("Subir archivo CSV", type=["csv"])
    
    if st.button("Limpiar Historial", use_container_width=True):
        st.session_state.historial = []
        st.rerun()

    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
            content = content.replace("Inactivo_40_\r\n70", "Inactivo_40_70").replace("Inactivo_40_\n70", "Inactivo_40_70")
            df_upload = pd.read_csv(io.StringIO(content), sep=None, engine='python')

            if st.button("🚀 Procesar Archivo CSV", use_container_width=True):
                nuevos = procesar_datos(df_upload)
