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
                if nuevos:
                    st.session_state.historial.extend(nuevos)
                    st.success(f"✅ {len(nuevos)} clientes procesados.")
                    st.rerun() 
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

st.title("🏦 Churn Insight Banking")
st.divider()

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
        res_list = procesar_datos(df_m)
        if res_list:
            res = res_list[0]
            st.session_state.historial.append(res)
            g1, g2 = st.columns([2, 1])
            with g1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=res['% Riesgo'],
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": res['color_hex']}}))
                st.plotly_chart(fig, use_container_width=True)
            with g2:
                st.markdown(f"### {res['Estado']}\n{res['Plan de Acción']}")

if st.session_state.historial:
    st.divider()
    st.subheader("📊 Panel de Control y Resumen Ejecutivo")
    df_metriz = pd.DataFrame(st.session_state.historial)
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Tasa de Riesgo Promedio", f"{df_metriz['% Riesgo'].mean():.2f}%")
    with m2:
        alto = len(df_metriz[df_metriz["% Riesgo"] >= 58])
        medio = len(df_metriz[(df_metriz["% Riesgo"] >= 40) & (df_metriz["% Riesgo"] < 58)])
        st.metric("Riesgo (Alto/Medio)", f"{alto} / {medio}")
    with m3:
        st.metric("Total Analizados", len(df_metriz))

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown("#### 🌎 Distribución por País")
        conteo_pais = df_metriz["País"].value_counts().reset_index()
        conteo_pais.columns = ["País", "Cantidad"]
        
        # Colores personalizados solicitados
        mapa_colores = {
            "España": "#FFD700",   # Amarillo
            "Alemania": "#8B4513", # Café
            "Francia": "#87CEEB"   # Celeste
        }
        colores_barras = [mapa_colores.get(p, "#004a99") for p in conteo_pais["País"]]
        
        fig_pais = go.Figure(go.Bar(
            x=conteo_pais["País"], 
            y=conteo_pais["Cantidad"], 
            marker_color=colores_barras
        ))
        st.plotly_chart(fig_pais, use_container_width=True)

    with g_col2:
        st.markdown("#### 📈 Niveles de Riesgo")
        conteo_estado = df_metriz["Estado"].value_counts().reset_index()
        conteo_estado.columns = ["Nivel", "Total"]
        custom_colors = {"🔴 Riesgo Alto": "#e24b4a", "🟡 Riesgo Medio": "#f5a623", "🟢 Seguro": "#3fc47a"}
        plot_colors = [custom_colors[label] for label in conteo_estado["Nivel"]]
        fig_pie = go.Figure(go.Pie(labels=conteo_estado["Nivel"], values=conteo_estado["Total"], hole=.4, marker_colors=plot_colors))
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- SECCIÓN DE DESCARGA PDF ---
    try:
        pdf_bytes = generar_pdf(df_metriz, fig_pais, fig_pie)
        st.download_button(
            label="📄 Descargar Reporte en PDF",
            data=pdf_bytes,
            file_name="Reporte_Churn_Insight.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(f"Error al generar el reporte: {e}")

    st.divider()
    st.markdown("#### 🗒️ Detalle Individual de Clientes")
    df_h = pd.DataFrame(st.session_state.historial[::-1])
    st.dataframe(df_h.drop(columns=['color_hex'], errors='ignore'), use_container_width=True, hide_index=True)
