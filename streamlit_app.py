import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import io
from fpdf import FPDF

def generar_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    
    # Título del Reporte
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Reporte de Riesgo de Churn - Alura Bank", ln=True, align='C')
    pdf.ln(10)
    
    # Resumen Ejecutivo
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Resumen de la Sesion:", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(200, 10, txt=f"Total de clientes analizados: {len(df)}", ln=True)
    pdf.cell(200, 10, txt=f"Riesgo promedio: {df['% Riesgo'].mean():.2f}%", ln=True)
    pdf.ln(5)

    # Encabezados de Tabla
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(40, 10, "ID Cliente", 1)
    pdf.cell(30, 10, "% Riesgo", 1)
    pdf.cell(120, 10, "Estado y Plan de Accion", 1)
    pdf.ln()

    # Datos de los clientes
    pdf.set_font("Arial", '', 9)
    for i, row in df.iterrows():
        # Limpiamos tildes para evitar errores de codificación latin-1
        id_c = str(row['ID Cliente'])
        riesgo = f"{row['% Riesgo']}%"
        # Reemplazamos caracteres conflictivos para el PDF básico
        estado_plan = f"{row['Estado']} - {row['Plan de Acción']}".replace('ó', 'o').replace('í', 'i').replace('á', 'a').replace('é', 'e').replace('ú', 'u')
        
        pdf.cell(40, 10, id_c, 1)
        pdf.cell(30, 10, riesgo, 1)
        pdf.cell(120, 10, estado_plan[:70], 1) # Cortamos si es muy largo
        pdf.ln()
    
    # Retornamos el PDF como bytes
    return pdf.output(dest='S').encode('latin-1', errors='replace')

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
        background-color: #004a99; color: white; width: 100%;
        border-radius: 10px; height: 3.5em; font-weight: bold; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("modelo_Banco_Alura_churn.pkl")

model = load_model()

# --- FUNCIÓN PROCESADORA ---
def procesar_datos(df_input):
    # Limpiar nombres de columnas por si vienen con saltos de línea o espacios
    df_input.columns = df_input.columns.str.replace(r'\r|\n', '', regex=True).str.strip()
    
    # Columnas requeridas por el modelo
    required = ['Age', 'NumOfProducts', 'Inactivo_40_70', 'Products_Risk_Flag', 'Country_Risk_Flag']
    
    # Verificar si faltan columnas
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
            estado, rec, color = "🔴 Riesgo Alto", "Atención prioritaria: Oferta de retención inmediata. Ofrecer promociones.", "#e24b4a"
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

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
    st.title("Panel de Control")
    st.subheader("📁 Carga Masiva")
    uploaded_file = st.file_uploader("Subir archivo CSV", type=["csv"])
    if st.button("Limpiar Historial"):
        st.session_state.historial = []
        st.rerun()

# --- LÓGICA DE CARGA CSV ---
if uploaded_file is not None:
    try:
        # Leer el contenido para limpiar posibles saltos de línea raros en los encabezados
        content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        # Reemplazar saltos de línea que ocurren dentro de las comillas o mal formateados
        content = content.replace("Inactivo_40_\r\n70", "Inactivo_40_70").replace("Inactivo_40_\n70", "Inactivo_40_70")
        
        # Leer con pandas detectando el separador (coma o punto y coma)
        df_upload = pd.read_csv(io.StringIO(content), sep=None, engine='python')
        
        if st.button("🚀 Procesar Archivo CSV"):
            nuevos = procesar_datos(df_upload)
            if nuevos:
                st.session_state.historial.extend(nuevos)
                st.success(f"✅ {len(nuevos)} clientes procesados.")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")

# --- HEADER ---
st.title("🏦 Churn Insight Banking")
st.divider()

if "historial" not in st.session_state:
    st.session_state.historial = []

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
        res_list = procesar_datos(df_m)
        if res_list:
            res = res_list[0]
            st.session_state.historial.append(res)
            # Gráfico y Resultado...
            g1, g2 = st.columns([2, 1])
            with g1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=res['% Riesgo'],
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": res['color_hex']}}))
                st.plotly_chart(fig, use_container_width=True)
            with g2:
                st.markdown(f"### {res['Estado']}\n{res['Plan de Acción']}")

# --- BUSCA LA SECCIÓN DE TABLA RESUMEN  ---
if st.session_state.historial:
    st.divider()
    st.subheader("📊 Panel de Control y Resumen Ejecutivo")
    
    # Creamos un DataFrame con todo el historial para los cálculos
    df_metriz = pd.DataFrame(st.session_state.historial)
    
    # --- 1. FILA DE MÉTRICAS (Tarjetas) ---
    m1, m2, m3 = st.columns(3)
    
    with m1:
        avg_risk = df_metriz["% Riesgo"].mean()
        st.metric("Tasa de Riesgo Promedio", f"{avg_risk:.2f}%")
        
    with m2:
        # Conteo por categorías basado en los emojis o el texto del estado
        alto = len(df_metriz[df_metriz["% Riesgo"] >= 58])
        medio = len(df_metriz[(df_metriz["% Riesgo"] >= 40) & (df_metriz["% Riesgo"] < 58)])
        st.metric("Clientes en Riesgo (Alto/Medio)", f"{alto} / {medio}", delta=f"{alto} Críticos", delta_color="inverse")
        
    with m3:
        st.metric("Total de Clientes Analizados", len(df_metriz))

   # --- BUSCA ESTA SECCIÓN (FILA DE GRÁFICOS) Y REEMPLÁZALA CON ESTA ---
    # --- 2. FILA DE GRÁFICOS (Corrección de Colores) ---
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown("#### 🌎 Distribución por País")
        conteo_pais = df_metriz["País"].value_counts().reset_index()
        conteo_pais.columns = ["País", "Cantidad"] 
        
        fig_pais = go.Figure(go.Bar(
            x=conteo_pais["País"], 
            y=conteo_pais["Cantidad"],
            marker_color='#004a99' # Azul bancario
        ))
        fig_pais.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pais, use_container_width=True)

    with g_col2:
        st.markdown("#### 📈 Niveles de Riesgo")
        # Forzamos los nombres de las columnas para evitar el KeyError
        conteo_estado = df_metriz["Estado"].value_counts().reset_index()
        conteo_estado.columns = ["Nivel", "Total"]
        
        # --- CORRECCIÓN DE COLORES (Intuitivo) ---
        # Asignamos colores específicos a cada etiqueta para que no dependa del orden
        custom_colors = {
            "🔴 Riesgo Alto": "#e24b4a",  # Rojo Alarma
            "🟡 Riesgo Medio": "#f5a623", # Amarillo Precaución
            "🟢 Seguro": "#3fc47a"       # Verde Seguridad
        }
        
        # Obtenemos la lista ordenada de colores según las etiquetas presentes
        plot_colors = [custom_colors[label] for label in conteo_estado["Nivel"]]
        
        fig_pie = go.Figure(go.Pie(
            labels=conteo_estado["Nivel"], 
            values=conteo_estado["Total"],
            hole=.4,
            marker_colors=plot_colors # Usamos la lista de colores corregida
        ))
        fig_pie.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
      
# --- BOTÓN DE PDF (Añádelo aquí) ---
        st.divider()
        df_reporte = pd.DataFrame(st.session_state.historial)
        
        try:
            pdf_bytes = generar_pdf(df_reporte)
            st.download_button(
                label="📄 Descargar Reporte en PDF",
                data=pdf_bytes,
                file_name="Reporte_Churn_AluraBank.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Error al generar el reporte: {e}")

        # --- 3. TABLA DETALLADA (Lo que ya tienes en tu imagen 6310ab) ---
        st.divider()
        st.markdown("#### 🗒️ Detalle Individual de Clientes")
        # Invertimos el historial para ver lo más reciente arriba
        df_h = pd.DataFrame(st.session_state.historial[::-1])
        st.dataframe(df_h, use_container_width=True)

    
    # --- 3. TABLA DETALLADA (Se mantiene justo debajo) ---
    st.markdown("#### 📑 Detalle Individual de Clientes")
    # Invertimos el historial para ver lo más reciente arriba
    df_h = pd.DataFrame(st.session_state.historial[::-1])
    st.dataframe(
        df_h.drop(columns=['color_hex'], errors='ignore'), 
        use_container_width=True, 
        hide_index=True
    )
