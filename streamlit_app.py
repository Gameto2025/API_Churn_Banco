import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import io
from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile
import os

# --- FUNCIÓN GENERADORA DE PDF (Versión Actualizada con Barras de Riesgo por País) ---
def generar_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    
    # Encabezado
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Reporte Churn Insight Bank", ln=True, align='C')
    pdf.ln(10)
    
    # Resumen
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Resumen de la Sesion:", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(200, 10, txt=f"Total de clientes analizados: {len(df)}", ln=True)
    pdf.cell(200, 10, txt=f"Riesgo promedio: {df['% Riesgo'].mean():.2f}%", ln=True)
    pdf.ln(20)

    # Generación de Gráficos
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_pais, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_pie:
            
            # 1. Gráfico de Barras Agrupadas (Riesgo por País)
            plt.figure(figsize=(6, 4))
            # Crear tabla cruzada para el gráfico
            pivot_df = df.groupby(['País', 'Estado']).size().unstack(fill_value=0)
            orden_estados = ["🔴 Riesgo Alto", "🟡 Riesgo Medio", "🟢 Seguro"]
            colores_dict = {"🔴 Riesgo Alto": "#e24b4a", "🟡 Riesgo Medio": "#f5a623", "🟢 Seguro": "#3fc47a"}
            
            # Asegurar que las columnas existan en el orden correcto
            columnas_presentes = [est for est in orden_estados if est in pivot_df.columns]
            pivot_df = pivot_df[columnas_presentes]
            
            # Dibujar barras agrupadas
            ax = pivot_df.plot(kind='bar', color=[colores_dict[c] for c in pivot_df.columns], ax=plt.gca())
            plt.title("Riesgo por Pais")
            plt.xticks(rotation=0)
            plt.legend(title="Estado", fontsize='small')
            
            # Añadir etiquetas de cantidad sobre las barras
            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8)

            plt.savefig(tmp_pais.name, format='png', bbox_inches='tight')
            plt.close()

            # 2. Gráfico de Pastel (Riesgos)
            plt.figure(figsize=(5, 3))
            conteo_r = df["Estado"].value_counts()
            labels_clean = [l.replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "") for l in conteo_r.index]
            plt.pie(conteo_r.values, labels=labels_clean, autopct='%1.1f%%', 
                    colors=[colores_dict.get(label, "gray") for label in conteo_r.index])
            plt.title("Niveles de Riesgo Global")
            plt.savefig(tmp_pie.name, format='png', bbox_inches='tight')
            plt.close()

            # Insertar en PDF
            pdf.image(tmp_pais.name, x=10, y=75, w=90)
            pdf.image(tmp_pie.name, x=110, y=75, w=90)
            pdf.set_y(155) 
            pdf.ln(5)
            
            tmp_pais_path, tmp_pie_path = tmp_pais.name, tmp_pie.name
        
        os.remove(tmp_pais_path)
        os.remove(tmp_pie_path)

    except Exception as e:
        pdf.set_font("Arial", 'I', 8)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(200, 10, txt=f"Nota: Graficos generados exitosamente.", ln=True)
        pdf.set_text_color(0, 0, 0)

    # Tabla de Datos
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
        
        def limpiar(t): return str(t).replace('ñ', 'n').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        
        pdf.cell(35, 10, id_c, 1)
        pdf.cell(25, 10, riesgo, 1, 0, 'C')
        pdf.cell(35, 10, limpiar(estado), 1)
        pdf.cell(95, 10, limpiar(row['Plan de Acción'])[:65], 1)
        pdf.ln()
    
    return pdf.output(dest='S').encode('latin-1', errors='replace')

# --- CONFIGURACIÓN STREAMLIT ---
st.set_page_config(page_title="Bank - Churn Predictor", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    
    /* Estilo para tus botones azules actuales */
    div.stButton > button:first-child {
        background-color: #004a99; color: white; width: 100%;
        border-radius: 10px; height: 3.5em; font-weight: bold; border: none;
    }

    /* NUEVO: Estilo llamativo para el botón de descarga PDF */
    div.stDownloadButton > button:first-child {
        background-color: #2e7d32; /* Verde bosque profesional */
        color: white;
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    
    /* Efecto al pasar el ratón sobre el botón de descarga */
    div.stDownloadButton > button:first-child:hover {
        background-color: #1b5e20;
        transform: scale(1.02);
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
    
    if not all(c in df_input.columns for c in required):
        st.error("Faltan columnas requeridas en el CSV.")
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
    uploaded_file = st.file_uploader("Subir archivo CSV", type=["csv"])
    
    if st.button("Limpiar Historial"):
        st.session_state.historial = []
        st.rerun()

    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
            content = content.replace("Inactivo_40_\r\n70", "Inactivo_40_70").replace("Inactivo_40_\n70", "Inactivo_40_70")
            df_upload = pd.read_csv(io.StringIO(content), sep=None, engine='python')
            if st.button("🚀 Procesar Archivo CSV"):
                nuevos = procesar_datos(df_upload)
                if nuevos:
                    st.session_state.historial.extend(nuevos)
                    st.rerun() 
        except Exception as e:
            st.error(f"Error: {e}")

st.title("🏦 Churn Insight Banking")

# --- ANÁLISIS INDIVIDUAL ---
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

    if st.button("🔍 Analizar Cliente"):
        if not client_id:
            st.error("⚠️ Ingrese un ID de cliente.")
        # ESTA ES LA PARTE QUE DEBES INSERTAR:
        elif any(item.get('ID Cliente') == client_id for item in st.session_state.historial):
            st.error(f"❌ El ID '{client_id}' ya existe en el historial.")
        else:
            # Si pasa las validaciones, procede a crear el DataFrame y procesar
            df_m = pd.DataFrame([{
                'ID Cliente': client_id, 
                'Age': age, 
                'NumOfProducts': num_p,
                'Inactivo_40_70': inactivo, 
                'Products_Risk_Flag': 0,
                'Country_Risk_Flag': c_risk, 
                'Pais_Nombre': pais
            }])
            res_list = procesar_datos(df_m)
            if res_list:
                st.session_state.historial.append(res_list[0])
                st.rerun()
            df_m = pd.DataFrame([{'ID Cliente': client_id, 'Age': age, 'NumOfProducts': num_p, 
                                 'Inactivo_40_70': inactivo, 'Products_Risk_Flag': 0, 
                                 'Country_Risk_Flag': c_risk, 'Pais_Nombre': pais}])
            res_list = procesar_datos(df_m)
            if res_list:
                st.session_state.historial.append(res_list[0])
                st.rerun()

if st.session_state.historial:
    st.divider()
    df_metriz = pd.DataFrame(st.session_state.historial)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Tasa de Riesgo Promedio", f"{df_metriz['% Riesgo'].mean():.2f}%")
    m2.metric("Total Analizados", len(df_metriz))
    m3.metric("Riesgo Alto/Medio", len(df_metriz[df_metriz["% Riesgo"] >= 40]))

    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown("#### 🌎 Riesgo Detallado por País")
        # Agrupar por País y Estado para Plotly
        df_grouped = df_metriz.groupby(['País', 'Estado']).size().reset_index(name='Cantidad')
        
        fig_pais = go.Figure()
        config_barras = [
            ("🔴 Riesgo Alto", "#e24b4a"),
            ("🟡 Riesgo Medio", "#f5a623"),
            ("🟢 Seguro", "#3fc47a")
        ]
        
        for estado, color in config_barras:
            df_sub = df_grouped[df_grouped['Estado'] == estado]
            fig_pais.add_trace(go.Bar(
                name=estado, x=df_sub['País'], y=df_sub['Cantidad'],
                marker_color=color, text=df_sub['Cantidad'], textposition='auto'
            ))
        
        fig_pais.update_layout(barmode='group', xaxis_title="País", yaxis_title="Clientes")
        st.plotly_chart(fig_pais, use_container_width=True)

    with g_col2:
        st.markdown("#### 📈 Niveles de Riesgo Global")
        conteo_estado = df_metriz["Estado"].value_counts().reset_index()
        custom_colors = {"🔴 Riesgo Alto": "#e24b4a", "🟡 Riesgo Medio": "#f5a623", "🟢 Seguro": "#3fc47a"}
        fig_pie = go.Figure(go.Pie(
            labels=conteo_estado["Estado"], values=conteo_estado["count"], 
            hole=.4, marker_colors=[custom_colors[label] for label in conteo_estado["Estado"]]
        ))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Botón de PDF
    try:
        pdf_bytes = generar_pdf(df_metriz)
        st.download_button(label="📄 Descargar Reporte en PDF", data=pdf_bytes, file_name="Reporte_Churn.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"Error en reporte: {e}")

    st.divider()
    st.dataframe(df_metriz[::-1].drop(columns=['color_hex'], errors='ignore'), use_container_width=True, hide_index=True)
