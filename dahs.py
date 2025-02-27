import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import chi2_contingency
import numpy as np
import requests
from io import BytesIO

# ------------------- CONFIGURACIÓN DE STREAMLIT -------------------
st.set_page_config(page_title="Dashboard de Créditos", page_icon="💰", layout="wide")

# ------------------- DESCARGAR LOS DATOS DESDE GOOGLE DRIVE -------------------
@st.cache_data
def load_data():
    url = "https://docs.google.com/uc?export=download&id=1afSPsXZISWfhzFWtW9mhP40gPIEvR9Fg"
    response = requests.get(url)
    file = BytesIO(response.content)

    df = pd.read_excel(file, sheet_name='2024', engine='openpyxl',
                       usecols=["SEGMENTO", "PROVINCIA", "CANTON", "INSTRUCCION", "SEXO",
                                "RANGO EDAD", "RANGO MONTO CREDITO CONCEDIDO", "RANGO PLAZO ORIGINAL CONCESION",
                                "TIPO PERSONA", "TIPO DE CRÉDITO"])  # 🔹 Corregido nombre de variable
    return df

df = load_data()

# ------------------- FILTROS DE DATOS -------------------
st.sidebar.header("📊 Filtros de Datos")
segmento = st.sidebar.selectbox("Segmento", ["Todos"] + list(df["SEGMENTO"].unique()))
provincia = st.sidebar.selectbox("Provincia", ["Todos"] + list(df["PROVINCIA"].unique()))
canton = st.sidebar.selectbox("Cantón", ["Todos"] + list(df["CANTON"].unique()))

# Aplicar filtros
df_filtered = df.copy()
if segmento != "Todos":
    df_filtered = df_filtered[df_filtered["SEGMENTO"] == segmento]
if provincia != "Todos":
    df_filtered = df_filtered[df_filtered["PROVINCIA"] == provincia]
if canton != "Todos":
    df_filtered = df_filtered[df_filtered["CANTON"] == canton]

# ------------------- PESTAÑAS EN STREAMLIT -------------------
tab1, tab2, tab3 = st.tabs(["📊 Análisis Descriptivo", "🔍 PCA (Componentes Principales)", "📈 Análisis Chi-Cuadrado"])

# ------------------- ANÁLISIS DESCRIPTIVO -------------------
with tab1:
    st.subheader("📊 Análisis Descriptivo de Créditos")

    # Gráficos descriptivos
    fig_monto = px.bar(df_filtered["RANGO MONTO CREDITO CONCEDIDO"].value_counts().reset_index(),
                       x="index", y="RANGO MONTO CREDITO CONCEDIDO",
                       title="Distribución del Monto de Crédito",
                       labels={"index": "Rango de Monto", "RANGO MONTO CREDITO CONCEDIDO": "Cantidad"},
                       width=1000, height=500)
    st.plotly_chart(fig_monto, use_container_width=True)

# ------------------- ANÁLISIS DE COMPONENTES PRINCIPALES (PCA) -------------------
with tab2:
    st.subheader("🔍 Análisis de Componentes Principales (PCA)")

    st.markdown("""
    🔹 **Explicación de la Codificación:**  
    - Para aplicar PCA, convertimos todas las variables categóricas en valores numéricos usando **Label Encoding**.
    - Esto significa que cada categoría dentro de una variable se asigna a un número entero.
    - Ejemplo: `"SEXO"` → {"HOMBRE": 0, "MUJER": 1}.
    - Aunque los valores son numéricos, siguen representando categorías.  
    """)

    categorical_columns = ["INSTRUCCION", "SEXO", "RANGO EDAD", "RANGO MONTO CREDITO CONCEDIDO",
                           "TIPO PERSONA", "TIPO DE CRÉDITO"]

    # Convertir variables categóricas en numéricas
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_filtered[col] = le.fit_transform(df_filtered[col].astype(str))
        label_encoders[col] = le

    # Normalización y PCA
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_filtered[categorical_columns])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)

    df_filtered["PCA1"] = pca_result[:, 0]
    df_filtered["PCA2"] = pca_result[:, 1]

    # Gráfico PCA
    fig_pca = px.scatter(df_filtered, x="PCA1", y="PCA2",
                         color=df_filtered["TIPO DE CRÉDITO"].astype(str),
                         hover_data=categorical_columns,
                         title="Visualización de Asociaciones con PCA",
                         width=1000, height=600)
    st.plotly_chart(fig_pca, use_container_width=True)

# ------------------- ANÁLISIS CHI-CUADRADO -------------------
with tab3:
    st.subheader("📈 Análisis de Asociación entre Variables Categóricas (Chi-Cuadrado)")

    st.markdown("""
    🔹 **Explicación del Test de Chi-Cuadrado:**  
    - Se usa para determinar si **dos variables categóricas están asociadas**.  
    - Un **p-valor bajo (< 0.05)** indica que existe una relación significativa.  
    - Un **p-valor alto (> 0.05)** significa que la relación es probablemente aleatoria.  
    """)

    # Variables para Chi-Cuadrado
    chi_vars = ["INSTRUCCION", "SEXO", "RANGO EDAD", "RANGO MONTO CREDITO CONCEDIDO",
                "TIPO PERSONA", "TIPO DE CRÉDITO"]

    results = []
    for var1 in chi_vars:
        for var2 in chi_vars:
            if var1 != var2:
                contingency_table = pd.crosstab(df_filtered[var1], df_filtered[var2])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                significance = "Significativa" if p < 0.05 else "No Significativa"
                results.append({"Variable 1": var1, "Variable 2": var2,
                                "Chi2": round(chi2, 4),
                                "p-valor": f"{p:.6f}",  # 🔹 Más decimales en el p-valor
                                "Asociación": significance})  # 🔹 Nueva columna de interpretación

    df_chi_results = pd.DataFrame(results)
    df_chi_results = df_chi_results.sort_values(by="Chi2", ascending=False)

    # Mostrar tabla con resultados del Chi-Cuadrado
    st.write("📌 **Resultados del Test de Chi-Cuadrado**")
    st.dataframe(df_chi_results)

    # Gráfico de Chi-Cuadrado
    fig_chi = px.bar(df_chi_results, x="Variable 1", y="Chi2",
                     color="Variable 2",
                     title="Resultados del Test de Chi-Cuadrado entre Variables",
                     width=1000, height=500)
    st.plotly_chart(fig_chi, use_container_width=True)

st.success("¡Dashboard completado con éxito! 🚀")