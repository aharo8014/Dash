import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import chi2_contingency
import numpy as np

# ------------------- CONFIGURACIÓN DE STREAMLIT -------------------
st.set_page_config(page_title="Dashboard de Créditos", page_icon="💰", layout="wide")

# ------------------- CARGAR LOS DATOS -------------------
@st.cache_data
def load_data():
    file_path = "2024.xlsx"
    df = pd.read_excel(file_path, sheet_name='2024', engine='openpyxl',
                       usecols=["SEGMENTO", "PROVINCIA", "CANTON", "INSTRUCCION", "SEXO",
                                "RANGO EDAD", "RANGO MONTO CREDITO CONCEDIDO", "RANGO PLAZO ORIGINAL CONCESION",
                                "TIPO PERSONA", "TIPO DE CRÃ‰DITO"])
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

    # Gráfico de Distribución de Créditos por Fecha de Corte
    df_monto = df_filtered["RANGO MONTO CREDITO CONCEDIDO"].value_counts().reset_index()
    df_monto.columns = ["RANGO MONTO CREDITO CONCEDIDO", "count"]

    fig_monto = px.bar(df_monto, x="RANGO MONTO CREDITO CONCEDIDO", y="count",
                       title="Distribución del Monto de Crédito",
                       labels={"count": "Cantidad de Créditos"},
                       width=1000, height=500)
    st.plotly_chart(fig_monto, use_container_width=True)

    # Otros gráficos de interés
    fig_instruccion = px.bar(df_filtered.groupby("INSTRUCCION").size().reset_index(name="Cantidad"),
                             x="INSTRUCCION", y="Cantidad",
                             title="Créditos por Nivel de Instrucción")
    st.plotly_chart(fig_instruccion, use_container_width=True)

    fig_sexo = px.bar(df_filtered.groupby("SEXO").size().reset_index(name="Cantidad"),
                      x="SEXO", y="Cantidad",
                      title="Créditos por Sexo")
    st.plotly_chart(fig_sexo, use_container_width=True)

# ------------------- ANÁLISIS DE COMPONENTES PRINCIPALES (PCA) -------------------
with tab2:
    st.subheader("🔍 Análisis de Componentes Principales (PCA)")

    categorical_columns = ["INSTRUCCION", "SEXO", "RANGO EDAD", "RANGO MONTO CREDITO CONCEDIDO",
                           "TIPO PERSONA", "TIPO DE CRÃ‰DITO"]

    # Convertir variables categóricas en numéricas
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_filtered[col] = le.fit_transform(df_filtered[col].astype(str))
        label_encoders[col] = le

    # Normalización de datos
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_filtered[categorical_columns])

    # Aplicación del PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)

    df_filtered["PCA1"] = pca_result[:, 0]
    df_filtered["PCA2"] = pca_result[:, 1]

    # Gráfico de PCA con etiquetas
    fig_pca = px.scatter(df_filtered, x="PCA1", y="PCA2",
                         color=df_filtered["TIPO DE CRÃ‰DITO"].astype(str),
                         hover_data=categorical_columns,
                         title="Visualización de Asociaciones con PCA",
                         labels={"PCA1": "Componente Principal 1", "PCA2": "Componente Principal 2"},
                         width=1000, height=600)
    st.plotly_chart(fig_pca, use_container_width=True)

    # Importancia de las Variables en los Componentes Principales
    st.subheader("📊 Importancia de Variables en PCA")

    df_pca_importance = pd.DataFrame(pca.components_.T, index=categorical_columns, columns=["PC1", "PC2"])
    df_pca_importance["Importancia Absoluta"] = df_pca_importance.abs().sum(axis=1)
    df_pca_importance = df_pca_importance.sort_values(by="Importancia Absoluta", ascending=False)

    fig_importance = px.bar(df_pca_importance, x=df_pca_importance.index, y="Importancia Absoluta",
                            title="Importancia de Variables en PCA",
                            labels={"x": "Variable", "Importancia Absoluta": "Contribución al PCA"},
                            width=1000, height=500)

    st.plotly_chart(fig_importance, use_container_width=True)

# ------------------- ANÁLISIS CHI-CUADRADO -------------------
with tab3:
    st.subheader("📈 Análisis de Asociación entre Variables Categóricas (Chi-Cuadrado)")

    # Excluir "Destino Financiero" (No está en los datos filtrados)
    chi_vars = ["INSTRUCCION", "SEXO", "RANGO EDAD", "RANGO MONTO CREDITO CONCEDIDO",
                "TIPO PERSONA", "TIPO DE CRÃ‰DITO"]

    results = []
    for var1 in chi_vars:
        for var2 in chi_vars:
            if var1 != var2:
                contingency_table = pd.crosstab(df_filtered[var1], df_filtered[var2])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                results.append({"Variable 1": var1, "Variable 2": var2, "Chi2": chi2, "p-valor": p})

    df_chi_results = pd.DataFrame(results)
    df_chi_results = df_chi_results.sort_values(by="Chi2", ascending=False)

    # Mostrar tabla con resultados del Chi-Cuadrado
    st.write("📌 **Resultados del Test de Chi-Cuadrado**")
    st.dataframe(df_chi_results)

    # Gráfico de Chi-Cuadrado
    fig_chi = px.bar(df_chi_results, x="Variable 1", y="Chi2",
                     color="Variable 2",
                     title="Resultados del Test de Chi-Cuadrado entre Variables",
                     labels={"Chi2": "Valor Chi-Cuadrado"},
                     width=1000, height=500)
    st.plotly_chart(fig_chi, use_container_width=True)

st.success("¡Dashboard completado con éxito! 🚀")
