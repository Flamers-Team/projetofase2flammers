import streamlit as st
import pandas as pd
import plotly.express as px

# Configura a página do Streamlit
st.set_page_config(page_title="Dashboard de Pacientes", layout="wide")

# Título do App
st.title("📊 Distribuição de Pacientes por Região (DF)")

# Carrega os dados (Usando cache para não recarregar o CSV a cada clique)
@st.cache_data
def carregar_dados():
    # Substitua pelo caminho do seu CSV
    return pd.read_csv('pacientes_df.csv')

df = carregar_dados()

# -----------------
# 1. Preparação dos Dados
# -----------------
# Conta os pacientes e reseta o index para criar um DataFrame limpo para o Plotly
df_contagem = df['regiao_administrativa'].value_counts().reset_index()
df_contagem.columns = ['Região Administrativa', 'Quantidade de Pacientes']

# -----------------
# 2. Exibição na Tela
# -----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Tabela de Dados")
    # Exibe um dataframe interativo
    st.dataframe(df_contagem, use_container_width=True)

with col2:
    st.subheader("Gráfico de Barras")
    # Cria o gráfico com Plotly
    fig_barras = px.bar(
        df_contagem, 
        x='Região Administrativa', 
        y='Quantidade de Pacientes',
        text_auto=True, # Mostra o número em cima da barra
        title="Pacientes por RA"
    )
    # Renderiza o gráfico no Streamlit
    st.plotly_chart(fig_barras, use_container_width=True)