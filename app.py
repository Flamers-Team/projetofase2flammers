
import streamlit as st
import pandas as pd
import time

from src.populacao import gerar_populacao_inicial
from src.aptidao import avaliar_fitness, decodificar_rotas
from src.selecao import selecionar_progenitores
from src.cruzamento import crossover_ordem_ox
from src.mutacao import aplicar_mutacao_troca
from src.substituicao import ordenar_e_aplicar_elitismo
from src.metricas import avaliar_eficiencia_geracao

st.set_page_config(page_title="Otimizador GA - VRP", layout="wide")
st.title("🧬 Painel Modular do Algoritmo Genético (VRP - DF)")


# 1. Configuração de Parâmetros na Barra Lateral
st.sidebar.header("🎛️ Parâmetros das Etapas")
TAM_POPULACAO = st.sidebar.slider("Tamanho da População (populacao.py)", 10, 200, 50)
GERACOES = st.sidebar.slider("Número de Gerações", 10, 300, 100)
PROB_MUTACAO = st.sidebar.slider("Taxa de Mutação (mutacao.py)", 0.0, 1.0, 0.2, step=0.05)
NUM_VEICULOS = st.sidebar.number_input("Frota de Veículos", 1, 10, 4)
CAPACIDADE_MAX = st.sidebar.number_input("Capacidade de Carga (Caixas)", 10, 100, 30)


# Carrega a base local
try:
    df_pacientes = pd.read_csv('pacientes_df.csv')
    # Limitando para teste rápido na interface se necessário
    df_filtrado = df_pacientes.head(30) 
except FileNotFoundError:
    st.error("❌ Arquivo 'pacientes_df.csv' não encontrado. Rode o gerador primeiro!")
    st.stop()

if st.button("🚀 Iniciar Ciclo Evolutivo Interativo"):
    # Estado inicial
    COORD_HOSPITAL = (-15.7984, -47.8864) # Hospital de Base
    num_pacientes = len(df_filtrado)
    
    # ETAPA 1: População Inicial
    populacao = gerar_populacao_inicial(num_pacientes, TAM_POPULACAO)
    
    historico_melhor_fitness = []
    
    # Containers para atualização em tempo real na tela
    grafico_placeholder = st.empty()
    status_placeholder = st.empty()
    
    for g in range(GERACOES):
        # ETAPA 2: Aptidão
        fitness_valores = [avaliar_fitness(ind, df_filtrado, COORD_HOSPITAL, NUM_VEICULOS, CAPACIDADE_MAX) for ind in populacao]
        
        # ETAPA 6: Ordenação e Elitismo para Substituição
        populacao, fitness_valores, melhor_individuo = ordenar_e_aplicar_elitismo(populacao, fitness_valores)
        
        # ETAPA 7: Coleta de Métricas da geração corrente
        metricas_gen = avaliar_eficiencia_geracao(fitness_valores)
        historico_melhor_fitness.append(metricas_gen["melhor_score"])
        
        # Atualiza o gráfico de linha interativo do Streamlit na tela
        df_plot = pd.DataFrame({"Score de Fitness (Menor é melhor)": historico_melhor_fitness})
        grafico_placeholder.line_chart(df_plot)
        status_placeholder.metric(label=f"Geração {g+1}/{GERACOES}", value=f"{metricas_gen['melhor_score']:.2f} km/score")
        
        # Monta a nova geração estruturando o ciclo evolutivo completo
        nova_geracao = [melhor_individuo] # Mantém o melhor pelo Elitismo
        
        while len(nova_geracao) < TAM_POPULACAO:
            # ETAPA 3: Seleção
            p1, p2 = selecionar_progenitores(populacao, fitness_valores)
            # ETAPA 4: Cruzamento
            filho = crossover_ordem_ox(p1, p2)
            # ETAPA 5: Mutação
            filho = aplicar_mutacao_troca(filho, PROB_MUTACAO)
            
            nova_geracao.append(filho)
            
        populacao = nova_geracao
        time.sleep(0.01) # Pequena pausa para os olhos humanos acompanharem a convergência gráfica
        
    st.success("🏆 Otimização concluída com sucesso! Verifique as curvas de convergência acima.")