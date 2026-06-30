# app.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import time
import random

# Importando os módulos estruturados da sua pasta src
from src.populacao import gerar_populacao_inicial
from src.aptidao import avaliar_fitness, decodificar_rotas
from src.selecao import selecionar_progenitores
from src.cruzamento import crossover_ordem_ox
from src.mutacao import aplicar_mutacao_troca
from src.substituicao import ordenar_e_aplicar_elitismo
from src.metricas import avaliar_eficiencia_geracao

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Roteirizador de Saúde da Mulher", layout="wide")

# 2. BANCO DE DADOS DE HOSPITAIS (Coordenadas Reais do DF)
HOSPITAIS = {
    "Hospital de Base (Plano Piloto)": (-15.7984, -47.8864),
    "Hospital Regional de Ceilândia": (-15.8239, -48.1152),
    "Hospital Regional de Taguatinga": (-15.8335, -48.0628),
    "Hospital Regional de Samambaia": (-15.8741, -48.0848)
}

# 3. INICIALIZAÇÃO DA MEMÓRIA DO APP (st.session_state)
if 'df_ativo' not in st.session_state:
    try:
        # Carrega os seus pacientes do arquivo CSV
        st.session_state.df_ativo = pd.read_csv('data/pacientes_df.csv').head(25) # Limitado a 25 para visualização limpa
    except FileNotFoundError:
        st.error("❌ Arquivo 'pacientes_df.csv' não encontrado na raiz do projeto. Garanta que ele exista!")
        st.stop()

if 'rotas_finais' not in st.session_state:
    st.session_state.rotas_finais = None
if 'historico_convergencia' not in st.session_state:
    st.session_state.historico_convergencia = []

# --- BARRA LATERAL: PARÂMETROS E CONTROLES ---
st.sidebar.header("🏥 Configuração da Central")
hospital_selecionado = st.sidebar.selectbox("Selecione o Hospital de Origem (Depósito):", list(HOSPITAIS.keys()))
COORD_HOSPITAL = HOSPITAIS[hospital_selecionado]

st.sidebar.markdown("---")
st.sidebar.header("🧬 Parâmetros do Algoritmo Genético")
TAM_POPULACAO = st.sidebar.slider("Tamanho da População", 20, 150, 50)
GERACOES = st.sidebar.slider("Número de Gerações", 10, 200, 80)
PROB_MUTACAO = st.sidebar.slider("Taxa de Mutação", 0.0, 1.0, 0.2, step=0.05)
NUM_VEICULOS = st.sidebar.number_input("Veículos na Frota", 1, 6, 3)
CAPACIDADE_MAX = st.sidebar.number_input("Capacidade por Veículo (Caixas)", 10, 50, 25)

# --- FUNÇÃO PARA EXECUTAR O ALGORITMO GENÉTICO ---
def executar_otimizacao():
    df = st.session_state.df_ativo
    num_pacientes = len(df)
    
    # Etapa 1: População Inicial (módulo populacao.py)
    populacao = gerar_populacao_inicial(num_pacientes, TAM_POPULACAO)
    
    historico = []
    progresso_barra = st.progress(0)
    status_texto = st.empty()
    
    for g in range(GERACOES):
        # Etapa 2: Avaliação de Aptidão (módulo aptidao.py)
        fitness_valores = [avaliar_fitness(ind, df, COORD_HOSPITAL, NUM_VEICULOS, CAPACIDADE_MAX) for ind in populacao]
        
        # Etapa 6: Ordenação e Elitismo (módulo substituicao.py)
        populacao, fitness_valores, melhor_individuo = ordenar_e_aplicar_elitismo(populacao, fitness_valores)
        
        # Etapa 7: Coleta de Métricas (módulo metricas.py)
        metricas = avaliar_eficiencia_geracao(fitness_valores)
        historico.append(metricas["melhor_score"])
        
        # Atualiza o progresso visual na tela
        progresso_barra.progress((g + 1) / GERACOES)
        status_texto.text(f"Evoluindo Geração {g+1}/{GERACOES} | Melhor Rota Atual: {metricas['melhor_score']:.2f} km/custo")
        
        # Montagem da nova população seguindo o fluxo de reprodução
        nova_populacao = [melhor_individuo]
        while len(nova_populacao) < TAM_POPULACAO:
            # Etapa 3: Seleção (módulo selecao.py)
            p1, p2 = selecionar_progenitores(populacao, fitness_valores)
            # Etapa 4: Cruzamento (módulo cruzamento.py)
            filho = crossover_ordem_ox(p1, p2)
            # Etapa 5: Mutação (módulo mutacao.py)
            filho = aplicar_mutacao_troca(filho, PROB_MUTACAO)
            nova_populacao.append(filho)
            
        populacao = nova_populacao
    
    # Salva os resultados finais na memória da sessão
    st.session_state.rotas_finais = decodificar_rotas(populacao[0], df, NUM_VEICULOS, CAPACIDADE_MAX)
    st.session_state.historico_convergencia = historico
    st.success("🏁 Rotas Otimizadas com Sucesso!")

# Botão para disparar o cálculo inicial do dia
if st.sidebar.button("🚀 Calcular Rotas Otimizadas", use_container_width=True):
    executar_otimizacao()

# --- BOTÃO DE URGÊNCIA EM TEMPO REAL ---
st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Eventos Adversos")
if st.sidebar.button("🚨 Adicionar Emergência Obstétrica", type="primary", use_container_width=True):    # Simula a chegada repentina de uma paciente grave em Ceilândia ou Taguatinga
    nova_urgencia = {
        'id_paciente': len(st.session_state.df_ativo) + 1,
        'nome_ficticio': f"URGÊNCIA_PACIENTE_{random.randint(100, 999)}",
        'regiao_administrativa': random.choice(['Ceilândia', 'Taguatinga']),
        'latitude': COORD_HOSPITAL[0] + random.uniform(-0.04, 0.04),
        'longitude': COORD_HOSPITAL[1] + random.uniform(-0.04, 0.04),
        'tipo_atendimento': 'emergencia_obstetrica',
        'prioridade': 1, # Máxima urgência médica
        'janela_inicio': 0,
        'janela_fim': 24,
        'demanda_caixas': 1,
        'temperatura_controlada': False,
        'status': 'pendente'
    }
    # Insere a nova paciente no banco de dados ativo e força o recálculo do AG imediatamente
    st.session_state.df_ativo = pd.concat([st.session_state.df_ativo, pd.DataFrame([nova_urgencia])], ignore_index=True)
    st.sidebar.warning("🚨 Nova emergência detectada! Recalculando frotas...")
    executar_otimizacao()

# --- ÁREA PRINCIPAL DA INTERFACE VISUAL ---
st.title("🏥 Sistema de Otimização de Rotas: Saúde da Mulher (DF)")

col_esquerda, col_direita = st.columns([2, 1])

with col_esquerda:
    st.subheader("🗺️ Mapa de Roteirização da Frota")
    
    # Inicializa o mapa centralizado nas coordenadas do hospital escolhido
    mapa = folium.Map(location=COORD_HOSPITAL, zoom_start=12)
    
    # Desenha o Hospital de Origem (Marcador Grande e Azul)
    folium.Marker(
        COORD_HOSPITAL, 
        popup=f"<b>BASE: {hospital_selecionado}</b>", 
        icon=folium.Icon(color="blue", icon="home", prefix="fa")
    ).add_to(mapa)
    
    df_atual = st.session_state.df_ativo
    
    # Cores fixas para identificar cada veículo/rota na tela
    cores_veiculos = ['red', 'purple', 'orange', 'green', 'cadetblue', 'darkred']
    
    # Se as rotas já tiverem sido calculadas, desenha os caminhos na tela
    if st.session_state.rotas_finais is not None:
        for idx_veiculo, rota in enumerate(st.session_state.rotas_finais):
            if not rota: continue
            
            cor = cores_veiculos[idx_veiculo % len(cores_veiculos)]
            coordenadas_caminho = [COORD_HOSPITAL] # Inicia no Hospital
            
            for ordem, idx_paciente in enumerate(rota):
                paciente = df_atual.iloc[idx_paciente]
                pos_paciente = (paciente['latitude'], paciente['longitude'])
                coordenadas_caminho.append(pos_paciente)
                
                # Ícones baseados na gravidade da situação médica da mulher
                icone = "exclamation-triangle" if paciente['prioridade'] == 1 else "medkit"
                
                folium.Marker(
                    pos_paciente,
                    popup=f"Veículo {idx_veiculo+1} | Parada {ordem+1}<br><b>{paciente['nome_ficticio']}</b><br>{paciente['tipo_atendimento'].upper()}",
                    icon=folium.Icon(color=cor, icon=icone, prefix="fa")
                ).add_to(mapa)
                
            coordenadas_caminho.append(COORD_HOSPITAL) # Obriga a fechar o circuito de volta ao Hospital
            
            # Desenha a linha conectando os pontos da rota desse veículo específico
            folium.PolyLine(coordenadas_caminho, color=cor, weight=4, opacity=0.8).add_to(mapa)
    else:
        # Se ainda não calculou as rotas, apenas plota os pacientes soltos como pontos pendentes
        for _, paciente in df_atual.iterrows():
            folium.CircleMarker(
                location=(paciente['latitude'], paciente['longitude']),
                radius=6,
                color="gray",
                fill=True,
                popup=f"{paciente['nome_ficticio']} (Aguardando Rota)"
            ).add_to(mapa)
            
    # Renderiza o mapa Folium dentro da interface do Streamlit
    st_folium(mapa, width=800, height=500, returned_objects=[])

with col_direita:
    st.subheader("📈 Curva de Convergência (Eficácia)")
    if st.session_state.historico_convergencia:
        # Plota o gráfico de linha provando a evolução matemática da taxa de aptidão
        st.line_chart(st.session_state.historico_convergencia)
        st.caption("O gráfico decrescente comprova que o Algoritmo Genético está eliminando caminhos ruins e reduzindo a distância total da frota.")
    else:
        st.info("Clique em 'Calcular Rotas Otimizadas' para iniciar a evolução e visualizar o gráfico de convergência.")

# --- SEÇÃO INFERIOR: PAINEL DE DADOS ---
st.markdown("---")
st.subheader("📋 Lista de Atendimentos Agendados para o Dia")
st.dataframe(st.session_state.df_ativo[['id_paciente', 'nome_ficticio', 'regiao_administrativa', 'tipo_atendimento', 'prioridade', 'demanda_caixas', 'status']], use_container_width=True)