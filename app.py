# app.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import time
import random
import numpy as np

# Importando os módulos estruturados da pasta src
from src.populacao import gerar_populacao_inicial
from src.aptidao import avaliar_fitness, decodificar_rotas, calcular_distancia_haversine
from src.selecao import selecionar_progenitores
from src.cruzamento import crossover_ordem_ox
from src.mutacao import aplicar_mutacao_troca
from src.substituicao import ordenar_e_aplicar_elitismo
from src.metricas import avaliar_eficiencia_geracao

st.set_page_config(page_title="DevTools - AG VRP", layout="wide", initial_sidebar_state="expanded")

HOSPITAIS = {
    "Hospital de Base (Plano Piloto)": (-15.7984, -47.8864),
    "Hospital Regional de Ceilândia": (-15.8239, -48.1152),
    "Hospital Regional de Taguatinga": (-15.8335, -48.0628)
}

@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv('data/pacientes_df.csv')
        return df.head(35) 
    except FileNotFoundError:
        st.error("Arquivo 'pacientes_df.csv' não encontrado.")
        st.stop()

df_pacientes = carregar_dados()

if 'resultado_ag' not in st.session_state:
    st.session_state.resultado_ag = None

# ==========================================
# CÁLCULO DA RÉGUA TEÓRICA (LOWER BOUND)
# ==========================================
def calcular_lower_bound_teorico(df: pd.DataFrame, coord_hospital: tuple, capacidade: int) -> float:
    """
    Calcula o Limite Inferior Contínuo para o VRP (Aproximação matemática otimista).
    Fórmula: Soma de (2 * Distância Direta * (Demanda do Paciente / Capacidade do Veículo))
    Isso assume que o caminhão é divisível em frações perfeitas e voa em linha reta.
    """
    lb_total = 0.0
    for _, paciente in df.iterrows():
        dist_direta = calcular_distancia_haversine(
            coord_hospital[0], coord_hospital[1], 
            paciente['latitude'], paciente['longitude']
        )
        fracao_carga = paciente['demanda_caixas'] / float(capacidade)
        lb_total += (2 * dist_direta * fracao_carga)
    return round(lb_total, 2)

# ==========================================
# BARRA LATERAL
# ==========================================
st.sidebar.markdown("### ⚙️ Painel de Engenharia")

# A Trava de Reprodutibilidade (Visual Limpo)
SEMENTE_FIXA = st.sidebar.checkbox("Fixar Semente Aleatória (Reprodutibilidade)", value=True)

# Lógica Oculta: Usa 42 se marcado, ou None se desmarcado.
VALOR_SEMENTE = 42 if SEMENTE_FIXA else None

hospital_selecionado = st.sidebar.selectbox("Depósito", list(HOSPITAIS.keys()))
COORD_HOSPITAL = HOSPITAIS[hospital_selecionado]

st.sidebar.markdown("#### Motor Evolutivo")
TAM_POPULACAO = st.sidebar.number_input("Tamanho da População", min_value=10, max_value=500, value=50, step=10)
GERACOES_MAX = st.sidebar.number_input("Gerações Máximas", min_value=10, max_value=1000, value=200, step=10)
PACIENCIA = st.sidebar.slider("Critério de Parada (Paciência)", 5, 100, 30, help="Para a execução se o custo não melhorar em X gerações seguidas.")
PROB_MUTACAO = st.sidebar.slider("Taxa de Mutação", 0.0, 1.0, 0.25, step=0.05)

st.sidebar.markdown("#### Restrições VRP")
NUM_VEICULOS = st.sidebar.slider("Quantidade de Veículos", 1, 10, 4)
CAPACIDADE_MAX = st.sidebar.slider("Capacidade de Carga", 10, 100, 30)

# ==========================================
# NÚCLEO DE EXECUÇÃO
# ==========================================
def executar_teste_benchmark():
    if VALOR_SEMENTE is not None:
        random.seed(VALOR_SEMENTE)
        np.random.seed(VALOR_SEMENTE)
        
    num_pacientes = len(df_pacientes)
    tempo_inicio = time.time() 
    
    populacao = gerar_populacao_inicial(num_pacientes, TAM_POPULACAO)
    historico = []
    
    melhor_score_global = float('inf')
    geracoes_estagnadas = 0
    geracao_final = 0
    
    progresso = st.progress(0)
    
    for g in range(GERACOES_MAX):
        # Certifique-se de que sua função avaliar_fitness suporta a AUTONOMIA_MAX_KM (ex: 120.0) se você a adicionou!
        # Se não, remova o 120.0 do final da chamada abaixo.
        fitness_valores = [avaliar_fitness(ind, df_pacientes, COORD_HOSPITAL, NUM_VEICULOS, CAPACIDADE_MAX) for ind in populacao]
        populacao, fitness_valores, melhor_ind = ordenar_e_aplicar_elitismo(populacao, fitness_valores)
        
        score_atual = fitness_valores[0]
        historico.append(score_atual)
        
        if score_atual < melhor_score_global - 0.01:
            melhor_score_global = score_atual
            geracoes_estagnadas = 0
        else:
            geracoes_estagnadas += 1
            
        progresso.progress((g + 1) / GERACOES_MAX)
        
        if geracoes_estagnadas >= PACIENCIA:
            geracao_final = g
            st.toast(f"🛑 Convergência alcançada. Early stopping na geração {g}.", icon="📉")
            break
            
        geracao_final = g
        nova_populacao = [melhor_ind]
        while len(nova_populacao) < TAM_POPULACAO:
            p1, p2 = selecionar_progenitores(populacao, fitness_valores)
            filho = crossover_ordem_ox(p1, p2)
            filho = aplicar_mutacao_troca(filho, PROB_MUTACAO)
            nova_populacao.append(filho)
            
        populacao = nova_populacao
        
    tempo_fim = time.time() 
    rotas_finais = decodificar_rotas(populacao[0], df_pacientes, NUM_VEICULOS, CAPACIDADE_MAX)
    
    lb_teorico = calcular_lower_bound_teorico(df_pacientes, COORD_HOSPITAL, CAPACIDADE_MAX)
    
    st.session_state.resultado_ag = {
        "metadados": {
            "tempo_processamento_segundos": round(tempo_fim - tempo_inicio, 3),
            "geracoes_processadas": geracao_final + 1,
            "motivo_parada": "Early Stopping" if geracoes_estagnadas >= PACIENCIA else "Limite de Gerações",
            "pacientes_atendidos": num_pacientes
        },
        "score_fitness_final": round(melhor_score_global, 2),
        "lower_bound_teorico": lb_teorico,
        "historico_convergencia": historico,
        "rotas": rotas_finais
    }

if st.sidebar.button("⚙️ Rodar Benchmark do Algoritmo", type="primary", use_container_width=True):
    executar_teste_benchmark()

# ==========================================
# ÁREA PRINCIPAL: VISUALIZAÇÃO
# ==========================================
st.title("🔬 Validação do Algoritmo Genético")

aba_mapa, aba_metricas, aba_api = st.tabs(["🗺️ MAPA das Rotas", "📈 Análise de Convergência", "💻  API"])

# --- ABA 1: MAPA ---
with aba_mapa:
    st.markdown("### 🗺️ Visualização Cartográfica das Rotas")
    st.markdown("Verificação de sanidade geométrica: As rotas estão se cruzando muito? O agrupamento (clusterização) por veículo faz sentido geográfico?")
    
    # 1. Inicializa o mapa centralizado no Hospital (usando um fundo claro para destacar as linhas)
    mapa = folium.Map(location=COORD_HOSPITAL, zoom_start=11, tiles="cartodbpositron")
    
    # 2. Marca o Hospital (Depósito Central)
    folium.Marker(
        COORD_HOSPITAL, 
        popup="🏥 BASE (Depósito)", 
        tooltip="Ponto de Partida",
        icon=folium.Icon(color="black", icon="building", prefix="fa")
    ).add_to(mapa)
    
    # 3. Se houver rotas processadas, desenha no mapa
    if st.session_state.resultado_ag:
        # Paleta de cores para diferenciar os veículos visualmente
        cores_veiculos = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen']
        
        rotas_geradas = st.session_state.resultado_ag['rotas']
        
        for id_veiculo, rota in enumerate(rotas_geradas):
            if not rota: # Pula os veículos que ficaram vazios/ociosos na garagem
                continue
                
            cor = cores_veiculos[id_veiculo % len(cores_veiculos)]
            coords_trajeto = [COORD_HOSPITAL] # O trajeto de cada carro sempre começa no hospital
            
            for idx_paciente in rota:
                # Busca os dados daquele paciente específico no DataFrame
                paciente = df_pacientes.iloc[idx_paciente]
                coord_paciente = (paciente['latitude'], paciente['longitude'])
                coords_trajeto.append(coord_paciente)
                
                # Dicionário visual para auditoria das categorias
                icone_dict = {
                    'violencia_domestica': 'user-secret',
                    'medicamento_hormonal': 'snowflake-o',
                    'pos_parto': 'child',
                    'atencao_basica': 'stethoscope'
                }
                icone_selecionado = icone_dict.get(paciente['tipo_atendimento'], 'info-circle')
                
                # Plota a casa do paciente no mapa
                folium.Marker(
                    location=coord_paciente,
                    popup=f"<b>{paciente['tipo_atendimento'].replace('_', ' ').title()}</b><br>Prioridade: {paciente['prioridade']}",
                    tooltip=f"Paciente #{paciente['id_paciente']}",
                    icon=folium.Icon(color=cor, icon=icone_selecionado, prefix="fa")
                ).add_to(mapa)
                
            # O trajeto sempre termina voltando ao hospital
            coords_trajeto.append(COORD_HOSPITAL)
            
            # Desenha a linha conectando todos os pontos do veículo
            folium.PolyLine(
                coords_trajeto, 
                color=cor, 
                weight=4, 
                opacity=0.8,
                tooltip=f"Rota do Veículo {id_veiculo + 1}"
            ).add_to(mapa)
            
    else:
        st.info("👆 Ajuste os parâmetros na barra lateral e clique em 'Rodar Benchmark' para traçar as rotas.")
        
    # Renderiza o componente Folium dentro da interface do Streamlit
    st_folium(mapa, width=1000, height=600, returned_objects=[])

# --- ABA 2: MÉTRICAS E AUDITORIA ---
with aba_metricas:
    if st.session_state.resultado_ag:
        res = st.session_state.resultado_ag
        
        score_ag = res['score_fitness_final']
        lb = res['lower_bound_teorico']
        gap_percentual = ((score_ag - lb) / lb) * 100 if lb > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Score de Fitness Final (Z)", f"{score_ag}")
        col2.metric("Lower Bound (Teórico)", f"{lb}")
        col3.metric("Eficiência (GAP do Teórico)", f"+{gap_percentual:.1f}%", delta_color="inverse")
        col4.metric("Hardware e CPU", f"{res['metadados']['tempo_processamento_segundos']} s")
        
        st.markdown(f"**Critério de Parada:** {res['metadados']['motivo_parada']} na Geração {res['metadados']['geracoes_processadas']}.")
        
        st.markdown("### Curva de Minimização de Custo")
        st.line_chart(res['historico_convergencia'])
    else:
        st.info("Execute o algoritmo para visualizar as métricas.")

# --- ABA 3: CONTRATO DA API ---
with aba_api:
    if st.session_state.resultado_ag:
        st.json(st.session_state.resultado_ag)