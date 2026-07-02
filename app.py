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

# --- CONFIGURAÇÃO DO AMBIENTE DEV ---
st.set_page_config(page_title="DevTools - AG VRP", layout="wide", initial_sidebar_state="expanded")

HOSPITAIS = {
    "Hospital de Base (Plano Piloto)": (-15.7984, -47.8864),
    "Hospital Regional de Ceilândia": (-15.8239, -48.1152),
    "Hospital Regional de Taguatinga": (-15.8335, -48.0628)
}

# ==========================================
# CARREGAMENTO DO DATASET DINÂMICO
# ==========================================
@st.cache_data
def carregar_dataset_completo():
    try:
        # Carrega o banco de dados completo sem limitar com .head()
        df = pd.read_csv('data/pacientes_df.csv')
        return df
    except FileNotFoundError:
        st.error("Arquivo 'pacientes_df.csv' não encontrado. Verifique a pasta data.")
        st.stop()

df_banco_completo = carregar_dataset_completo()
TOTAL_PACIENTES_BANCO = len(df_banco_completo)

if 'resultado_ag' not in st.session_state:
    st.session_state.resultado_ag = None

# ==========================================
# CÁLCULO DA RÉGUA TEÓRICA (LOWER BOUND)
# ==========================================
def calcular_lower_bound_teorico(df: pd.DataFrame, coord_hospital: tuple, capacidade: int) -> float:
    """
    Calcula o Limite Inferior Contínuo para o VRP (Aproximação matemática otimista).
    Fórmula: Soma de (2 * Distância Direta * (Demanda do Paciente / Capacidade do Veículo))
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
# BARRA LATERAL: CALIBRADOR DE CENÁRIOS
# ==========================================
st.sidebar.markdown("### ⚙️ Painel de Engenharia")

# 1. Trava de Reprodutibilidade
SEMENTE_FIXA = st.sidebar.checkbox("Fixar Semente Aleatória (Reprodutibilidade)", value=True)
VALOR_SEMENTE = 42 if SEMENTE_FIXA else None

hospital_selecionado = st.sidebar.selectbox("Depósito (Ponto de Partida)", list(HOSPITAIS.keys()))
COORD_HOSPITAL = HOSPITAIS[hospital_selecionado]

# 2. Controle do Volume da Operação
st.sidebar.markdown("#### Demanda Diária")

# Trava de segurança: O teto é 250, ou o total do banco (o que for menor)
LIMITE_MAXIMO_PACIENTES = min(250, TOTAL_PACIENTES_BANCO)

VOLUME_PACIENTES = st.sidebar.slider(
    "Quantidade de Entregas (Pacientes)", 
    min_value=5, 
    max_value=LIMITE_MAXIMO_PACIENTES, 
    value=35,
    step=1
)

# Filtra o DataFrame dinamicamente com base no slider
df_pacientes = df_banco_completo.head(VOLUME_PACIENTES)

# 3. Configuração da Frota
st.sidebar.markdown("#### Restrições VRP (Frota)")
NUM_VEICULOS = st.sidebar.slider("Quantidade de Veículos", 1, 10, 4)
CAPACIDADE_MAX = st.sidebar.slider("Capacidade de Carga por Veículo", 10, 100, 30)

# VALIDAÇÃO MATEMÁTICA DE INFRAESTRUTURA
demanda_total_dia = df_pacientes['demanda_caixas'].sum()
capacidade_total_frota = NUM_VEICULOS * CAPACIDADE_MAX

if demanda_total_dia > capacidade_total_frota:
    st.sidebar.error(f"⚠️ Risco de Colapso Logístico! A demanda de hoje é de {demanda_total_dia} caixas, mas a frota suporta no máximo {capacidade_total_frota}. Aumente os veículos ou a capacidade.")

# 4. Motor Evolutivo (IA)
st.sidebar.markdown("#### Motor Evolutivo (IA)")
TAM_POPULACAO = st.sidebar.number_input("Tamanho da População", min_value=10, max_value=500, value=50, step=10)

ESTRATEGIA_INICIAL = st.sidebar.selectbox(
    "Estratégia de População Inicial", 
    options=["100% Aleatória", "Híbrida: Semente Vizinho Mais Próximo"],
    help="Alterne aqui para testar o impacto de inicializar a Geração 0 com uma heurística determinística."
)

# Seleção Dinâmica do Algoritmo de Seleção de Progenitores
metodo_selecao = st.sidebar.selectbox(
    "Método de Seleção dos Pais",
    options=["Roleta Inversa", "Ranking"],
    help="Roleta Inversa usa valores proporcionais ao inverso do fitness. O Ranking avalia a posição relativa dos indivíduos."
)

GERACOES_MAX = st.sidebar.number_input("Gerações Máximas", min_value=10, max_value=1000, value=200, step=10)
PACIENCIA = st.sidebar.slider("Critério de Parada (Paciência)", 5, 100, 30)
PROB_MUTACAO = st.sidebar.slider("Taxa de Mutação", 0.0, 1.0, 0.25, step=0.05)

# ==========================================
# NÚCLEO DE EXECUÇÃO
# ==========================================
def executar_teste_benchmark():
    # Trava a semente (Seed) se o usuário marcou a caixa
    if VALOR_SEMENTE is not None:
        random.seed(VALOR_SEMENTE)
        np.random.seed(VALOR_SEMENTE)
        
    num_pacientes = len(df_pacientes)
    tempo_inicio = time.time() 
    
    # Inicia a população com a estratégia escolhida
    populacao = gerar_populacao_inicial(
        num_pacientes=num_pacientes, 
        tam_populacao=TAM_POPULACAO,
        estrategia=ESTRATEGIA_INICIAL,
        df_pacientes=df_pacientes,
        coord_hospital=COORD_HOSPITAL
    )
    
    historico = []
    melhor_score_global = float('inf')
    geracoes_estagnadas = 0
    geracao_final = 0
    
    progresso = st.progress(0)
    
    for g in range(GERACOES_MAX):
        # NOTA: Se você adicionou o parâmetro de autonomia no aptidao.py, passe-o aqui também.
        fitness_valores = [avaliar_fitness(ind, df_pacientes, COORD_HOSPITAL, NUM_VEICULOS, CAPACIDADE_MAX) for ind in populacao]
        populacao, fitness_valores, melhor_ind = ordenar_e_aplicar_elitismo(populacao, fitness_valores)
        
        score_atual = fitness_valores[0]
        historico.append(score_atual)
        
        # Auditoria de Avanço (Early Stopping)
        if score_atual < melhor_score_global - 0.01:
            melhor_score_global = score_atual
            geracoes_estagnadas = 0
        else:
            geracoes_estagnadas += 1
            
        progresso.progress((g + 1) / GERACOES_MAX)
        
        # Interrompe se atingir o limite de paciência
        if geracoes_estagnadas >= PACIENCIA:
            geracao_final = g
            st.toast(f"🛑 Convergência alcançada. Early stopping na geração {g}.", icon="📉")
            break
            
        geracao_final = g
        nova_populacao = [melhor_ind]
        
        while len(nova_populacao) < TAM_POPULACAO:
            # Passando o método de seleção definido na interface gráfica para o módulo interno
            p1, p2 = selecionar_progenitores(populacao, fitness_valores, metodo=metodo_selecao)
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

# Botão desabilitado se houver risco de colapso logístico
desabilitar_botao = demanda_total_dia > capacidade_total_frota

if st.sidebar.button("⚙️ Rodar Benchmark do Algoritmo", type="primary", use_container_width=True, disabled=desabilitar_botao):
    executar_teste_benchmark()

# ==========================================
# ÁREA PRINCIPAL: VISUALIZAÇÃO
# ==========================================
st.title("🔬 DevTools: Validação do Algoritmo Genético")

aba_mapa, aba_metricas, aba_api = st.tabs(["🗺️ Prova Visual", "📈 Análise de Convergência", "💻 Contrato API"])

# --- ABA 1: MAPA ---
with aba_mapa:
    st.markdown("### 🗺️ Visualização Cartográfica das Rotas")
    st.markdown("Verificação de sanidade geométrica: As rotas estão se cruzando muito? O agrupamento (clusterização) por veículo faz sentido geográfico?")
    
    mapa = folium.Map(location=COORD_HOSPITAL, zoom_start=11, tiles="cartodbpositron")
    
    folium.Marker(
        COORD_HOSPITAL, 
        popup="🏥 BASE (Depósito)", 
        tooltip="Ponto de Partida",
        icon=folium.Icon(color="black", icon="building", prefix="fa")
    ).add_to(mapa)
    
    if st.session_state.resultado_ag:
        cores_veiculos = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen']
        rotas_geradas = st.session_state.resultado_ag['rotas']
        
        for id_veiculo, rota in enumerate(rotas_geradas):
            if not rota: 
                continue # Não desenha carros que ficaram vazios na garagem
                
            cor = cores_veiculos[id_veiculo % len(cores_veiculos)]
            coords_trajeto = [COORD_HOSPITAL]
            
            for idx_paciente in rota:
                paciente = df_pacientes.iloc[idx_paciente]
                coord_paciente = (paciente['latitude'], paciente['longitude'])
                coords_trajeto.append(coord_paciente)
                
                icone_dict = {
                    'violencia_domestica': 'user-secret',
                    'medicamento_hormonal': 'snowflake-o',
                    'pos_parto': 'child',
                    'atencao_basica': 'stethoscope'
                }
                icone_selecionado = icone_dict.get(paciente['tipo_atendimento'], 'info-circle')
                
                folium.Marker(
                    location=coord_paciente,
                    popup=f"<b>{paciente['tipo_atendimento'].replace('_', ' ').title()}</b><br>Prioridade: {paciente['prioridade']}",
                    tooltip=f"Paciente #{paciente['id_paciente']} (Veículo {id_veiculo + 1})",
                    icon=folium.Icon(color=cor, icon=icone_selecionado, prefix="fa")
                ).add_to(mapa)
                
            coords_trajeto.append(COORD_HOSPITAL)
            
            folium.PolyLine(
                coords_trajeto, 
                color=cor, 
                weight=4, 
                opacity=0.8,
                tooltip=f"Rota do Veículo {id_veiculo + 1}"
            ).add_to(mapa)
            
    else:
        st.info("👆 Ajuste os parâmetros na barra lateral e clique em 'Rodar Benchmark' para traçar as rotas.")
        
    st_folium(mapa, width=1000, height=600, returned_objects=[])

# --- ABA 2: MÉTRICAS E AUDITORIA ---
with aba_metricas:
    if st.session_state.resultado_ag:
        res = st.session_state.resultado_ag
        
        score_ag = res['score_fitness_final']
        lb = res['lower_bound_teorico']
        gap_percentual = ((score_ag - lb) / lb) * 100 if lb > 0 else 0
        
        st.info("ℹ️ **Régua de Negócios:** A eficiência atual compara o custo gerado pela Inteligência Artificial contra a heurística do **Vizinho Mais Próximo (Nearest Neighbor)**, que simula o comportamento da operação humana padrão, e o **Lower Bound (Limiar Teórico)**.")
        
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
        st.markdown("Este é o modelo do Response Payload (JSON) devolvido para a interface.")
        st.json(st.session_state.resultado_ag)