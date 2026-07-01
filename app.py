# app.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import time
import json

# Importando os módulos estruturados da pasta src
from src.populacao import gerar_populacao_inicial
from src.aptidao import avaliar_fitness, decodificar_rotas
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

# --- CARREGAMENTO DO DATASET ATUALIZADO ---
@st.cache_data
def carregar_dados():
    try:
        # Carrega a base que geramos com "atencao_basica"
        df = pd.read_csv('data/pacientes_df.csv')
        return df.head(35) # Limitado a 35 paradas para testes rápidos de convergência
    except FileNotFoundError:
        st.error("Arquivo 'pacientes_df.csv' não encontrado.")
        st.stop()

df_pacientes = carregar_dados()

# Variáveis de sessão para guardar os testes
if 'resultado_ag' not in st.session_state:
    st.session_state.resultado_ag = None

# ==========================================
# BARRA LATERAL: CALIBRADOR DE HIPERPARÂMETROS
# ==========================================
st.sidebar.markdown("### ⚙️ Painel de Engenharia")
st.sidebar.caption("Ajuste os parâmetros para simular o payload da futura API.")

hospital_selecionado = st.sidebar.selectbox("Ponto de Partida (Depósito)", list(HOSPITAIS.keys()))
COORD_HOSPITAL = HOSPITAIS[hospital_selecionado]

st.sidebar.markdown("#### Motor Evolutivo")
TAM_POPULACAO = st.sidebar.number_input("Tamanho da População", min_value=10, max_value=500, value=50, step=10)
GERACOES = st.sidebar.number_input("Número de Gerações", min_value=10, max_value=1000, value=100, step=10)
PROB_MUTACAO = st.sidebar.slider("Taxa de Mutação (Exploração)", 0.0, 1.0, 0.25, step=0.05)

st.sidebar.markdown("#### Restrições VRP (Frota)")
NUM_VEICULOS = st.sidebar.slider("Quantidade de Veículos", 1, 10, 4)
CAPACIDADE_MAX = st.sidebar.slider("Capacidade de Carga (Caixas)", 10, 100, 30)

# ==========================================
# NÚCLEO DE EXECUÇÃO (MOCK DA API)
# ==========================================
def executar_teste_benchmark():
    num_pacientes = len(df_pacientes)
    tempo_inicio = time.time() # Inicia o cronômetro de hardware
    
    populacao = gerar_populacao_inicial(num_pacientes, TAM_POPULACAO)
    historico = []
    
    progresso = st.progress(0)
    
    for g in range(GERACOES):
        fitness_valores = [avaliar_fitness(ind, df_pacientes, COORD_HOSPITAL, NUM_VEICULOS, CAPACIDADE_MAX) for ind in populacao]
        populacao, fitness_valores, melhor_ind = ordenar_e_aplicar_elitismo(populacao, fitness_valores)
        
        metricas = avaliar_eficiencia_geracao(fitness_valores)
        historico.append(metricas["melhor_score"])
        progresso.progress((g + 1) / GERACOES)
        
        nova_populacao = [melhor_ind]
        while len(nova_populacao) < TAM_POPULACAO:
            p1, p2 = selecionar_progenitores(populacao, fitness_valores)
            filho = crossover_ordem_ox(p1, p2)
            filho = aplicar_mutacao_troca(filho, PROB_MUTACAO)
            nova_populacao.append(filho)
            
        populacao = nova_populacao
        
    tempo_fim = time.time() # Para o cronômetro
    
    rotas_finais = decodificar_rotas(populacao[0], df_pacientes, NUM_VEICULOS, CAPACIDADE_MAX)
    
    # Monta a estrutura que será devolvida pela API para o frontend React
    st.session_state.resultado_ag = {
        "metadados": {
            "tempo_processamento_segundos": round(tempo_fim - tempo_inicio, 3),
            "geracoes_processadas": GERACOES,
            "pacientes_atendidos": num_pacientes
        },
        "score_fitness_final": round(historico[-1], 2),
        "historico_convergencia": historico,
        "rotas": rotas_finais
    }
    
    st.sidebar.success(f"Benchmark finalizado em {st.session_state.resultado_ag['metadados']['tempo_processamento_segundos']}s")

if st.sidebar.button("⚙️ Rodar Benchmark do Algoritmo", type="primary", use_container_width=True):
    executar_teste_benchmark()

# ==========================================
# ÁREA PRINCIPAL: VISUALIZAÇÃO DE DEV
# ==========================================
st.title("🔬 DevTools: Validação do Algoritmo Genético")

# Cria abas organizadas para separar a análise visual da análise de dados
aba_mapa, aba_metricas, aba_api, aba_dados = st.tabs([
    "🗺️ Prova Visual (Mapa)", 
    "📈 Análise de Convergência", 
    "💻 Contrato API (Output JSON)", 
    "🗃️ Dataset (Atenção Básica)"
])

# --- ABA 1: MAPA ---
with aba_mapa:
    st.markdown("Verificação de sanidade geométrica (as rotas estão se cruzando muito? Os veículos estão voltando à base?)")
    mapa = folium.Map(location=COORD_HOSPITAL, zoom_start=11)
    
    folium.Marker(COORD_HOSPITAL, popup="DEPOSITO", icon=folium.Icon(color="black", icon="building", prefix="fa")).add_to(mapa)
    
    if st.session_state.resultado_ag:
        cores = ['red', 'blue', 'green', 'purple', 'orange', 'darkred']
        
        for idx_v, rota in enumerate(st.session_state.resultado_ag['rotas']):
            if not rota: continue
            cor = cores[idx_v % len(cores)]
            coords_rota = [COORD_HOSPITAL]
            
            for idx_p in rota:
                p = df_pacientes.iloc[idx_p]
                coords_rota.append((p['latitude'], p['longitude']))
                
                # Tratamento visual das novas categorias (incluindo atencao_basica)
                icone_dict = {
                    'violencia_domestica': 'user-secret',
                    'medicamento_hormonal': 'snowflake-o',
                    'pos_parto': 'child',
                    'atencao_basica': 'stethoscope'
                }
                icone_selecionado = icone_dict.get(p['tipo_atendimento'], 'info')
                
                folium.Marker(
                    (p['latitude'], p['longitude']),
                    popup=f"[{p['prioridade']}] {p['tipo_atendimento']}",
                    icon=folium.Icon(color=cor, icon=icone_selecionado, prefix="fa")
                ).add_to(mapa)
                
            coords_rota.append(COORD_HOSPITAL)
            folium.PolyLine(coords_rota, color=cor, weight=3, opacity=0.7).add_to(mapa)
            
    st_folium(mapa, width=900, height=500, returned_objects=[])

# --- ABA 2: MÉTRICAS ---
with aba_metricas:
    if st.session_state.resultado_ag:
        col1, col2 = st.columns(2)
        col1.metric("Score de Fitness Final (km + penalidades)", f"{st.session_state.resultado_ag['score_fitness_final']}")
        col2.metric("Tempo de Processamento (Hardware)", f"{st.session_state.resultado_ag['metadados']['tempo_processamento_segundos']} s")
        
        st.markdown("### Curva de Minimização de Custo")
        st.line_chart(st.session_state.resultado_ag['historico_convergencia'])
    else:
        st.info("Execute o algoritmo para visualizar as métricas.")

# --- ABA 3: CONTRATO DA API ---
with aba_api:
    st.markdown("Este é o modelo exato do **Response Payload** que o Python deverá retornar para o React quando o endpoint `/api/v1/otimizar` for chamado.")
    if st.session_state.resultado_ag:
        # Formata os dados numéricos brutos para um JSON legível para os devs do Front
        mock_response = {
            "status": "success",
            "metadata": st.session_state.resultado_ag['metadados'],
            "optimization_result": {
                "total_fitness_score": st.session_state.resultado_ag['score_fitness_final'],
                "fleet_routes": []
            }
        }
        
        # Mapeia as rotas para o JSON
        for id_veiculo, rota in enumerate(st.session_state.resultado_ag['rotas']):
            ids_pacientes = [int(df_pacientes.iloc[i]['id_paciente']) for i in rota]
            mock_response["optimization_result"]["fleet_routes"].append({
                "vehicle_id": id_veiculo + 1,
                "stop_sequence_ids": ids_pacientes,
                "total_stops": len(ids_pacientes)
            })
            
        st.json(mock_response)
    else:
        st.info("Execute o algoritmo para gerar o mock do Payload JSON.")

# --- ABA 4: AUDITORIA DO DATASET ---
with aba_dados:
    st.markdown("Verificação da nova distribuição de categorias:")
    # Mostra a porcentagem de cada categoria para provar que a atenção básica está lá
    st.dataframe(df_pacientes['tipo_atendimento'].value_counts(normalize=True) * 100)
    st.dataframe(df_pacientes)