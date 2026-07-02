import pandas as pd
import time
import random
import numpy as np

# Importando rigorosamente os módulos separados que já construímos!
from src.populacao import gerar_populacao_inicial
from src.aptidao import avaliar_fitness, decodificar_rotas, calcular_distancia_haversine
from src.selecao import selecionar_progenitores
from src.cruzamento import crossover_ordem_ox
from src.mutacao import aplicar_mutacao_troca
from src.substituicao import ordenar_e_aplicar_elitismo

def calcular_lower_bound_teorico(df: pd.DataFrame, coord_hospital: tuple, capacidade: int) -> float:
    """
    Função Auxiliar da API: Calcula o custo base impossível (Lower Bound).
    Necessário para o frontend exibir a Eficiência (GAP) da rota gerada.
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

def rodar_otimizacao_api(
    lista_pacientes: list,          
    coordenadas_hospital: tuple,    
    config_algoritmo: dict,         
    config_frota: dict              
) -> dict:
    """
    Controlador da API: Orquestra a evolução, impõe limites de infraestrutura (Early Stopping),
    garante a idempotência (Seed) e devolve um payload de observabilidade rico.
    """
    
    # 0. Trava de Determinismo (Idempotência da API)
    semente = config_algoritmo.get("semente", 42)
    if semente is not None:
        random.seed(semente)
        np.random.seed(semente)
    
    # 1. Parsing dos Dados
    df_pacientes = pd.DataFrame(lista_pacientes)
    num_pacientes = len(df_pacientes)
    
    # 2. Extração Segura de Parâmetros (com valores default de fallback)
    TAM_POP = config_algoritmo.get("populacao", 50)
    GERACOES_MAX = config_algoritmo.get("geracoes", 200)
    PACIENCIA = config_algoritmo.get("paciencia", 30) 
    PROB_MUTACAO = config_algoritmo.get("mutacao", 0.25)
    
    # NOVO: Recebendo os novos parâmetros do frontend
    ESTRATEGIA_INICIAL = config_algoritmo.get("estrategia_inicial", "100% Aleatória")
    METODO_SELECAO = config_algoritmo.get("metodo_selecao", "Roleta Inversa")
    
    NUM_VEICULOS = config_frota.get("veiculos", 4)
    CAPACIDADE = config_frota.get("capacidade", 30)
    AUTONOMIA = config_frota.get("autonomia_km", 120.0)

    tempo_inicio = time.time()
    
    # 3. Inicialização (NOVO: Passando todos os parâmetros exigidos pela função híbrida)
    populacao = gerar_populacao_inicial(
        num_pacientes=num_pacientes, 
        tam_populacao=TAM_POP,
        estrategia=ESTRATEGIA_INICIAL,
        df_pacientes=df_pacientes,
        coord_hospital=coordenadas_hospital
    )
    
    melhor_score_global = float('inf')
    geracoes_estagnadas = 0
    geracao_final = 0
    
    # 4. O Ciclo Evolutivo com Proteção de Servidor (Early Stopping)
    for g in range(GERACOES_MAX):
        fitness_valores = [avaliar_fitness(ind, df_pacientes, coordenadas_hospital, NUM_VEICULOS, CAPACIDADE, AUTONOMIA) for ind in populacao]
        
        populacao, fitness_valores, melhor_ind = ordenar_e_aplicar_elitismo(populacao, fitness_valores)
        score_atual = fitness_valores[0]
        
        # Auditoria de Avanço
        if score_atual < melhor_score_global - 0.01:
            melhor_score_global = score_atual
            geracoes_estagnadas = 0
        else:
            geracoes_estagnadas += 1
            
        # Fusível de parada
        if geracoes_estagnadas >= PACIENCIA:
            geracao_final = g
            break
            
        geracao_final = g
        nova_populacao = [melhor_ind]
        
        # Reprodução
        while len(nova_populacao) < TAM_POP:
            # NOVO: Passando o método de seleção (Roleta vs Ranking)
            p1, p2 = selecionar_progenitores(populacao, fitness_valores, metodo=METODO_SELECAO)
            filho = crossover_ordem_ox(p1, p2)
            filho = aplicar_mutacao_troca(filho, PROB_MUTACAO)
            nova_populacao.append(filho)
            
        populacao = nova_populacao
        
    tempo_fim = time.time()
    
    # 5. Fechamento e Auditoria de Qualidade
    rotas_finais = decodificar_rotas(populacao[0], df_pacientes, NUM_VEICULOS, CAPACIDADE)
    lb_teorico = calcular_lower_bound_teorico(df_pacientes, coordenadas_hospital, CAPACIDADE)
    gap_eficiencia = round(((melhor_score_global - lb_teorico) / lb_teorico) * 100, 2) if lb_teorico > 0 else 0.0
    
    # 6. Contrato de Resposta (JSON Payload para o React/Flutter)
    return {
        "status": "sucesso",
        "metadados_processamento": {
            "tempo_segundos": round(tempo_fim - tempo_inicio, 3),
            "geracoes_processadas": geracao_final + 1,
            "motivo_parada": "estagnacao_early_stopping" if geracoes_estagnadas >= PACIENCIA else "limite_geracoes_atingido",
            "semente_utilizada": semente,
            "estrategia_inicial_usada": ESTRATEGIA_INICIAL,
            "metodo_selecao_usado": METODO_SELECAO
        },
        "auditoria_qualidade": {
            "score_fitness_final": round(melhor_score_global, 2),
            "lower_bound_teorico": lb_teorico,
            "gap_ineficiencia_percentual": gap_eficiencia
        },
        "frota": {
            "veiculos_utilizados": sum(1 for rota in rotas_finais if len(rota) > 0),
            "rotas_detalhadas": rotas_finais
        }
    }