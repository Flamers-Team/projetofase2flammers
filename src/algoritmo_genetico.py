import random
import math
import copy
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def calcular_distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula a distância real em quilômetros entre duas coordenadas geográficas."""
    R = 6371.0  # Raio da Terra em km
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def decodificar_rotas(chromossomo: List[int], df_pacientes: pd.DataFrame, num_veiculos: int, capacidade_max_caixas: int) -> List[List[int]]:
    """
    Divide a sequência única de pacientes (cromossomo) entre os veículos disponíveis,
    respeitando o limite máximo de caixas que cada veículo consegue carregar.
    """
    rotas = [[] for _ in range(num_veiculos)]
    veiculo_atual = 0
    carga_atual = 0
    
    for idx_paciente in chromossomo:
        # Puxa os dados do paciente baseado no índice do cromossomo
        paciente_row = df_pacientes.iloc[idx_paciente]
        demanda = paciente_row['demanda_caixas']
        
        # Se estourar a capacidade do veículo atual, passa para o próximo veículo
        if carga_atual + demanda > capacidade_max_caixas and veiculo_atual < num_veiculos - 1:
            veiculo_atual += 1
            carga_atual = 0
            
        rotas[veiculo_atual].append(idx_paciente)
        carga_atual += demanda
        
    return rotas

def calcular_fitness_vrp(chromossomo: List[int], df_pacientes: pd.DataFrame, hospital_coord: Tuple[float, float], num_veiculos: int, capacidade_max_caixas: int) -> float:
    """
    Função de fitness adaptada para VRP. Calcula a distância total percorrida pela frota
    e aplica penalidades severas caso restrições de tempo, capacidade ou prioridade sejam quebradas.
    """
    rotas = decodificar_rotas(chromossomo, df_pacientes, num_veiculos, capacidade_max_caixas)
    custo_total = 0.0
    
    penalidade_janela_tempo = 0
    penalidade_prioridade = 0
    
    velocidade_media_kmh = 40.0  # Velocidade média simulada para o tráfego do DF
    
    for rota in rotas:
        if not rota:
            continue
            
        # O veículo sempre sai do hospital (Depósito)
        ponto_anterior_lat, ponto_anterior_lon = hospital_coord
        tempo_atual_horas = 8.0  # Frota sai às 08:00 da manhã
        
        prioridade_maxima_vista = 4  # Acompanha se estamos atendendo os casos mais graves primeiro
        
        for idx_paciente in rota:
            paciente = df_pacientes.iloc[idx_paciente]
            
            # 1. Distância até a parada
            dist = calcular_distancia_haversine(ponto_anterior_lat, ponto_anterior_lon, paciente['latitude'], paciente['longitude'])
            custo_total += dist
            
            # Atualiza o tempo com base na distância percorrida
            tempo_atual_horas += dist / velocidad_media_kmh
            
            # 2. Restrição de Janela de Tempo (Pós-parto, etc)
            if tempo_atual_horas < paciente['janela_inicio']:
                # Se chegou antes, o veículo espera abrir a janela
                tempo_atual_horas = paciente['janela_inicio']
            elif tempo_atual_horas > paciente['janela_fim']:
                # Se chegou depois, aplica penalidade de atraso (100km de custo por hora de atraso)
                penalidade_janela_tempo += (tempo_atual_horas - paciente['janela_fim']) * 100
                
            # Tempo gasto no atendimento médico/entrega (aproximadamente 15 minutos = 0.25h)
            tempo_atual_horas += 0.25
            
            # 3. Restrição de Ordem de Prioridade
            # Prioridade 1 (Emergência) deve vir antes de Prioridade 4 (Pós-parto)
            if paciente['prioridade'] < prioridade_maxima_vista:
                # Significa que atendemos alguém mais urgente DEPOIS de alguém menos urgente nesta mesma rota
                penalidade_prioridade += 150  # Penalidade pesada por inversão de prioridade
            else:
                prioridade_maxima_vista = paciente['prioridade']
                
            ponto_anterior_lat, ponto_anterior_lon = paciente['latitude'], paciente['longitude']
            
        # O veículo deve retornar ao hospital ao fim da rota
        dist_retorno = calcular_distancia_haversine(ponto_anterior_lat, ponto_anterior_lon, hospital_coord[0], hospital_coord[1])
        custo_total += dist_retorno

    # O objetivo do algoritmo genético é MINIMIZAR o retorno desta função (Menor distância + Sem erros)
    return custo_total + penalidade_janela_tempo + penalidade_prioridade

def order_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """Operador OX idêntico ao do professor (trabalhando com índices numéricos puros)."""
    length = len(parent1)
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    child = parent1[start_index:end_index]
    remaining_positions = [i for i in range(length) if i < start_index or i >= end_index]
    remaining_genes = [gene for gene in parent2 if gene not in child]

    for position, gene in zip(remaining_positions, remaining_genes):
        child.insert(position, gene)

    return child

def mutate(solution: List[int], mutation_probability: float) -> List[int]:
    """Operador de mutação por troca simples do professor."""
    mutated_solution = copy.deepcopy(solution)
    if random.random() < mutation_probability:
        if len(solution) < 2:
            return solution
        index = random.randint(0, len(solution) - 2)
        mutated_solution[index], mutated_solution[index + 1] = solution[index + 1], solution[index]   
    return mutated_solution

def rodar_otimizacao_vrp(df_pacientes: pd.DataFrame, hospital_coord: Tuple[float, float], num_veiculos: int, capacidade_max_caixas: int, tam_populacao: int = 50, geracoes: int = 100, prob_mutacao: float = 0.2) -> Dict:
    """Função mestre que executa o ciclo evolutivo completo do AG para o problema de frotas."""
    num_pacientes = len(df_pacientes)
    # Cromossomo representa a ordem sequencial de leitura dos índices das linhas do Pandas
    base_indices = list(range(num_pacientes))
    
    # Geração da população inicial
    populacao = [random.sample(base_indices, num_pacientes) for _ in range(tam_populacao)]
    historico_fitness = []
    
    for gen in range(geracoes):
        # Avaliação da população
        populacao_fitness = [calcular_fitness_vrp(ind, df_pacientes, hospital_coord, num_veiculos, capacidade_max_caixas) for ind in populacao]
        
        # Ordenação por melhor resultado (menor custo/distância)
        combinado = sorted(list(zip(populacao, populacao_fitness)), key=lambda x: x[1])
        populacao, populacao_fitness = zip(*combinado)
        populacao = list(populacao)
        
        historico_fitness.append(populacao_fitness[0])
        
        # Elitismo: mantém o melhor intocado
        nova_populacao = [populacao[0]]
        
        while len(nova_populacao) < tam_populacao:
            # Seleção baseada em torneio ou ranking dos melhores
            p1, p2 = random.choices(populacao[:10], k=2)
            
            filho = order_crossover(p1, p2)
            filho = mutate(filho, prob_mutacao)
            nova_populacao.append(filho)
            
        populacao = nova_populacao
        
    melhor_solucao = populacao[0]
    rotas_finais = decodificar_rotas(melhor_solucao, df_pacientes, num_veiculos, capacidade_max_caixas)
    
    return {
        "rotas_por_veiculo": rotas_finais,
        "melhor_fitness": historico_fitness[-1],
        "historico_evolucao": historico_fitness
    }