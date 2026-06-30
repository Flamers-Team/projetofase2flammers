# src/aptidao.py
import math
import pandas as pd
from typing import List, Tuple

def calcular_distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0  # Raio da Terra em km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def decodificar_rotas(chromossomo: List[int], df_pacientes: pd.DataFrame, num_veiculos: int, capacidade_max: int) -> List[List[int]]:
    rotas = [[] for _ in range(num_veiculos)]
    veiculo_atual = 0
    carga_atual = 0
    
    for idx_paciente in chromossomo:
        demanda = df_pacientes.iloc[idx_paciente]['demanda_caixas']
        if carga_atual + demanda > capacidade_max and veiculo_atual < num_veiculos - 1:
            veiculo_atual += 1
            carga_atual = 0
        rotas[veiculo_atual].append(idx_paciente)
        carga_atual += demanda
    return rotas

def avaliar_fitness(chromossomo: List[int], df_pacientes: pd.DataFrame, hospital_coord: Tuple[float, float], num_veiculos: int, capacidade_max: int) -> float:
    rotas = decodificar_rotas(chromossomo, df_pacientes, num_veiculos, capacidade_max)
    custo_total = 0.0
    penalidades = 0.0
    velocidade_media = 40.0  # km/h simulados no tráfego do DF
    
    for rota in rotas:
        if not rota: continue
        ponto_anterior_lat, ponto_anterior_lon = hospital_coord
        tempo_atual = 8.0  # Início às 08:00
        prioridade_maxima = 4
        
        for idx_paciente in rota:
            paciente = df_pacientes.iloc[idx_paciente]
            dist = calcular_distancia_haversine(ponto_anterior_lat, ponto_anterior_lon, paciente['latitude'], paciente['longitude'])
            custo_total += dist
            tempo_atual += dist / velocidade_media
            
            # Restrição de Janela de Tempo
            if tempo_atual < paciente['janela_inicio']:
                tempo_atual = paciente['janela_inicio']
            elif tempo_atual > paciente['janela_fim']:
                penalidades += (tempo_atual - paciente['janela_fim']) * 120.0  # Penalidade por atraso
                
            tempo_atual += 0.25  # 15 min de atendimento
            
            # Restrição de Prioridade (Emergências primeiro)
            if paciente['prioridade'] < prioridade_maxima:
                penalidades += 200.0  # Penalidade grave por inversão de prioridade
            else:
                prioridade_maxima = paciente['prioridade']
                
            ponto_anterior_lat, ponto_anterior_lon = paciente['latitude'], paciente['longitude']
            
        # Retorno ao depósito
        custo_total += calcular_distancia_haversine(ponto_anterior_lat, ponto_anterior_lon, hospital_coord[0], hospital_coord[1])
        
    return custo_total + penalidades