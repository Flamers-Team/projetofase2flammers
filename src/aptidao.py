import math
import pandas as pd
from typing import List, Tuple

def calcular_distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0  
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

def avaliar_fitness(
    chromossomo: List[int], 
    df_pacientes: pd.DataFrame, 
    hospital_coord: Tuple[float, float], 
    num_veiculos: int, 
    capacidade_max: int,
    autonomia_max_km: float = 120.0
) -> float:
    
    rotas = decodificar_rotas(chromossomo, df_pacientes, num_veiculos, capacidade_max)
    
    # --- HIPERPARÂMETROS DE CUSTO NORMALIZADO ---
    # Estes valores convertem distância, tempo ocioso e atrasos em uma única métrica
    PESO_KM = 1.0                # Custo base de 1 unidade por km rodado
    PESO_ESPERA = 5.0            # Penalidade leve por deixar o motorista ocioso aguardando janela
    FATOR_ATRASO_BASE = 50.0     # Penalidade base por hora de atraso
    PENALIDADE_LETAL = 5000.0    # Multa massiva para quebra de restrições rígidas (Autonomia)
    
    # Mapeamento do impacto do atraso (SLA) com base na nova tabela de prioridades
    # Menor número = Maior gravidade = Peso exponencial no atraso
    multiplicador_prioridade = {
        2: 5.0,  # Violência Doméstica (Atraso é inaceitável)
        3: 2.0,  # Medicamento Hormonal (Importante, mas com leve tolerância)
        4: 1.2,  # Pós-parto (Rotina com horário marcado)
        5: 1.0   # Atenção Básica (Rotina flexível, peso base)
    }
    
    custo_total_normalizado = 0.0
    velocidade_media = 35.0  # Ajustado para trânsito urbano mais realista
    
    for rota in rotas:
        if not rota: continue
        ponto_anterior_lat, ponto_anterior_lon = hospital_coord
        tempo_atual = 8.0 
        distancia_percorrida_veiculo = 0.0 
        
        for idx_paciente in rota:
            paciente = df_pacientes.iloc[idx_paciente]
            dist = calcular_distancia_haversine(ponto_anterior_lat, ponto_anterior_lon, paciente['latitude'], paciente['longitude'])
            
            distancia_percorrida_veiculo += dist
            tempo_atual += dist / velocidade_media
            
            # --- NOVA LÓGICA DE JANELA E SLA PONDERADO ---
            if tempo_atual < paciente['janela_inicio']:
                # Penaliza o tempo ocioso (Otimiza a escala de trabalho)
                horas_espera = paciente['janela_inicio'] - tempo_atual
                custo_total_normalizado += horas_espera * PESO_ESPERA
                tempo_atual = paciente['janela_inicio']
                
            elif tempo_atual > paciente['janela_fim']:
                # Aplica a multa ponderada pela gravidade da paciente
                horas_atraso = tempo_atual - paciente['janela_fim']
                peso_paciente = multiplicador_prioridade.get(paciente['prioridade'], 1.0)
                
                custo_total_normalizado += (horas_atraso * FATOR_ATRASO_BASE * peso_paciente)
                
            tempo_atual += 0.25 # 15 min de consulta
            ponto_anterior_lat, ponto_anterior_lon = paciente['latitude'], paciente['longitude']
            
        # Retorno ao depósito
        dist_retorno = calcular_distancia_haversine(ponto_anterior_lat, ponto_anterior_lon, hospital_coord[0], hospital_coord[1])
        distancia_percorrida_veiculo += dist_retorno
        
        # Custo espacial
        custo_total_normalizado += (distancia_percorrida_veiculo * PESO_KM)
        
        # Restrição Hard de Autonomia
        if distancia_percorrida_veiculo > autonomia_max_km:
            km_excedido = distancia_percorrida_veiculo - autonomia_max_km
            custo_total_normalizado += (km_excedido * PENALIDADE_LETAL)
            
    return custo_total_normalizado