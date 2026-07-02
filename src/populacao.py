import random
import pandas as pd
from typing import List, Tuple

# Importamos a fórmula matemática que já criamos para garantir consistência nas distâncias
from src.aptidao import calcular_distancia_haversine

def gerar_rota_vizinho_mais_proximo(df_pacientes: pd.DataFrame, coord_hospital: Tuple[float, float]) -> List[int]:
    """
    Heurística Construtiva (Nearest Neighbor): 
    Cria uma rota lógica baseada estritamente na distância física mais curta a cada passo.
    Atua como um "humano imaginário" despachando o carro de forma gulosa.
    """
    pacientes_nao_visitados = list(range(len(df_pacientes)))
    rota_heuristica = []
    
    # O carro liga o motor no hospital
    lat_atual, lon_atual = coord_hospital
    
    while pacientes_nao_visitados:
        menor_distancia = float('inf')
        paciente_mais_proximo = -1
        
        # O motorista olha no mapa todos os pacientes que ainda faltam
        # e escolhe aquele que está fisicamente mais perto de onde ele está parado agora
        for idx in pacientes_nao_visitados:
            paciente = df_pacientes.iloc[idx]
            dist = calcular_distancia_haversine(
                lat_atual, lon_atual, 
                paciente['latitude'], paciente['longitude']
            )
            
            if dist < menor_distancia:
                menor_distancia = dist
                paciente_mais_proximo = idx
                
        # Adiciona o paciente mais perto na prancheta da rota
        rota_heuristica.append(paciente_mais_proximo)
        
        # Risca o paciente da lista de pendências
        pacientes_nao_visitados.remove(paciente_mais_proximo)
        
        # O carro "dirige" até lá. A nova posição de partida vira a casa desse paciente.
        paciente_escolhido = df_pacientes.iloc[paciente_mais_proximo]
        lat_atual, lon_atual = paciente_escolhido['latitude'], paciente_escolhido['longitude']
        
    return rota_heuristica

def gerar_populacao_inicial(
    num_pacientes: int, 
    tam_populacao: int, 
    estrategia: str = "100% Aleatória", 
    df_pacientes: pd.DataFrame = None, 
    coord_hospital: Tuple[float, float] = None
) -> List[List[int]]:
    """
    Gera a Geração 0 (População Inicial) do Algoritmo Genético.
    Pode gerar uma matriz 100% caótica ou injetar o 'Indivíduo Semente' (Híbrida).
    """
    populacao = []
    
    # 1. INJEÇÃO HÍBRIDA (O Gênio no meio da multidão)
    if estrategia == "Híbrida: Semente Vizinho Mais Próximo" and df_pacientes is not None and coord_hospital is not None:
        individuo_semente = gerar_rota_vizinho_mais_proximo(df_pacientes, coord_hospital)
        populacao.append(individuo_semente)
    
    # 2. EXPLORAÇÃO ESTOCÁSTICA (Preenchendo o restante das vagas)
    # Garante a Diversidade Genética gerando o resto da população de forma totalmente aleatória
    while len(populacao) < tam_populacao:
        ind = list(range(num_pacientes))
        random.shuffle(ind)
        populacao.append(ind)
        
    return populacao