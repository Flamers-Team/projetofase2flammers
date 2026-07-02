import random
import numpy as np
from typing import List, Tuple

def _roleta_inversa(populacao: List[List[int]], fitness_valores: List[float], k: int) -> List[List[int]]:
    """Aplica a seleção por roleta inversa."""
    atenuacao = 1.0 / (np.array(fitness_valores) + 1e-6)
    pesos = atenuacao / np.sum(atenuacao)
    return random.choices(populacao, weights=pesos, k=k)

def _selecao_ranking(populacao: List[List[int]], fitness_valores: List[float], k: int) -> List[List[int]]:
    """
    Aplica a seleção por Ranking Linear para MINIMIZAÇÃO.
    Indivíduos com MENOR fitness recebem a MAIOR probabilidade de seleção.
    """
    n = len(populacao)
    
    # Ordena os índices do pior (maior fitness) para o melhor (menor fitness)
    # np.argsort nos dá os índices do menor para o maior. 
    # Ao inverter com [::-1], o primeiro da lista será o pior (maior fitness) e o último o melhor.
    indices_pior_para_melhor = np.argsort(fitness_valores)[::-1]
    
    # Atribui os pesos linearmente: o pior ganha peso 1, o segundo pior peso 2, ..., o melhor ganha peso n
    pesos = np.zeros(n)
    for rank, idx in enumerate(indices_pior_para_melhor, start=1):
        pesos[idx] = rank
        
    # Normaliza os pesos para obter as probabilidades de seleção
    probabilidades = pesos / np.sum(pesos)
    
    return random.choices(populacao, weights=probabilidades, k=k)

def selecionar_progenitores(
    populacao: List[List[int]], 
    fitness_valores: List[float], 
    metodo: str = "Roleta Inversa", 
    k: int = 2
) -> Tuple[List[int], List[int]]:
    """
    Função principal de seleção que atua como interface para o algoritmo genético.
    Aceita os métodos: 'Roleta Inversa' ou 'Ranking'.
    """
    if metodo == "Roleta Inversa":
        escolhidos = _roleta_inversa(populacao, fitness_valores, k)
    elif metodo == "Ranking":
        escolhidos = _selecao_ranking(populacao, fitness_valores, k)
    else:
        raise ValueError(f"Método de seleção desconhecido: {metodo}. Escolha 'Roleta Inversa' ou 'Ranking'.")
        
    return escolhidos[0], escolhidos[1]