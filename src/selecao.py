# src/selecao.py
import random
import numpy as np
from typing import List, Tuple

def selecionar_progenitores(populacao: List[List[int]], fitness_valores: List[float], k: int = 2) -> Tuple[List[int], List[int]]:
    """
    Seleção por Roleta Inversa (proporcional ao inverso do fitness, já que buscamos a MINIMIZAÇÃO).
    """
    atenuacao = 1.0 / (np.array(fitness_valores) + 1e-6)
    pesos = atenuacao / np.sum(atenuacao)
    
    escolhidos = random.choices(populacao, weights=pesos, k=k)
    return escolhidos[0], escolhidos[1]