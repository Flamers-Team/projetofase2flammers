
import random
import copy
from typing import List

def aplicar_mutacao_troca(solucao: List[int], probabilidade_mutacao: float) -> List[int]:

    mutada = copy.deepcopy(solucao)
    if random.random() < probabilidade_mutacao:
        if len(solucao) < 2:
            return solucao
        idx = random.randint(0, len(solucao) - 2)
        mutada[idx], mutada[idx + 1] = solucao[idx + 1], solucao[idx]
    
    return mutada