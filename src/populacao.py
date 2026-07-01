import random
from typing import List


def gerar_populacao_inicial(num_pacientes: int, tamanho_populacao: int) -> List[List[int]]:
    
    base_indices = list(range(num_pacientes))
    return [random.sample(base_indices, num_pacientes) for _ in range(tamanho_populacao)]