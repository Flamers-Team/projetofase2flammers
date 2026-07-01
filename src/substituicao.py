from typing import List, Tuple

def ordenar_e_aplicar_elitismo(populacao: List[List[int]], fitness_valores: List[float]) -> Tuple[List[List[int]], List[float], List[int]]:
    """
    Ordena a população atual e separa o melhor indivíduo absoluto (Elitismo)
    para compor o primeiro elemento da próxima geração.
    """
    combinado = sorted(list(zip(populacao, fitness_valores)), key=lambda x: x[1])
    populacao_ordenada, fitness_ordenado = zip(*combinado)
    
    return list(populacao_ordenada), list(fitness_ordenado), copy_individual(populacao_ordenada[0])

def copy_individual(ind: List[int]) -> List[int]:
    import copy
    return copy.deepcopy(ind)