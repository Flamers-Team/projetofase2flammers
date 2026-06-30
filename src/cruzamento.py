# src/cruzamento.py
import random
from typing import List

def crossover_ordem_ox(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Operador de Cruzamento por Ordem (OX) original do professor.
    Preserva trechos contíguos de rotas evitando duplicidade de pacientes.
    """
    length = len(parent1)
    start_idx = random.randint(0, length - 1)
    end_idx = random.randint(start_idx + 1, length)

    child = parent1[start_idx:end_idx]
    remaining_positions = [i for i in range(length) if i < start_idx or i >= end_idx]
    remaining_genes = [gene for gene in parent2 if gene not in child]

    for position, gene in zip(remaining_positions, remaining_genes):
        child.insert(position, gene)

    return child