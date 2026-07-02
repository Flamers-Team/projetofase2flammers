
import numpy as np
from typing import List, Dict

def avaliar_eficiencia_geracao(fitness_valores: List[float]) -> Dict[str, float]:
    """
    Analisa os dados estatísticos da geração corrente para monitorar a convergência.
    """
    return {
        "melhor_score": float(np.min(fitness_valores)),
        "media_score": float(np.mean(fitness_valores)),
        "pior_score": float(np.max(fitness_valores))
    }