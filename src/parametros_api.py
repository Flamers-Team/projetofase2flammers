import pandas as pd
import time

# Importando rigorosamente os módulos separados que já construímos!
from src.populacao import gerar_populacao_inicial
from src.aptidao import avaliar_fitness, decodificar_rotas
from src.selecao import selecionar_progenitores
from src.cruzamento import crossover_ordem_ox
from src.mutacao import aplicar_mutacao_troca
from src.substituicao import ordenar_e_aplicar_elitismo

def rodar_otimizacao_api(
    lista_pacientes: list,          # Recebe o array de objetos JSON do React
    coordenadas_hospital: tuple,    # (lat, lon)
    config_algoritmo: dict,         # {"populacao": 50, "mutacao": 0.25, "geracoes": 100}
    config_frota: dict              # {"veiculos": 4, "capacidade": 30}
) -> dict:
    """
    Função Maestro: Orquestra os módulos do Algoritmo Genético, 
    processa os dados puros da API e devolve um dicionário (JSON) de resposta.
    """
    
    # 1. Converte a lista de dicionários do React para um DataFrame
    # (Assim os seus arquivos como 'aptidao.py' continuam funcionando sem mudar 1 linha!)
    df_pacientes = pd.DataFrame(lista_pacientes)
    num_pacientes = len(df_pacientes)
    
    # 2. Extrai os parâmetros recebidos
    TAM_POP = config_algoritmo["populacao"]
    GERACOES = config_algoritmo["geracoes"]
    PROB_MUTACAO = config_algoritmo["mutacao"]
    NUM_VEICULOS = config_frota["veiculos"]
    CAPACIDADE = config_frota["capacidade"]

    tempo_inicio = time.time()
    
    # 3. O Ciclo Evolutivo Modular 
    populacao = gerar_populacao_inicial(num_pacientes, TAM_POP)
    
    for _ in range(GERACOES):
        fitness_valores = [avaliar_fitness(ind, df_pacientes, coordenadas_hospital, NUM_VEICULOS, CAPACIDADE) for ind in populacao]
        
        populacao, fitness_valores, melhor_ind = ordenar_e_aplicar_elitismo(populacao, fitness_valores)
        
        nova_populacao = [melhor_ind]
        while len(nova_populacao) < TAM_POP:
            p1, p2 = selecionar_progenitores(populacao, fitness_valores)
            filho = crossover_ordem_ox(p1, p2)
            filho = aplicar_mutacao_troca(filho, PROB_MUTACAO)
            nova_populacao.append(filho)
            
        populacao = nova_populacao
        
    tempo_fim = time.time()
    
    # 4. Decodifica as rotas do melhor indivíduo encontrado
    rotas_finais = decodificar_rotas(populacao[0], df_pacientes, NUM_VEICULOS, CAPACIDADE)
    
    # 5. Retorna o Payload de Resposta limpo para o React
    return {
        "status": "sucesso",
        "metadados": {
            "tempo_processamento_segundos": round(tempo_fim - tempo_inicio, 3),
            "pacientes_processados": num_pacientes
        },
        "score_fitness_final": round(fitness_valores[0], 2),
        "rotas": rotas_finais
    }