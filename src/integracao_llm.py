
import google.generativeai as genai
import pandas as pd
from typing import List

def inicializar_llm(api_key: str):
    """Configura e valida as credenciais da API do Gemini."""
    genai.configure(api_key=api_key)

def gerar_manuais_e_roteiros_llm(rotas_otimizadas: List[List[int]], df_pacientes: pd.DataFrame, hospital_nome: str) -> str:
    """
    Formata o resultado do algoritmo genético em um JSON estruturado e solicita
    à LLM a criação do Manual Prático do Motorista e Roteiro Detalhado.
    """
    # 1. Serializar a rota estruturada para texto que a LLM compreenda facilmente
    contexto_logistico = []
    
    for v_id, rota in enumerate(rotas_otimizadas):
        paradas_veiculo = []
        for ordem, idx_paciente in enumerate(rota):
            p = df_pacientes.iloc[idx_paciente]
            paradas_veiculo.append({
                "ordem_visita": ordem + 1,
                "identificador": p['nome_ficticio'],
                "regiao": p['regiao_administrativa'],
                "atendimento": p['tipo_atendimento'].replace('_', ' ').title(),
                "prioridade_nivel": p['prioridade'],
                "caixas_medicamento": p['demanda_caixas'],
                "refrigerado": "Sim" if p['temperatura_controlada'] else "Não"
            })
        contexto_logistico.append({
            "veiculo": f"Veículo {v_id + 1}",
            "roteiro_de_paradas": paradas_veiculo
        })

    # 2. Desenhar o prompt especializado com protocolos sensíveis à saúde da mulher
    prompt = f"""
    Você é um especialista em logística médica e segurança comunitária em saúde pública da mulher.
    Abaixo está a rota de distribuição diária de medicamentos e suporte domiciliar otimizada para o Distrito Federal, partindo do {hospital_nome}.
    
    Dados brutos da rota de frotas:
    {contexto_logistico}
    
    Sua tarefa é criar um documento oficial estruturado contendo duas seções principais:
    
    1. MANUAL DE INSTRUÇÕES DA EQUIPE DE TRANSPORTE (Focado em Conduta e Protocolos):
       Para cada tipo de atendimento presente na rota, forneça regras claras baseadas nestes direcionamentos:
       - Casos de Violência Doméstica: Exigir discrição absoluta. Não usar sirenes ou identificações ostensivas do veículo na porta da residência. Manter privacidade e seguir o protocolo silencioso de entrega.
       - Medicamentos Hormonais / Temperatura Controlada: Instruir a checagem do termômetro digital da caixa térmica a cada parada, garantindo a conservação.
       - Emergências Obstétricas: Prioridade de condução imediata com contato rádio prévio com a central.
       - Atendimentos Pós-parto: Empatia, verificação de bem-estar básico e registro rigoroso na planilha de acompanhamento.
    
    2. ROTEIRO DETALHADO DE VISITAS (Prático para visualização de campo):
       Escreva uma listagem fluida e legível para cada Veículo, detalhando a sequência exata de quem eles visitarão, em qual região administrativa do DF e o que será entregue ou realizado em cada parada.
    
    Formate a resposta inteiramente em Markdown amigável e profissional para ser exibido diretamente na tela do Streamlit. Responda em português.
    """

    # 3. Chamar o modelo generativo
    # Usamos o gemini-1.5-flash por ser rápido, contextual e altamente econômico
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Erro ao processar integração com a LLM: {str(e)}"