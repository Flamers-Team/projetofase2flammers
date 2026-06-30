import pandas as pd
import random

def gerar_base_com_atencao_basica(quantidade_pacientes=1000):
    regioes = [
        ('Ceilândia', -15.8194, -48.1119, 0.04),
        ('Taguatinga', -15.8331, -48.0564, 0.03),
        ('Samambaia', -15.8741, -48.0848, 0.04),
        ('Plano Piloto', -15.7797, -47.9297, 0.05),
        ('Gama', -16.0171, -48.0642, 0.03)
    ]

    # Nova estrutura de atendimentos programáveis
    # Formato: (Nome, Prioridade, Janela de Início, Janela de Fim, Requer Refrigeração)
    tipos_agendados = [
        ('violencia_domestica', 2, 0, 24, False),     # Prioridade 2 (Pode ser visitada a qualquer hora com discrição)
        ('medicamento_hormonal', 3, 8, 18, True),     # Prioridade 3 (Refrigerado, Horário Comercial)
        ('pos_parto', 4, 14, 18, False),              # Prioridade 4 (Horário da Tarde)
        ('atencao_basica', 5, 8, 18, False)           # Prioridade 5 (Rotina padrão, Horário Comercial)
    ]

    pacientes_lista = []
    for i in range(1, quantidade_pacientes + 1):
        regiao = random.choice(regioes)
        nome_ra = regiao[0]
        
        lat = regiao[1] + random.uniform(-regiao[3], regiao[3])
        lon = regiao[2] + random.uniform(-regiao[3], regiao[3])
        
        tipo = random.choice(tipos_agendados)
        
        pacientes_lista.append({
            'id_paciente': i, 
            'nome_ficticio': f"Paciente_{random.randint(1000, 9999)}",
            'regiao_administrativa': nome_ra,
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'tipo_atendimento': tipo[0],
            'prioridade': tipo[1],
            'janela_inicio': tipo[2],
            'janela_fim': tipo[3],
            'demanda_caixas': random.randint(1, 3),
            'temperatura_controlada': tipo[4],
            'status': 'pendente'
        })

    df_pacientes = pd.DataFrame(pacientes_lista)
    df_pacientes.to_csv('pacientes_df.csv', index=False, encoding='utf-8')
    print(f"✅ Nova base gerada com {quantidade_pacientes} atendimentos agendados!")
    print("\nDistribuição Realista do Planejamento:")
    print(df_pacientes['tipo_atendimento'].value_counts(normalize=True) * 100)

# Executa a geração
gerar_base_com_atencao_basica(1000)