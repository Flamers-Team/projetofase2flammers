import pandas as pd
import random

def gerar_arquivos_csv(quantidade_pacientes=1000):
    # 1. Base de Dados dos Hospitais (Os pontos de partida e chegada)
    hospitais_dados = [
        {'id_hospital': 'HBB', 'nome': 'Hospital de Base (Plano Piloto)', 'latitude': -15.7984, 'longitude': -47.8864},
        {'id_hospital': 'HRC', 'nome': 'Hospital Regional de Ceilândia', 'latitude': -15.8239, 'longitude': -48.1152},
        {'id_hospital': 'HRT', 'nome': 'Hospital Regional de Taguatinga', 'latitude': -15.8335, 'longitude': -48.0628},
        {'id_hospital': 'HRSAM', 'nome': 'Hospital Regional de Samambaia', 'latitude': -15.8741, 'longitude': -48.0848}
    ]
    df_hospitais = pd.DataFrame(hospitais_dados)
    
    # Salva o arquivo CSV de hospitais
    df_hospitais.to_csv('hospitais_df.csv', index=False, encoding='utf-8')
    print("✅ Arquivo 'hospitais_df.csv' gerado com sucesso!")

    # 2. Zonas e Regras para os Pacientes
    regioes = [
        ('Ceilândia', -15.8194, -48.1119, 0.04),
        ('Taguatinga', -15.8331, -48.0564, 0.03),
        ('Samambaia', -15.8741, -48.0848, 0.04),
        ('Plano Piloto', -15.7797, -47.9297, 0.05),
        ('Gama', -16.0171, -48.0642, 0.03)
    ]

    tipos_atendimento = [
        ('emergencia_obstetrica', 1, 0, 24, False), # (tipo, prioridade, janela_inicio, janela_fim, controla_temperatura)
        ('violencia_domestica', 2, 0, 24, False),
        ('medicamento_hormonal', 3, 8, 18, True),
        ('pos_parto', 4, 14, 18, False)
    ]

    # 3. Gerando os Pacientes Aleatoriamente
    pacientes_lista = []
    for i in range(1, quantidade_pacientes + 1):
        regiao = random.choice(regioes)
        nome_ra = regiao[0]
        
        # Cria uma variação segura em volta da coordenada central da Região Administrativa
        lat = regiao[1] + random.uniform(-regiao[3], regiao[3])
        lon = regiao[2] + random.uniform(-regiao[3], regiao[3])
        
        tipo = random.choice(tipos_atendimento)
        
        pacientes_lista.append({
            'id_paciente': i, # Um ID numérico limpo para o algoritmo genético
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

    # Salva o arquivo CSV de pacientes
    df_pacientes = pd.DataFrame(pacientes_lista)
    df_pacientes.to_csv('pacientes_df.csv', index=False, encoding='utf-8')
    print("✅ Arquivo 'pacientes_df.csv' gerado com sucesso!")
    
    return df_hospitais, df_pacientes

# --- Execução ---
if __name__ == "__main__":
    df_h, df_p = gerar_arquivos_csv(quantidade_pacientes=1000)