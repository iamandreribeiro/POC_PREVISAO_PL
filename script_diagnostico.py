import pandas as pd

def map_season_string(season_str):
    """Converte '2019-2020' para 2020"""
    if pd.isna(season_str):
        return 0
    try:
        # Pega o segundo ano (ex: 2020 de "2019-2020")
        return int(season_str.split('-')[1])
    except (IndexError, ValueError):
        return 0

print("Iniciando diagnóstico de nomes de times...")

# 1. Carregar nomes do PremierLeague.csv (Arquivo de Partidas)
try:
    df_matches = pd.read_csv('PremierLeague.csv')
    df_matches['season'] = df_matches['Season'].apply(map_season_string)
    
    # Pegar nomes das temporadas relevantes (2020-2024)
    df_matches_relevant = df_matches[df_matches['season'].isin([2020, 2021, 2022, 2023, 2024])]
    
    # Usamos set() para obter valores únicos de ambas as colunas
    match_names = set(df_matches_relevant['HomeTeam'].unique()) | set(df_matches_relevant['AwayTeam'].unique())
    print(f"\n--- Nomes encontrados em 'PremierLeague.csv' (Partidas) --- \n{sorted(match_names)}\n")
    
except FileNotFoundError:
    print("Erro: 'PremierLeague.csv' não encontrado.")
    match_names = set()

# 2. Carregar nomes do data_accp_with_posse (1).csv (Arquivo de Stats)
try:
    df_stats = pd.read_csv('data_accp_with_posse (1).csv')
    df_stats_relevant = df_stats[df_stats['season'].isin([2020, 2021, 2022, 2023, 2024])]
    stats_names = set(df_stats_relevant['team_name'].unique())
    print(f"\n--- Nomes encontrados em 'data_accp_with_posse (1).csv' (Stats) --- \n{sorted(stats_names)}\n")
    
except FileNotFoundError:
    print("Erro: 'data_accp_with_posse (1).csv' não encontrado.")
    stats_names = set()

# 3. Comparar as listas
if not match_names or not stats_names:
    print("Não foi possível comparar os nomes pois um dos arquivos faltou.")
else:
    print("="*50)
    print("           DIAGNÓSTICO DE NOMES DE TIMES")
    print("="*50)
    
    # Nomes que estão no CSV de partidas mas não no CSV de stats
    only_in_matches = match_names - stats_names
    if only_in_matches:
        print(f"\n>>> NOMES QUE ESTÃO APENAS EM 'PremierLeague.csv' (precisam ser corrigidos):")
        print(sorted(only_in_matches))
    else:
        print("\n[SUCESSO] Todos os nomes de 'PremierLeague.csv' estão em 'data_accp_with_posse (1).csv'.")

    # Nomes que estão no CSV de stats mas não no CSV de partidas
    only_in_stats = stats_names - match_names
    if only_in_stats:
        print(f"\n>>> NOMES QUE ESTÃO APENAS EM 'data_accp_with_posse (1).csv' (referência):")
        print(sorted(only_in_stats))
    else:
        print("\n[SUCESSO] Todos os nomes de 'data_accp_with_posse (1).csv' estão em 'PremierLeague.csv'.")
        
    if not only_in_matches and not only_in_stats:
         print("\nSUCESSO! Os nomes dos times parecem corresponder perfeitamente.")

print("\nDiagnóstico concluído.")