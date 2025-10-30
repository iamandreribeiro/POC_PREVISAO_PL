import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.model_selection import GridSearchCV
import warnings

# Ignorar avisos futuros do pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

# Lista de features que vamos usar para criar as "diferenças"
FEATURES_PARA_LAG = [
    'points', 'position', 'points_per_match', 'xG', 'xGA', 'npxG', 'npxGA', 'xPts', 
    'xPts_per_match', 'status_time_points', 'Passes', 'long_pass_accuracy', 
    'cross_accuracy', 'dribble_accuracy', 'teams_salary', 'Shot_Difference', 
    'SOT_Difference', 'gf_xG_performance_final', 'posse_percent'
]

def get_stats_lookup_table():
    """
    Esta função executa a preparação de dados dos Protótipos 1 e 2.
    Ela retorna o DataFrame 'df_final' que usaremos como uma
    tabela de consulta para os stats T-1 de cada time.
    """
    print("Carregando Tabela de Stats (data_accp_with_posse (1).csv)...")
    try:
        df = pd.read_csv('data_accp_with_posse (1).csv')
    except FileNotFoundError:
        print("Erro: 'data_accp_with_posse (1).csv' não encontrado.")
        return None

    df = df.sort_values(by=['team_name', 'season'])

    # 1. Criar features T-1
    for col in FEATURES_PARA_LAG:
        df[f'{col}_T-1'] = df.groupby('team_name')[col].shift(1)

    # 2. Remover temporada 2019 (não temos T-1 para ela)
    df_base = df[df['season'] >= 2020].copy()

    # 3. Preencher NaNs (times promovidos)
    temporadas = sorted(df_base['season'].unique())
    df_final_list = []

    for season in temporadas:
        df_season_T = df_base[df_base['season'] == season].copy()
        df_season_T_minus_1 = df[df['season'] == (season - 1)]

        rebaixados_stats = df_season_T_minus_1.loc[
            df_season_T_minus_1['position'] > 17, FEATURES_PARA_LAG
        ].mean()

        fill_values = {}
        for col in FEATURES_PARA_LAG:
            if col in rebaixados_stats:
                fill_values[f'{col}_T-1'] = rebaixados_stats[col]

        df_season_T.fillna(value=fill_values, inplace=True)
        df_final_list.append(df_season_T)

    df_final = pd.concat(df_final_list, ignore_index=True)
    
    # Retornamos apenas as colunas que precisamos para a consulta
    cols_to_keep = ['season', 'team_name'] + [f'{col}_T-1' for col in FEATURES_PARA_LAG]
    
    # Adicionamos 'points' (pontos reais T) para o cálculo final do RMSE
    cols_to_keep.append('points') 
    
    print("Tabela de Stats T-1 criada com sucesso.")
    return df_final[cols_to_keep]


def load_and_prepare_match_data(stats_lookup_table):
    """
    Carrega o 'PremierLeague.csv' e o combina
    com a tabela de stats T-1 para criar as features de diferença.
    """
    print("Carregando dados de partidas (PremierLeague.csv)...")
    try:
        df_matches = pd.read_csv('PremierLeague.csv') 
    except FileNotFoundError:
        print("ERRO: Arquivo 'PremierLeague.csv' não encontrado.")
        return None

    # --- Limpeza (Ajustado para seu CSV) ---
    df_matches.rename(columns={
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FullTimeResult': 'result_string',
        'Season': 'season_string'
    }, inplace=True)

    # --- CORREÇÃO: Dicionário de Mapeamento de Nomes ---
    # Traduz os nomes do 'PremierLeague.csv' (chave) para os nomes do 'data_accp...' (valor)
    name_map = {
        'Man City': 'Manchester City',
        'Man United': 'Manchester United',
        'Newcastle': 'Newcastle United',
        "Nott'm Forest": 'Nottingham Forest',
        'West Brom': 'West Bromwich Albion',
        'Wolves': 'Wolverhampton Wanderers'
        # Nomes que já batem (ex: 'Arsenal') não precisam ser listados
    }

    # Aplica a correção aos nomes dos times. 
    # .get(x, x) significa: "tente encontrar 'x' no mapa, se não encontrar, use 'x' mesmo"
    df_matches['home_team'] = df_matches['home_team'].apply(lambda x: name_map.get(x, x))
    df_matches['away_team'] = df_matches['away_team'].apply(lambda x: name_map.get(x, x))
    
    print("Nomes de times corrigidos (mapeados).")
    # --- FIM DA CORREÇÃO ---

    # --- Criar o Alvo (Y) ---
    # 2 = Home Win (H), 1 = Draw (D), 0 = Away Win (A)
    result_map = {'H': 2, 'D': 1, 'A': 0}
    df_matches['result'] = df_matches['result_string'].map(result_map)
    
    # --- Mapear a 'season_string' para o nosso 'season' inteiro ---
    def map_season(season_str):
        if pd.isna(season_str):
            return 0
        try:
            return int(season_str.split('-')[1]) # Pega o segundo ano (ex: 2020 de "2019-2020")
        except (IndexError, ValueError):
            return 0
            
    df_matches['season'] = df_matches['season_string'].apply(map_season)

    # Remover temporadas que não podemos usar (2019 e anteriores, ou 2025 e futuras)
    # Vamos usar 2020-2023 para treinar, 2024 para testar
    df_matches = df_matches[df_matches['season'].isin([2020, 2021, 2022, 2023, 2024])].copy()

    # --- Criar Features de Diferença (O passo mais importante) ---
    print("Mesclando stats T-1 com os dados das partidas...")
    
    # Juntar stats T-1 do time da casa
    df_merged = pd.merge(
        df_matches, 
        stats_lookup_table,
        left_on=['season', 'home_team'],
        right_on=['season', 'team_name'],
        how='left'
    )
    
    # Juntar stats T-1 do time de fora
    df_merged = pd.merge(
        df_merged,
        stats_lookup_table,
        left_on=['season', 'away_team'],
        right_on=['season', 'team_name'],
        how='left',
        suffixes=('_home', '_away') # Sufixos _home e _away
    )

    # Criar as colunas _diff
    diff_features = []
    for col in FEATURES_PARA_LAG:
        col_t1 = f'{col}_T-1'
        col_diff = f'{col}_diff'
        df_merged[col_diff] = df_merged[f'{col_t1}_home'] - df_merged[f'{col_t1}_away']
        diff_features.append(col_diff)

    # Lidar com NaNs (partidas onde o merge pode ter falhado)
    initial_rows = len(df_merged)
    df_merged.dropna(subset=diff_features, inplace=True)
    final_rows = len(df_merged)
    
    if initial_rows > final_rows:
        # A única equipa que deve falhar agora é 'Ipswich' na season 2024,
        # pois está no ficheiro de stats mas não no de partidas.
        print(f"Atenção: {initial_rows - final_rows} partidas removidas (provavelmente 'Ipswich' e outras de 2024).")
    
    print("Features de diferença criadas.")
    return df_merged, diff_features

def run_simulation(df_processed, diff_features, stats_lookup_table):
    """
    Treina o classificador e o usa para simular a temporada final,
    calculando a Acurácia da Partida (AccuracyM) e o RMSE da Tabela.
    """
    
    # --- Divisão Cronológica ---
    # Treinar em 2020-2023, Testar em 2024
    df_train = df_processed[df_processed['season'] < 2024]
    df_test = df_processed[df_processed['season'] == 2024]

    X_train = df_train[diff_features]
    y_train = df_train['result']
    X_test = df_test[diff_features]
    y_test_actual = df_test['result'] # Resultados reais das partidas

    if len(X_train) == 0 or len(X_test) == 0:
        print("Erro: Falha na divisão de treino/teste. Verifique os dados.")
        return

    # --- Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Treinar o Classificador (XGBoost) ---
    print(f"\nTreinando Classificador de Partidas em {len(X_train)} jogos (2020-2023)...")
    
    # Otimização com GridSearchCV para o classificador de 3 classes
    print("Iniciando GridSearchCV para XGBClassifier (pode demorar)...")
    param_grid_xgb = {
        'n_estimators': [100, 150],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.7, 1.0]
    }
    xgb_model_base = XGBClassifier(
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='mlogloss' # Métrica para classificação multiclasse
    )
    
    grid_search_xgb = GridSearchCV(
        estimator=xgb_model_base, 
        param_grid=param_grid_xgb, 
        cv=5, # 5-fold cross-validation
        scoring='accuracy', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search_xgb.fit(X_train_scaled, y_train)
    
    print(f"Melhores parâmetros encontrados para XGBClassifier: {grid_search_xgb.best_params_}")
    
    # Usar o MELHOR modelo encontrado pelo grid search
    best_xgb_model = grid_search_xgb.best_estimator_

    # --- Avaliação 1: Acurácia da Partida (AccuracyM) ---
    y_pred_matches = best_xgb_model.predict(X_test_scaled)
    accuracy_m = accuracy_score(y_test_actual, y_pred_matches)
    
    print("\n" + "="*50)
    print("--- RESULTADOS DO PROTÓTIPO 3 (SIMULAÇÃO DE PARTIDAS) ---")
    print("="*50)
    print(f"\nAcurácia da Partida (AccuracyM): {accuracy_m * 100:.2f}%")
    print("Benchmark do Artigo (AccuracyM): 57%")
    print("\nRelatório de Classificação de Partidas (Season 2024):\n")
    print(classification_report(y_test_actual, y_pred_matches, target_names=['Vitória Fora (0)', 'Empate (1)', 'Vitória Casa (2)']))

    # --- Avaliação 2: Simulação da Tabela e RMSE ---
    print("Calculando RMSE da Tabela Final (simulando partidas)...")
    
    # Adicionar previsões ao dataframe de teste
    df_test_sim = df_test.copy()
    df_test_sim['prediction'] = y_pred_matches
    
    # Calcular pontos simulados
    simulated_points = {}
    
    for _, row in df_test_sim.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        prediction = row['prediction']
        
        # Inicializar times na tabela se não existirem
        if home_team not in simulated_points: simulated_points[home_team] = 0
        if away_team not in simulated_points: simulated_points[away_team] = 0
        
        # Atribuir pontos
        if prediction == 2: # Vitória Casa
            simulated_points[home_team] += 3
        elif prediction == 1: # Empate
            simulated_points[home_team] += 1
            simulated_points[away_team] += 1
        elif prediction == 0: # Vitória Fora
            simulated_points[away_team] += 3
            
    # Criar tabela simulada
    df_sim_table = pd.DataFrame(
        list(simulated_points.items()), columns=['team_name', 'pontos_simulados']
    )
    df_sim_table = df_sim_table.sort_values(by='pontos_simulados', ascending=False)
    
    # Obter tabela real da nossa tabela de stats T-1
    df_real_table = stats_lookup_table[
        stats_lookup_table['season'] == 2024
    ][['team_name', 'points']]
    
    # Juntar tabelas (usando 'inner' merge para caso de times faltantes como 'Ipswich')
    df_final_table = pd.merge(
        df_real_table.rename(columns={'points': 'pontos_reais'}),
        df_sim_table,
        on='team_name',
        how='inner' # 'inner' garante que só comparamos times presentes em AMBOS os dataframes
    )
    
    # Calcular RMSE Final
    rmse = np.sqrt(mean_squared_error(df_final_table['pontos_reais'], df_final_table['pontos_simulados']))
    
    print(f"\nRMSE da Tabela Final (Simulação): {rmse:.2f} pontos")
    print("Benchmark do Artigo (RMSE): 9.0 pontos")
    
    print("\nTabela Final Real vs. Simulada (Season 2024):\n")
    # Adicionar posições
    df_final_table_sorted = df_final_table.sort_values(by='pontos_simulados', ascending=False)
    df_final_table_sorted['pos_simulada'] = range(1, len(df_final_table_sorted) + 1)
    print(df_final_table_sorted[['pos_simulada', 'team_name', 'pontos_reais', 'pontos_simulados']])

# --- Função Principal ---
if __name__ == "__main__":
    # 1. Obter a tabela de stats T-1 (do 'data_accp_with_posse (1).csv')
    stats_lookup = get_stats_lookup_table()
    
    if stats_lookup is not None:
        # 2. Carregar dados das partidas (do 'PremierLeague.csv') e criar features de diferença
        df_processed, diff_features = load_and_prepare_match_data(stats_lookup)
        
        if df_processed is not None:
            # 3. Rodar a simulação
            run_simulation(df_processed, diff_features, stats_lookup)