import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURAÇÕES ---
FILE_PATH = 'data_accp_with_posse_webscrap.csv'

FEATURES = [
    'points', 'position', 'points_per_match', 'xG', 'xGA', 'npxG', 'npxGA', 'xPts', 
    'xPts_per_match', 'status_time_points', 'Passes', 'long_pass_accuracy', 
    'cross_accuracy', 'dribble_accuracy', 'teams_salary', 'Shot_Difference', 
    'SOT_Difference', 'gf_xG_performance_final', 'posse_percent'
]

def load_and_prep_data(filepath):
    print(f"Carregando dados de: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("ERRO: Arquivo não encontrado.")
        return pd.DataFrame()

    # 1. Garantir que todas as FEATURES existam
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    if 'season' in df.columns:
        df['season'] = pd.to_numeric(df['season'], errors='coerce').fillna(0)

    df = df.sort_values(by=['team_name', 'season'])

    # 2. Criar Features de Lag (T-1)
    print("Criando features de Lag (T-1)...")
    lag_cols = []
    for col in FEATURES:
        lag_col = f'{col}_T-1'
        df[lag_col] = df.groupby('team_name')[col].shift(1)
        lag_cols.append(lag_col)

    # Filtrar dados (2020 em diante)
    df_model = df[df['season'] >= 2020].copy()
    
    # Preencher Lags vazios para times promovidos
    seasons = sorted(df_model['season'].unique())
    df_filled_list = []

    for season in seasons:
        df_current = df_model[df_model['season'] == season].copy()
        prev_season = season - 1
        df_prev = df[df['season'] == prev_season]
        
        if not df_prev.empty:
            relegated = df_prev[df_prev['position'] > 17]
            if not relegated.empty:
                relegated_stats = relegated[FEATURES].mean()
                fill_values = {f'{c}_T-1': relegated_stats[c] for c in FEATURES}
                df_current.fillna(value=fill_values, inplace=True)
        
        df_filled_list.append(df_current)

    if not df_filled_list:
        print("Erro: Nenhum dado processado.")
        return pd.DataFrame()

    df_final = pd.concat(df_filled_list, ignore_index=True)
    df_final[lag_cols] = df_final[lag_cols].fillna(0)
    
    print(f"Dataset pronto: {len(df_final)} linhas.")
    return df_final

def get_zone_label(rank):
    """Define a zona baseada na posição."""
    if rank <= 4: return "UCL"
    if rank == 5: return "UEL"
    if 6 <= rank <= 7: return "UECL"
    if rank >= 18: return "Relegation"
    return "MidTable"

def plot_results(results_df):
    print("\nGerando gráficos...")
    sns.set_theme(style="whitegrid")
    
    # 1. Comparação de Pontos
    plt.figure(figsize=(14, 7))
    x = range(len(results_df))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], results_df['predicted_points'], width, label='Previsto', color='#1f77b4', alpha=0.8)
    plt.bar([i + width/2 for i in x], results_df['points'], width, label='Real', color='#ff7f0e', alpha=0.8)
    
    plt.xlabel('Times')
    plt.ylabel('Pontos')
    plt.title('Comparação: Pontos Previstos vs Reais - Premier League 2024')
    plt.xticks(x, results_df['team_name'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparacao_pontos.png')
    print("-> Gráfico salvo: comparacao_pontos.png")

    # 2. Scatter Plot
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=results_df, x='points', y='predicted_points', s=100, hue='team_name', legend=False)
    
    min_val = min(results_df['points'].min(), results_df['predicted_points'].min()) - 5
    max_val = max(results_df['points'].max(), results_df['predicted_points'].max()) + 5
    plt.plot([min_val, max_val], [min_val, max_val], ls="--", c=".3", label='Previsão Perfeita')
    
    for i in range(len(results_df)):
        row = results_df.iloc[i]
        if abs(row['predicted_points'] - row['points']) > 10:
            plt.text(row['points']+0.5, row['predicted_points'], row['team_name'], fontsize=9)
            
    plt.title('Correlação: Pontos Reais vs Previstos')
    plt.xlabel('Pontos Reais')
    plt.ylabel('Pontos Previstos')
    plt.legend()
    plt.tight_layout()
    plt.savefig('scatter_pontos.png')
    print("-> Gráfico salvo: scatter_pontos.png")

    # 3. Erro Residual
    results_df['error'] = results_df['predicted_points'] - results_df['points']
    plt.figure(figsize=(14, 7))
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in results_df['error']]
    sns.barplot(x='team_name', y='error', data=results_df, palette=colors)
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Erro Residual (Previsto - Real)')
    plt.ylabel('Diferença de Pontos (Previsto - Real)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('erro_residual.png')
    print("-> Gráfico salvo: erro_residual.png")

def calculate_baseline_metrics(df):
    df_2024 = df[df['season'] == 2024].copy()
    
    if df_2024.empty:
        print("Aviso: Sem dados de 2024 para calcular o baseline.")
        return

    df_2024['baseline_points'] = df_2024['points_T-1']
    df_2024 = df_2024.sort_values(by='baseline_points', ascending=False)
    df_2024['baseline_predicted_rank'] = range(1, len(df_2024) + 1)
    
    real_ranking = df_2024[['team_name', 'points']].sort_values(by='points', ascending=False)
    real_ranking['actual_rank'] = range(1, len(real_ranking) + 1)
    
    results = df_2024.merge(real_ranking[['team_name', 'actual_rank']], on='team_name')

    rmse_base = np.sqrt(mean_squared_error(results['points'], results['baseline_points']))
    
    accM_base = (abs(results['baseline_predicted_rank'] - results['actual_rank']) <= 2).mean()
    
    results['actual_zone'] = results['actual_rank'].apply(get_zone_label)
    results['baseline_zone'] = results['baseline_predicted_rank'].apply(get_zone_label)
    accQ_base = (results['baseline_zone'] == results['actual_zone']).mean()
    
    print("\n" + "="*40)
    print("      METRICAS DO BASELINE (ANO ANTERIOR)")
    print("="*40)
    print(f"RMSE Baseline: {rmse_base:.2f}")
    print(f"AccM Baseline: {accM_base*100:.1f}%")
    print(f"AccQ Baseline: {accQ_base*100:.1f}%")
    print("="*40)

def train_and_optimize(df):
    train_mask = (df['season'] >= 2020) & (df['season'] < 2024)
    test_mask = (df['season'] == 2024)
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    if df_train.empty or df_test.empty:
        print("Erro: Dados insuficientes.")
        return pd.DataFrame()

    X_cols = [f'{col}_T-1' for col in FEATURES]
    y_col = 'points'

    X_train = df_train[X_cols]
    y_train = df_train[y_col]
    X_test = df_test[X_cols]
    y_test = df_test[y_col]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nOtimizando modelo XGBoost...")
    xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
        'subsample': [0.8, 1.0]
    }
    
    grid = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_
    print(f"Melhores parâmetros: {grid.best_params_}")

    df_test['predicted_points'] = best_model.predict(X_test_scaled)

    df_test = df_test.sort_values(by='predicted_points', ascending=False)
    df_test['predicted_rank'] = range(1, len(df_test) + 1)
    
    real_ranking = df_test[['team_name', 'points']].sort_values(by='points', ascending=False)
    real_ranking['actual_rank'] = range(1, len(real_ranking) + 1)
    
    results = df_test.merge(real_ranking[['team_name', 'actual_rank']], on='team_name')
    
    results['actual_zone'] = results['actual_rank'].apply(get_zone_label)
    results['predicted_zone'] = results['predicted_rank'].apply(get_zone_label)

    accP = (results['predicted_rank'] == results['actual_rank']).mean()
    accM = (abs(results['predicted_rank'] - results['actual_rank']) <= 2).mean()
    accQ = (results['predicted_zone'] == results['actual_zone']).mean()
    rmse = np.sqrt(mean_squared_error(results['points'], results['predicted_points']))
    mae_rank = mean_absolute_error(results['actual_rank'], results['predicted_rank'])

    print("\n" + "="*60)
    print("               RELATÓRIO FINAL (MODELO PROPOSTO)")
    print("="*60)
    print(f"RMSE (Erro de Pontos): {rmse:.2f}")
    print(f"MAE (Erro Médio de Posição): {mae_rank:.2f}")
    print("-" * 30)
    print(f"accP (Acerto Posição Exata):  {accP*100:.1f}%")
    print(f"accM (Acerto Margem <= 2):    {accM*100:.1f}%")
    print(f"accQ (Acerto Qualificação):   {accQ*100:.1f}%")
    print("="*60)
    
    print("\n--- Tabela Comparativa (2024) ---")
    cols = ['predicted_rank', 'team_name', 'predicted_points', 'actual_rank', 'points', 'predicted_zone', 'actual_zone']
    display_df = results[cols].rename(columns={'points': 'actual_pts'})
    print(display_df.to_string(index=False))

    return results

if __name__ == "__main__":
    df_processed = load_and_prep_data(FILE_PATH)
    if not df_processed.empty:
        results_model = train_and_optimize(df_processed)
        
        if not results_model.empty:
            plot_results(results_model)
            
        calculate_baseline_metrics(df_processed)
    else:
        print("Não foi possível processar os dados.")