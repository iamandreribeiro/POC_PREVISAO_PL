import soccerdata as sd
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURAÇÕES ---
LEAGUE = "ENG-Premier League"

start_year = 17 
end_year = 24   
SEASONS = [f"{y:02d}-{y+1:02d}" for y in range(start_year, end_year + 1)]

print(f"Iniciando scraper para {LEAGUE} nas temporadas: {SEASONS}")
fb = sd.FBref(leagues=LEAGUE, seasons=SEASONS)

print("Baixando estatísticas (Performance, Passing, Possession, Defense)...")
try:
    std_stats = fb.read_team_season_stats(stat_type="standard")
    pass_stats = fb.read_team_season_stats(stat_type="passing")
    poss_stats = fb.read_team_season_stats(stat_type="possession")
    opp_stats = fb.read_team_season_stats(stat_type="standard", opponent_stats=True)
except Exception as e:
    print(f"Erro ao baixar estatísticas: {e}")
    exit()

print("Baixando calendário...")
schedule = fb.read_schedule()

if 'home_score' not in schedule.columns or 'away_score' not in schedule.columns:
    print("Ajustando colunas de placar...")
    if 'score' in schedule.columns:
        schedule[['home_score', 'away_score']] = schedule['score'].astype(str).str.split(r'[–-]', expand=True)
        schedule['home_score'] = pd.to_numeric(schedule['home_score'], errors='coerce')
        schedule['away_score'] = pd.to_numeric(schedule['away_score'], errors='coerce')
    else:
        schedule['home_score'] = np.nan
        schedule['away_score'] = np.nan

def calculate_standings(schedule_df):
    standings_list = []
    played_games = schedule_df.dropna(subset=['home_score', 'away_score']).copy()
    
    if played_games.empty:
        return pd.DataFrame()

    if 'league' not in played_games.columns:
        played_games = played_games.reset_index()

    for (league, season), games in played_games.groupby(['league', 'season']):
        teams = set(games['home_team']).union(set(games['away_team']))
        table = {team: {'played': 0, 'won': 0, 'draw': 0, 'lost': 0, 'gf': 0, 'ga': 0, 'points': 0} for team in teams}
        
        for _, game in games.iterrows():
            h_team, a_team = game['home_team'], game['away_team']
            try:
                h_score = int(game['home_score'])
                a_score = int(game['away_score'])
            except ValueError:
                continue 
            
            table[h_team]['played'] += 1
            table[h_team]['gf'] += h_score
            table[h_team]['ga'] += a_score
            table[a_team]['played'] += 1
            table[a_team]['gf'] += a_score
            table[a_team]['ga'] += h_score
            
            if h_score > a_score:
                table[h_team]['won'] += 1
                table[h_team]['points'] += 3
                table[a_team]['lost'] += 1
            elif a_score > h_score:
                table[a_team]['won'] += 1
                table[a_team]['points'] += 3
                table[h_team]['lost'] += 1
            else:
                table[h_team]['draw'] += 1
                table[h_team]['points'] += 1
                table[a_team]['draw'] += 1
                table[a_team]['points'] += 1
        
        df_season = pd.DataFrame.from_dict(table, orient='index').reset_index().rename(columns={'index': 'team'})
        df_season['gd'] = df_season['gf'] - df_season['ga']
        df_season = df_season.sort_values(by=['points', 'gd', 'gf'], ascending=False)
        df_season['position'] = range(1, len(df_season) + 1)
        df_season['league'] = league
        df_season['season'] = season
        standings_list.append(df_season)
        
    if not standings_list:
        return pd.DataFrame()
        
    return pd.concat(standings_list, ignore_index=True)

df_standings = calculate_standings(schedule)

print("Consolidando dados...")

def flatten_and_clean(df):
    df = df.reset_index()
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            name = '_'.join([str(c) for c in col if str(c) != '']).strip()
        else:
            name = str(col).strip()
        new_cols.append(name)
    df.columns = new_cols
    return df

std_stats = flatten_and_clean(std_stats)
pass_stats = flatten_and_clean(pass_stats)
poss_stats = flatten_and_clean(poss_stats)
opp_stats = flatten_and_clean(opp_stats)

def safe_select(df, col_map):
    subset = pd.DataFrame(index=df.index)
    for key in ['league', 'season', 'team']:
        subset[key] = df[key]
    for original_col, new_col in col_map.items():
        if original_col in df.columns:
            subset[new_col] = df[original_col]
        else:
            subset[new_col] = np.nan
    return subset

map_std = {
    'Per 90 Minutes_Gls': 'Gls_per_90m',
    'Per 90 Minutes_Ast': 'Ast_90m',
    'Expected_xG': 'xG_total',
    'Expected_npxG': 'npxG_total',
    'Poss': 'posse_percent',
    'Performance_Gls': 'gf_check'
}
df_main = safe_select(std_stats, map_std)

map_opp = {
    'Per 90 Minutes_Gls': 'GA/90',
    'Expected_xG': 'xGA_total',
    'Expected_npxG': 'npxGA_total'
}
df_opp_clean = safe_select(opp_stats, map_opp)

map_pass = {
    'Total_Cmp': 'Passes',
    'Long_Cmp%': 'long_pass_accuracy'
}
df_pass_clean = safe_select(pass_stats, map_pass)

map_poss = {
    'Take-Ons_Succ%': 'dribble_accuracy'
}
df_poss_clean = safe_select(poss_stats, map_poss)

final_df = df_standings.merge(df_main, on=['league', 'season', 'team'], how='left')
final_df = final_df.merge(df_opp_clean, on=['league', 'season', 'team'], how='left')
final_df = final_df.merge(df_pass_clean, on=['league', 'season', 'team'], how='left')
final_df = final_df.merge(df_poss_clean, on=['league', 'season', 'team'], how='left')

def convert_season(s):
    try:
        return int("20" + str(s)[:2])
    except:
        return s

if not final_df.empty:
    final_df['season'] = final_df['season'].apply(convert_season)
    final_df['competition'] = 'PL'
    final_df['team_name'] = final_df['team']
    final_df['points_per_match'] = final_df['points'] / final_df['played']    
    final_df['xG'] = final_df['xG_total']
    final_df['xGA'] = final_df['xGA_total']
    final_df['npxG'] = final_df['npxG_total']
    final_df['npxGA'] = final_df['npxGA_total']    
    final_df['teams_salary'] = 0 
    final_df['status_time_points'] = 0
    final_df['xPts'] = 0
    final_df['xPts_per_match'] = 0
    final_df['cross_accuracy'] = 0
    final_df['Shot_Difference'] = 0
    final_df['SOT_Difference'] = 0  
    final_df['gf_xG_performance_final'] = final_df['gf'] - final_df['xG'].fillna(0)
    cols_order = [
        'competition', 'season', 'team_name', 
        'Gls_per_90m', 'Ast_90m', 'GA/90', 'position', 'played', 'won', 'draw', 'lost', 
        'gf', 'ga', 'gd', 'points', 'points_per_match', 
        'xG', 'xGA', 'npxG', 'npxGA', 
        'xPts', 'xPts_per_match', 'status_time_points',
        'Passes', 'long_pass_accuracy', 'cross_accuracy', 'dribble_accuracy', 
        'teams_salary', 'Shot_Difference', 'SOT_Difference', 
        'gf_xG_performance_final', 'posse_percent'
    ]

    for col in cols_order:
        if col not in final_df.columns:
            final_df[col] = 0

    final_export = final_df[cols_order]
    final_export = final_export.dropna(subset=['team_name', 'points'])

    output_file = "data_accp_with_posse_webscrap.csv"
    final_export.to_csv(output_file, index=False)
    print(f"\nSucesso! Arquivo '{output_file}' gerado com compatibilidade total.")
    print(f"Linhas: {len(final_export)} | Colunas: {len(final_export.columns)}")
    print(final_export.head())

else:
    print("O DataFrame final está vazio. Verifique se o download funcionou.")