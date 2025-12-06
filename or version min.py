import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CHARGEMENT ET NETTOYAGE ROBUSTE
# ==========================================
print("--- CHARGEMENT DE LA BASE DE DONNÉES COMPLÈTE ---")

files_config = {
    # MAROC
    'IAM.xlsx': {'ticker': 'IAM', 'type': 'maroc'},
    'CIH.xlsx': {'ticker': 'CIH', 'type': 'maroc'},
    # USA (ACTIONS)
    'Apple_data_1an.xlsx': {'ticker': 'AAPL', 'type': 'yahoo_xlsx'},
    'Tesla_data_1an.xlsx': {'ticker': 'TSLA', 'type': 'yahoo_xlsx'},
    # OBLIGATION
    'obligation_5Y_sample.csv': {'ticker': 'OBLIG', 'type': 'csv_oblig'},
    # OR
    'gold_futures_history.csv': {'ticker': 'GOLD', 'type': 'yahoo_csv'} 
}

all_data_frames = []

for filename, info in files_config.items():
    try:
        ticker = info['ticker']
        file_type = info['type']
        df_temp = pd.DataFrame()

        if file_type == 'maroc':
            df = pd.read_excel(filename)
            df_temp = df[['Séance', 'Cours ajusté']].copy()
            df_temp['Date'] = pd.to_datetime(df_temp['Séance'], dayfirst=True)
            df_temp.rename(columns={'Cours ajusté': ticker}, inplace=True)
            
        elif file_type == 'yahoo_xlsx': 
            df = pd.read_excel(filename, header=0, skiprows=[1, 2]) 
            df_temp = df[['Price', 'Close']].copy()
            df_temp['Date'] = pd.to_datetime(df_temp['Price'])
            df_temp.rename(columns={'Close': ticker}, inplace=True)

        elif file_type == 'yahoo_csv':
            df = pd.read_csv(filename, header=0, skiprows=[1, 2])
            df_temp = df.iloc[:, [0, 1]].copy() 
            df_temp.columns = ['Date', ticker]
            df_temp['Date'] = pd.to_datetime(df_temp['Date'])
            
        elif file_type == 'csv_oblig':
            df = pd.read_csv(filename)
            df_temp = df[['Date', 'Obligation_5Y']].copy()
            df_temp['Date'] = pd.to_datetime(df_temp['Date']).dt.normalize()
            df_temp.rename(columns={'Obligation_5Y': ticker}, inplace=True)

        # Nettoyage
        df_temp.set_index('Date', inplace=True)
        df_temp = df_temp[~df_temp.index.duplicated(keep='first')]
        df_temp = df_temp[[ticker]]
        
        all_data_frames.append(df_temp)

    except Exception as e:
        print(f"⚠️ Erreur sur {filename} : {e}")

# Fusion et Nettoyage Final
global_df = pd.concat(all_data_frames, axis=1, join='outer')
global_df = global_df.ffill().dropna()

# Force numeric pour éviter les erreurs de texte
global_df = global_df.apply(pd.to_numeric, errors='coerce')
global_df = global_df.dropna()

global_returns = global_df.pct_change().dropna()
print(f"\nBase de données prête : {len(global_returns)} jours communs.")


# ==========================================
# 2. FONCTION D'ANALYSE (AVEC NOUVEL AFFICHAGE)
# ==========================================
def analyser_scenario(titre, actifs):
    print("\n" + "━"*80)
    print(f"   📘 {titre}")
    print(f"   Actifs : {actifs}")
    print("━"*80)
    
    # A. Données
    returns = global_returns[actifs]
    trading_days = 252
    
    # B. Métriques Individuelles
    mean_ret = returns.mean() * trading_days
    cov_mat = returns.cov() * trading_days
    corr_mat = returns.corr()
    vol_ind = returns.std() * np.sqrt(trading_days)
    
    print(f"\n--- 1. ANALYSE DES COMPOSANTS ---")
    print(f"{'ACTIF':<8} | {'RENDEMENT':<10} | {'RISQUE':<10}")
    print("-" * 40)
    for t in actifs:
        print(f"{t:<8} | {mean_ret[t]:<10.2%} | {vol_ind[t]:<10.2%}")

    print(f"\n--- 2. CORRÉLATIONS ---")
    print(corr_mat.round(2).to_string())

    # C. Simulation Monte Carlo
    print(f"\n--- 3. OPTIMISATION (Calcul en cours...) ---")
    sim_results = []
    for _ in range(15000):
        w = np.random.random(len(actifs))
        w /= np.sum(w)
        p_ret = np.dot(w, mean_ret)
        p_var = np.dot(w.T, np.dot(cov_mat, w))
        p_std = np.sqrt(p_var)
        p_sharpe = (p_ret - 0.03) / p_std
        sim_results.append((p_ret, p_std, p_sharpe, w))
    
    # Extraction
    results_array = np.array([x[:3] for x in sim_results])
    weights_array = np.array([x[3] for x in sim_results])
    
    # Meilleur Sharpe
    idx_max = results_array[:, 2].argmax()
    best_sharpe = results_array[idx_max]
    best_w = weights_array[idx_max]
    
    # Variance Minimale
    idx_min = results_array[:, 1].argmin()
    min_risk = results_array[idx_min]
    min_w = weights_array[idx_min]

    # D. Résultats Finaux (AMÉLIORÉS)
    print(f"\n🏆 MEILLEUR PORTEFEUILLE (Sharpe Max) :")
    print(f"   Sharpe : {best_sharpe[2]:.2f} | Rendement : {best_sharpe[0]:.2%} | Risque : {best_sharpe[1]:.2%}")
    print(f"   Allocation :")
    for i, asset in enumerate(actifs):
        print(f"      - {asset:<5}: {best_w[i]:.2%}")

    # --- C'EST ICI QUE J'AI AJOUTÉ L'AFFICHAGE DES POIDS ---
    print(f"\n🛡️ PORTEFEUILLE SÉCURITÉ (Variance Min) :")
    print(f"   Sharpe : {min_risk[2]:.2f} | Rendement : {min_risk[0]:.2%} | Risque : {min_risk[1]:.2%}")
    print(f"   Allocation :")
    for i, asset in enumerate(actifs):
        print(f"      - {asset:<5}: {min_w[i]:.2%}")
    # -------------------------------------------------------

    # E. Graphique
    plt.figure(figsize=(10, 6))
    plt.scatter(results_array[:, 1], results_array[:, 0], c=results_array[:, 2], cmap='viridis', s=2, alpha=0.5)
    plt.colorbar(label='Sharpe')
    plt.scatter(best_sharpe[1], best_sharpe[0], c='red', s=200, marker='*', label='Optimal')
    plt.scatter(min_risk[1], min_risk[0], c='blue', s=200, marker='*', label='Sécurité')
    plt.title(f"Frontière Efficiente : {titre}")
    plt.xlabel("Risque (Volatilité)")
    plt.ylabel("Rendement")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================================
# 3. EXÉCUTION
# ==========================================

analyser_scenario("SCÉNARIO 1 : Actions Maroc", ['IAM', 'CIH'])
analyser_scenario("SCÉNARIO 2 : Diversification Géo", ['IAM', 'CIH', 'AAPL', 'TSLA'])
analyser_scenario("SCÉNARIO 3 : + Obligation", ['IAM', 'CIH', 'AAPL', 'TSLA', 'OBLIG'])
analyser_scenario("SCÉNARIO 4 : + Or (Ultime)", ['IAM', 'CIH', 'AAPL', 'TSLA', 'OBLIG', 'GOLD'])