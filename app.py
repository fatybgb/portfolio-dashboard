import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
#   CONFIG GÉNÉRALE
# =========================

DATA_DIR = Path(__file__).parent / "data_gp"

st.set_page_config(
    page_title="Portfolio Analytics Dashboard",
    layout="wide",
    page_icon="📊",
)

# ============ STYLE GLOBAL ============

st.markdown(
    """
    <style>
    /* Police globale */
    html, body, [class*="css"] {
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 14px;
    }

    /* En-tête principale */
    .top-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.7rem 0 1rem 0;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .top-bar-left {
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }
    .app-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #111827;
    }
    .app-subtitle {
        color: #6B7280;
        font-size: 0.92rem;
    }
    .brand-pill {
        padding: 0.1rem 0.6rem;
        border-radius: 999px;
        border: 1px solid #E5E7EB;
        font-size: 0.75rem;
        color: #6B7280;
    }

    /* Cartes KPI */
    .kpi-card {
        background: #FFFFFF;
        border-radius: 0.9rem;
        border: 1px solid #E5E7EB;
        padding: 0.9rem 1rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        color: #6B7280;
        letter-spacing: 0.06em;
        margin-bottom: 0.1rem;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 600;
        color: #111827;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #9CA3AF;
        margin-top: 0.1rem;
    }

    /* Padding des tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
#   FONCTIONS UTILITAIRES
# =========================

@st.cache_data
def charger_prix():
    """Charge et prépare les séries de prix pour tous les actifs."""

    # Actions Maroc : IAM
    iam = pd.read_excel(DATA_DIR / "IAM.xlsx")
    iam["Date"] = pd.to_datetime(iam["Séance"], dayfirst=True)
    iam = iam[["Date", "Cours ajusté"]].rename(columns={"Cours ajusté": "IAM"})

    # Actions Maroc : CIH
    cih = pd.read_excel(DATA_DIR / "CIH.xlsx")
    cih["Date"] = pd.to_datetime(cih["Séance"], dayfirst=True)
    cih = cih[["Date", "Cours ajusté"]].rename(columns={"Cours ajusté": "CIH"})

    # Apple
    aapl = pd.read_excel(DATA_DIR / "Apple_data_1an.xlsx")
    aapl = aapl.iloc[2:].copy()
    aapl["Date"] = pd.to_datetime(aapl["Price"])
    aapl = aapl[["Date", "Close"]].rename(columns={"Close": "AAPL"})
    aapl["AAPL"] = aapl["AAPL"].astype(float)

    # Tesla
    tsla = pd.read_excel(DATA_DIR / "Tesla_data_1an.xlsx")
    tsla = tsla.iloc[2:].copy()
    tsla["Date"] = pd.to_datetime(tsla["Price"])
    tsla = tsla[["Date", "Close"]].rename(columns={"Close": "TSLA"})
    tsla["TSLA"] = tsla["TSLA"].astype(float)

    # Or (futures)
    gold = pd.read_csv(DATA_DIR / "gold_futures_history.csv")
    gold = gold.iloc[2:].copy()
    gold["Date"] = pd.to_datetime(gold["Price"])
    gold = gold[["Date", "Close"]].rename(columns={"Close": "GOLD"})
    gold["GOLD"] = gold["GOLD"].astype(float)

    # Obligation 5Y
    oblig = pd.read_csv(DATA_DIR / "obligation_5Y_sample.csv")
    oblig["Date"] = pd.to_datetime(oblig["Date"])
    oblig = oblig.rename(columns={"Obligation_5Y": "OBLIG"})[["Date", "OBLIG"]]

    # Fusion
    df = (
        iam.merge(cih, on="Date", how="outer")
           .merge(aapl, on="Date", how="outer")
           .merge(tsla, on="Date", how="outer")
           .merge(gold, on="Date", how="outer")
           .merge(oblig, on="Date", how="outer")
    )

    df = df.sort_values("Date").set_index("Date")
    df = df.ffill().dropna()
    return df


def calculer_rendements(df_prix: pd.DataFrame) -> pd.DataFrame:
    return df_prix.pct_change().dropna()


def stats_annuelles(rendements: pd.DataFrame):
    mu = rendements.mean() * 252
    sigma = rendements.std() * np.sqrt(252)
    return mu, sigma


def portefeuille_variance_min(rendements: pd.DataFrame, actifs: list):
    cov = rendements[actifs].cov() * 252
    ones = np.ones(len(actifs))
    inv_cov = np.linalg.inv(cov.values)
    w = inv_cov.dot(ones) / (ones.T @ inv_cov @ ones)
    w = pd.Series(w, index=actifs)

    mu = rendements[actifs].mean() * 252
    r_p = float((mu * w).sum())
    sigma_p = float(np.sqrt(w.T @ cov @ w))
    return w, r_p, sigma_p, cov


def simuler_frontiere(rendements: pd.DataFrame, actifs: list,
                      n_portfolios: int = 1500, r_f: float = 0.0):
    cov = rendements[actifs].cov() * 252
    mu = rendements[actifs].mean() * 252

    vols, rets, sharpes = [], [], []
    for _ in range(n_portfolios):
        w = np.random.rand(len(actifs))
        w /= w.sum()
        r_p = float(np.dot(w, mu))
        s_p = float(np.sqrt(np.dot(w, np.dot(cov, w))))
        sharpe = (r_p - r_f) / s_p if s_p != 0 else 0.0
        vols.append(s_p)
        rets.append(r_p)
        sharpes.append(sharpe)

    return np.array(vols), np.array(rets), np.array(sharpes)


# =========================
#   APPLICATION
# =========================

def main():
    # ----- HEADER PRO -----
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.markdown(
            """
            <div class="top-bar">
              <div class="top-bar-left">
                <div class="app-title">Portfolio Analytics Dashboard</div>
                <div class="app-subtitle">
                  Outil d’analyse de portefeuilles multi-actifs (actions marocaines, US, obligations et or).
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_head2:
        st.markdown(
            """
            <div class="top-bar" style="justify-content: flex-end;">
                <div class="brand-pill">Corporate Finance • Internal Tool</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ----- DONNÉES -----
    df_prix = charger_prix()
    df_rend = calculer_rendements(df_prix)

    # ----- SIDEBAR (configuration) -----
    st.sidebar.title("Paramètres du portefeuille")

    tous_les_actifs = ["IAM", "CIH", "AAPL", "TSLA", "OBLIG", "GOLD"]

    actifs = st.sidebar.multiselect(
        "Actifs à inclure :",
        options=tous_les_actifs,
        default=["IAM", "CIH", "GOLD"],
        help="Sélectionnez la combinaison d’actifs que vous souhaitez analyser."
    )

    r_f = st.sidebar.number_input(
        "Taux sans risque (%)",
        value=2.0,
        min_value=-5.0,
        max_value=15.0,
        step=0.25,
    ) / 100.0

    n_portfolios = st.sidebar.slider(
        "Nombre de portefeuilles simulés",
        min_value=300,
        max_value=5000,
        value=1500,
        step=100,
    )

    afficher_prix = st.sidebar.checkbox("Afficher les dernières observations de prix")

    if len(actifs) < 2:
        st.warning("Veuillez sélectionner **au moins deux actifs** pour construire un portefeuille.")
        st.stop()

    # ----- ANALYSE PRINCIPALE (MVP) -----
    w_mvp, r_mvp, s_mvp, cov = portefeuille_variance_min(df_rend, actifs)
    sharpe_mvp = (r_mvp - r_f) / s_mvp if s_mvp != 0 else 0.0

    # Cartes KPI
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Rendement attendu</div>
              <div class="kpi-value">{r_mvp*100:.2f} %</div>
              <div class="kpi-sub">Portefeuille à variance minimum</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Risque (volatilité)</div>
              <div class="kpi-value">{s_mvp*100:.2f} %</div>
              <div class="kpi-sub">Sur une base annualisée</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Ratio de Sharpe</div>
              <div class="kpi-value">{sharpe_mvp:.2f}</div>
              <div class="kpi-sub">Taux sans risque : {r_f*100:.2f} %</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if afficher_prix:
        st.markdown("### Dernières observations de prix")
        st.dataframe(df_prix[actifs].tail(10), use_container_width=True)

    # ----- ONGLETs (comme menu) -----
    tab_overview, tab_corr, tab_opt, tab_scenarios = st.tabs(
        ["Vue d'ensemble", "Corrélations & risque", "Optimisation", "Scénarios types"]
    )

    # ===== TAB 1 : Vue d'ensemble =====
    with tab_overview:
        st.subheader("Statistiques annuelles par actif")
        mu, sigma = stats_annuelles(df_rend[actifs])
        stats_actifs = pd.DataFrame({
            "Rendement annuel (%)": np.round(mu * 100, 2),
            "Risque (volatilité annuelle, %)": np.round(sigma * 100, 2),
        })
        st.dataframe(stats_actifs, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            ax1.bar(stats_actifs.index, stats_actifs["Rendement annuel (%)"])
            ax1.set_title("Rendements annuels (%)")
            ax1.set_ylabel("%")
            plt.xticks(rotation=25)
            st.pyplot(fig1, use_container_width=False)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.bar(stats_actifs.index, stats_actifs["Risque (volatilité annuelle, %)"])
            ax2.set_title("Risque annuel (%)")
            ax2.set_ylabel("%")
            plt.xticks(rotation=25)
            st.pyplot(fig2, use_container_width=False)

        st.caption(
            "Cette vue permet d’identifier rapidement les actifs les plus rémunérateurs "
            "et les plus volatils."
        )

    # ===== TAB 2 : Corrélations & risque =====
    with tab_corr:
        st.subheader("Matrice de corrélation")

        corr = df_rend[actifs].corr()
        c1, c2 = st.columns([1.3, 1])

        with c1:
            fig_corr, ax_corr = plt.subplots(figsize=(4.8, 3.8))
            im = ax_corr.imshow(corr.values, vmin=-1, vmax=1)
            ax_corr.set_xticks(range(len(actifs)))
            ax_corr.set_yticks(range(len(actifs)))
            ax_corr.set_xticklabels(actifs, rotation=45, ha="right")
            ax_corr.set_yticklabels(actifs)
            for i in range(len(actifs)):
                for j in range(len(actifs)):
                    ax_corr.text(
                        j, i, f"{corr.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=8, color="black"
                    )
            ax_corr.set_title("Corrélations entre actifs")
            plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
            st.pyplot(fig_corr, use_container_width=False)

        with c2:
            st.write("Corrélations numériques")
            st.dataframe(np.round(corr, 2), use_container_width=True)
            st.caption(
                "Des corrélations faibles ou négatives (proches de 0 ou < 0) "
                "sont intéressantes pour la diversification."
            )

    # ===== TAB 3 : Optimisation =====
    with tab_opt:
        st.subheader("Portefeuille à variance minimum (MVP)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Allocations optimales (%)**")
            alloc = pd.DataFrame(
                {"Allocation (%)": np.round(w_mvp * 100, 2)}
            )
            st.dataframe(alloc, use_container_width=True)

        with col2:
            tableau_mvp = pd.DataFrame(
                {
                    "Rendement annuel (%)": [np.round(r_mvp * 100, 2)],
                    "Risque annuel (%)": [np.round(s_mvp * 100, 2)],
                    "Sharpe": [np.round(sharpe_mvp, 2)],
                },
                index=["Portefeuille MVP"],
            )
            st.markdown("**Synthèse du portefeuille MVP**")
            st.table(tableau_mvp)

        st.markdown("---")
        st.subheader("Frontière efficiente (simulation)")

        vols, rets, sharpes = simuler_frontiere(
            df_rend, actifs, n_portfolios=n_portfolios, r_f=r_f
        )

        fig_front, ax_front = plt.subplots(figsize=(6, 4))
        scatter = ax_front.scatter(vols, rets, c=sharpes, s=8, alpha=0.7)
        ax_front.scatter(s_mvp, r_mvp, marker="*", s=140,
                         edgecolor="black", label="MVP")
        ax_front.set_xlabel("Risque (volatilité annuelle)")
        ax_front.set_ylabel("Rendement annuel")
        ax_front.set_title("Portefeuilles simulés")
        ax_front.legend()
        plt.colorbar(scatter, ax=ax_front, label="Sharpe (approx.)")
        st.pyplot(fig_front, use_container_width=False)

    # ===== TAB 4 : Scénarios types =====
    with tab_scenarios:
        st.subheader("Scénarios prédéfinis (portefeuilles MVP)")

        scenarios = {
            "Maroc (IAM, CIH)": ["IAM", "CIH"],
            "Maroc + Or": ["IAM", "CIH", "GOLD"],
            "Maroc + US Tech": ["IAM", "CIH", "AAPL", "TSLA"],
            "Diversifié complet": ["IAM", "CIH", "AAPL", "TSLA", "OBLIG", "GOLD"],
        }

        lignes = []
        for nom, act in scenarios.items():
            act_ok = [a for a in act if a in df_rend.columns]
            if len(act_ok) < 2:
                continue
            w, r, s, _ = portefeuille_variance_min(df_rend, act_ok)
            lignes.append(
                {
                    "Scénario": nom,
                    "Nb d’actifs": len(act_ok),
                    "Rendement MVP (%)": np.round(r * 100, 2),
                    "Risque MVP (%)": np.round(s * 100, 2),
                }
            )

        if lignes:
            df_comp = pd.DataFrame(lignes).set_index("Scénario")
            st.dataframe(df_comp, use_container_width=True)
            st.caption(
                "Comparaison de quelques profils types de portefeuilles. "
                "Utile pour une présentation à un comité d’investissement."
            )
        else:
            st.info("Aucun scénario prédéfini valide pour l’instant.")


if __name__ == "__main__":
    main()
