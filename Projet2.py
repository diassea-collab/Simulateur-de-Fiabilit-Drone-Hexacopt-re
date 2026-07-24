import streamlit as st
import numpy as np
from math import comb
import matplotlib.pyplot as plt

# =====================================================================
# CONFIGURATION GENERALE
# =====================================================================
st.set_page_config(
    page_title="Fiabilité Drone - Étude technique",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# STYLE — Thème clair avec textes grand format et très foncés
# =====================================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+Pro:wght@400;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* BASE & TEXTE GENERAL - Très grands et très foncés */
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
        font-size: 1.2rem !important; 
        color: #0f172a !important; 
    }
    
    p, li, label, div {
        color: #0f172a !important;
        font-weight: 500 !important;
    }

    h1, h2, h3, .hero-title, .card-title-lg { 
        font-family: 'Source Serif Pro', serif; 
        color: #092540 !important;
        font-weight: 800 !important;
    }

    .stApp {
        background-color: #f7f8fa;
        background-image:
            radial-gradient(circle at 100% 0%, rgba(14,124,134,0.05) 0%, transparent 40%),
            radial-gradient(circle at 0% 100%, rgba(29,53,87,0.04) 0%, transparent 40%);
        background-attachment: fixed;
    }

    /* SIDEBAR - Textes agrandis et plus sombres */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #cbd5e1;
    }
    section[data-testid="stSidebar"] * { 
        color: #0f172a !important; 
        font-size: 1rem !important;
        font-weight: 600 !important;
    }

    /* EN-TETE / HERO */
    .hero-wrap {
        padding: 44px 46px;
        border-radius: 10px;
        background: linear-gradient(135deg, #ffffff 0%, #f4f8f8 100%);
        border: 2px solid #cbd5e1;
        position: relative;
        overflow: hidden;
        margin-bottom: 30px;
    }
    .hero-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #085f67 !important;
        line-height: 1.15;
        margin-bottom: 0.6rem;
    }
    .hero-title-accent {
        width: 100px;
        height: 6px;
        background: linear-gradient(90deg, #0e7c86 0%, #1d3557 100%);
        border-radius: 3px;
        margin-bottom: 20px;
    }
    .hero-subtitle {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        max-width: 850px;
        line-height: 1.6;
    }
    .pill {
        display: inline-block; 
        padding: 6px 14px; 
        border-radius: 4px;
        background: #e2e8f0; 
        border: 1px solid #94a3b8;
        color: #0f172a !important; 
        font-size: 0.9rem !important; 
        font-weight: 800 !important; 
        letter-spacing: 0.03em;
        margin-right: 8px; 
        margin-bottom: 14px; 
        text-transform: uppercase;
    }

    /* CARTES & METRIQUES */
    .feature-card {
        background: #ffffff;
        border: 2px solid #cbd5e1;
        border-radius: 8px;
        padding: 26px;
        height: 100%;
    }
    .card-title-lg { 
        font-size: 1.5rem !important; 
        font-weight: 800 !important; 
        margin-bottom: 12px; 
        color: #0f172a !important; 
    }

    .big-badge {
        display: inline-block; 
        padding: 6px 16px; 
        border-radius: 4px;
        font-size: 1.2rem !important; 
        font-weight: 800 !important;
    }

    .metric-card {
        background: #ffffff;
        border: 2px solid #cbd5e1;
        border-top: 4px solid #1d3557;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .metric-label { 
        color: #334155 !important; 
        font-size: 1rem !important; 
        font-weight: 800 !important; 
        text-transform: uppercase; 
        letter-spacing: 0.05em; 
    }
    .metric-value { 
        color: #0f172a !important; 
        font-size: 2.5rem !important; 
        font-weight: 800 !important; 
        margin-top: 6px; 
        font-family: 'Source Serif Pro', serif; 
    }

    .formula-box {
        background: #e2e8f0;
        border-left: 4px solid #1d3557;
        padding: 14px 18px;
        border-radius: 4px;
        font-weight: 700 !important;
        font-size: 2rem !important;
        color: #0f172a !important;
        font-family: 'Source Serif Pro', serif;
    }

    hr { border-color: #cbd5e1 !important; border-width: 2px; }
    
    .stButton>button {
        background-color: #1d3557; 
        color: #ffffff !important; 
        border: none; 
        border-radius: 6px;
        font-weight: 700 !important; 
        font-size: 1.5rem !important;
        padding: 0.6em 1.4em;
    }
    .stButton>button:hover { background-color: #0f172a; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 1. MODELE METIER
# =====================================================================
class DroneReliability:
    """Modèle de calcul de fiabilité du drone."""

    def __init__(self, lambda_batt=1e-4, lambda_carte=0.5e-4,
                 beta_moteur=2.0, eta_moteur=1000.0, k=3, n=4):
        self.lam_batt = lambda_batt
        self.lam_carte = lambda_carte
        self.beta = beta_moteur
        self.eta = eta_moteur
        self.k = int(k)
        self.n = int(n)

    def r_weibull(self, t, beta=None, eta=None):
        b = self.beta if beta is None else beta
        e = self.eta if eta is None else eta
        return np.exp(-((t / e) ** b))

    def r_k_parmi_n(self, t, k=None, n=None, beta=None, eta=None):
        kk = self.k if k is None else k
        nn = self.n if n is None else n
        r_m = self.r_weibull(t, beta, eta)
        return sum(comb(nn, i) * (r_m ** i) * ((1 - r_m) ** (nn - i)) for i in range(kk, nn + 1))

    def r_propulsion(self, t):
        return self.r_k_parmi_n(t)

    def r_drone(self, t):
        return np.exp(-self.lam_batt * t) * np.exp(-self.lam_carte * t) * self.r_propulsion(t)

    def simuler_mc(self, t_mission=100, n_sims=100000, seed=42):
        rng = np.random.default_rng(seed)
        ok_batt = rng.exponential(1 / self.lam_batt, n_sims) > t_mission
        ok_carte = rng.exponential(1 / self.lam_carte, n_sims) > t_mission
        ok_prop = (self.eta * rng.weibull(self.beta, (n_sims, self.n)) > t_mission).sum(axis=1) >= self.k

        ok_drone = ok_batt & ok_carte & ok_prop
        echecs = ~ok_drone

        return {
            "r_mc": ok_drone.mean(),
            "echec_batt": int((~ok_batt & echecs).sum()),
            "echec_carte": int((ok_batt & ~ok_carte & echecs).sum()),
            "echec_prop": int((ok_batt & ok_carte & ~ok_prop & echecs).sum())
        }


def style_ax(ax, fig):
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors="#0f172a", labelsize=11)
    ax.xaxis.label.set_color("#0f172a")
    ax.xaxis.label.set_fontsize(12)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_color("#0f172a")
    ax.yaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontweight("bold")
    ax.title.set_color("#092540")
    ax.title.set_fontsize(14)
    ax.title.set_fontweight("bold")
    ax.grid(True, linestyle="--", alpha=0.35, color="#64748b")
    for spine in ax.spines.values():
        spine.set_color('#94a3b8')
        spine.set_linewidth(1.5)
    leg = ax.legend(facecolor='#ffffff', edgecolor='#94a3b8', labelcolor='#0f172a', fontsize=11)
    return leg


# =====================================================================
# 2. NAVIGATION
# =====================================================================
if 'page' not in st.session_state:
    st.session_state.page = "Accueil"

st.sidebar.markdown("## Fiabilité Drone Lab")
page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Simulateur & Analyse", "Comparaison & Sensibilité"],
    index=["Accueil", "Simulateur & Analyse", "Comparaison & Sensibilité"].index(st.session_state.page),
    key="nav_radio"
)
st.session_state.page = page
st.sidebar.markdown("---")
st.sidebar.caption("Mission : transport médical par drone hexacoptère · durée nominale 100h")


# =====================================================================
# PAGE 1 : ACCUEIL
# =====================================================================
if st.session_state.page == "Accueil":

    st.markdown("""
        <div class="hero-wrap">
            <span class="pill">MODÉLISATION</span><span class="pill">MONTE CARLO</span><span class="pill">WEIBULL & REDONDANCE</span>
            <div class="hero-title-accent"></div>
            <div class="hero-title">Fiabilité drone hexacoptère</div>
            <div class="hero-subtitle">
                Plateforme de modélisation prédictive et de simulation Monte Carlo pour l'évaluation
                de la fiabilité d'un drone hexacoptère médical, avec redondance active partielle
                sur la propulsion (k parmi n).
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- NAVIGATION CENTRALE ----------
    st.markdown("<h3 style='text-align:center; color:#092540; font-weight:800; font-size:1.5rem;'>Navigation</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#1e293b; font-size:1.2rem; font-weight:600; margin-bottom:24px;'>Choisissez un module pour poursuivre l'étude.</p>", unsafe_allow_html=True)

    spacer_l, nav1, nav2, nav3, spacer_r = st.columns([0.5, 2, 2, 2, 0.5])

    with nav1:
        st.markdown("""
            <div class="feature-card" style="border-left:5px solid #0e7c86;">
                <div style="color:#0e7c86; font-weight:800; font-size:0.95rem; letter-spacing:0.08em; margin-bottom:8px;">01 &nbsp;·&nbsp; VOUS ÊTES ICI</div>
                <div class="card-title-lg">Accueil</div>
                <p style="color:#0f172a; font-size:1.1rem; line-height:1.6; font-weight:600;">
                    Présentation du contexte, de la plateforme et des lois de fiabilité utilisées.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with nav2:
        st.markdown("""
            <div class="feature-card" style="border-left:5px solid #1d3557;">
                <div style="color:#1d3557; font-weight:800; font-size:0.95rem; letter-spacing:0.08em; margin-bottom:8px;">02 &nbsp;·&nbsp; MODULE</div>
                <div class="card-title-lg">Simulateur &amp; Analyse</div>
                <p style="color:#0f172a; font-size:1.1rem; line-height:1.6; font-weight:600;">
                    Calcul analytique et simulation de Monte Carlo pour une configuration donnée du drone.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with nav3:
        st.markdown("""
            <div class="feature-card" style="border-left:5px solid #6b4c6e;">
                <div style="color:#6b4c6e; font-weight:800; font-size:0.95rem; letter-spacing:0.08em; margin-bottom:8px;">03 &nbsp;·&nbsp; MODULE</div>
                <div class="card-title-lg">Comparaison &amp; Sensibilité</div>
                <p style="color:#0f172a; font-size:1.1rem; line-height:1.6; font-weight:600;">
                    Comparaison des architectures de redondance et étude de sensibilité aux paramètres β et η.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<p style='text-align:center; color:#334155; font-size:1.05rem; font-weight:700; margin-top:20px;'>Utilisez le menu latéral pour changer de module.</p>", unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # ---------- LOIS UTILISEES ----------
    st.markdown("<h2 style='margin-top:6px; font-size:2rem;'>Lois de fiabilité utilisées</h2>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("""
            <div class="feature-card" style="border-top:5px solid #0e7c86;">
                <div class="card-title-lg" style="color:#085f67 !important;">Taux constant (λ) — Exponentielle</div>
                <p style="font-size: 1.15rem; color:#0f172a; line-height: 1.6; font-weight:600;">
                    Utilisée pour la batterie et la carte de vol : pannes aléatoires indépendantes
                    du vieillissement du composant.
                </p>
                <div class="formula-box">R(t) = e<sup>-λt</sup></div>
            </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
            <div class="feature-card" style="border-top:5px solid #6b4c6e;">
                <div class="card-title-lg" style="color:#6b4c6e !important;">Usure moteur (β, η) — Weibull & redondance</div>
                <p style="font-size: 1.5rem; color:#0f172a; line-height: 1.6; font-weight:600;">
                    Dégradation mécanique des moteurs, couplée à une architecture tolérante
                    aux pannes de type k parmi n.
                </p>
                <div class="formula-box">R<sub>prop</sub>(t) = Σ C(n,i)·R<sub>m</sub>(t)<sup>i</sup>·(1-R<sub>m</sub>(t))<sup>n-i</sup></div>
            </div>
        """, unsafe_allow_html=True)


# =====================================================================
# PAGE 2 : SIMULATEUR & ANALYSE
# =====================================================================
elif st.session_state.page == "Simulateur & Analyse":

    st.markdown("""
        <div class="hero-wrap" style="padding:28px 36px;">
            <div class="hero-title" style="font-size: 2.8rem !important;">Simulateur temps réel</div>
            <div class="hero-subtitle" style="font-size: 1.5rem !important;">Ajustez les paramètres pour recalculer la fiabilité du système.</div>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("Configuration système")
    st.sidebar.subheader("Composants électroniques")
    t_mission = st.sidebar.number_input("Durée de mission (h)", value=100.0, step=10.0)
    lam_batt = st.sidebar.number_input("λ Batterie (h⁻¹)", value=0.0001, format="%.5f")
    lam_carte = st.sidebar.number_input("λ Carte de vol (h⁻¹)", value=0.00005, format="%.5f")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Propulsion (Weibull k/n)")
    beta = st.sidebar.number_input("β Moteur (forme)", value=2.0, step=0.1)
    eta = st.sidebar.number_input("η Moteur (échelle, h)", value=1000.0, step=50.0)

    col_k, col_n = st.sidebar.columns(2)
    k = col_k.number_input("Requis (k)", value=3, min_value=1, step=1)
    n = col_n.number_input("Total (n)", value=4, min_value=1, step=1)

    if k > n:
        st.sidebar.error("k ne peut pas dépasser n.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Monte Carlo")
    n_sims = st.sidebar.select_slider(
        "Nombre de missions simulées",
        options=[10000, 50000, 100000, 250000, 500000],
        value=100000,
        format_func=lambda x: f"{x:,}".replace(",", " ")
    )

    drone = DroneReliability(lambda_batt=lam_batt, lambda_carte=lam_carte,
                             beta_moteur=beta, eta_moteur=eta, k=k, n=n)

    r_th = drone.r_drone(t_mission)
    mc = drone.simuler_mc(t_mission=t_mission, n_sims=n_sims)

    if r_th >= 0.95:
        status_html = '<span class="big-badge" style="background:#dcfce7; color:#14532d; border:2px solid #86efac;">Fiabilité optimale</span>'
    elif r_th >= 0.80:
        status_html = '<span class="big-badge" style="background:#fef3c7; color:#78350f; border:2px solid #fde047;">Fiabilité modérée</span>'
    else:
        status_html = '<span class="big-badge" style="background:#fee2e2; color:#7f1d1d; border:2px solid #fca5a5;">Risque élevé</span>'

    col_res, col_chart = st.columns([1, 1.25], gap="large")

    with col_res:
        st.markdown("<h3 style='font-size:1.3rem;'>Résultats à T_mission</h3>", unsafe_allow_html=True)
        kpi1, kpi2 = st.columns(2)
        with kpi1:
            st.markdown(f"""
                <div class="metric-card" style="border-top: 5px solid #1d3557;">
                    <div class="metric-label">Théorique (analytique)</div>
                    <div class="metric-value">{r_th * 100:.2f}%</div>
                    <div style="margin-top:12px;">{status_html}</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi2:
            ecart = abs(r_th - mc['r_mc']) * 100
            st.markdown(f"""
                <div class="metric-card" style="border-top: 5px solid #6b4c6e;">
                    <div class="metric-label">Monte Carlo ({n_sims//1000}k)</div>
                    <div class="metric-value">{mc['r_mc'] * 100:.2f}%</div>
                    <div style="margin-top:14px; font-weight:800; color:#6b4c6e; font-size:1.15rem;">Écart : {ecart:.3f}%</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size:1.2rem;'>Origine des pannes (simulation)</h3>", unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        for col, label, val, color in [
            (p1, "Batterie", mc['echec_batt'], "#991b1b"),
            (p2, "Carte de vol", mc['echec_carte'], "#92400e"),
            (p3, "Propulsion", mc['echec_prop'], "#1d3557"),
        ]:
            col.markdown(f"""
                <div class="metric-card" style="border-top: 5px solid {color};">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color: {color} !important;">{val}</div>
                </div>
            """, unsafe_allow_html=True)

    with col_chart:
        st.markdown("<h3 style='font-size:1.4rem;'>📈 Courbes de fiabilité R(t)</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5.2))
        t = np.linspace(0.1, max(150.0, t_mission * 1.5), 250)
        ax.plot(t, np.exp(-drone.lam_batt * t), "--", color="#991b1b", linewidth=2, label="Batterie")
        ax.plot(t, np.exp(-drone.lam_carte * t), "--", color="#92400e", linewidth=2, label="Carte de vol")
        ax.plot(t, [drone.r_propulsion(ti) for ti in t], "--", color="#1d3557", linewidth=2, label=f"Propulsion ({drone.k}/{drone.n})")
        ax.plot(t, [drone.r_drone(ti) for ti in t], "-", color="#15803d", linewidth=3.5, label="Drone total")
        ax.axvline(t_mission, color="#991b1b", linestyle=":", linewidth=2.5, label=f"Mission ({t_mission:.0f}h)")
        ax.set_xlabel("Temps (heures)")
        ax.set_ylabel("Fiabilité R(t)")
        style_ax(ax, fig)
        st.pyplot(fig)


# =====================================================================
# PAGE 3 : COMPARAISON D'ARCHITECTURES & SENSIBILITE
# =====================================================================
elif st.session_state.page == "Comparaison & Sensibilité":

    st.markdown("""
        <div class="hero-wrap" style="padding:28px 36px;">
            <div class="hero-title" style="font-size: 2.4rem !important;">Comparaison d'architectures & Sensibilité</div>
            <div class="hero-subtitle" style="font-size: 1.25rem !important;">
                Comparez plusieurs stratégies de redondance k/n et étudiez l'impact des paramètres β et η
                de la loi de Weibull sur la fiabilité du système de propulsion.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("Paramètres de comparaison")
    t_comp = st.sidebar.number_input("Durée de mission (h)", value=100.0, step=10.0, key="t_comp")
    beta_comp = st.sidebar.number_input("β nominal", value=2.0, step=0.1, key="beta_comp")
    eta_comp = st.sidebar.number_input("η nominal (h)", value=1000.0, step=50.0, key="eta_comp")

    # ---------- BLOC 1 : COMPARAISON D'ARCHITECTURES ----------
    st.markdown("<h2 style='font-size:4rem;'>Comparaison des architectures de propulsion</h2>", unsafe_allow_html=True)

    dc = DroneReliability(beta_moteur=beta_comp, eta_moteur=eta_comp)
    r_moteur = dc.r_weibull(t_comp)

    architectures = [
        ("4 en série (k=4, n=4)", 4, 4),
        ("3 parmi 4 (config. réelle)", 3, 4),
        ("2 parmi 4", 2, 4),
        ("4 parmi 5", 4, 5),
    ]
    noms = [a[0] for a in architectures]
    valeurs = [dc.r_k_parmi_n(t_comp, k=a[1], n=a[2]) * 100 for a in architectures]

    col_tab, col_graph = st.columns([1, 1.3], gap="large")

    with col_tab:
        st.markdown(f"""
            <div class="feature-card">
                <p style="color:#0f172a; font-size:1.5rem; font-weight:700; margin-bottom:16px;">
                    Fiabilité d'un moteur seul à t={t_comp:.0f}h : <b style="color:#1d3557; font-size:1.3rem;">{r_moteur*100:.4f}%</b>
                </p>
        """, unsafe_allow_html=True)
        for nom, val in zip(noms, valeurs):
            highlight = "border-top:4px solid #15803d;" if "réelle" in nom else "border-top:4px solid #94a3b8;"
            st.markdown(f"""
                <div class="metric-card" style="{highlight} margin-bottom:12px; text-align:left; display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; color:#0f172a; font-size:1.5rem;">{nom}</span>
                    <span style="font-weight:800; font-size:1.5rem; color:#0f172a;">{val:.4f}%</span>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_graph:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        colors = ["#991b1b", "#15803d", "#1d3557", "#6b4c6e", "#92400e"]
        bars = ax.barh(noms, valeurs, color=colors)
        ax.set_xlabel("Fiabilité R_sys (%)")
        ax.set_xlim(min(valeurs) - 2, 100.5)
        for bar, val in zip(bars, valeurs):
            ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f"{val:.3f}%",
                    va="center", color="#0f172a", fontsize=11, fontweight="bold")
        ax.set_title(f"Fiabilité par architecture (t={t_comp:.0f}h)")
        style_ax(ax, fig)
        st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Lecture** : Le ratio de tolérance de panne (n−k)/n compte davantage que le nombre absolu de composants — "
            "un système 4-parmi-5 peut être moins fiable qu'un 3-parmi-4, malgré un composant redondant de plus.")

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # ---------- BLOC 2 : ANALYSE DE SENSIBILITE ----------
    st.markdown("<h2 style='font-size:2rem;'>Analyse de sensibilité sur β et η</h2>", unsafe_allow_html=True)

    col_beta, col_eta = st.columns(2, gap="large")

    betas = np.linspace(max(0.5, beta_comp - 1.2), beta_comp + 1.2, 40)
    r_moteur_beta = [dc.r_weibull(t_comp, beta=b, eta=eta_comp) for b in betas]
    r_prop_beta = [dc.r_k_parmi_n(t_comp, beta=b, eta=eta_comp) * 100 for b in betas]

    etas = np.linspace(max(100, eta_comp * 0.5), eta_comp * 1.5, 40)
    r_moteur_eta = [dc.r_weibull(t_comp, beta=beta_comp, eta=e) for e in etas]
    r_prop_eta = [dc.r_k_parmi_n(t_comp, beta=beta_comp, eta=e) * 100 for e in etas]

    with col_beta:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(betas, r_prop_beta, color="#1d3557", linewidth=3)
        ax.axvline(beta_comp, color="#991b1b", linestyle=":", linewidth=2.5, label=f"β nominal = {beta_comp}")
        ax.set_xlabel("β (paramètre de forme)")
        ax.set_ylabel("R_propulsion (%)")
        ax.set_title("Sensibilité à β (η fixe)")
        style_ax(ax, fig)
        st.pyplot(fig)

    with col_eta:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(etas, r_prop_eta, color="#6b4c6e", linewidth=3)
        ax.axvline(eta_comp, color="#991b1b", linestyle=":", linewidth=2.5, label=f"η nominal = {eta_comp:.0f}h")
        ax.set_xlabel("η (paramètre d'échelle, h)")
        ax.set_ylabel("R_propulsion (%)")
        ax.set_title("Sensibilité à η (β fixe)")
        style_ax(ax, fig)
        st.pyplot(fig)

    # Coefficients de sensibilité (dérivée numérique normalisée)
    d = 1e-3
    R0 = dc.r_weibull(t_comp, beta=beta_comp, eta=eta_comp)
    dR_dbeta = (dc.r_weibull(t_comp, beta=beta_comp + d, eta=eta_comp) -
                dc.r_weibull(t_comp, beta=beta_comp - d, eta=eta_comp)) / (2 * d)
    dR_deta = (dc.r_weibull(t_comp, beta=beta_comp, eta=eta_comp + 1) -
               dc.r_weibull(t_comp, beta=beta_comp, eta=eta_comp - 1)) / 2
    S_beta = dR_dbeta * (beta_comp / R0)
    S_eta = dR_deta * (eta_comp / R0)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""
        <div class="metric-card" style="border-top:5px solid #1d3557;">
            <div class="metric-label">Sensibilité relative S_β</div>
            <div class="metric-value">{S_beta:.4f}</div>
        </div>
    """, unsafe_allow_html=True)
    c2.markdown(f"""
        <div class="metric-card" style="border-top:5px solid #6b4c6e;">
            <div class="metric-label">Sensibilité relative S_η</div>
            <div class="metric-value">{S_eta:.4f}</div>
        </div>
    """, unsafe_allow_html=True)
    ratio = abs(S_beta / S_eta) if S_eta != 0 else float("inf")
    c3.markdown(f"""
        <div class="metric-card" style="border-top:5px solid #15803d;">
            <div class="metric-label">β est plus influent que η par</div>
            <div class="metric-value" style="color:#15803d !important;">×{ratio:.1f}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # ---------- BLOC 3 : PISTE COUT / OPTIMISATION ----------
    st.markdown("<h2 style='font-size:2rem;'>Piste coût vs fiabilité</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div class="feature-card">
            <p style="color:#0f172a; font-size:1.2rem; line-height:1.7; font-weight:600;">
                Chaque moteur additionnel augmente le coût, le poids et la consommation du drone.
                Le graphique de comparaison ci-dessus montre que les gains de fiabilité entre
                <b style="color:#1d3557; font-size:1.25rem;">3-parmi-4</b> et <b style="color:#6b4c6e; font-size:1.25rem;">2-parmi-4</b> sont marginaux
                (quelques centièmes de %), alors que le passage de <b style="color:#991b1b; font-size:1.25rem;">série</b> à
                <b style="color:#1d3557; font-size:1.25rem;">3-parmi-4</b> apporte un gain massif. L'architecture 2-parmi-4
                représente donc le meilleur compromis coût / fiabilité / faisabilité physique de vol.
            </p>
        </div>
    """, unsafe_allow_html=True)
