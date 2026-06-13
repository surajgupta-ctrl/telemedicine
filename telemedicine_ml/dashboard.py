import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from model import predict_churn, predict_risk

st.set_page_config(page_title="MediAI — Telemedicine Platform", layout="wide", page_icon="🏥")

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Background ── */
[data-testid="stAppViewContainer"] {
    background: #0a0f1e;
    color: #e2e8f0;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a0f1e 100%);
    border-right: 1px solid #1e2d45;
    padding-top: 10px;
}
[data-testid="stSidebar"] * { color: #8898aa !important; }
[data-testid="stSidebarNav"] { display: none; }

/* ── Sidebar logo ── */
.sidebar-logo {
    text-align: center;
    padding: 18px 0 10px;
}
.sidebar-logo-icon {
    font-size: 2.8rem;
    display: block;
    margin-bottom: 4px;
}
.sidebar-logo-text {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-logo-sub {
    font-size: 0.72rem;
    color: #475569 !important;
    margin-top: 2px;
}

/* ── Sidebar nav items ── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 11px 18px;
    border-radius: 10px;
    margin: 3px 10px;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 500;
    color: #64748b;
    transition: all 0.2s;
    text-decoration: none;
}
.nav-item:hover { background: #1e2d45; color: #94a3b8 !important; }
.nav-item.active {
    background: linear-gradient(90deg, #1e3a5f, #1e2d45);
    color: #38bdf8 !important;
    border-left: 3px solid #38bdf8;
}
.nav-icon { font-size: 1.1rem; }

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #0d1f3c 0%, #0f2847 50%, #0d1f3c 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, #2563eb22 0%, transparent 70%);
    border-radius: 50%;
}
.page-header-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f1f5f9;
    margin: 0;
}
.page-header-sub {
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 6px;
}
.page-header-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #38bdf8;
    border: 1px solid #2563eb44;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 10px;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.kpi-card {
    background: linear-gradient(135deg, #0d1f3c, #111827);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, linear-gradient(90deg, #38bdf8, #2563eb));
    border-radius: 0 0 14px 14px;
}
.kpi-icon {
    font-size: 1.6rem;
    margin-bottom: 10px;
    display: block;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1;
}
.kpi-label {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 5px;
    font-weight: 500;
}
.kpi-delta {
    font-size: 0.78rem;
    margin-top: 6px;
    font-weight: 600;
}

/* ── Section card ── */
.section-card {
    background: #0d1526;
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
}
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #cbd5e1;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Form card ── */
.form-section {
    background: #0d1526;
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
}
.form-section-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2d45;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #0d1f3c, #0f172a);
    border-radius: 16px;
    padding: 28px;
    margin-top: 20px;
    border: 1px solid #1e3a5f;
}
.result-prob-value {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
}
.result-label {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 6px;
    font-weight: 500;
}
.gauge-bg {
    background: #1e2d45;
    border-radius: 100px;
    height: 12px;
    overflow: hidden;
    margin: 10px 0;
}
.gauge-fill {
    height: 12px;
    border-radius: 100px;
    transition: width 0.6s ease;
}
.tag {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.tag-danger { background: #f43f5e22; color: #f43f5e; border: 1px solid #f43f5e44; }
.tag-success { background: #22c55e22; color: #22c55e; border: 1px solid #22c55e44; }
.tag-warning { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b44; }

/* ── Vital badge ── */
.vital-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
}
.vital-value { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; }
.vital-label { font-size: 0.78rem; color: #64748b; margin-top: 3px; }
.vital-status { font-size: 0.78rem; margin-top: 5px; font-weight: 600; }

/* ── Patient table ── */
.patient-row {
    background: #0d1526;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* ── Streamlit overrides ── */
[data-testid="metric-container"] {
    background: #0d1526 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 1.5rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.8rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 30px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    letter-spacing: 0.02em;
    transition: all 0.2s !important;
}
div.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px); }

[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    background: #111827 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
}
[data-testid="stSlider"] > div { color: #e2e8f0 !important; }
label, .stSlider label { color: #94a3b8 !important; font-size: 0.83rem !important; }

[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

div[data-testid="stAlert"] { border-radius: 12px !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
if not os.path.exists("data/patients.csv"):
    st.error("⚠️ Run `python model.py` first to generate data and train models.")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv("data/patients.csv"), pd.read_csv("data/consultations.csv")

patients, consultations = load_data()

CHART = dict(
    plot_bgcolor="#0d1526", paper_bgcolor="#0d1526",
    font_color="#64748b", title_font_color="#cbd5e1",
    title_font_size=13, margin=dict(l=10, r=10, t=40, b=10),
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="sidebar-logo-icon">🏥</span>
        <div class="sidebar-logo-text">MediAI</div>
        <div class="sidebar-logo-sub">Telemedicine Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1e2d45;margin:12px 0'>", unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🏠  Home", "📊  Analytics", "🔮  Churn Predictor", "⚕️  Risk Classifier", "🗃️  Patient Explorer"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1e2d45;margin:12px 0'>", unsafe_allow_html=True)

    # live stats
    st.markdown(f"""
    <div style='padding:0 10px'>
        <div style='font-size:0.72rem;color:#334155;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px'>Live Stats</div>
        <div style='display:flex;justify-content:space-between;margin-bottom:7px'>
            <span style='font-size:0.8rem;color:#475569'>Patients</span>
            <span style='font-size:0.8rem;color:#38bdf8;font-weight:600'>{len(patients):,}</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:7px'>
            <span style='font-size:0.8rem;color:#475569'>Consultations</span>
            <span style='font-size:0.8rem;color:#818cf8;font-weight:600'>{len(consultations):,}</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:7px'>
            <span style='font-size:0.8rem;color:#475569'>Churn Rate</span>
            <span style='font-size:0.8rem;color:#f43f5e;font-weight:600'>{patients["churned"].mean()*100:.1f}%</span>
        </div>
        <div style='display:flex;justify-content:space-between'>
            <span style='font-size:0.8rem;color:#475569'>High Risk</span>
            <span style='font-size:0.8rem;color:#f59e0b;font-weight:600'>{consultations["high_risk"].sum():,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2d45;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#1e3a5f;font-size:0.72rem;text-align:center'>India Digital Healthcare © 2024</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    # Hero
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1f3c 0%,#0f2847 60%,#0d1f3c 100%);
                border:1px solid #1e3a5f;border-radius:20px;padding:48px 40px;
                margin-bottom:28px;position:relative;overflow:hidden;'>
        <div style='position:absolute;top:-80px;right:-60px;width:350px;height:350px;
                    background:radial-gradient(circle,#2563eb18,transparent 70%);border-radius:50%'></div>
        <div style='position:absolute;bottom:-60px;left:30%;width:250px;height:250px;
                    background:radial-gradient(circle,#818cf810,transparent 70%);border-radius:50%'></div>
        <div style='font-size:0.8rem;font-weight:700;color:#38bdf8;text-transform:uppercase;
                    letter-spacing:.1em;margin-bottom:14px'>🇮🇳 India Telemedicine Intelligence</div>
        <div style='font-size:2.8rem;font-weight:800;color:#f1f5f9;line-height:1.15;margin-bottom:16px'>
            AI-Powered Healthcare<br>
            <span style='background:linear-gradient(90deg,#38bdf8,#818cf8);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                         background-clip:text'>Decision Platform</span>
        </div>
        <div style='font-size:1rem;color:#64748b;max-width:560px;line-height:1.6'>
            Predict patient churn, assess clinical risk, and explore analytics across 
            India's post-pandemic telemedicine ecosystem — powered by machine learning.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    churn_rate = patients["churned"].mean() * 100
    avg_sat    = patients["satisfaction_score"].mean()
    high_risk  = consultations["high_risk"].sum()
    risk_rate  = consultations["high_risk"].mean() * 100

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card" style="--accent:linear-gradient(90deg,#38bdf8,#2563eb)">
            <span class="kpi-icon">👥</span>
            <div class="kpi-value">{len(patients):,}</div>
            <div class="kpi-label">Total Patients</div>
            <div class="kpi-delta" style="color:#38bdf8">Across all platforms</div>
        </div>
        <div class="kpi-card" style="--accent:linear-gradient(90deg,#f43f5e,#e11d48)">
            <span class="kpi-icon">📉</span>
            <div class="kpi-value">{churn_rate:.1f}%</div>
            <div class="kpi-label">Churn Rate</div>
            <div class="kpi-delta" style="color:#f43f5e">Platform dropout risk</div>
        </div>
        <div class="kpi-card" style="--accent:linear-gradient(90deg,#22c55e,#16a34a)">
            <span class="kpi-icon">⭐</span>
            <div class="kpi-value">{avg_sat:.2f}</div>
            <div class="kpi-label">Avg Satisfaction</div>
            <div class="kpi-delta" style="color:#22c55e">Out of 5.0</div>
        </div>
        <div class="kpi-card" style="--accent:linear-gradient(90deg,#f59e0b,#d97706)">
            <span class="kpi-icon">🚨</span>
            <div class="kpi-value">{high_risk:,}</div>
            <div class="kpi-label">High Risk Cases</div>
            <div class="kpi-delta" style="color:#f59e0b">{risk_rate:.1f}% of consultations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="section-card" style="border-top:3px solid #38bdf8;text-align:center;padding:28px">
            <div style="font-size:2.5rem;margin-bottom:12px">📊</div>
            <div style="font-size:1rem;font-weight:700;color:#f1f5f9;margin-bottom:8px">Analytics Dashboard</div>
            <div style="font-size:0.83rem;color:#64748b;line-height:1.6">
                Explore churn trends, platform distribution, satisfaction scores, and risk patterns across city tiers.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="section-card" style="border-top:3px solid #818cf8;text-align:center;padding:28px">
            <div style="font-size:2.5rem;margin-bottom:12px">🔮</div>
            <div style="font-size:1rem;font-weight:700;color:#f1f5f9;margin-bottom:8px">Churn Predictor</div>
            <div style="font-size:0.83rem;color:#64748b;line-height:1.6">
                Predict which patients are at risk of dropping the platform using ML-powered scoring.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="section-card" style="border-top:3px solid #f59e0b;text-align:center;padding:28px">
            <div style="font-size:2.5rem;margin-bottom:12px">⚕️</div>
            <div style="font-size:1rem;font-weight:700;color:#f1f5f9;margin-bottom:8px">Risk Classifier</div>
            <div style="font-size:0.83rem;color:#64748b;line-height:1.6">
                Assess clinical risk from symptoms and vitals to triage patients for in-person care.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Quick charts
    st.markdown("<br>", unsafe_allow_html=True)
    qa, qb = st.columns(2)
    with qa:
        fig = px.histogram(patients, x="city_tier", color="churned", barmode="group",
                           title="Churn by City Tier",
                           color_discrete_sequence=["#38bdf8","#f43f5e"],
                           labels={"churned":"Churned","city_tier":"City Tier"})
        fig.update_layout(**CHART)
        st.plotly_chart(fig, use_container_width=True)
    with qb:
        spec = consultations.groupby("specialty")["high_risk"].mean().reset_index()
        spec.columns = ["Specialty","Risk Rate"]
        fig2 = px.bar(spec.sort_values("Risk Rate"), x="Risk Rate", y="Specialty",
                      orientation="h", title="Risk Rate by Specialty",
                      color="Risk Rate", color_continuous_scale="Reds")
        fig2.update_layout(**CHART, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Analytics":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-badge">📊 ANALYTICS</div>
        <div class="page-header-title">Platform Analytics</div>
        <div class="page-header-sub">Deep-dive into patient behaviour, churn drivers, and health risk trends</div>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    with st.expander("🔧 Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        tier_filter = f1.multiselect("City Tier", ["Tier-1","Tier-2","Tier-3"], default=["Tier-1","Tier-2","Tier-3"])
        plat_filter = f2.multiselect("Platform", patients["platform"].unique().tolist(), default=patients["platform"].unique().tolist())
        inet_filter = f3.multiselect("Internet Quality", ["Good","Average","Poor"], default=["Good","Average","Poor"])

    filt = patients[
        patients["city_tier"].isin(tier_filter) &
        patients["platform"].isin(plat_filter) &
        patients["internet_quality"].isin(inet_filter)
    ]

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Filtered Patients", f"{len(filt):,}", f"{len(filt)-len(patients):,} vs total")
    k2.metric("Churn Rate", f"{filt['churned'].mean()*100:.1f}%")
    k3.metric("Avg Wait Time", f"{filt['avg_wait_time'].mean():.1f} min")
    k4.metric("Avg Satisfaction", f"{filt['satisfaction_score'].mean():.2f} / 5")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1
    r1a, r1b = st.columns(2)
    with r1a:
        fig = px.histogram(filt, x="city_tier", color="churned", barmode="group",
                           title="Churn Distribution by City Tier",
                           color_discrete_sequence=["#38bdf8","#f43f5e"])
        fig.update_layout(**CHART)
        st.plotly_chart(fig, use_container_width=True)
    with r1b:
        pc = filt["platform"].value_counts().reset_index()
        pc.columns = ["platform","count"]
        fig2 = px.pie(pc, names="platform", values="count",
                      title="Patient Share by Platform", hole=0.45,
                      color_discrete_sequence=["#38bdf8","#818cf8","#22c55e","#f59e0b"])
        fig2.update_layout(**CHART)
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2
    r2a, r2b = st.columns(2)
    with r2a:
        fig3 = px.box(filt, x="internet_quality", y="satisfaction_score",
                      color="internet_quality", title="Satisfaction by Internet Quality",
                      color_discrete_sequence=["#22c55e","#f59e0b","#f43f5e"])
        fig3.update_layout(**CHART, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    with r2b:
        fig4 = px.scatter(filt, x="avg_wait_time", y="satisfaction_score",
                          color="churned", title="Wait Time vs Satisfaction",
                          color_discrete_sequence=["#38bdf8","#f43f5e"],
                          opacity=0.6, size_max=6)
        fig4.update_layout(**CHART)
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3
    r3a, r3b = st.columns(2)
    with r3a:
        fig5 = px.histogram(filt, x="age", color="churned", nbins=25,
                            title="Age Distribution — Churned vs Retained",
                            color_discrete_sequence=["#38bdf8","#f43f5e"])
        fig5.update_layout(**CHART)
        st.plotly_chart(fig5, use_container_width=True)
    with r3b:
        lang_churn = filt.groupby("language")["churned"].mean().reset_index()
        lang_churn.columns = ["Language","Churn Rate"]
        fig6 = px.bar(lang_churn.sort_values("Churn Rate", ascending=True),
                      x="Churn Rate", y="Language", orientation="h",
                      title="Churn Rate by Language",
                      color="Churn Rate", color_continuous_scale="RdBu_r")
        fig6.update_layout(**CHART, coloraxis_showscale=False)
        st.plotly_chart(fig6, use_container_width=True)

    # Consultation analytics
    st.markdown("<hr style='border-color:#1e2d45;margin:8px 0 20px'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1rem;font-weight:700;color:#cbd5e1;margin-bottom:16px'>⚕️ Consultation Risk Analytics</div>", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        spo2_bins = pd.cut(consultations["spo2"], bins=[80,92,95,98,100], labels=["<92","92-95","95-98","98-100"])
        spo2_risk = consultations.copy()
        spo2_risk["spo2_range"] = spo2_bins
        spo2_agg = spo2_risk.groupby("spo2_range", observed=True)["high_risk"].mean().reset_index()
        fig7 = px.bar(spo2_agg, x="spo2_range", y="high_risk",
                      title="High Risk Rate by SpO2 Range",
                      color="high_risk", color_continuous_scale="Reds",
                      labels={"spo2_range":"SpO2 (%)","high_risk":"High Risk Rate"})
        fig7.update_layout(**CHART, coloraxis_showscale=False)
        st.plotly_chart(fig7, use_container_width=True)
    with cb:
        bp_bins = pd.cut(consultations["bp_systolic"], bins=[80,120,140,160,200], labels=["Normal","Elevated","High","Very High"])
        bp_risk = consultations.copy()
        bp_risk["bp_category"] = bp_bins
        bp_agg = bp_risk.groupby("bp_category", observed=True)["high_risk"].mean().reset_index()
        fig8 = px.bar(bp_agg, x="bp_category", y="high_risk",
                      title="High Risk Rate by BP Category",
                      color="high_risk", color_continuous_scale="Oranges",
                      labels={"bp_category":"BP Category","high_risk":"High Risk Rate"})
        fig8.update_layout(**CHART, coloraxis_showscale=False)
        st.plotly_chart(fig8, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — CHURN PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Churn Predictor":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-badge">🔮 ML MODEL</div>
        <div class="page-header-title">Patient Churn Predictor</div>
        <div class="page-header-sub">Enter patient details to predict their likelihood of leaving the platform</div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([3, 2], gap="large")

    with col_form:
        st.markdown('<div class="form-section"><div class="form-section-title">👤 Demographics</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        age       = fc1.slider("Age", 18, 80, 40)
        city_tier = fc2.selectbox("City Tier", ["Tier-1","Tier-2","Tier-3"])
        language  = fc3.selectbox("Language", ["Hindi","English","Tamil","Bengali","Telugu","Marathi"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section"><div class="form-section-title">💻 Platform & Access</div>', unsafe_allow_html=True)
        fp1, fp2, fp3 = st.columns(3)
        platform      = fp1.selectbox("Platform", ["eSanjeevani","Practo","mFine","Other"])
        internet      = fp2.selectbox("Internet Quality", ["Good","Average","Poor"])
        has_insurance = fp3.selectbox("Insurance", [1,0], format_func=lambda x: "✅ Yes" if x else "❌ No")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section"><div class="form-section-title">📋 Consultation History</div>', unsafe_allow_html=True)
        fh1, fh2, fh3 = st.columns(3)
        num_cons      = fh1.number_input("# Consultations", 0, 30, 3)
        avg_wait      = fh2.number_input("Avg Wait (min)", 1.0, 60.0, 15.0)
        satisfaction  = fh3.slider("Satisfaction", 1.0, 5.0, 3.5, step=0.1)
        chronic = st.selectbox("Chronic Condition", [0,1], format_func=lambda x: "✅ Yes" if x else "❌ No")
        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button("🔮 Run Churn Prediction")

    with col_result:
        st.markdown("""
        <div style='background:#0d1526;border:1px dashed #1e3a5f;border-radius:14px;
                    padding:40px 20px;text-align:center;margin-top:0px'>
            <div style='font-size:3rem;margin-bottom:12px'>🔮</div>
            <div style='color:#334155;font-size:0.9rem'>Fill in patient details and click<br><strong style="color:#38bdf8">Run Churn Prediction</strong></div>
        </div>
        """, unsafe_allow_html=True)

        if predict_btn:
            patient = {
                "age": age, "internet_quality": internet, "city_tier": city_tier,
                "language": language, "platform": platform,
                "num_consultations": int(num_cons), "avg_wait_time": avg_wait,
                "satisfaction_score": satisfaction, "has_insurance": has_insurance,
                "chronic_condition": chronic,
            }
            result = predict_churn(patient)
            prob   = result["churn_probability"]
            pct    = prob * 100
            is_churn = result["will_churn"]
            color  = "#f43f5e" if is_churn else "#22c55e"
            label  = "HIGH RISK" if is_churn else "LOW RISK"
            tag_cls= "tag-danger" if is_churn else "tag-success"
            emoji  = "⚠️" if is_churn else "✅"

            st.markdown(f"""
            <div class="result-card">
                <div style='text-align:center;margin-bottom:20px'>
                    <div style='font-size:0.8rem;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px'>Churn Probability</div>
                    <div class="result-prob-value" style='color:{color}'>{pct:.1f}%</div>
                    <div style='margin-top:10px'><span class='tag {tag_cls}'>{emoji} {label}</span></div>
                </div>
                <div class="gauge-bg"><div class="gauge-fill" style='width:{pct}%;background:linear-gradient(90deg,{color}88,{color})'></div></div>
                <div style='display:flex;justify-content:space-between;font-size:0.75rem;color:#334155;margin-bottom:20px'>
                    <span>0%</span><span>50%</span><span>100%</span>
                </div>
                <hr style='border-color:#1e2d45;margin-bottom:16px'>
                <div style='font-size:0.8rem;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px'>Key Inputs</div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.82rem'>
                    <div style='color:#475569'>Platform</div><div style='color:#94a3b8;text-align:right'>{platform}</div>
                    <div style='color:#475569'>Internet</div><div style='color:#94a3b8;text-align:right'>{internet}</div>
                    <div style='color:#475569'>City Tier</div><div style='color:#94a3b8;text-align:right'>{city_tier}</div>
                    <div style='color:#475569'>Satisfaction</div><div style='color:#94a3b8;text-align:right'>{satisfaction}/5</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if is_churn:
                st.error("⚠️ **High Churn Risk** — Recommend: personalised outreach, discount offer, or follow-up call.")
            else:
                st.success("✅ **Low Churn Risk** — Patient is likely to continue using the platform.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — RISK CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚕️  Risk Classifier":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-badge">⚕️ CLINICAL AI</div>
        <div class="page-header-title">Health Risk Classifier</div>
        <div class="page-header-sub">Assess clinical risk from symptoms and vitals to triage patients effectively</div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([3, 2], gap="large")

    with col_form:
        st.markdown('<div class="form-section"><div class="form-section-title">🧑‍⚕️ Patient Info</div>', unsafe_allow_html=True)
        ri1, ri2 = st.columns(2)
        p_age     = ri1.slider("Age", 18, 90, 50)
        specialty = ri2.selectbox("Specialty", ["General","Cardiology","Dermatology","Pediatrics","Orthopedics"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section"><div class="form-section-title">💊 Vitals</div>', unsafe_allow_html=True)
        rv1, rv2, rv3 = st.columns(3)
        bp_sys = rv1.number_input("BP Systolic (mmHg)", 80, 200, 120)
        bp_dia = rv2.number_input("BP Diastolic (mmHg)", 50, 120, 80)
        spo2   = rv3.number_input("SpO2 (%)", 80.0, 100.0, 97.0, step=0.5)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section"><div class="form-section-title">🤒 Symptoms</div>', unsafe_allow_html=True)
        rs1, rs2, rs3, rs4, rs5 = st.columns(5)
        fever  = int(rs1.checkbox("🌡️ Fever"))
        cough  = int(rs2.checkbox("😷 Cough"))
        fatigue= int(rs3.checkbox("😴 Fatigue"))
        breath = int(rs4.checkbox("🫁 Breathlessness"))
        chest  = int(rs5.checkbox("💔 Chest Pain"))
        st.markdown('</div>', unsafe_allow_html=True)

        risk_btn = st.button("⚕️ Assess Clinical Risk")

    with col_result:
        st.markdown("""
        <div style='background:#0d1526;border:1px dashed #1e3a5f;border-radius:14px;
                    padding:40px 20px;text-align:center'>
            <div style='font-size:3rem;margin-bottom:12px'>⚕️</div>
            <div style='color:#334155;font-size:0.9rem'>Enter patient vitals and symptoms,<br>then click <strong style="color:#38bdf8">Assess Clinical Risk</strong></div>
        </div>
        """, unsafe_allow_html=True)

        if risk_btn:
            consultation = {
                "age": p_age, "symptoms_fever": fever, "symptoms_cough": cough,
                "symptoms_fatigue": fatigue, "symptoms_breathlessness": breath,
                "symptoms_chest_pain": chest, "bp_systolic": bp_sys,
                "bp_diastolic": bp_dia, "spo2": spo2, "specialty": specialty,
            }
            result   = predict_risk(consultation)
            prob     = result["risk_probability"]
            pct      = prob * 100
            is_risk  = result["high_risk"]
            color    = "#f43f5e" if is_risk else "#22c55e"
            label    = "HIGH RISK" if is_risk else "LOW RISK"
            tag_cls  = "tag-danger" if is_risk else "tag-success"
            emoji    = "🚨" if is_risk else "✅"

            bp_status   = ("🔴 High", "#f43f5e") if bp_sys > 140 else (("🟡 Borderline", "#f59e0b") if bp_sys > 120 else ("🟢 Normal", "#22c55e"))
            spo2_status = ("🔴 Low", "#f43f5e") if spo2 < 95 else ("🟢 Normal", "#22c55e")
            sym_count   = fever + cough + fatigue + breath + chest

            st.markdown(f"""
            <div class="result-card">
                <div style='text-align:center;margin-bottom:20px'>
                    <div style='font-size:0.8rem;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px'>Risk Score</div>
                    <div class="result-prob-value" style='color:{color}'>{pct:.1f}%</div>
                    <div style='margin-top:10px'><span class='tag {tag_cls}'>{emoji} {label}</span></div>
                </div>
                <div class="gauge-bg"><div class="gauge-fill" style='width:{pct}%;background:linear-gradient(90deg,{color}88,{color})'></div></div>
                <div style='display:flex;justify-content:space-between;font-size:0.75rem;color:#334155;margin-bottom:20px'>
                    <span>0%</span><span>50%</span><span>100%</span>
                </div>
                <hr style='border-color:#1e2d45;margin-bottom:14px'>
                <div style='font-size:0.8rem;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px'>Vitals Summary</div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.82rem;margin-bottom:14px'>
                    <div style='color:#475569'>Blood Pressure</div>
                    <div style='color:{bp_status[1]};text-align:right;font-weight:600'>{bp_sys}/{bp_dia} {bp_status[0]}</div>
                    <div style='color:#475569'>SpO2</div>
                    <div style='color:{spo2_status[1]};text-align:right;font-weight:600'>{spo2}% {spo2_status[0]}</div>
                    <div style='color:#475569'>Active Symptoms</div>
                    <div style='color:#94a3b8;text-align:right'>{sym_count} / 5</div>
                    <div style='color:#475569'>Age Group</div>
                    <div style='color:#94a3b8;text-align:right'>{"Senior (60+)" if p_age >= 60 else "Adult"}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if is_risk:
                st.error("🚨 **High Risk** — Immediate in-person consultation recommended.")
            else:
                st.success("✅ **Low Risk** — Telemedicine follow-up is appropriate.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PATIENT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗃️  Patient Explorer":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-badge">🗃️ DATA EXPLORER</div>
        <div class="page-header-title">Patient Explorer</div>
        <div class="page-header-sub">Browse, filter, and search through the patient dataset</div>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    ef1, ef2, ef3, ef4 = st.columns(4)
    tier_f  = ef1.multiselect("City Tier", ["Tier-1","Tier-2","Tier-3"], default=["Tier-1","Tier-2","Tier-3"])
    plat_f  = ef2.multiselect("Platform", patients["platform"].unique().tolist(), default=patients["platform"].unique().tolist())
    churn_f = ef3.selectbox("Churn Status", ["All","Churned","Retained"])
    age_f   = ef4.slider("Age Range", 18, 80, (18, 80))

    df = patients.copy()
    df = df[df["city_tier"].isin(tier_f) & df["platform"].isin(plat_f)]
    df = df[(df["age"] >= age_f[0]) & (df["age"] <= age_f[1])]
    if churn_f == "Churned":  df = df[df["churned"] == 1]
    elif churn_f == "Retained": df = df[df["churned"] == 0]

    # stats
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Matching Records", f"{len(df):,}")
    s2.metric("Churn Rate", f"{df['churned'].mean()*100:.1f}%")
    s3.metric("Avg Satisfaction", f"{df['satisfaction_score'].mean():.2f}")
    s4.metric("Avg Wait Time", f"{df['avg_wait_time'].mean():.1f} min")

    st.markdown("<br>", unsafe_allow_html=True)

    # Styled table
    display_df = df[["age","city_tier","platform","internet_quality","language",
                      "num_consultations","avg_wait_time","satisfaction_score",
                      "has_insurance","chronic_condition","churned"]].copy()
    display_df.columns = ["Age","City Tier","Platform","Internet","Language",
                          "Consultations","Wait (min)","Satisfaction","Insurance","Chronic","Churned"]
    display_df["Churned"] = display_df["Churned"].map({1:"⚠️ Yes", 0:"✅ No"})
    display_df["Insurance"] = display_df["Insurance"].map({1:"✅ Yes", 0:"❌ No"})
    display_df["Chronic"] = display_df["Chronic"].map({1:"Yes", 0:"No"})

    st.dataframe(display_df.head(200), use_container_width=True, height=420)
    st.caption(f"Showing up to 200 of {len(df):,} matching records")

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data (CSV)", csv, "filtered_patients.csv", "text/csv")
