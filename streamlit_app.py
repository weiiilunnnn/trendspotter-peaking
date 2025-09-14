import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import squarify
import numpy as np

st.set_page_config(page_title="TrendSpotter Hybrid Model", layout="wide")
st.title("📊 TrendSpotter — Hybrid Beauty Trend Analysis - Group: Peaking")

st.markdown("""
Explore **Hair, Makeup, Skincare, and Fragrance** trends with:
- Hybrid scoring (token + topic normalization)
- Family-level clustering
- Cohort split (Gen Z vs Millennials)
""")

# -------------------------------
# DATA LOADING
# -------------------------------
@st.cache_data
def load_data():
    hybrid_monthly = pd.read_csv("data/hybrid_monthly.csv")
    hybrid_sustained = pd.read_csv("data/hybrid_sustained.csv")
    fam_weekly = pd.read_csv("data/fam_weekly.csv")         # ["week","family","count"]
    wk_cohort_family = pd.read_csv("data/wk_cohort_family.csv") # ["week","cohort","family","count"]

    # 🔑 Ensure week is datetime
    fam_weekly["week"] = pd.to_datetime(fam_weekly["week"])
    wk_cohort_family["week"] = pd.to_datetime(wk_cohort_family["week"])

    # 🔑 Apply volume floor (same as notebook)
    min_avg_weekly = 10
    valid_fams = (
        fam_weekly.groupby("family")["count"].mean()
        .loc[lambda x: x >= min_avg_weekly]
        .index.tolist()
    )
    fam_weekly = fam_weekly[fam_weekly["family"].isin(valid_fams)]
    wk_cohort_family = wk_cohort_family[wk_cohort_family["family"].isin(valid_fams)]

    return hybrid_monthly, hybrid_sustained, fam_weekly, wk_cohort_family

hybrid_monthly, hybrid_sustained, fam_weekly, wk_cohort_family = load_data()

# -------------------------------
# TREEMAPS
# -------------------------------
def plot_treemap(df, title):
    if df.empty:
        st.warning("No data available")
        return
    labels = [f"{row['token']}\n({row['hybrid_score']:.2f})" for _, row in df.iterrows()]
    sizes = df["hybrid_score"].values
    colors = plt.cm.viridis(df["hybrid_score"].rank(pct=True))

    fig, ax = plt.subplots(figsize=(10, 5))
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, pad=True, text_kwargs={'fontsize':10})
    plt.axis("off"); plt.title(title, fontsize=14, fontweight="bold")
    st.pyplot(fig)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Hybrid Monthly")
    plot_treemap(hybrid_monthly, "Hybrid Monthly Trends Treemap")
with col2:
    st.subheader("Hybrid Sustained")
    plot_treemap(hybrid_sustained, "Hybrid Sustained Trends Treemap")

# -------------------------------
# FAMILY OVERALL RANKING
# -------------------------------
st.subheader("🏆 Family Overall Ranking")

ranking_data = {
    "family": ["hair", "makeup", "fragrance", "skincare"],
    "monthly_h": [0.85, 0.00, 0.5667, 0.00],
    "sustained_h": [0.605, 1.000, 0.000, 0.620],
    "overall_score": [0.752, 0.400, 0.340, 0.248],
    "avg_weekly_recent": [405.08, 1064.17, 29.75, 320.08]
}
df_rank = pd.DataFrame(ranking_data).set_index("family")

st.dataframe(df_rank.style.highlight_max(axis=0, color="lightgreen"))

# -------------------------------
# FAMILY-LEVEL TRENDS
# -------------------------------
st.subheader("📈 Family Trends (last 26 weeks)")
families = fam_weekly["family"].unique().tolist()
selected_families = st.multiselect("Select families:", families, default=["makeup","hair"])

def plot_family_trends(fam_weekly, families, weeks=26):
    df = fam_weekly.copy().sort_values("week")
    end = df["week"].max()
    start = end - pd.Timedelta(weeks=weeks)
    df = df[(df["week"]>start) & (df["week"]<=end)]
    fig, ax = plt.subplots(figsize=(12,6))
    for fam in families:
        sub = df[df["family"]==fam]
        ax.plot(sub["week"], sub["count"].rolling(3, min_periods=1).mean(), marker="o", label=fam)
    ax.set_title("Family Mentions (3-pt smooth)")
    ax.set_xlabel("Week"); ax.set_ylabel("Mentions")
    ax.legend(); ax.grid(True)
    st.pyplot(fig)

plot_family_trends(fam_weekly, selected_families)

# -------------------------------
# QUADRANT MATRIX
# -------------------------------
st.subheader("🟦 Quadrant Matrix (Volume vs Quality)")

st.markdown("""
| Quadrant | Description | Families |
|----------|-------------|-----------|
| **High Volume / High Quality** | Large and sustainable — strong leaders in both visibility and resilience. | *(None clearly fits)* |
| **High Volume / Low Quality** | Very visible but less sustainable — might plateau or decline without innovation. | **Makeup** |
| **Low Volume / High Quality** | Strong structural potential but under-discussed — opportunity to amplify. | **Hair** |
| **Low Volume / Low Quality** | Niche or episodic — low mentions and low hybrid resilience. | **Skincare, Fragrance** |
""")

# -------------------------------
# COHORT SPLIT
# -------------------------------
st.subheader("👥 Cohort View (Gen Z vs Millennials)")
family_for_cohort = st.selectbox("Choose family:", families, index=0)

def plot_family_by_cohort(wk, family, last_weeks=26):
    sub = wk[wk["family"]==family].copy()
    if sub.empty:
        st.warning(f"No data for family '{family}'")
        return
    end = sub["week"].max()
    start = end - pd.Timedelta(weeks=last_weeks)
    sub = sub[(sub["week"]>=start)&(sub["week"]<=end)]

    fig, ax = plt.subplots(figsize=(12,5))
    for coh, g in sub.groupby("cohort"):
        g = g.sort_values("week")
        ax.plot(g["week"], g["count"].rolling(3, min_periods=1).mean(),
                 marker="o", label=str(coh))
    ax.set_title(f"{family} — Gen Z vs Millennials")
    ax.set_xlabel("Week"); ax.set_ylabel("Mentions (smoothed)")
    ax.grid(True); ax.legend()
    st.pyplot(fig)

plot_family_by_cohort(wk_cohort_family, family_for_cohort)


# -------------------------------
# INSIGHTS PANEL
# -------------------------------
st.markdown("""
## ✅ Strategic Insights
- **Hair (Millennials-led, but Decaying):** Transformation-driven, but momentum fading. Needs short-form hybrids to capture Gen Z.  
- **Makeup (Gen Z powerhouse, cooling):** Still dominant, but growth slowing. Requires creator-led short-form + AR try-ons.  
- **Skincare (Gen Z rising):** Strong upward momentum. Best opportunity for daily rituals + GRWM crossovers.  
- **Fragrance (Low volume, emerging in Gen Z):** Small but stable. Needs storytelling bundles with skincare/makeup.  
""")