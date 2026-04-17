"""
app.py
------
Activity 27 — Startup Funding Analysis
Big Data Activity-Based Learning Model

Dataset: Startup Funding (Kaggle)
Tools: Python, Pandas, Matplotlib, Seaborn, Streamlit

Tasks:
    1. Top funded sectors
    2. Investor trends
    3. Year-wise growth

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ── Page config ────────────────────────────────
st.set_page_config(
    page_title="Startup Funding Analysis",
    page_icon="🚀",
    layout="wide"
)

# ── Title ──────────────────────────────────────
st.title("🚀 Startup Funding Analysis Dashboard")
st.markdown("**Activity 27 | Big Data Activity-Based Learning Model**")
st.markdown("**Dataset:** [Indian Startup Funding — Kaggle](https://www.kaggle.com/datasets/sudalairajkumar/indian-startup-funding)")
st.markdown("---")

# ── File Upload ────────────────────────────────
st.sidebar.header("📂 Upload Dataset")
st.sidebar.markdown("Download from Kaggle: `startup_funding.csv`")
uploaded_file = st.sidebar.file_uploader("Upload startup_funding.csv", type=["csv"])

# ── Load Data ──────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    # Normalize common column name variations
    rename_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if 'amount' in cl:
            rename_map[col] = 'AmountInUSD'
        elif 'sector' in cl or 'vertical' in cl or 'industry' in cl:
            rename_map[col] = 'IndustryVertical'
        elif 'investor' in cl:
            rename_map[col] = 'InvestorsName'
        elif 'year' in cl and 'date' not in cl:
            rename_map[col] = 'Year'
        elif 'date' in cl:
            rename_map[col] = 'Date'
        elif 'startup' in cl or 'company' in cl:
            rename_map[col] = 'StartupName'
        elif 'city' in cl or 'location' in cl:
            rename_map[col] = 'CityLocation'
        elif 'type' in cl or 'round' in cl:
            rename_map[col] = 'InvestmentType'
    df.rename(columns=rename_map, inplace=True)

    # Extract Year from Date if Year column not present
    if 'Year' not in df.columns and 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df['Year'] = df['Date'].dt.year

    # Clean AmountInUSD
    if 'AmountInUSD' in df.columns:
        df['AmountInUSD'] = (
            df['AmountInUSD']
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('undisclosed', '0', case=False)
            .str.replace('unknown', '0', case=False)
        )
        df['AmountInUSD'] = pd.to_numeric(df['AmountInUSD'], errors='coerce').fillna(0)

    df.fillna('Unknown', inplace=True)
    return df


# ── Sample Data Generator (fallback) ──────────
def generate_sample_data():
    np.random.seed(42)
    sectors = ['Technology', 'E-Commerce', 'FinTech', 'HealthTech', 'EdTech',
               'FoodTech', 'LogisTech', 'AgriTech', 'Real Estate', 'Gaming']
    investors = ['Sequoia Capital', 'Accel Partners', 'Tiger Global',
                 'SoftBank', 'Nexus Venture', 'Kalaari Capital',
                 'Matrix Partners', 'Blume Ventures']
    cities = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai', 'Pune']
    types = ['Seed Funding', 'Series A', 'Series B', 'Series C', 'Private Equity']
    years = list(range(2015, 2021))

    n = 500
    data = {
        'StartupName': [f'Startup_{i}' for i in range(n)],
        'Year': np.random.choice(years, n),
        'IndustryVertical': np.random.choice(sectors, n, p=[0.25,0.2,0.15,0.1,0.1,0.07,0.05,0.04,0.02,0.02]),
        'InvestorsName': np.random.choice(investors, n),
        'AmountInUSD': np.random.exponential(scale=5_000_000, size=n).astype(int),
        'CityLocation': np.random.choice(cities, n),
        'InvestmentType': np.random.choice(types, n),
    }
    return pd.DataFrame(data)


# ── Main App ───────────────────────────────────
if uploaded_file:
    df = load_data(uploaded_file)
    st.success(f"✅ Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
else:
    st.info("📊 No file uploaded — showing **sample data** for demonstration. Upload your CSV from the sidebar.")
    df = generate_sample_data()

# Dataset Preview
with st.expander("🔍 Dataset Preview"):
    st.dataframe(df.head(10))
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    if 'AmountInUSD' in df.columns:
        st.write(f"**Null values:**")
        st.write(df.isnull().sum())

st.markdown("---")

# ── TASK 1: Top Funded Sectors ─────────────────
st.subheader("📌 Task 1 — Top Funded Sectors")

col1, col2 = st.columns(2)

with col1:
    if 'IndustryVertical' in df.columns and 'AmountInUSD' in df.columns:
        sector_funding = (
            df[df['AmountInUSD'] > 0]
            .groupby('IndustryVertical')['AmountInUSD']
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        bars = ax1.barh(sector_funding['IndustryVertical'],
                        sector_funding['AmountInUSD'] / 1e6,
                        color=sns.color_palette("Blues_r", len(sector_funding)))
        ax1.set_xlabel("Total Funding (Million USD)")
        ax1.set_title("Top 10 Sectors by Total Funding")
        ax1.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig1)
        # removed local savefig
        plt.close()

with col2:
    if 'IndustryVertical' in df.columns:
        sector_count = df['IndustryVertical'].value_counts().head(10)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.bar(sector_count.index, sector_count.values,
                color=sns.color_palette("Oranges_r", len(sector_count)))
        ax2.set_xlabel("Sector")
        ax2.set_ylabel("Number of Deals")
        ax2.set_title("Top 10 Sectors by Number of Deals")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

st.markdown("---")

# ── TASK 2: Investor Trends ────────────────────
st.subheader("📌 Task 2 — Investor Trends")

col3, col4 = st.columns(2)

with col3:
    if 'InvestorsName' in df.columns:
        top_investors = (
            df[df['InvestorsName'] != 'Unknown']
            ['InvestorsName']
            .value_counts()
            .head(10)
        )
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ax3.barh(top_investors.index, top_investors.values,
                 color=sns.color_palette("Greens_r", len(top_investors)))
        ax3.set_xlabel("Number of Investments")
        ax3.set_title("Top 10 Most Active Investors")
        ax3.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig3)
        # removed local savefig
        plt.close()

with col4:
    if 'InvestmentType' in df.columns:
        inv_type = df['InvestmentType'].value_counts().head(8)
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        wedges, texts, autotexts = ax4.pie(
            inv_type.values,
            labels=inv_type.index,
            autopct='%1.1f%%',
            colors=sns.color_palette("Set2", len(inv_type)),
            startangle=140
        )
        ax4.set_title("Investment Type Distribution")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

st.markdown("---")

# ── TASK 3: Year-wise Growth ───────────────────
st.subheader("📌 Task 3 — Year-wise Growth")

col5, col6 = st.columns(2)

with col5:
    if 'Year' in df.columns:
        yearly = (
            df[df['Year'].apply(lambda x: str(x).isdigit() if x != 'Unknown' else False)]
            .copy()
        )
        yearly['Year'] = yearly['Year'].astype(int)
        yearly = yearly[(yearly['Year'] >= 2010) & (yearly['Year'] <= 2025)]
        year_count = yearly.groupby('Year').size().reset_index(name='Deals')

        fig5, ax5 = plt.subplots(figsize=(7, 5))
        ax5.plot(year_count['Year'], year_count['Deals'],
                 marker='o', color='steelblue', linewidth=2.5, markersize=7)
        ax5.fill_between(year_count['Year'], year_count['Deals'], alpha=0.15, color='steelblue')
        ax5.set_xlabel("Year")
        ax5.set_ylabel("Number of Deals")
        ax5.set_title("Year-wise Number of Funding Deals")
        ax5.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig5)
        # removed local savefig
        plt.close()

with col6:
    if 'Year' in df.columns and 'AmountInUSD' in df.columns:
        yearly2 = (
            df[(df['Year'].apply(lambda x: str(x).isdigit() if x != 'Unknown' else False)) &
               (df['AmountInUSD'] > 0)]
            .copy()
        )
        yearly2['Year'] = yearly2['Year'].astype(int)
        yearly2 = yearly2[(yearly2['Year'] >= 2010) & (yearly2['Year'] <= 2025)]
        year_amt = yearly2.groupby('Year')['AmountInUSD'].sum().reset_index()

        fig6, ax6 = plt.subplots(figsize=(7, 5))
        ax6.bar(year_amt['Year'], year_amt['AmountInUSD'] / 1e6,
                color=sns.color_palette("viridis", len(year_amt)))
        ax6.set_xlabel("Year")
        ax6.set_ylabel("Total Funding (Million USD)")
        ax6.set_title("Year-wise Total Funding Amount")
        ax6.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig6)
        # removed local savefig
        plt.close()

st.markdown("---")

# ── KPI Summary ────────────────────────────────
st.subheader("📊 Key Insights Summary")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Startups", f"{df['StartupName'].nunique():,}" if 'StartupName' in df.columns else "N/A")
with k2:
    total_funding = df['AmountInUSD'].sum() if 'AmountInUSD' in df.columns else 0
    st.metric("Total Funding", f"${total_funding/1e9:.2f}B" if total_funding > 1e9 else f"${total_funding/1e6:.1f}M")
with k3:
    top_sector = df['IndustryVertical'].value_counts().idxmax() if 'IndustryVertical' in df.columns else "N/A"
    st.metric("Top Sector", top_sector)
with k4:
    top_inv = df['InvestorsName'].value_counts().idxmax() if 'InvestorsName' in df.columns else "N/A"
    st.metric("Most Active Investor", str(top_inv)[:20])

st.markdown("---")
st.caption("Submitted for Activity 27 — Big Data Activity-Based Learning Model | Tools: Python · Pandas · Matplotlib · Seaborn · Streamlit")
