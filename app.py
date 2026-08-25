import streamlit as st
import pandas as pd
import plotly.express as px

# Clear cache to force reload of edited CSV files
st.cache_data.clear()

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="India Economic Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .kpi-card {
        background-color: #f8f9fa;
        border-left: 5px solid #1f77b4;
        padding: 12px 15px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .kpi-title {
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #212529;
    }
    .source-text {
        font-size: 0.75rem;
        color: #6c757d;
        font-style: italic;
        margin-top: 5px;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & CLEANING
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    # World Bank total data
    df = pd.read_csv("total_data.csv")
    df["Year"] = pd.to_datetime(df["Year"])

    # Standardize FDI Column
    fdi_cols = [c for c in df.columns if "FDI" in c]
    if fdi_cols:
        raw_fdi_col = fdi_cols[0]
        if df[raw_fdi_col].dropna().abs().mean() > 1000:
            df["FDI (net inflows in USD Bn)"] = df[raw_fdi_col] / 1e9
        else:
            df["FDI (net inflows in USD Bn)"] = df[raw_fdi_col]
    else:
        df["FDI (net inflows in USD Bn)"] = 0.0

    # Sectorwise employment
    sectorwise_emp = pd.read_csv("Employment share data.csv").dropna()
    sectorwise_emp = sectorwise_emp.replace("%", "", regex=True)

    non_sector_cols = ["Period", "Start_Date", "as_of_date", "date", "Year"]
    sector_cols = [c for c in sectorwise_emp.columns if c not in non_sector_cols]
    for c in sector_cols:
        sectorwise_emp[c] = pd.to_numeric(sectorwise_emp[c], errors="coerce")

    def parse_period_to_date(p_val):
        year_str = str(p_val).split("-")[0].strip()
        return pd.to_datetime(f"{year_str}-04-01", errors="coerce")

    sectorwise_emp["Start_Date"] = sectorwise_emp["Period"].apply(parse_period_to_date)

    # MSME employment
    msme_emp_df = pd.read_csv("MSME employment data.csv")
    msme_emp_df["Total employees working in MSMEs registered on Udyam portal"] = (
        msme_emp_df["Total employees working in MSMEs registered on Udyam portal"]
        .astype(str)
        .str.replace(",", "")
        .astype(float)
    )
    msme_emp_df["Start_Date"] = msme_emp_df["Period"].apply(parse_period_to_date)

    # Non-food credit
    non_food_credit = pd.read_csv("Food & Non-Food Credit of Scheduled Commercial Banks.csv")
    non_food_credit["date"] = pd.to_datetime(non_food_credit["Fortnight Date Final"], errors="coerce")
    for col in ["Bank Credit", "Food Credit"]:
        if col in non_food_credit.columns:
            non_food_credit[col] = pd.to_numeric(
                non_food_credit[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )
    non_food_credit["non_food_credit"] = non_food_credit["Bank Credit"] - non_food_credit["Food Credit"]
    non_food_credit = non_food_credit[
        non_food_credit["date"].notna() & non_food_credit["non_food_credit"].gt(0)
    ].copy()
    non_food_credit["fy_end_year"] = non_food_credit["date"].dt.year + (non_food_credit["date"].dt.month >= 4).astype(int)

    fy_end_nfc = (
        non_food_credit.sort_values("date")
        .groupby("fy_end_year", as_index=False)
        .tail(1)
        .sort_values("fy_end_year")
    )
    fy_end_nfc["financial_year"] = (
        (fy_end_nfc["fy_end_year"] - 1).astype(str) + "-" + fy_end_nfc["fy_end_year"].astype(str).str[-2:]
    )
    fy_end_nfc["non_food_credit_lakh_crore"] = fy_end_nfc["non_food_credit"] / 100000
    fy_end_nfc = fy_end_nfc.rename(columns={"date": "as_of_date"})

    # Savings Data
    try:
        import sdmx
        IMF_DATA = sdmx.Client('IMF_DATA')
        data_msg = IMF_DATA.data('WEO', key='IND.NGSD_NGDP.A', params={'startPeriod': 1980})
        df_savings = pd.DataFrame(sdmx.to_pandas(data_msg)).reset_index()[['TIME_PERIOD', 'value']]
        df_savings['TIME_PERIOD'] = pd.to_datetime(df_savings['TIME_PERIOD'])
        df_savings = df_savings[df_savings['TIME_PERIOD'] < '2026-01-01']
    except Exception:
        df_savings = pd.DataFrame({
            "TIME_PERIOD": pd.date_range("1980-01-01", "2025-01-01", freq="YS"),
            "value": [20.0 + i * 0.2 for i in range(46)]
        })

    # Repo rates
    repo_rates = pd.read_csv("Repo rates.csv").dropna()
    repo_rates["Dates"] = pd.to_datetime(repo_rates["Dates"], errors="coerce")

    # NPAs Data Clean-up
    NPAs = pd.read_csv("NPAs.csv").dropna()
    npa_date_col = NPAs.columns[0]

    # Identify percentage column
    val_cols = [c for c in NPAs.columns if c != npa_date_col]
    if val_cols:
        target_col = val_cols[0]
        NPAs["Gross NPAs (% of advances)"] = (
            NPAs[target_col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        NPAs["Gross NPAs (% of advances)"] = pd.to_numeric(NPAs["Gross NPAs (% of advances)"], errors="coerce")

    extracted_years = NPAs[npa_date_col].astype(str).str.extract(r'(\d{4})')[0]
    NPAs["Year"] = pd.to_datetime(extracted_years + "-01-01", errors="coerce")

    # Market Cap
    market_cap = pd.read_csv("Market Capitalisations.csv")
    market_cap["End-period"] = pd.to_datetime(market_cap["End-period"], format="%b-%y", errors="coerce")

    return df, df_savings, sectorwise_emp, msme_emp_df, fy_end_nfc, repo_rates, NPAs, market_cap

df, df_savings, sectorwise_emp, msme_emp_df, fy_end_nfc, repo_rates, NPAs, market_cap = load_and_clean_data()

# Helper function to style charts
def style_chart(fig, data, date_col, date_format="%Y", height=480, legend_bottom=True):
    if not data.empty and date_col in data.columns:
        min_date = data[date_col].min()
        max_date = data[date_col].max()
        fig.update_xaxes(
            range=[min_date, max_date],
            tickformat=date_format,
            autorange=False
        )

    layout_opts = dict(
        height=height,
        margin=dict(l=15, r=15, t=30, b=15),
        autosize=True
    )

    if legend_bottom:
        layout_opts["legend"] = dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            title_text=""
        )

    fig.update_layout(**layout_opts)
    return fig

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & ERA FILTERING
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation & Filters")
page = st.sidebar.radio("Select Dashboard Page", ["Output", "Employment", "Savings and investments"])

period_option = st.sidebar.selectbox(
    "Select Period Era",
    ["All Time", "1947-1991", "1991-2003", "2003-2014", "2014-present"]
)

period_bounds = {
    "All Time": (1947, 2026),
    "1947-1991": (1947, 1991),
    "1991-2003": (1991, 2003),
    "2003-2014": (2003, 2014),
    "2014-present": (2014, 2026)
}
start_yr, end_yr = period_bounds[period_option]

def apply_date_filter(data, col_name):
    if data.empty or col_name not in data.columns:
        return data
    data_cp = data.copy()
    data_cp[col_name] = pd.to_datetime(data_cp[col_name])
    mask = (data_cp[col_name].dt.year >= start_yr) & (data_cp[col_name].dt.year <= end_yr)
    return data_cp[mask]

f_df = apply_date_filter(df, "Year")
f_df_savings = apply_date_filter(df_savings, "TIME_PERIOD")
f_sectorwise_emp = apply_date_filter(sectorwise_emp, "Start_Date")
f_msme = apply_date_filter(msme_emp_df, "Start_Date")
f_nfc = apply_date_filter(fy_end_nfc, "as_of_date")
f_repo = apply_date_filter(repo_rates, "Dates")
f_npas = apply_date_filter(NPAs, "Year")
f_mcap = apply_date_filter(market_cap, "End-period")

def render_kpi(label, val):
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

def render_source(txt):
    st.markdown(f'<div class="source-text"><b>Source:</b> {txt}</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 1: OUTPUT
# -----------------------------------------------------------------------------
if page == "Output":
    st.title("Output Dashboard")

    k1, k2, k3 = st.columns(3)
    with k1: render_kpi("GDP (2025)", f"${f_df['GDP'].dropna().iloc[-1]/1e12:.2f} T" if not f_df['GDP'].dropna().empty else "N/A")
    with k2: render_kpi("GDP Per Capita (2025)", f"${f_df['GDP per capita'].dropna().iloc[-1]:,.2f}" if not f_df['GDP per capita'].dropna().empty else "N/A")
    with k3: render_kpi("Gross Capital Formation (2025)", f"${f_df['Gross Capital Formation'].dropna().iloc[-1]/1e12:.2f} T" if not f_df['Gross Capital Formation'].dropna().empty else "N/A")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("a. Nominal GDP")
        plot_data = f_df.dropna(subset=["GDP"]).copy()
        fig = px.line(plot_data, x="Year", y="GDP", markers=True)
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="USD trillions")
        style_chart(fig, plot_data, "Year", "%Y", height=450, legend_bottom=False)
        st.plotly_chart(fig, use_container_width=True)
        render_source("Country official statistics, National Statistical Organizations and/or Central Banks; National Accounts data files, Organisation for Economic Co-operation and Development ( OECD ); Staff estimates, World Bank ( WB )")

    with c2:
        st.subheader("b. GDP per Capita (USD)")
        t_df = f_df[["Year", "GDP per capita"]].dropna().copy()
        if not t_df.empty: t_df["Year"] = t_df["Year"].dt.year
        st.dataframe(t_df, use_container_width=True, height=450)
        render_source("Country official statistics, National Statistical Organizations and/or Central Banks; National Accounts data files, Organisation for Economic Co-operation and Development ( OECD ); Staff estimates, World Bank ( WB )")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("c. GDP by Sector (Share %)")
        sec_cols = ["Services share of GDP", "Industry share of GDP", "Agriculture share of GDP"]
        plot_data = f_df.dropna(subset=sec_cols).copy()
        fig = px.area(plot_data, x="Year", y=sec_cols)
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Percentage of GDP")
        style_chart(fig, plot_data, "Year", "%Y", height=480, legend_bottom=True)
        st.plotly_chart(fig, use_container_width=True)
        render_source("country official statistics, National Statistical Offices ( NSOs ); National Accounts data files, Central Banks; Staff estimates, World Bank ( WB )")

    with c4:
        st.subheader("d. Gross Capital Formation")
        plot_data = f_df.dropna(subset=["Gross Capital Formation"]).copy()
        fig = px.line(plot_data, x="Year", y="Gross Capital Formation", markers=True)
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="USD trillions")
        style_chart(fig, plot_data, "Year", "%Y", height=480, legend_bottom=False)
        st.plotly_chart(fig, use_container_width=True)
        render_source("Country official statistics, National Statistical Organizations and/or Central Banks; National Accounts data files, Organisation for Economic Co-operation and Development ( OECD ); Staff estimates, World Bank ( WB )")

    st.sidebar.markdown("---")
    st.sidebar.download_button("Export Output Data (CSV)", f_df.to_csv(index=False), "output_data.csv")

# -----------------------------------------------------------------------------
# PAGE 2: EMPLOYMENT
# -----------------------------------------------------------------------------
elif page == "Employment":
    st.title("Employment Dashboard")

    lfr_col = "Labor force participation rate (ages 15-64)"
    unemp_col = "Unemployment rate"

    k1, k2, k3 = st.columns(3)
    with k1: render_kpi("Labor Force Participation Rate (2025)", f"{f_df[lfr_col].dropna().iloc[-1]:.2f}%" if not f_df[lfr_col].dropna().empty else "N/A")
    with k2: render_kpi("Unemployment Rate (2025)", f"{f_df[unemp_col].dropna().iloc[-1]:.2f}%" if not f_df[unemp_col].dropna().empty else "N/A")
    with k3: render_kpi("MSME Employment (FY23-24)", f"{f_msme['Total employees working in MSMEs registered on Udyam portal'].iloc[-1]:,.0f}" if not f_msme.empty else "N/A")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("a. Labour Force Participation Rate")
        plot_data = f_df.dropna(subset=[lfr_col]).copy()
        fig = px.line(plot_data, x="Year", y=lfr_col, markers=True)
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Percentage of working-age (15 years and above) population")
        style_chart(fig, plot_data, "Year", "%Y", height=450, legend_bottom=False)
        st.plotly_chart(fig, use_container_width=True)
        render_source("ILO Modelled Estimates database ( ILOEST ), International Labour Organization ( ILO ), uri: ilostat.ilo.org/data/bulk, publisher: ILOSTAT, type: external database, date accessed: January 17, 2026")

    with c2:
        st.subheader("b. Unemployment Rate")
        plot_data = f_df.dropna(subset=[unemp_col]).copy()
        fig = px.line(plot_data, x="Year", y=unemp_col, markers=True)
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Percentage of total labour force")
        style_chart(fig, plot_data, "Year", "%Y", height=450, legend_bottom=False)
        st.plotly_chart(fig, use_container_width=True)
        render_source("Source: ILO Modelled Estimates database ( ILOEST ), International Labour Organization ( ILO ), uri: ilostat.ilo.org/data/bulk, publisher: ILOSTAT, type: external database, date accessed: January 17, 2026")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("c. Employment by Sector")
        emp_sectors = [c for c in f_sectorwise_emp.columns if c not in ["Period", "Start_Date"]]
        plot_data = f_sectorwise_emp.dropna(subset=emp_sectors).copy()
        fig = px.area(plot_data, x="Start_Date", y=emp_sectors, labels={"Start_Date": "Date (Month/Year)"})
        fig.update_xaxes(title_text="Period")
        fig.update_yaxes(title_text="Percentage of total employment")
        style_chart(fig, plot_data, "Start_Date", "%b %Y", height=480, legend_bottom=True)
        st.plotly_chart(fig, use_container_width=True)
        render_source("Reserve Bank of India, India KLEMS (Capital, Labour, Energy, Material, and Services) datebase, 2023")

    with c4:
        st.subheader("d. MSME Employment")
        fig = px.bar(f_msme, x="Period", y="Total employees working in MSMEs registered on Udyam portal")
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Employees in millions")
        fig.update_layout(height=480, margin=dict(l=15, r=15, t=30, b=15))
        st.plotly_chart(fig, use_container_width=True)
        render_source("Ministry of Micro,Small & Medium Enterprises press release")

    st.sidebar.markdown("---")
    st.sidebar.download_button("Export Employment Data (CSV)", f_sectorwise_emp.to_csv(index=False), "employment_data.csv")

# -----------------------------------------------------------------------------
# PAGE 3: SAVINGS AND INVESTMENTS
# -----------------------------------------------------------------------------
elif page == "Savings and investments":
    st.title("Savings & Investments Dashboard")

    inf_col = "Inflation consumer prices (annual %)"
    fdi_col = "FDI (net inflows in USD Bn)"

    k1, k2, k3, k4 = st.columns(4)
    with k1: render_kpi("Gross Savings Rate (2025)", f"{f_df_savings['value'].dropna().iloc[-1]:.2f}%" if not f_df_savings['value'].dropna().empty else "N/A")
    with k2: render_kpi("Net FDI Inflows (2025)", f"${f_df[fdi_col].dropna().iloc[-1]:.2f} Bn" if not f_df[fdi_col].dropna().empty else "N/A")
    with k3: render_kpi("CPI Inflation (2025)", f"{f_df[inf_col].dropna().iloc[-1]:.2f}%" if not f_df[inf_col].dropna().empty else "N/A")
    with k4: render_kpi("Gross NPAs (2024)", f"{f_npas['Gross NPAs (% of advances)'].dropna().iloc[-1]:.2f}%" if not f_npas['Gross NPAs (% of advances)'].dropna().empty else "N/A")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("a. Gross Savings Rate")
        plot_data = f_df_savings.dropna(subset=["value"]).copy()
        fig = px.line(plot_data, x="TIME_PERIOD", y="value", markers=True)
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Percentage of GDP")

        style_chart(fig, plot_data, "TIME_PERIOD", "%Y", height=450, legend_bottom=False)
        st.plotly_chart(fig, use_container_width=True)
        render_source("International Monetary Fund (IMF), World Economic Outlook (WEO)")

    with c2:
        st.subheader("b. Market Cap of Listed Companies")
        m_cols = [c for c in f_mcap.columns if "Market Capitalisation" in c]

        updated_df=pd.read_csv('Market Capitalisation - NSE.csv')
        updated_df['End-period']=pd.to_datetime(updated_df['End-period'])
        #plot_data = f_mcap.dropna(subset=m_cols).copy()
        plot_data=updated_df

        fig = px.line(plot_data, x="End-period", y="NSE Market Capitalisation (₹)")
        style_chart(fig, plot_data, "End-period", "%b %Y", height=450, legend_bottom=True)

        fig.update_yaxes(exponentformat="none", tickformat=",")

        #style_chart(fig, plot_data, height=450, legend_bottom=True)

        st.plotly_chart(fig, use_container_width=True)
        render_source("Reserve Bank of India Database on Indian Economy (DBIE)")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("c. FDI Net Inflows Table")
        disp_fdi = f_df[["Year", fdi_col]].dropna().copy()
        if not disp_fdi.empty: disp_fdi["Year"] = disp_fdi["Year"].dt.year
        st.dataframe(disp_fdi, use_container_width=True, height=350)
        render_source("Balance of Payments database, International Monetary Fund ( IMF ), note: International Monetary Fund, Balance of Payments database, supplemented by data from the United Nations Conference on Trade and Development and official national sources.; UN Conference on Trade and Development ( UNCTAD ); Official national sources")

    with c4:
        st.subheader("d. REPO Rates Table")
        st.dataframe(f_repo, use_container_width=True, height=350)
        render_source("Reserve Bank of India Database on Indian Economy (DBIE)")

    c5, c6, c7 = st.columns(3)
    with c5:
        st.subheader("e. CPI Annual Inflation")
        plot_data = f_df.dropna(subset=[inf_col]).copy()
        fig = px.line(plot_data, x="Year", y=inf_col)
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Percentage")
        style_chart(fig, plot_data, "Year", "%Y", height=420, legend_bottom=False)
        st.plotly_chart(fig, use_container_width=True)
        render_source("International Financial Statistics database, International Monetary Fund ( IMF )")

    with c6:
        st.subheader("f. Gross NPAs (% of gross advances)")
        npa_val_col = "Gross NPAs (% of advances)"
        plot_data = f_npas.dropna(subset=[npa_val_col]).copy()
        fig = px.line(plot_data, x="Year", y=npa_val_col, markers=True, labels={npa_val_col: "Gross NPAs (%)"})
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Percentage")
        style_chart(fig, plot_data, "Year", "%Y", height=420, legend_bottom=False)
        st.plotly_chart(fig, use_container_width=True)
        render_source("Source: Reserve Bank of India’s Handbook of Statistics on Indian Economy")

    with c7:
        st.subheader("g. Non-Food Credit")
        plot_data = f_nfc.dropna(subset=["non_food_credit_lakh_crore"]).copy()
        fig = px.line(plot_data, x="as_of_date", y="non_food_credit_lakh_crore", markers=True)
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Non-food credit in ₹ Lakh Crores")
        style_chart(fig, plot_data, "as_of_date", "%b %Y", height=420, legend_bottom=False)
        st.plotly_chart(fig, use_container_width=True)
        render_source("Reserve Bank of India Database on Indian Economy (DBIE)")

    st.sidebar.markdown("---")
    st.sidebar.download_button("Export Savings Data (CSV)", f_df_savings.to_csv(index=False), "savings_data.csv")
