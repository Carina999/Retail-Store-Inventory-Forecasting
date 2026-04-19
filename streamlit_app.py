"""
Retail Demand Forecasting — Streamlit App
==========================================
Reads from RDS MySQL (populated by retail_xgb_sagemaker.ipynb).

Run locally:
    pip install streamlit sqlalchemy pymysql plotly pandas
    streamlit run streamlit_app.py

On Streamlit Cloud, set secrets in .streamlit/secrets.toml:
    [rds]
    host     = "YOUR-RDS-ENDPOINT.rds.amazonaws.com"
    port     = 3306
    db       = "retail_forecast"
    user     = "admin"
    password = "YOUR-PASSWORD"
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecast",
    page_icon="📦",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-label  { font-size: 12px; color: #6b7280; font-weight: 500; }
    .metric-value  { font-size: 24px; font-weight: 600; color: #111827; line-height: 1.2; }
    .metric-delta  { font-size: 12px; margin-top: 2px; }
    .section-title { font-size: 16px; font-weight: 600; color: #111827; margin-bottom: 4px; }
    .section-sub   { font-size: 12px; color: #9ca3af; margin-bottom: 16px; }
    div[data-testid="stAlert"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── DB connection ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    cfg = st.secrets["rds"]
    url = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['db']}"
    )
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(ttl=300)   # refresh every 5 min
def load_predictions() -> pd.DataFrame:
    engine = get_engine()
    sql = text("""
        SELECT
            p.obs_date,
            p.store_id,
            p.product_id,
            p.category,
            p.region,
            p.actual_units,
            p.predicted_units,
            p.predicted_lower,
            p.predicted_upper,
            p.abs_error,
            a.price,
            a.discount,
            a.is_promo
        FROM predictions p
        JOIN actuals a
          ON  a.store_id   = p.store_id
          AND a.product_id = p.product_id
          AND a.obs_date   = p.obs_date
        WHERE p.run_timestamp = (SELECT MAX(run_timestamp) FROM predictions)
        ORDER BY p.obs_date, p.store_id, p.product_id
    """)
    df = pd.read_sql(sql, engine)
    df['obs_date'] = pd.to_datetime(df['obs_date'])
    return df


@st.cache_data(ttl=300)
def load_metrics() -> pd.DataFrame:
    engine = get_engine()
    sql = text("""
        SELECT rmse, mae, r2, mape_pct, train_time_s, split_date, run_timestamp
        FROM metrics
        WHERE run_timestamp = (SELECT MAX(run_timestamp) FROM metrics)
    """)
    return pd.read_sql(sql, engine)


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df_raw   = load_predictions()
    df_met   = load_metrics()
    db_ok    = True
except Exception as e:
    st.error(f"Cannot connect to RDS: {e}")
    st.info("Update your `.streamlit/secrets.toml` with valid RDS credentials and re-run.")
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📦 Retail Demand Forecast")
st.markdown(
    f"XGBoost predictions · test period "
    f"`{df_raw['obs_date'].min().date()}` → `{df_raw['obs_date'].max().date()}`"
)
st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Filters
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Filters")

    all_stores   = sorted(df_raw['store_id'].unique())
    all_products = sorted(df_raw['product_id'].unique())
    all_cats     = sorted(df_raw['category'].unique())
    all_regions  = sorted(df_raw['region'].unique())

    sel_stores = st.multiselect(
        "Store", all_stores, default=all_stores,
        help="Select one or more stores"
    )
    sel_cats = st.multiselect(
        "Category", all_cats, default=all_cats
    )
    sel_regions = st.multiselect(
        "Region", all_regions, default=all_regions
    )

    # Dynamic product list based on category selection
    available_products = sorted(
        df_raw[df_raw['category'].isin(sel_cats)]['product_id'].unique()
    )
    sel_products = st.multiselect(
        "Product", available_products, default=available_products,
        help="Filtered by selected categories"
    )

    date_range = st.date_input(
        "Date range",
        value=(df_raw['obs_date'].min().date(), df_raw['obs_date'].max().date()),
        min_value=df_raw['obs_date'].min().date(),
        max_value=df_raw['obs_date'].max().date(),
    )

    st.divider()
    alert_days = st.slider(
        "Inventory alert threshold (days of supply)",
        min_value=1, max_value=14, value=5,
        help="SKUs with fewer days of forecasted supply than this will be flagged"
    )

    st.divider()
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    # Model KPIs in sidebar
    if not df_met.empty:
        row = df_met.iloc[0]
        st.markdown("### Model performance")
        st.metric("RMSE",   f"{row['rmse']:.2f}")
        st.metric("MAE",    f"{row['mae']:.2f}")
        st.metric("R²",     f"{row['r2']:.4f}")
        st.metric("MAPE",   f"{row['mape_pct']:.2f}%")


# ── Apply filters ─────────────────────────────────────────────────────────────
# Resolve date_range (may be a single date if user clicks once)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    d_start, d_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    d_start = d_end = pd.Timestamp(date_range)

df = df_raw[
    df_raw['store_id'].isin(sel_stores) &
    df_raw['product_id'].isin(sel_products) &
    df_raw['category'].isin(sel_cats) &
    df_raw['region'].isin(sel_regions) &
    (df_raw['obs_date'] >= d_start) &
    (df_raw['obs_date'] <= d_end)
].copy()

if df.empty:
    st.warning("No data matches the current filters. Adjust sidebar selections.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Forecast Explorer
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Forecast Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Daily aggregated actual vs XGBoost prediction with 80% confidence interval</div>',
    unsafe_allow_html=True
)

# KPI row
daily = df.groupby('obs_date').agg(
    actual    =('actual_units',    'sum'),
    predicted =('predicted_units', 'sum'),
    abs_error =('abs_error',       'sum'),
).reset_index()

# CI computed at daily-aggregated level from daily residuals.
# Summing per-record lower/upper would over-inflate the band by ~sqrt(n).
# Instead: residual = actual_daily - predicted_daily, take 10th/90th percentile.
daily['residual'] = daily['actual'] - daily['predicted']
ci_lo = float(np.percentile(daily['residual'], 10))
ci_hi = float(np.percentile(daily['residual'], 90))
daily['lower'] = (daily['predicted'] + ci_lo).clip(lower=0)
daily['upper'] = (daily['predicted'] + ci_hi).clip(lower=0)

total_actual    = daily['actual'].sum()
total_predicted = daily['predicted'].sum()
overall_err_pct = abs(total_actual - total_predicted) / total_actual * 100 if total_actual else 0
avg_daily_err   = daily['abs_error'].mean()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total actual units",     f"{total_actual:,.0f}")
with k2:
    st.metric("Total predicted units",  f"{total_predicted:,.0f}")
with k3:
    st.metric("Overall error",          f"{overall_err_pct:.1f}%")
with k4:
    st.metric("Avg daily MAE",          f"{avg_daily_err:,.1f}")

# ── Chart 1: Daily time series with CI ────────────────────────────────────────
fig = go.Figure()

# CI band
fig.add_trace(go.Scatter(
    x=pd.concat([daily['obs_date'], daily['obs_date'][::-1]]),
    y=pd.concat([daily['upper'], daily['lower'][::-1]]),
    fill='toself',
    fillcolor='rgba(52, 152, 219, 0.12)',
    line=dict(color='rgba(0,0,0,0)'),
    name='80% CI',
    hoverinfo='skip',
))
# Predicted line
fig.add_trace(go.Scatter(
    x=daily['obs_date'], y=daily['predicted'],
    mode='lines',
    line=dict(color='#3498db', width=2, dash='dash'),
    name='XGBoost predicted',
))
# Actual line
fig.add_trace(go.Scatter(
    x=daily['obs_date'], y=daily['actual'],
    mode='lines',
    line=dict(color='#2c3e50', width=2.5),
    name='Actual',
))

fig.update_layout(
    height=340,
    margin=dict(l=0, r=0, t=8, b=0),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor='#f3f4f6', title='Units sold'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    hovermode='x unified',
)
st.plotly_chart(fig, use_container_width=True)


# ── Chart 2: By category (bar) ────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    cat_agg = df.groupby('category').agg(
        actual    =('actual_units',    'sum'),
        predicted =('predicted_units', 'sum'),
    ).reset_index()

    fig_cat = go.Figure()
    fig_cat.add_bar(x=cat_agg['category'], y=cat_agg['actual'],    name='Actual',    marker_color='#2c3e50')
    fig_cat.add_bar(x=cat_agg['category'], y=cat_agg['predicted'], name='Predicted', marker_color='#3498db', opacity=0.8)
    fig_cat.update_layout(
        title=dict(text='Total units by category', font=dict(size=13)),
        barmode='group', height=280,
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='#f3f4f6'),
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with c2:
    region_agg = df.groupby('region').agg(
        actual    =('actual_units',    'sum'),
        predicted =('predicted_units', 'sum'),
    ).reset_index()

    fig_reg = go.Figure()
    fig_reg.add_bar(x=region_agg['region'], y=region_agg['actual'],    name='Actual',    marker_color='#2c3e50')
    fig_reg.add_bar(x=region_agg['region'], y=region_agg['predicted'], name='Predicted', marker_color='#3498db', opacity=0.8)
    fig_reg.update_layout(
        title=dict(text='Total units by region', font=dict(size=13)),
        barmode='group', height=280,
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='#f3f4f6'),
    )
    st.plotly_chart(fig_reg, use_container_width=True)


# ── Chart 3: Error distribution ────────────────────────────────────────────────
with st.expander("Error distribution (click to expand)"):
    fig_err = px.histogram(
        df, x='abs_error', nbins=50,
        color_discrete_sequence=['#3498db'],
        title='Absolute error distribution across all records',
        labels={'abs_error': 'Absolute error (units)', 'count': 'Records'},
    )
    fig_err.update_layout(
        height=260, margin=dict(l=0, r=0, t=36, b=0),
        plot_bgcolor='white', paper_bgcolor='white',
        bargap=0.05,
    )
    st.plotly_chart(fig_err, use_container_width=True)


st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Inventory Alerts
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Inventory Alerts</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Based on latest test-period predictions — SKUs sorted by urgency</div>',
    unsafe_allow_html=True
)

# Build per-SKU summary: use latest available date per SKU
latest_date = df.groupby(['store_id', 'product_id'])['obs_date'].max().reset_index()
latest_date.columns = ['store_id', 'product_id', 'latest_date']

df_latest = df.merge(latest_date, on=['store_id', 'product_id'])
df_latest = df_latest[df_latest['obs_date'] == df_latest['latest_date']].copy()

# Aggregate to SKU level (sum over the filtered date)
sku_agg = df.groupby(['store_id', 'product_id', 'category', 'region']).agg(
    total_actual    =('actual_units',    'sum'),
    total_predicted =('predicted_units', 'sum'),
    avg_predicted   =('predicted_units', 'mean'),
).reset_index()

# Days of supply: total_actual / avg daily predicted demand
sku_agg['days_of_supply'] = (
    sku_agg['total_actual'] / sku_agg['avg_predicted'].replace(0, np.nan)
).round(1)

# Alert level
def alert_level(days):
    if pd.isna(days) or days <= alert_days * 0.4:
        return 'Critical'
    elif days <= alert_days:
        return 'Low stock'
    else:
        return 'Adequate'

sku_agg['alert'] = sku_agg['days_of_supply'].apply(alert_level)
sku_agg = sku_agg.sort_values(['alert', 'days_of_supply'], ascending=[True, True])

# Summary counts
n_critical  = (sku_agg['alert'] == 'Critical').sum()
n_low       = (sku_agg['alert'] == 'Low stock').sum()
n_adequate  = (sku_agg['alert'] == 'Adequate').sum()

a1, a2, a3 = st.columns(3)
with a1:
    st.metric("🔴 Critical SKUs",  n_critical,  help=f"Days of supply < {int(alert_days * 0.4)}")
with a2:
    st.metric("🟡 Low stock SKUs", n_low,       help=f"Days of supply {int(alert_days * 0.4)}–{alert_days}")
with a3:
    st.metric("🟢 Adequate SKUs",  n_adequate,  help=f"Days of supply > {alert_days}")

# ── Donut chart ───────────────────────────────────────────────────────────────
col_donut, col_table = st.columns([1, 2])

with col_donut:
    labels = ['Critical', 'Low stock', 'Adequate']
    values = [n_critical, n_low, n_adequate]
    colors = ['#e74c3c', '#f39c12', '#27ae60']

    fig_donut = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.6,
        marker_colors=colors,
        textinfo='percent+label',
        textfont_size=12,
    ))
    fig_donut.update_layout(
        height=240, margin=dict(l=0, r=0, t=8, b=0),
        showlegend=False,
        paper_bgcolor='white',
        annotations=[dict(
            text=f'{len(sku_agg)}<br>SKUs', x=0.5, y=0.5,
            font_size=14, showarrow=False
        )]
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_table:
    # Show all alerts or just at-risk
    show_all = st.toggle("Show all SKUs", value=False)
    display_df = sku_agg if show_all else sku_agg[sku_agg['alert'] != 'Adequate']

    # Colour-coded alert column
    def style_alert(val):
        colors = {
            'Critical': 'background-color:#fde8e8; color:#991b1b;',
            'Low stock': 'background-color:#fef3c7; color:#92400e;',
            'Adequate':  'background-color:#d1fae5; color:#065f46;',
        }
        return colors.get(val, '')

    styled = (
        display_df[[
            'store_id', 'product_id', 'category', 'region',
            'total_actual', 'avg_predicted', 'days_of_supply', 'alert'
        ]]
        .rename(columns={
            'store_id':      'Store',
            'product_id':    'Product',
            'category':      'Category',
            'region':        'Region',
            'total_actual':  'Stock (units)',
            'avg_predicted': 'Daily forecast',
            'days_of_supply':'Days of supply',
            'alert':         'Status',
        })
        .style
        .map(style_alert, subset=['Status'])
        .format({'Stock (units)': '{:,.0f}', 'Daily forecast': '{:.1f}', 'Days of supply': '{:.1f}'})
    )
    st.dataframe(styled, use_container_width=True, height=220)


# ── Days-of-supply bar chart ───────────────────────────────────────────────────
at_risk = sku_agg[sku_agg['alert'] != 'Adequate'].head(20).copy()
at_risk['sku'] = at_risk['store_id'] + ' · ' + at_risk['product_id']
at_risk['color'] = at_risk['alert'].map({'Critical': '#e74c3c', 'Low stock': '#f39c12'})

if not at_risk.empty:
    fig_bar = go.Figure(go.Bar(
        x=at_risk['sku'],
        y=at_risk['days_of_supply'],
        marker_color=at_risk['color'],
        text=at_risk['days_of_supply'].round(1),
        textposition='outside',
    ))
    fig_bar.add_hline(
        y=alert_days, line_dash='dash', line_color='#6b7280',
        annotation_text=f'Threshold ({alert_days}d)', annotation_position='top right'
    )
    fig_bar.update_layout(
        title=dict(text='Days of supply — at-risk SKUs (top 20)', font=dict(size=13)),
        height=300, margin=dict(l=0, r=0, t=36, b=0),
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(showgrid=False, tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#f3f4f6', title='Days of supply'),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.success("No at-risk SKUs with current filters and threshold.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
if not df_met.empty:
    row = df_met.iloc[0]
    st.caption(
        f"Model: XGBoost · Split date: {row['split_date']} · "
        f"Run: {row['run_timestamp']} · "
        f"RMSE {row['rmse']:.2f} · MAE {row['mae']:.2f} · R² {row['r2']:.4f}"
    )
