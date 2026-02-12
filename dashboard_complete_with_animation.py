"""
Equipment Asset Optimization Dashboard
Chicago Department of Procurement Services

Interactive dashboard for exploring 15-year simulation results.

Installation:
    pip install streamlit plotly pandas numpy

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx

# Page config
st.set_page_config(
    page_title="Equipment Optimization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (simplified to avoid conflicts)
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #028090;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 18px;
        color: #64748B;
        margin-bottom: 30px;
    }
    /* Remove conflicting metric styles */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ADOPTION CURVES FUNCTIONS
# ============================================================================

def s_curve(year, ramp_up_years=3, max_multiplier=1.0, steepness=1.0):
    """
    S-curve (logistic function) for adoption rate.
    
    Parameters:
    - year: Years since start (0, 1, 2, ...)
    - ramp_up_years: Years to reach 50% of max
    - max_multiplier: Maximum multiplier (1.0 = current savings, 1.5 = 50% more)
    - steepness: How steep the curve is (higher = faster adoption)
    
    Returns:
    - Multiplier to apply to baseline savings (0.0 to max_multiplier)
    """
    import numpy as np
    
    # Logistic function: f(x) = L / (1 + e^(-k*(x - x0)))
    L = max_multiplier  # Maximum value
    x0 = ramp_up_years  # Midpoint
    k = steepness       # Steepness
    
    if year <= 0:
        return 0.0  # Year 0: no savings yet
    
    multiplier = L / (1 + np.exp(-k * (year - x0)))
    return multiplier

def apply_adoption_curves(simulation_df, pillar_configs, enable_curves=True):
    """
    Apply adoption curves to simulation data.
    
    Parameters:
    - simulation_df: Original simulation data
    - pillar_configs: Dict with config for each pillar
    - enable_curves: If False, return original data
    
    Returns:
    - DataFrame with adjusted savings
    """
    if not enable_curves:
        return simulation_df
    
    adjusted_df = simulation_df.copy()
    
    # Get start year
    start_year = adjusted_df['Year'].min()
    
    for _, row in adjusted_df.iterrows():
        year = row['Year']
        years_since_start = year - start_year
        idx = row.name
        
        # Apply curve to each pillar
        for pillar_name, config in pillar_configs.items():
            if config['enabled']:
                multiplier = s_curve(
                    years_since_start,
                    ramp_up_years=config['ramp_up'],
                    max_multiplier=config['max_multiplier'],
                    steepness=config['steepness']
                )
                
                # Get column name for this pillar
                col_mapping = {
                    'Predictive Analytics': 'Savings_Predictive_Analytics_Total',
                    'AI Procurement': 'Savings_AI_Procurement_Total',
                    'Asset Lifecycle': 'Savings_Asset_Lifecycle_Refurb',
                    'Automation': 'Savings_Automation',
                    'Energy Efficiency': 'Savings_Energy_Efficiency_Net',
                    'Circular Economy': 'Savings_Circular_Economy_Total',
                    'Bulk Purchasing': 'Savings_Bulk_Purchasing'
                }
                
                col_name = col_mapping.get(pillar_name)
                if col_name and col_name in adjusted_df.columns:
                    adjusted_df.at[idx, col_name] = row[col_name] * multiplier
    
    # Recalculate totals
    adjusted_df['Total_Savings_All_Pillars'] = (
        adjusted_df['Savings_Predictive_Analytics_Total'] +
        adjusted_df['Savings_AI_Procurement_Total'] +
        adjusted_df['Savings_Asset_Lifecycle_Refurb'] +
        adjusted_df['Savings_Automation'] +
        adjusted_df['Savings_Energy_Efficiency_Net'] +
        adjusted_df['Savings_Circular_Economy_Total'] +
        adjusted_df['Savings_Bulk_Purchasing']
    )
    
    adjusted_df['Total_Spend_Optimized'] = (
        adjusted_df['Total_Spend_Baseline'] - adjusted_df['Total_Savings_All_Pillars']
    )
    
    return adjusted_df

# ============================================================================
# HELPER FUNCTIONS FOR NETWORK VISUALIZATIONS
# ============================================================================

@st.cache_data(show_spinner=False)
def load_enriched_data():
    """Load enriched data with Super Structure columns"""
    try:
        df = pd.read_excel("Equipment_Simulation_SuperStructure_Optimized.xlsx", 
                          sheet_name="Enriched_Data")
        return df
    except FileNotFoundError:
        return None

@st.cache_data(show_spinner=False)
def calculate_network_layout(_df, top_n_programs=20):
    """Calculate force-directed network layout (cached for performance)"""
    program_totals = _df.groupby('L3_Program').agg({
        'Total_Spend_Baseline': 'sum'
    }).reset_index().sort_values('Total_Spend_Baseline', ascending=False)
    
    top_programs = program_totals.head(top_n_programs)['L3_Program'].tolist()
    df_filtered = _df[_df['L3_Program'].isin(top_programs)].copy()
    
    G = nx.DiGraph()
    
    for org in df_filtered['L0_Organization'].unique():
        G.add_node(org, level='L0')
    for sector in df_filtered['L1_Sector'].unique():
        G.add_node(sector, level='L1')
    for dept in df_filtered['L2_Department'].unique():
        G.add_node(dept, level='L2')
    for prog in top_programs:
        G.add_node(prog, level='L3')
    for stakeholder in df_filtered['L5_Stakeholder'].unique():
        G.add_node(stakeholder, level='L5')
    
    for sector in df_filtered['L1_Sector'].unique():
        G.add_edge('City of Chicago', sector)
        for dept in df_filtered[df_filtered['L1_Sector'] == sector]['L2_Department'].unique():
            G.add_edge(sector, dept)
    
    for dept in df_filtered['L2_Department'].unique():
        for prog in df_filtered[df_filtered['L2_Department'] == dept]['L3_Program'].unique():
            if prog in top_programs:
                G.add_edge(dept, prog)
    
    for prog in top_programs:
        for stakeholder in df_filtered[df_filtered['L3_Program'] == prog]['L5_Stakeholder'].unique():
            G.add_edge(prog, stakeholder)
    
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
    
    return G, pos, df_filtered

def get_savings_color(savings_pct):
    """Return color based on savings percentage"""
    if savings_pct < 10:
        return '#FFC107'
    elif savings_pct < 20:
        return '#8BC34A'
    elif savings_pct < 30:
        return '#4CAF50'
    else:
        return '#2E7D32'

# ============================================================================
# LOAD DATA FUNCTION (user must provide their simulation results)
# ============================================================================

@st.cache_data
def load_data():
    """
    Load simulation data from Excel file.
    
    User should have these sheets:
    - Simulation_Annual (main output)
    - Summary_Annual
    - Summary_Category
    - Summary_Department
    - Pillar_Contribution
    """
    
    # IMPORTANT: Replace with actual file path
    file_path = "Equipment_Simulation_Results_Complete.xlsx"
    
    try:
        simulation = pd.read_excel(file_path, sheet_name="Simulation_Annual")
        annual_summary = pd.read_excel(file_path, sheet_name="Summary_Annual")
        category_summary = pd.read_excel(file_path, sheet_name="Summary_Category")
        dept_summary = pd.read_excel(file_path, sheet_name="Summary_Department")
        pillar_contribution = pd.read_excel(file_path, sheet_name="Pillar_Contribution")
        
        return {
            "simulation": simulation,
            "annual": annual_summary,
            "category": category_summary,
            "department": dept_summary,
            "pillars": pillar_contribution
        }
    except FileNotFoundError:
        st.error(f"❌ File not found: {file_path}")
        st.info("Please update the file_path in load_data() function")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# Load data
data = load_data()

# ============================================================================
# SIDEBAR - FILTERS
# ============================================================================

st.sidebar.markdown("## 🎛️ Filters")

# Year range slider
year_min = int(data["simulation"]["Year"].min())
year_max = int(data["simulation"]["Year"].max())

selected_years = st.sidebar.slider(
    "Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

# Category filter
categories = ["All"] + sorted(data["simulation"]["Equipment_Category"].unique().tolist())
selected_category = st.sidebar.selectbox("Equipment Category", categories)

# Department filter
departments = ["All"] + sorted(data["simulation"]["Department"].unique().tolist())
selected_department = st.sidebar.selectbox("Department", departments)

# Pillar toggles
st.sidebar.markdown("### 🎯 Optimization Pillars")
pillar_toggles = {
    "Predictive Analytics": st.sidebar.checkbox("Predictive Analytics", value=True),
    "AI Procurement": st.sidebar.checkbox("AI Procurement", value=True),
    "Asset Lifecycle": st.sidebar.checkbox("Asset Lifecycle", value=True),
    "Automation": st.sidebar.checkbox("Automation", value=True),
    "Energy Efficiency": st.sidebar.checkbox("Energy Efficiency", value=True),
    "Circular Economy": st.sidebar.checkbox("Circular Economy", value=True),
    "Bulk Purchasing": st.sidebar.checkbox("Bulk Purchasing", value=True),
}

st.sidebar.markdown("---")

# Adoption Curves Configuration
st.sidebar.markdown("### 📈 Adoption Curves")
enable_curves = st.sidebar.checkbox("Enable Adoption Curves", value=False, 
    help="Apply S-curve adoption model instead of constant savings")

if enable_curves:
    st.sidebar.info("💡 Tip: Lower ramp-up = faster adoption. Higher max = more savings.")
    
    # Expandable sections for each pillar
    pillar_configs = {}
    
    with st.sidebar.expander("⚙️ Predictive Analytics"):
        pillar_configs['Predictive Analytics'] = {
            'enabled': pillar_toggles['Predictive Analytics'],
            'ramp_up': st.slider("Ramp-up (years)", 1, 5, 3, key='pred_ramp'),
            'max_multiplier': st.slider("Max Savings Multiplier", 0.5, 2.0, 1.75, 0.25, key='pred_max'),
            'steepness': st.slider("Curve Steepness", 0.5, 2.0, 1.0, 0.25, key='pred_steep')
        }
    
    with st.sidebar.expander("⚙️ AI Procurement"):
        pillar_configs['AI Procurement'] = {
            'enabled': pillar_toggles['AI Procurement'],
            'ramp_up': st.slider("Ramp-up (years)", 1, 5, 2, key='ai_ramp'),
            'max_multiplier': st.slider("Max Savings Multiplier", 0.5, 2.0, 1.5, 0.25, key='ai_max'),
            'steepness': st.slider("Curve Steepness", 0.5, 2.0, 1.2, 0.25, key='ai_steep')
        }
    
    with st.sidebar.expander("⚙️ Asset Lifecycle"):
        pillar_configs['Asset Lifecycle'] = {
            'enabled': pillar_toggles['Asset Lifecycle'],
            'ramp_up': st.slider("Ramp-up (years)", 1, 5, 2, key='asset_ramp'),
            'max_multiplier': st.slider("Max Savings Multiplier", 0.5, 2.0, 1.6, 0.25, key='asset_max'),
            'steepness': st.slider("Curve Steepness", 0.5, 2.0, 1.0, 0.25, key='asset_steep')
        }
    
    with st.sidebar.expander("⚙️ Automation"):
        pillar_configs['Automation'] = {
            'enabled': pillar_toggles['Automation'],
            'ramp_up': st.slider("Ramp-up (years)", 1, 5, 4, key='auto_ramp'),
            'max_multiplier': st.slider("Max Savings Multiplier", 0.5, 2.0, 1.3, 0.25, key='auto_max'),
            'steepness': st.slider("Curve Steepness", 0.5, 2.0, 0.8, 0.25, key='auto_steep')
        }
    
    with st.sidebar.expander("⚙️ Energy Efficiency"):
        pillar_configs['Energy Efficiency'] = {
            'enabled': pillar_toggles['Energy Efficiency'],
            'ramp_up': st.slider("Ramp-up (years)", 1, 5, 1, key='energy_ramp'),
            'max_multiplier': st.slider("Max Savings Multiplier", 0.5, 2.0, 1.8, 0.25, key='energy_max'),
            'steepness': st.slider("Curve Steepness", 0.5, 2.0, 1.5, 0.25, key='energy_steep')
        }
    
    with st.sidebar.expander("⚙️ Circular Economy"):
        pillar_configs['Circular Economy'] = {
            'enabled': pillar_toggles['Circular Economy'],
            'ramp_up': st.slider("Ramp-up (years)", 1, 5, 3, key='circ_ramp'),
            'max_multiplier': st.slider("Max Savings Multiplier", 0.5, 2.0, 1.4, 0.25, key='circ_max'),
            'steepness': st.slider("Curve Steepness", 0.5, 2.0, 0.9, 0.25, key='circ_steep')
        }
    
    with st.sidebar.expander("⚙️ Bulk Purchasing"):
        pillar_configs['Bulk Purchasing'] = {
            'enabled': pillar_toggles['Bulk Purchasing'],
            'ramp_up': st.slider("Ramp-up (years)", 1, 5, 1, key='bulk_ramp'),
            'max_multiplier': st.slider("Max Savings Multiplier", 0.5, 2.0, 1.2, 0.25, key='bulk_max'),
            'steepness': st.slider("Curve Steepness", 0.5, 2.0, 1.5, 0.25, key='bulk_steep')
        }
else:
    # Default config (no curves)
    pillar_configs = {
        name: {
            'enabled': enabled,
            'ramp_up': 0,
            'max_multiplier': 1.0,
            'steepness': 1.0
        }
        for name, enabled in pillar_toggles.items()
    }

# Reset button
if st.sidebar.button("🔄 Reset Filters"):
    st.rerun()

# Apply filters
filtered_data = data["simulation"].copy()
filtered_data = filtered_data[
    (filtered_data["Year"] >= selected_years[0]) &
    (filtered_data["Year"] <= selected_years[1])
]

if selected_category != "All":
    filtered_data = filtered_data[filtered_data["Equipment_Category"] == selected_category]

if selected_department != "All":
    filtered_data = filtered_data[filtered_data["Department"] == selected_department]

# Apply adoption curves if enabled
filtered_data = apply_adoption_curves(filtered_data, pillar_configs, enable_curves)

# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown('<div class="main-header">🏢 Equipment Asset Optimization Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Chicago Department of Procurement Services | 15-Year Simulation Results</div>', unsafe_allow_html=True)

# ============================================================================
# TAB NAVIGATION
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Overview",
    "📈 Trends",
    "🏷️ Categories",
    "🎯 Pillars",
    "📅 Replacement Planning",
    "🌊 Sankey Flow",
    "⚡ CAPEX Collision",
    "🎬 Collision Animation"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tab1:
    st.markdown("## Overview & Key Metrics")
    
    # KPI Cards (fixed styling)
    col1, col2, col3, col4 = st.columns(4)
    
    total_baseline = filtered_data["Total_Spend_Baseline"].sum()
    total_optimized = filtered_data["Total_Spend_Optimized"].sum()
    total_savings = filtered_data["Total_Savings_All_Pillars"].sum()
    savings_pct = (total_savings / total_baseline * 100) if total_baseline > 0 else 0
    
    with col1:
        st.markdown(f"""
        <div style='background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #028090; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <p style='color: #64748B; font-size: 14px; margin: 0;'>💰 Total Savings</p>
            <h2 style='color: #2C2C2C; font-size: 32px; margin: 10px 0;'>${total_savings/1e6:.0f}M</h2>
            <p style='color: #10B981; font-size: 14px; margin: 0;'>↑ {savings_pct:.1f}% reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_annual = total_savings / len(filtered_data["Year"].unique())
        st.markdown(f"""
        <div style='background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #00A896; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <p style='color: #64748B; font-size: 14px; margin: 0;'>📊 Avg Annual Savings</p>
            <h2 style='color: #2C2C2C; font-size: 32px; margin: 10px 0;'>${avg_annual/1e6:.0f}M</h2>
            <p style='color: #10B981; font-size: 14px; margin: 0;'>↑ Per year</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        top_pillar = data["pillars"].iloc[0]
        st.markdown(f"""
        <div style='background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #02C39A; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <p style='color: #64748B; font-size: 14px; margin: 0;'>🎯 Top Pillar</p>
            <h2 style='color: #2C2C2C; font-size: 24px; margin: 10px 0;'>{top_pillar["Pillar"]}</h2>
            <p style='color: #10B981; font-size: 14px; margin: 0;'>↑ {top_pillar["Pct_of_Total"]:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        replacements = filtered_data["Has_Replacement_This_Year"].sum()
        refurbs = filtered_data["Has_Refurbishment_This_Year"].sum()
        refurb_rate = (refurbs / (replacements + refurbs) * 100) if (replacements + refurbs) > 0 else 0
        st.markdown(f"""
        <div style='background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #1E2761; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <p style='color: #64748B; font-size: 14px; margin: 0;'>🔧 Refurbishment Rate</p>
            <h2 style='color: #2C2C2C; font-size: 32px; margin: 10px 0;'>{refurb_rate:.1f}%</h2>
            <p style='color: #10B981; font-size: 14px; margin: 0;'>↑ {int(refurbs)} refurbs</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main chart: Baseline vs Optimized
    st.markdown("### Baseline vs Optimized Spend Over Time")
    
    # Show info if curves are enabled
    if enable_curves:
        st.success("📈 **Adoption Curves Active** - Savings grow non-linearly following S-curve model. Adjust parameters in sidebar.")
    else:
        st.warning("⚠️ **Constant Savings Model** - Enable 'Adoption Curves' in sidebar for realistic non-linear growth.")
    
    annual_data = filtered_data.groupby("Year").agg({
        "Total_Spend_Baseline": "sum",
        "Total_Spend_Optimized": "sum",
        "Total_Savings_All_Pillars": "sum"
    }).reset_index()
    
    fig_main = go.Figure()
    
    # Baseline line
    fig_main.add_trace(go.Scatter(
        x=annual_data["Year"],
        y=annual_data["Total_Spend_Baseline"] / 1e6,
        mode='lines+markers',
        name='Baseline',
        line=dict(color='#E63946', width=3),
        marker=dict(size=8)
    ))
    
    # Optimized line
    fig_main.add_trace(go.Scatter(
        x=annual_data["Year"],
        y=annual_data["Total_Spend_Optimized"] / 1e6,
        mode='lines+markers',
        name='Optimized',
        line=dict(color='#028090', width=3),
        marker=dict(size=8)
    ))
    
    # Savings area
    fig_main.add_trace(go.Scatter(
        x=annual_data["Year"],
        y=annual_data["Total_Spend_Baseline"] / 1e6,
        fill='tonexty',
        mode='none',
        name='Savings',
        fillcolor='rgba(2, 128, 144, 0.2)',
        showlegend=True
    ))
    
    fig_main.update_layout(
        xaxis_title="Year",
        yaxis_title="Total Spend ($ Millions)",
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_main, use_container_width=True)
    
    # Show adoption curve visualization if enabled
    if enable_curves:
        st.markdown("### 📈 Adoption Curves by Pillar")
        st.caption("Visualización de cómo cada pilar incrementa sus savings con el tiempo")
        
        # Create visualization of adoption curves
        years_range = list(range(0, 16))  # 0-15 years
        
        fig_curves = go.Figure()
        
        colors_map = {
            'Predictive Analytics': '#028090',
            'AI Procurement': '#00A896',
            'Asset Lifecycle': '#02C39A',
            'Automation': '#14B8A6',
            'Energy Efficiency': '#5EEAD4',
            'Circular Economy': '#99F6E4',
            'Bulk Purchasing': '#CCFBF1'
        }
        
        for pillar_name, config in pillar_configs.items():
            if config['enabled']:
                multipliers = [
                    s_curve(y, config['ramp_up'], config['max_multiplier'], config['steepness']) 
                    for y in years_range
                ]
                
                fig_curves.add_trace(go.Scatter(
                    x=years_range,
                    y=multipliers,
                    mode='lines+markers',
                    name=pillar_name,
                    line=dict(width=3, color=colors_map.get(pillar_name, '#028090')),
                    marker=dict(size=6)
                ))
        
        fig_curves.update_layout(
            xaxis_title="Years Since Implementation",
            yaxis_title="Savings Multiplier",
            hovermode='x unified',
            height=400,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.15),
            showlegend=True
        )
        
        # Add reference line at 1.0
        fig_curves.add_hline(
            y=1.0, 
            line_dash="dash", 
            line_color="gray", 
            annotation_text="Baseline (1.0x)",
            annotation_position="right"
        )
        
        st.plotly_chart(fig_curves, use_container_width=True)
        
        st.info("💡 **Interpretation:** Multiplier 1.0 = baseline savings | 1.5 = 50% more savings | 2.0 = double savings")
    
    # Two-column charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Pillar Contribution")
        
        fig_pie = px.pie(
            data["pillars"],
            values="Total_Savings",
            names="Pillar",
            color_discrete_sequence=px.colors.sequential.Teal
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400)
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### OPEX vs CAPEX Breakdown")
        
        opex_capex = filtered_data.groupby("Year").agg({
            "OPEX_Optimized": "sum",
            "CAPEX_Optimized": "sum"
        }).reset_index()
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=opex_capex["Year"],
            y=opex_capex["OPEX_Optimized"] / 1e6,
            name='OPEX',
            marker_color='#028090'
        ))
        fig_bar.add_trace(go.Bar(
            x=opex_capex["Year"],
            y=opex_capex["CAPEX_Optimized"] / 1e6,
            name='CAPEX',
            marker_color='#02C39A'
        ))
        
        fig_bar.update_layout(
            barmode='stack',
            xaxis_title="Year",
            yaxis_title="$ Millions",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================================
# TAB 2: TRENDS
# ============================================================================

with tab2:
    st.markdown("## Trends Analysis")
    
    # Savings trend over time
    st.markdown("### Savings Trend Over Time")
    
    fig_trends = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Savings amount (bars)
    fig_trends.add_trace(
        go.Bar(
            x=annual_data["Year"],
            y=annual_data["Total_Savings_All_Pillars"] / 1e6,
            name="Savings ($M)",
            marker_color='#028090',
            opacity=0.7
        ),
        secondary_y=False
    )
    
    # Savings percentage (line)
    savings_pct_trend = (annual_data["Total_Savings_All_Pillars"] / annual_data["Total_Spend_Baseline"] * 100)
    fig_trends.add_trace(
        go.Scatter(
            x=annual_data["Year"],
            y=savings_pct_trend,
            name="Savings %",
            mode='lines+markers',
            line=dict(color='#E63946', width=3),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    fig_trends.update_xaxes(title_text="Year")
    fig_trends.update_yaxes(title_text="Savings ($ Millions)", secondary_y=False)
    fig_trends.update_yaxes(title_text="Savings (%)", secondary_y=True)
    fig_trends.update_layout(height=500, hovermode='x unified')
    
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # OPEX and CAPEX trends side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### OPEX Trend")
        
        opex_trend = filtered_data.groupby("Year").agg({
            "OPEX_Baseline": "sum",
            "OPEX_Optimized": "sum"
        }).reset_index()
        
        fig_opex = go.Figure()
        fig_opex.add_trace(go.Scatter(
            x=opex_trend["Year"],
            y=opex_trend["OPEX_Baseline"] / 1e6,
            name="Baseline",
            line=dict(color='#E63946', width=2)
        ))
        fig_opex.add_trace(go.Scatter(
            x=opex_trend["Year"],
            y=opex_trend["OPEX_Optimized"] / 1e6,
            name="Optimized",
            line=dict(color='#028090', width=2)
        ))
        
        fig_opex.update_layout(
            xaxis_title="Year",
            yaxis_title="$ Millions",
            height=400
        )
        
        st.plotly_chart(fig_opex, use_container_width=True)
    
    with col2:
        st.markdown("### CAPEX Trend")
        
        capex_trend = filtered_data.groupby("Year").agg({
            "CAPEX_Baseline": "sum",
            "CAPEX_Optimized": "sum"
        }).reset_index()
        
        fig_capex = go.Figure()
        fig_capex.add_trace(go.Scatter(
            x=capex_trend["Year"],
            y=capex_trend["CAPEX_Baseline"] / 1e6,
            name="Baseline",
            line=dict(color='#E63946', width=2)
        ))
        fig_capex.add_trace(go.Scatter(
            x=capex_trend["Year"],
            y=capex_trend["CAPEX_Optimized"] / 1e6,
            name="Optimized",
            line=dict(color='#028090', width=2)
        ))
        
        fig_capex.update_layout(
            xaxis_title="Year",
            yaxis_title="$ Millions",
            height=400
        )
        
        st.plotly_chart(fig_capex, use_container_width=True)
    
    # Cumulative savings
    st.markdown("### Cumulative Savings")
    
    annual_data["Cumulative_Savings"] = annual_data["Total_Savings_All_Pillars"].cumsum()
    
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(
        x=annual_data["Year"],
        y=annual_data["Cumulative_Savings"] / 1e6,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#028090', width=3),
        fillcolor='rgba(2, 128, 144, 0.3)'
    ))
    
    fig_cumulative.update_layout(
        xaxis_title="Year",
        yaxis_title="Cumulative Savings ($ Millions)",
        height=400
    )
    
    st.plotly_chart(fig_cumulative, use_container_width=True)

# ============================================================================
# TAB 3: CATEGORIES
# ============================================================================

with tab3:
    st.markdown("## Category Analysis")
    
    # Category performance bar chart
    st.markdown("### Top Categories by Savings")
    
    cat_data = data["category"].copy()
    cat_data["Savings_Pct"] = (cat_data["Total_Savings_All_Pillars"] / cat_data["Total_Spend_Baseline"] * 100)
    cat_data = cat_data.sort_values("Total_Savings_All_Pillars", ascending=True).tail(15)
    
    fig_cat = go.Figure()
    fig_cat.add_trace(go.Bar(
        y=cat_data.index,
        x=cat_data["Total_Savings_All_Pillars"] / 1e6,
        orientation='h',
        marker=dict(
            color=cat_data["Savings_Pct"],
            colorscale='Teal',
            showscale=True,
            colorbar=dict(title="Savings %")
        ),
        text=cat_data["Total_Savings_All_Pillars"].apply(lambda x: f"${x/1e6:.0f}M"),
        textposition='outside'
    ))
    
    fig_cat.update_layout(
        xaxis_title="Savings ($ Millions)",
        yaxis_title="Category",
        height=600,
        showlegend=False
    )
    
    st.plotly_chart(fig_cat, use_container_width=True)
    
    # Scatter plot
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Savings $ vs Savings %")
        
        fig_scatter = px.scatter(
            data["category"],
            x="Total_Savings_All_Pillars",
            y=(data["category"]["Total_Savings_All_Pillars"] / data["category"]["Total_Spend_Baseline"] * 100),
            size="Total_Spend_Baseline",
            hover_name=data["category"].index,
            labels={
                "x": "Total Savings ($)",
                "y": "Savings (%)"
            },
            color_discrete_sequence=['#028090']
        )
        
        fig_scatter.update_layout(height=400)
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.markdown("### Category Detail Table")
        
        display_cat = data["category"].copy()
        display_cat["Savings_Pct"] = (display_cat["Total_Savings_All_Pillars"] / display_cat["Total_Spend_Baseline"] * 100)
        display_cat = display_cat[["Total_Spend_Baseline", "Total_Spend_Optimized", "Total_Savings_All_Pillars", "Savings_Pct"]]
        display_cat = display_cat.sort_values("Total_Savings_All_Pillars", ascending=False).head(10)
        
        # Format as millions
        display_cat["Total_Spend_Baseline"] = display_cat["Total_Spend_Baseline"].apply(lambda x: f"${x/1e6:.1f}M")
        display_cat["Total_Spend_Optimized"] = display_cat["Total_Spend_Optimized"].apply(lambda x: f"${x/1e6:.1f}M")
        display_cat["Total_Savings_All_Pillars"] = display_cat["Total_Savings_All_Pillars"].apply(lambda x: f"${x/1e6:.1f}M")
        display_cat["Savings_Pct"] = display_cat["Savings_Pct"].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_cat, use_container_width=True, height=400)

# ============================================================================
# TAB 4: PILLARS
# ============================================================================

with tab4:
    st.markdown("## Optimization Pillars Deep Dive")
    
    # Pillar selector
    selected_pillar = st.selectbox(
        "Select a pillar to analyze",
        data["pillars"]["Pillar"].tolist()
    )
    
    # Pillar stats
    pillar_row = data["pillars"][data["pillars"]["Pillar"] == selected_pillar].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Savings",
            f"${pillar_row['Total_Savings']/1e6:.0f}M"
        )
    
    with col2:
        st.metric(
            "% of Total Savings",
            f"{pillar_row['Pct_of_Total']:.1f}%"
        )
    
    with col3:
        rank = data["pillars"][data["pillars"]["Pillar"] == selected_pillar].index[0] + 1
        st.metric(
            "Rank",
            f"#{rank} of 7"
        )
    
    st.markdown("---")
    
    # Pillar comparison radar chart
    st.markdown("### Pillar Comparison (All Pillars)")
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=data["pillars"]["Pct_of_Total"].tolist(),
        theta=data["pillars"]["Pillar"].tolist(),
        fill='toself',
        marker=dict(color='#028090')
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, data["pillars"]["Pct_of_Total"].max() * 1.1]
            )
        ),
        showlegend=False,
        height=500
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Pillar contribution by category
    st.markdown(f"### {selected_pillar} Impact by Category")
    
    # Note: This would require detailed breakdown data
    # For now, showing placeholder
    st.info("📊 Detailed pillar-by-category breakdown requires additional data processing")

# ============================================================================
# TAB 5: REPLACEMENT PLANNING
# ============================================================================

with tab5:
    st.markdown("## Replacement Planning & Schedule")
    
    # Replacement schedule
    st.markdown("### Replacement Schedule")
    
    replacement_data = filtered_data.groupby("Year").agg({
        "Has_Replacement_This_Year": "sum",
        "Has_Refurbishment_This_Year": "sum",
        "Replacement_CapEx_Baseline": "sum",
        "CAPEX_Optimized": "sum"
    }).reset_index()
    
    fig_replacement = go.Figure()
    
    fig_replacement.add_trace(go.Bar(
        x=replacement_data["Year"],
        y=replacement_data["Has_Replacement_This_Year"],
        name="Replacements",
        marker_color='#028090'
    ))
    
    fig_replacement.add_trace(go.Bar(
        x=replacement_data["Year"],
        y=replacement_data["Has_Refurbishment_This_Year"],
        name="Refurbishments",
        marker_color='#02C39A'
    ))
    
    fig_replacement.update_layout(
        barmode='stack',
        xaxis_title="Year",
        yaxis_title="Number of Batches",
        height=500
    )
    
    st.plotly_chart(fig_replacement, use_container_width=True)
    
    # CapEx comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Refurbishment Analysis")
        
        total_replacements = replacement_data["Has_Replacement_This_Year"].sum()
        total_refurbs = replacement_data["Has_Refurbishment_This_Year"].sum()
        refurb_rate = (total_refurbs / (total_replacements + total_refurbs) * 100) if (total_replacements + total_refurbs) > 0 else 0
        
        st.metric("Total Replacements", f"{total_replacements:.0f}")
        st.metric("Total Refurbishments", f"{total_refurbs:.0f}")
        st.metric("Refurbishment Rate", f"{refurb_rate:.1f}%")
    
    with col2:
        st.markdown("### CapEx Comparison")
        
        fig_capex_comp = go.Figure()
        
        fig_capex_comp.add_trace(go.Scatter(
            x=replacement_data["Year"],
            y=replacement_data["Replacement_CapEx_Baseline"] / 1e6,
            name="Baseline CapEx",
            line=dict(color='#E63946', width=2)
        ))
        
        fig_capex_comp.add_trace(go.Scatter(
            x=replacement_data["Year"],
            y=replacement_data["CAPEX_Optimized"] / 1e6,
            name="Optimized CapEx",
            line=dict(color='#028090', width=2)
        ))
        
        fig_capex_comp.update_layout(
            xaxis_title="Year",
            yaxis_title="$ Millions",
            height=300
        )
        
        st.plotly_chart(fig_capex_comp, use_container_width=True)
    
    # Upcoming replacements table
    st.markdown("### Upcoming Replacements (Next 3 Years)")
    
    current_year = filtered_data["Year"].max()
    upcoming = filtered_data[
        (filtered_data["Year"] >= current_year) &
        (filtered_data["Year"] <= current_year + 2) &
        (filtered_data["Has_Replacement_This_Year"] == True)
    ][["Year", "Batch_ID", "Equipment_Category", "Equipment_Name", "Replacement_CapEx_Baseline", "CAPEX_Optimized"]]
    
    upcoming = upcoming.sort_values("Replacement_CapEx_Baseline", ascending=False).head(20)
    
    st.dataframe(upcoming, use_container_width=True, height=400)

# ============================================================================

# ============================================================================
# TAB 6: SANKEY FLOW DIAGRAM
# ============================================================================

with tab6:
    st.markdown("## 🌊 Sankey Flow Diagram")
    st.markdown("Budget flow through organizational hierarchy: City → Sectors → Departments → Programs → Stakeholders")
    
    enriched_df = load_enriched_data()
    
    if enriched_df is None:
        st.warning("⚠️ This visualization requires enriched Super Structure data.")
        st.info("Run the enrichment notebook first to generate `Equipment_Simulation_SuperStructure_Optimized.xlsx`")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            show_year = st.selectbox(
                "Select Year",
                options=sorted(enriched_df['Year'].unique()),
                index=0,
                key="sankey_year"
            )
        
        with col2:
            top_n_sankey = st.selectbox(
                "Top N Programs",
                options=[10, 15, 20, 'All'],
                index=1,
                key="sankey_top_n"
            )
        
        year_data = enriched_df[enriched_df['Year'] == show_year].copy()
        
        program_totals = year_data.groupby('L3_Program').agg({
            'Total_Spend_Optimized': 'sum'
        }).reset_index().sort_values('Total_Spend_Optimized', ascending=False)
        
        if top_n_sankey != 'All':
            top_programs = program_totals.head(top_n_sankey)['L3_Program'].tolist()
            year_data = year_data[year_data['L3_Program'].isin(top_programs)]
        
        with st.spinner("Generating Sankey diagram..."):
            sector_agg = year_data.groupby(['L0_Organization', 'L1_Sector']).agg({
                'Total_Spend_Optimized': 'sum'
            }).reset_index()
            
            dept_agg = year_data.groupby(['L1_Sector', 'L2_Department']).agg({
                'Total_Spend_Optimized': 'sum'
            }).reset_index()
            
            program_agg = year_data.groupby(['L2_Department', 'L3_Program']).agg({
                'Total_Spend_Optimized': 'sum'
            }).reset_index()
            
            stakeholder_agg = year_data.groupby(['L3_Program', 'L5_Stakeholder']).agg({
                'Total_Spend_Optimized': 'sum'
            }).reset_index()
            
            nodes = []
            node_dict = {}
            
            def add_node(label):
                if label not in node_dict:
                    node_dict[label] = len(nodes)
                    nodes.append(label)
                return node_dict[label]
            
            for _, row in sector_agg.iterrows():
                add_node(row['L0_Organization'])
                add_node(row['L1_Sector'])
            for _, row in dept_agg.iterrows():
                add_node(row['L2_Department'])
            for _, row in program_agg.iterrows():
                add_node(row['L3_Program'])
            for _, row in stakeholder_agg.iterrows():
                add_node(row['L5_Stakeholder'])
            
            links = []
            
            for _, row in sector_agg.iterrows():
                links.append({
                    'source': node_dict[row['L0_Organization']],
                    'target': node_dict[row['L1_Sector']],
                    'value': row['Total_Spend_Optimized'] / 1e6
                })
            
            for _, row in dept_agg.iterrows():
                links.append({
                    'source': node_dict[row['L1_Sector']],
                    'target': node_dict[row['L2_Department']],
                    'value': row['Total_Spend_Optimized'] / 1e6
                })
            
            for _, row in program_agg.iterrows():
                links.append({
                    'source': node_dict[row['L2_Department']],
                    'target': node_dict[row['L3_Program']],
                    'value': row['Total_Spend_Optimized'] / 1e6
                })
            
            for _, row in stakeholder_agg.iterrows():
                links.append({
                    'source': node_dict[row['L3_Program']],
                    'target': node_dict[row['L5_Stakeholder']],
                    'value': row['Total_Spend_Optimized'] / 1e6
                })
            
            node_colors = []
            for n in nodes:
                if 'Chicago' in n:
                    node_colors.append('#1E2761')
                elif n in ['Program Delivery', 'Shared Services']:
                    node_colors.append('#028090')
                elif n in enriched_df['L2_Department'].unique():
                    node_colors.append('#00A896')
                elif n in enriched_df['L3_Program'].unique():
                    node_colors.append('#02C39A')
                else:
                    node_colors.append('#E63946')
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='white', width=2),
                    label=nodes,
                    color=node_colors
                ),
                link=dict(
                    source=[l['source'] for l in links],
                    target=[l['target'] for l in links],
                    value=[l['value'] for l in links],
                    color='rgba(2, 128, 144, 0.3)'
                )
            )])
            
            fig.update_layout(
                title=f"<b>Budget Flow - Year {show_year}</b><br><sub>Width represents spending in millions</sub>",
                font=dict(size=11),
                height=700,
                margin=dict(t=80, l=20, r=20, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        total_spend = year_data['Total_Spend_Optimized'].sum()
        total_savings = year_data['Total_Savings_All_Pillars'].sum()
        
        with col1:
            st.metric("Total Spending", f"${total_spend/1e6:.0f}M")
        with col2:
            st.metric("Total Savings", f"${total_savings/1e6:.0f}M")
        with col3:
            st.metric("Nodes Shown", len(nodes))

# ============================================================================

# ============================================================================
# TAB 7: CAPEX COLLISION NETWORK
# ============================================================================

def build_capex_collision_network(
    enriched_df: pd.DataFrame,
    node_level: str = "L2_Department",
    capex_col: str = "Replacement_CapEx_Baseline",
    years: list[int] | None = None,
    top_n: int = 25,
    min_overlap_pct_of_smaller: float = 0.20,
):
    """
    Build an undirected graph where edges represent competition/collision in CAPEX timing.

    Nodes: departments or programs (choose node_level).
    Node value: total CAPEX in selected years.
    Edge weight: sum_y min(CAPEX_i_y, CAPEX_j_y)  (overlap dollars).
    Edge is kept if overlap dollars >= min_overlap_pct_of_smaller * min(total_i, total_j).
    """
    if years is None:
        years = sorted(enriched_df["Year"].unique().tolist())

    df = enriched_df.loc[enriched_df["Year"].isin(years), [node_level, "Year", capex_col]].copy()
    df = df.dropna(subset=[node_level])

    # Aggregate CAPEX by node-year
    ny = (
        df.groupby([node_level, "Year"], dropna=False)[capex_col]
        .sum()
        .reset_index()
        .rename(columns={node_level: "Node", capex_col: "Capex"})
    )

    # Pivot to Node x Year matrix
    pivot = ny.pivot_table(index="Node", columns="Year", values="Capex", aggfunc="sum", fill_value=0.0)

    # Select top nodes by total CAPEX
    totals = pivot.sum(axis=1).sort_values(ascending=False)
    top_nodes = totals.head(top_n).index.tolist()
    pivot = pivot.loc[top_nodes]
    totals = totals.loc[top_nodes]

    # Build graph
    G = nx.Graph()
    for node in top_nodes:
        series = pivot.loc[node].values
        total = float(totals.loc[node])
        spike_year = int(pivot.columns[int(np.argmax(series))]) if len(series) else None
        volatility = float(np.std(series) / (np.mean(series) + 1e-9)) if np.mean(series) > 0 else 0.0

        G.add_node(
            node,
            total_capex=total,
            spike_year=spike_year,
            volatility=volatility,
        )

    # Pairwise overlap
    mat = pivot.values
    year_cols = pivot.columns.to_list()

    n = len(top_nodes)
    for i in range(n):
        a = mat[i]
        total_a = float(totals.iloc[i])
        if total_a <= 0:
            continue
        for j in range(i + 1, n):
            b = mat[j]
            total_b = float(totals.iloc[j])
            if total_b <= 0:
                continue

            overlap_by_year = np.minimum(a, b)
            overlap = float(overlap_by_year.sum())
            if overlap <= 0:
                continue

            smaller = min(total_a, total_b)
            if overlap < (min_overlap_pct_of_smaller * smaller):
                continue

            # collision year: where overlap is max
            idx_y = int(np.argmax(overlap_by_year))
            collision_year = int(year_cols[idx_y])
            collision_peak = float(overlap_by_year[idx_y])

            G.add_edge(
                top_nodes[i],
                top_nodes[j],
                overlap=overlap,
                overlap_pct_of_smaller=float(overlap / smaller) if smaller > 0 else 0.0,
                collision_year=collision_year,
                collision_peak=collision_peak,
            )

    # Layout
    pos = nx.spring_layout(G, k=0.8, seed=42)
    return G, pos


def plot_capex_collision_network(G: nx.Graph, pos: dict, title: str):
    # Edge traces (width ~ overlap)
    edge_x, edge_y, edge_w, edge_text = [], [], [], []
    overlaps = [G.edges[e]["overlap"] for e in G.edges()]
    if overlaps:
        lo, hi = float(np.min(overlaps)), float(np.max(overlaps))
    else:
        lo, hi = 0.0, 1.0

    def scale_w(v):
        # log scaling for nicer visuals
        v = max(v, 1e-6)
        denominator = (np.log10(max(hi, 1e-6)) - np.log10(max(lo, 1e-6)) + 1e-9)
        if abs(denominator) < 1e-9 or hi <= lo:
            return 0.5
        return 0.5 + 4.0 * (np.log10(v) - np.log10(max(lo, 1e-6))) / denominator

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_w.append(scale_w(d["overlap"]))
        edge_text.append(
            f"<b>Collision</b><br>{u} ↔ {v}"
            f"<br>Overlap: ${d['overlap']/1e6:.1f}M"
            f"<br>Peak Year: {d['collision_year']}"
            f"<br>Peak Overlap: ${d['collision_peak']/1e6:.1f}M"
            f"<br>Overlap (vs smaller): {d['overlap_pct_of_smaller']*100:.0f}%"
        )

    # Plotly can't vary width per-segment in one trace cleanly; use average width for a single trace,
    # then add invisible markers at midpoints for hover.
    avg_w = float(np.mean(edge_w)) if edge_w else 0.8

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=avg_w, color="rgba(120,120,120,0.35)"),
        hoverinfo="none",
        showlegend=False,
    )

    # Edge hover markers at midpoints
    mid_x, mid_y, mid_text = [], [], []
    for (u, v, d), txt in zip(G.edges(data=True), edge_text):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_text.append(txt)

    edge_hover = go.Scatter(
        x=mid_x,
        y=mid_y,
        mode="markers",
        marker=dict(size=8, color="rgba(0,0,0,0)"),
        hovertext=mid_text,
        hoverinfo="text",
        showlegend=False,
    )

    # Node trace
    node_x, node_y, node_size, node_color, node_text = [], [], [], [], []
    totals = [G.nodes[n].get("total_capex", 0.0) for n in G.nodes()]
    if totals:
        tmin, tmax = float(np.min(totals)), float(np.max(totals))
    else:
        tmin, tmax = 0.0, 1.0

    def scale_size(v):
        v = max(v, 0.0)
        # size from 18 to 60
        denominator = (np.log10(tmax + 1) - np.log10(tmin + 1) + 1e-9)
        if abs(denominator) < 1e-9 or tmax <= 0:
            return 18
        return 18 + 42 * (np.log10(v + 1) - np.log10(tmin + 1)) / denominator

    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        total = float(G.nodes[n].get("total_capex", 0.0))
        spike = G.nodes[n].get("spike_year", None)
        vol = float(G.nodes[n].get("volatility", 0.0))

        node_size.append(scale_size(total))
        node_color.append(vol)  # color by volatility

        label = n if len(n) < 42 else n[:39] + "..."
        node_text.append(
            f"<b>{label}</b>"
            f"<br>Total CAPEX: ${total/1e6:.1f}M"
            f"<br>Peak Year: {spike}"
            f"<br>Volatility: {vol:.2f}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[n if len(n) <= 18 else "" for n in G.nodes()],  # small labels only
        textposition="top center",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="CAPEX Volatility"),
            line=dict(width=1.5, color="rgba(30,30,30,0.7)"),
            opacity=0.92,
        ),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, edge_hover, node_trace])
    fig.update_layout(
        title=title,
        height=720,
        hovermode="closest",
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


with tab7:
    st.markdown("## ⚡ CAPEX Collision Network")
    st.markdown(
        "This network highlights **where replacement CAPEX timing overlaps** across departments/programs — "
        "i.e., where budget pressure is likely to collide in the same years."
    )

    enriched_df = load_enriched_data()
    if enriched_df is None:
        st.warning("⚠️ This visualization requires the `Enriched_Data` sheet.")
    else:
        # Controls
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])

        with c1:
            node_level_label = st.selectbox(
                "Node level",
                options=["Department", "Program"],
                index=0,
                help="Choose whether nodes represent Departments (L2) or Programs (L3).",
            )
            node_level = "L2_Department" if node_level_label == "Department" else "L3_Program"

        with c2:
            capex_metric = st.selectbox(
                "CAPEX metric",
                options=[
                    "Replacement CAPEX (Baseline)",
                    "Total CAPEX (Baseline)",
                    "Total CAPEX (Optimized)",
                ],
                index=0,
            )
            capex_col = {
                "Replacement CAPEX (Baseline)": "Replacement_CapEx_Baseline",
                "Total CAPEX (Baseline)": "CAPEX_Baseline",
                "Total CAPEX (Optimized)": "CAPEX_Optimized",
            }[capex_metric]

        with c3:
            top_n = st.slider("Top N nodes", min_value=10, max_value=60, value=25, step=5)

        with c4:
            min_overlap_pct = st.slider(
                "Min overlap (as % of smaller node CAPEX)",
                min_value=0.05,
                max_value=0.60,
                value=0.20,
                step=0.05,
                help="Higher values show only stronger CAPEX collisions.",
            )

        years_all = sorted(enriched_df["Year"].unique().tolist())
        yr_min, yr_max = int(min(years_all)), int(max(years_all))
        year_window = st.slider(
            "Year window",
            min_value=yr_min,
            max_value=yr_max,
            value=(yr_min, yr_max),
            step=1,
        )
        years = list(range(year_window[0], year_window[1] + 1))

        with st.spinner("Building collision network..."):
            G, pos = build_capex_collision_network(
                enriched_df=enriched_df,
                node_level=node_level,
                capex_col=capex_col,
                years=years,
                top_n=top_n,
                min_overlap_pct_of_smaller=min_overlap_pct,
            )

        st.caption(f"Nodes: {G.number_of_nodes()} • Edges (collisions): {G.number_of_edges()}")

        if G.number_of_edges() == 0:
            st.info("No collisions found at this threshold. Try lowering the minimum overlap or increasing Top N.")
        else:
            title = (
                f"<b>CAPEX Collision Network</b><br>"
                f"<sub>{node_level_label} nodes • {capex_metric} • Years {years[0]}–{years[-1]} • "
                f"Top {top_n} • Min overlap {int(min_overlap_pct*100)}% of smaller</sub>"
            )
            fig = plot_capex_collision_network(G, pos, title)
            st.plotly_chart(fig, use_container_width=True)

            # Edge table: top collisions
            edge_rows = []
            for u, v, d in G.edges(data=True):
                edge_rows.append(
                    {
                        "Node A": u,
                        "Node B": v,
                        "Overlap ($M)": d["overlap"] / 1e6,
                        "Peak Year": d["collision_year"],
                        "Peak Overlap ($M)": d["collision_peak"] / 1e6,
                        "Overlap vs smaller (%)": d["overlap_pct_of_smaller"] * 100,
                    }
                )
            edges_df = pd.DataFrame(edge_rows).sort_values("Overlap ($M)", ascending=False).head(15)
            st.markdown("### 🔝 Top CAPEX Collisions")
            st.dataframe(edges_df, use_container_width=True)

# ============================================================================
# TAB 8: CAPEX COLLISION ANIMATION
# ============================================================================

def build_capex_collision_network_single_year(
    enriched_df: pd.DataFrame,
    node_level: str,
    capex_col: str,
    year: int,
    top_n: int = 25,
    min_overlap_amt: float = 0.5e6,
):
    """
    Build a collision network for a single year.
    
    Edges represent nodes that BOTH have CAPEX in that same year.
    Edge weight = min(capex_A, capex_B) = overlap potential
    """
    df = enriched_df.loc[enriched_df["Year"] == year, [node_level, capex_col]].copy()
    df = df.dropna(subset=[node_level])
    
    # Aggregate CAPEX by node for this year
    node_capex = (
        df.groupby(node_level, dropna=False)[capex_col]
        .sum()
        .reset_index()
        .rename(columns={node_level: "Node", capex_col: "Capex"})
    )
    
    # Filter out zero/negative CAPEX
    node_capex = node_capex[node_capex["Capex"] > 0]
    
    # Check if we have any data
    if len(node_capex) == 0:
        # Return empty graph
        return nx.Graph()
    
    # Select top nodes by CAPEX in this year
    node_capex = node_capex.sort_values("Capex", ascending=False).head(top_n)
    top_nodes = node_capex["Node"].tolist()
    
    # Build graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for _, row in node_capex.iterrows():
        G.add_node(
            row["Node"],
            capex=float(row["Capex"]),
            year=year
        )
    
    # Add edges: connect nodes that both have CAPEX > threshold
    n = len(top_nodes)
    for i in range(n):
        node_a = top_nodes[i]
        capex_a = float(node_capex[node_capex["Node"] == node_a]["Capex"].iloc[0])
        
        for j in range(i + 1, n):
            node_b = top_nodes[j]
            capex_b = float(node_capex[node_capex["Node"] == node_b]["Capex"].iloc[0])
            
            # Overlap = minimum of the two (conservative estimate)
            overlap = min(capex_a, capex_b)
            
            # Only add edge if overlap is significant
            if overlap >= min_overlap_amt:
                G.add_edge(
                    node_a,
                    node_b,
                    overlap=overlap,
                    year=year
                )
    
    return G


def create_animation_frames(
    enriched_df: pd.DataFrame,
    node_level: str,
    capex_col: str,
    years: list,
    top_n: int = 25,
    min_overlap_amt: float = 0.5e6,
):
    """
    Create animation frames for each year.
    Uses a consistent layout across all frames.
    """
    
    # Step 1: Build network for ALL years combined to get consistent layout
    all_nodes = set()
    for year in years:
        df_year = enriched_df[enriched_df["Year"] == year]
        nodes_year = df_year[node_level].dropna().unique()
        all_nodes.update(nodes_year)
    
    # Create a super-graph with all nodes
    G_all = nx.Graph()
    for node in all_nodes:
        G_all.add_node(node)
    
    # Add some edges for layout (connect nodes that appear in same year at least once)
    for year in years:
        df_year = enriched_df[enriched_df["Year"] == year]
        year_nodes = df_year[node_level].dropna().unique().tolist()
        for i in range(len(year_nodes)):
            for j in range(i + 1, len(year_nodes)):
                if not G_all.has_edge(year_nodes[i], year_nodes[j]):
                    G_all.add_edge(year_nodes[i], year_nodes[j])
    
    # Calculate fixed layout
    pos = nx.spring_layout(G_all, k=1.2, iterations=100, seed=42)
    
    # Step 2: Create frames for each year
    frames = []
    
    for year in years:
        G_year = build_capex_collision_network_single_year(
            enriched_df, node_level, capex_col, year, top_n, min_overlap_amt
        )
        
        # Create plotly traces for this year
        # Edges
        edge_x, edge_y = [], []
        edge_widths = []
        
        if G_year.number_of_edges() > 0:
            overlaps = [G_year.edges[e]["overlap"] for e in G_year.edges()]
            max_overlap = max(overlaps) if overlaps else 1.0
            
            for u, v, d in G_year.edges(data=True):
                if u in pos and v in pos:
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
                    # Width proportional to overlap
                    width = 0.5 + 5.0 * (d["overlap"] / max_overlap)
                    edge_widths.append(width)
        
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=2, color='rgba(100, 100, 100, 0.4)'),
            hoverinfo='none',
            showlegend=False
        )
        
        # Nodes
        node_x, node_y, node_size, node_color, node_text = [], [], [], [], []
        
        if G_year.number_of_nodes() > 0:
            capexes = [G_year.nodes[n].get("capex", 0) for n in G_year.nodes()]
            max_capex = max(capexes) if capexes and max(capexes) > 0 else 1.0
            
            for node in G_year.nodes():
                if node in pos:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    capex = G_year.nodes[node].get("capex", 0)
                    size = 15 + 45 * (capex / max_capex) if max_capex > 0 else 15
                    node_size.append(size)
                    
                    # Color by number of connections (degree)
                    degree = G_year.degree(node)
                    node_color.append(degree)
                    
                    label = node if len(node) < 40 else node[:37] + "..."
                    node_text.append(
                        f"<b>{label}</b><br>"
                        f"CAPEX: ${capex/1e6:.1f}M<br>"
                        f"Collisions: {degree}"
                    )
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(
                    title="# Collisions",
                    x=1.02
                ),
                line=dict(width=1.5, color='rgba(50, 50, 50, 0.8)'),
                opacity=0.9
            ),
            hovertext=node_text,
            hoverinfo='text',
            showlegend=False
        )
        
        frame_data = [edge_trace, node_trace]
        frames.append(go.Frame(data=frame_data, name=str(year)))
    
    return frames, pos


with tab8:
    st.markdown("## 🎬 CAPEX Collision Animation")
    st.markdown(
        "Watch how **CAPEX collisions evolve year by year**. "
        "Nodes grow when CAPEX increases, connections appear when departments compete for budget."
    )
    
    enriched_df = load_enriched_data()
    if enriched_df is None:
        st.warning("⚠️ This visualization requires the `Enriched_Data` sheet.")
    else:
        # Controls
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.2])
        
        with c1:
            node_level_label_anim = st.selectbox(
                "Node level",
                options=["Department", "Program"],
                index=0,
                key="anim_node_level",
                help="Choose whether nodes represent Departments (L2) or Programs (L3).",
            )
            node_level_anim = "L2_Department" if node_level_label_anim == "Department" else "L3_Program"
        
        with c2:
            capex_metric_anim = st.selectbox(
                "CAPEX metric",
                options=[
                    "Replacement CAPEX (Baseline)",
                    "Total CAPEX (Baseline)",
                    "Total CAPEX (Optimized)",
                ],
                index=0,
                key="anim_capex_metric"
            )
            capex_col_anim = {
                "Replacement CAPEX (Baseline)": "Replacement_CapEx_Baseline",
                "Total CAPEX (Baseline)": "CAPEX_Baseline",
                "Total CAPEX (Optimized)": "CAPEX_Optimized",
            }[capex_metric_anim]
        
        with c3:
            top_n_anim = st.slider(
                "Top N nodes",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                key="anim_top_n"
            )
        
        with c4:
            animation_speed = st.slider(
                "Animation Speed (ms)",
                min_value=300,
                max_value=2000,
                value=800,
                step=100,
                key="anim_speed",
                help="Milliseconds per frame. Lower = faster."
            )
        
        years_all = sorted(enriched_df["Year"].unique().tolist())
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info("**How to use:** Click ▶️ Play to see collisions evolve. Larger nodes = more CAPEX. More connections = more budget conflicts.")
        with col_info2:
            st.info("**Color scale:** Darker red = more collisions with other departments. White/Light = isolated spending.")
        
        with st.spinner("Building animation frames (this may take 10-15 seconds)..."):
            frames, pos = create_animation_frames(
                enriched_df=enriched_df,
                node_level=node_level_anim,
                capex_col=capex_col_anim,
                years=years_all,
                top_n=top_n_anim,
                min_overlap_amt=0.5e6  # $500K minimum overlap
            )
        
        # Create figure with animation
        if len(frames) > 0:
            fig = go.Figure(
                data=frames[0]['data'],
                frames=frames
            )
            
            fig.update_layout(
                title=f"<b>CAPEX Collision Animation: {years_all[0]}-{years_all[-1]}</b><br>"
                      f"<sub>{node_level_label_anim} nodes • {capex_metric_anim} • "
                      f"Top {top_n_anim} by annual CAPEX</sub>",
                height=750,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'x': 0.05,
                    'y': 1.15,
                    'buttons': [
                        {
                            'label': '▶️ Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': animation_speed, 'redraw': True},
                                'fromcurrent': True,
                                'mode': 'immediate',
                                'transition': {'duration': 200}
                            }]
                        },
                        {
                            'label': '⏸️ Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        }
                    ]
                }],
                sliders=[{
                    'active': 0,
                    'yanchor': 'top',
                    'y': -0.05,
                    'xanchor': 'left',
                    'currentvalue': {
                        'prefix': 'Year: ',
                        'visible': True,
                        'xanchor': 'center'
                    },
                    'pad': {'b': 10, 't': 10},
                    'len': 0.9,
                    'x': 0.05,
                    'steps': [
                        {
                            'args': [
                                [f.name],
                                {
                                    'frame': {'duration': 0, 'redraw': True},
                                    'mode': 'immediate',
                                    'transition': {'duration': 0}
                                }
                            ],
                            'label': f.name,
                            'method': 'animate'
                        }
                        for f in frames
                    ]
                }]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            st.markdown("### 📊 Animation Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_unique_nodes = len(pos)
                st.metric("Total Unique Nodes", total_unique_nodes)
            
            with col2:
                avg_edges = np.mean([len(f['data'][0]['x'])//3 for f in frames if len(f['data']) > 0])
                st.metric("Avg Collisions/Year", f"{avg_edges:.0f}")
            
            with col3:
                st.metric("Years Animated", len(frames))
            
            # Key insights
            with st.expander("💡 How to Interpret"):
                st.markdown("""
                **What you're seeing:**
                - **Node size** = CAPEX amount in that year (larger = more spending)
                - **Node color** = Number of collisions (red = competing with many departments)
                - **Edges** = Both departments have CAPEX in the same year (potential budget conflict)
                - **Animation** = Shows how conflicts shift over time
                
                **Key patterns to look for:**
                1. **Years with many connections** = High budget pressure years
                2. **Nodes that are consistently red** = Always competing for budget
                3. **Nodes that disappear/reappear** = Cyclical replacement patterns
                4. **Sudden size increases** = Major replacement waves
                
                **Action items:**
                - **Heavy collision years** → Plan budget contingencies
                - **Consistently connected nodes** → Coordinate replacement schedules
                - **Isolated nodes** → Can be scheduled flexibly
                """)
        
        else:
            st.error("Unable to generate animation frames. Check data availability.")

# ============================================================================
# FOOTER
# ============================================================================
