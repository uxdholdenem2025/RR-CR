import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date

# Import utils and refactored apps
import cr_utils
import run_rate_utils
import cr_app_refactored
import run_rate_app_refactored

# ==============================================================================
# --- PAGE CONFIGURATION ---
# ==============================================================================
st.set_page_config(
    page_title="Manufacturing Analytics Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# --- SIDEBAR NAVIGATION ---
# ==============================================================================
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Select Module:",
    ["Combined Executive Report", "Capacity Risk Details", "Run Rate Analysis Details"]
)

st.sidebar.markdown("---")

# ==============================================================================
# --- HELPER: METRIC BADGE ---
# ==============================================================================
def metric_with_badge(label, value, badge_text, badge_color="green", help_text=None):
    """
    Custom component to display a metric with a colored badge below it.
    """
    color_map = {
        "green": "#77dd77",
        "red": "#ff6961",
        "orange": "#ffb347",
        "gray": "#d3d3d3"
    }
    bg_color = color_map.get(badge_color, "#d3d3d3")
    
    tooltip_html = f' title="{help_text}"' if help_text else ''
    
    st.markdown(f"""
    <div style="margin-bottom: 10px;"{tooltip_html}>
        <p style="font-size: 14px; margin-bottom: 0px; color: #888;">{label}</p>
        <p style="font-size: 28px; font-weight: bold; margin-bottom: 5px;">{value}</p>
        <span style="background-color: {bg_color}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">
            {badge_text}
        </span>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# --- MODULE 1: COMBINED EXECUTIVE REPORT ---
# ==============================================================================
if app_mode == "Combined Executive Report":
    st.title("🏭 Manufacturing Executive Dashboard")
    st.markdown("Combined performance analysis across Capacity Risk (CR) and Run Rate (RR) metrics.")

    # --- 1. Global Settings & Data Upload (Sidebar) ---
    st.sidebar.header("1. Global Configuration")
    
    # Shared Parameters
    with st.sidebar.expander("⚙️ Calculation Parameters", expanded=True):
        global_run_interval = st.slider(
            "Run Interval Threshold (Hours)", 1.0, 24.0, 8.0, 0.5,
            help="Gaps longer than this define a new 'Production Run'."
        )
        global_stop_gap = st.slider(
            "Stop Threshold / Gap (Seconds)", 0.0, 10.0, 2.0, 0.5,
            help="Idle time required to trigger a stop event."
        )
        global_ct_tolerance = st.slider(
            "Cycle Time Tolerance (%)", 0.01, 0.50, 0.05, 0.01,
            help="Variation allowed around Mode CT before flagging as abnormal."
        )

    # NEW: Cost Settings
    with st.sidebar.expander("💰 Cost Settings", expanded=True):
        machine_rate = st.number_input("Machine Rate ($/h)", value=170.0, step=10.0)
        labor_rate = st.number_input("Labor Cost ($/h)", value=10.0, step=5.0)

    # File Uploaders
    st.sidebar.header("2. Data Sources")
    cr_file = st.sidebar.file_uploader("Upload Capacity Risk Data (CSV/XLS)", type=["csv", "xlsx", "xls"], key="global_cr_upload")
    rr_file = st.sidebar.file_uploader("Upload Run Rate Data (XLSX)", type=["xlsx", "xls"], key="global_rr_upload")
    
    # --- 2. Data Loading & Date Filtering ---
    
    cr_data = pd.DataFrame()
    rr_data = pd.DataFrame()
    min_global_date = None
    max_global_date = None

    # Load CR Data
    if cr_file:
        try:
            df_raw_cr = cr_utils.load_data(cr_file)
            if df_raw_cr is not None and not df_raw_cr.empty:
                cr_data, cr_min, cr_max = cr_utils.get_preprocessed_data(df_raw_cr)
                if min_global_date is None or cr_min < min_global_date: min_global_date = cr_min
                if max_global_date is None or cr_max > max_global_date: max_global_date = cr_max
        except Exception as e:
            st.sidebar.error(f"Error loading CR file: {e}")

    # Load RR Data
    if rr_file:
        try:
            rr_data_raw = run_rate_utils.load_all_data([rr_file])
            if rr_data_raw is not None and not rr_data_raw.empty:
                rr_data = rr_data_raw.copy()
                rr_data['date_obj'] = rr_data['shot_time'].dt.date
                rr_min = rr_data['date_obj'].min()
                rr_max = rr_data['date_obj'].max()
                if min_global_date is None or rr_min < min_global_date: min_global_date = rr_min
                if max_global_date is None or rr_max > max_global_date: max_global_date = rr_max
        except Exception as e:
            st.sidebar.error(f"Error loading RR file: {e}")

    # Date Picker
    if not cr_data.empty or not rr_data.empty:
        st.subheader("📅 Analysis Period")
        c1, c2 = st.columns([1, 3])
        with c1:
            default_val = (min_global_date, max_global_date) if min_global_date and max_global_date else []
            date_range = st.date_input(
                "Select Date Range",
                value=default_val,
                min_value=min_global_date,
                max_value=max_global_date
            )
        
        start_date, end_date = date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (min_global_date, max_global_date)
        
        # Filter Data
        cr_filtered = pd.DataFrame()
        if not cr_data.empty:
            mask_cr = (cr_data['date'] >= start_date) & (cr_data['date'] <= end_date)
            cr_filtered = cr_data[mask_cr].copy()

        rr_filtered = pd.DataFrame()
        if not rr_data.empty:
            mask_rr = (rr_data['date_obj'] >= start_date) & (rr_data['date_obj'] <= end_date)
            rr_filtered = rr_data[mask_rr].copy()
    else:
        st.info("👈 Please upload data files in the sidebar to begin the Executive Summary.")
        st.stop()

    st.divider()

    # --- 3. Calculations & Data Prep ---
    
    cr_metrics = {}
    rr_metrics = {}
    
    # -- CR Calculation --
    cr_results_df = pd.DataFrame()
    if not cr_filtered.empty:
        cr_results_df, _ = cr_utils.run_capacity_calculation_cached_v2(
            cr_filtered, False, 2, 100.0, 
            global_ct_tolerance, global_stop_gap, global_run_interval
        )
        
        if not cr_results_df.empty:
            total_optimal = cr_results_df['Optimal Output (parts)'].sum()
            total_actual = cr_results_df['Actual Output (parts)'].sum()
            
            # Breakdown of losses for "Availability" vs "Efficiency"
            loss_downtime_parts = cr_results_df['Capacity Loss (downtime) (parts)'].sum()
            loss_slow_parts = cr_results_df['Capacity Loss (slow cycle time) (parts)'].sum()
            gain_fast_parts = cr_results_df['Capacity Gain (fast cycle time) (parts)'].sum()
            net_efficiency_loss_parts = loss_slow_parts - gain_fast_parts
            
            total_loss_parts = cr_results_df['Total Capacity Loss (parts)'].sum()
            total_loss_sec = cr_results_df['Total Capacity Loss (sec)'].sum()
            
            cr_perf = (total_actual / total_optimal) * 100 if total_optimal > 0 else 0
            
            cr_metrics = {
                "optimal": total_optimal,
                "actual": total_actual,
                "loss_total_parts": total_loss_parts,
                "loss_total_sec": total_loss_sec,
                "loss_availability_parts": loss_downtime_parts,
                "loss_efficiency_parts": net_efficiency_loss_parts,
                "perf": cr_perf
            }

    # -- RR Calculation --
    if not rr_filtered.empty:
        rr_calc = run_rate_utils.RunRateCalculator(
            rr_filtered, 
            global_ct_tolerance, 
            global_stop_gap, 
            analysis_mode='aggregate'
        )
        rr_res = rr_calc.results
        
        rr_metrics = {
            "mttr": rr_res.get('mttr_min', 0),
            "mtbf": rr_res.get('mtbf_min', 0),
            "stability": rr_res.get('stability_index', 0),
            "efficiency": rr_res.get('efficiency', 0) * 100,
        }

    # --- 4. TOOLING PERFORMANCE TABLE (The "Insight" View) ---
    
    st.subheader("📋 Tooling Production Performance")
    
    if cr_metrics:
        # Calculate Financials
        loss_hours = cr_metrics['loss_total_sec'] / 3600.0
        total_cost = loss_hours * (machine_rate + labor_rate)
        
        # Format values for the table
        
        # 1. Loss Hours
        row_loss_hrs = f"{loss_hours:,.1f} Hours"
        
        # 2. Cost
        row_cost = f"${total_cost:,.0f} *"
        
        # 3. Opportunity Lost
        row_opp_lost = f"{cr_metrics['loss_total_parts']:,.0f} parts"
        
        # 4. RR Metrics
        row_rr_eff = f"{rr_metrics.get('efficiency', 0):.1f}%" if rr_metrics else "N/A"
        row_rr_mtbf = f"{rr_metrics.get('mtbf', 0):.0f} Minutes" if rr_metrics else "N/A"
        row_rr_mttr = f"{rr_metrics.get('mttr', 0):.0f} Minutes" if rr_metrics else "N/A"
        
        # 5. Capacity Risk Breakdown
        # Formatting exactly like the screenshot: "Actual: X (Loss Y)"
        val_opt = f"{cr_metrics['optimal']:,.0f} parts"
        
        # Actual (Gap)
        gap_total = cr_metrics['actual'] - cr_metrics['optimal']
        val_act = f"{cr_metrics['actual']:,.0f} ({gap_total:+,.0f} parts)"
        
        # Availability (Gap) - This is Downtime Loss
        # Note: In your screenshot, Availability is negative, representing loss
        val_avail = f"-{cr_metrics['loss_availability_parts']:,.0f} parts"
        
        # Efficiency (Gap) - This is Cycle Time Loss
        val_eff = f"-{cr_metrics['loss_efficiency_parts']:,.0f} parts"

        # Build Dataframe
        data = {
            "KPI": [
                "Total Incurred Loss in Machine Hours",
                f"Total Incurred Costs (Machine Rate + Labor)*",
                "Parts Opportunity Lost",
                "Run Rate Efficiency",
                "Run Rate MTBF (Mean Time Between Failure)",
                "Run Rate MTTR (Mean Time To Repair)",
                "Capacity Risk: Optimal Output (100% OEE)",
                "Capacity Risk: Actual Output",
                "Capacity Risk: Availability Loss",
                "Capacity Risk: Efficiency Loss"
            ],
            "Selected Scope": [
                row_loss_hrs,
                row_cost,
                row_opp_lost,
                row_rr_eff,
                row_rr_mtbf,
                row_rr_mttr,
                val_opt,
                val_act,
                val_avail,
                val_eff
            ]
        }
        
        df_insight = pd.DataFrame(data)
        
        # Display
        st.table(df_insight)
        
        st.caption(f"*Based on a machine rate cost of ${machine_rate}/h and labor costs of ${labor_rate}/h")

    else:
        st.warning("Please upload Capacity Risk data to generate the Performance Table.")

    st.divider()

    # --- 5. Dashboard Graphs ---
    st.subheader("📈 Metric Dashboards")
    
    g1, g2 = st.columns(2)
    
    # Graph 1: Capacity Risk Waterfall
    with g1:
        if "actual" in cr_metrics:
            st.markdown("#### Capacity Loss Breakdown")
            
            fig_waterfall = go.Figure(go.Waterfall(
                name="Capacity", orientation="v",
                measure=["absolute", "relative", "relative", "total"],
                x=["Optimal", "Downtime Loss", "Speed Loss", "Actual"],
                y=[cr_metrics['optimal'], -cr_metrics['loss_availability_parts'], -cr_metrics['loss_efficiency_parts'], cr_metrics['actual']],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#ff6961"}},
                increasing={"marker": {"color": "#2ca02c"}},
                totals={"marker": {"color": "#3498DB"}}
            ))
            fig_waterfall.update_layout(height=350, title="Production Waterfall (Parts)")
            st.plotly_chart(fig_waterfall, use_container_width=True)

    # Graph 2: Run Rate Stability Gauge
    with g2:
        if "stability" in rr_metrics:
            st.markdown("#### Stability Performance")
            stab_val = rr_metrics["stability"]
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=stab_val,
                title={'text': "Stability Index (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 50], 'color': "#ff6961"},
                        {'range': [50, 70], 'color': "#ffb347"},
                        {'range': [70, 100], 'color': "#77dd77"}
                    ],
                    'bar': {'color': "black"}
                }
            ))
            fig_gauge.update_layout(height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)

# ==============================================================================
# --- MODULE 2 & 3: INDIVIDUAL APP LOADS ---
# ==============================================================================
elif app_mode == "Capacity Risk Details":
    cr.run_capacity_risk_ui()

elif app_mode == "Run Rate Analysis Details":
    run_rate_app_refactored.run_run_rate_ui()