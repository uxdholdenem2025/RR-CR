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
# --- MODULE 1: COMBINED EXECUTIVE REPORT ---
# ==============================================================================
if app_mode == "Combined Executive Report":
    st.title("🏭 Manufacturing Executive Dashboard")
    st.markdown("Combined performance analysis across Capacity Risk (CR) and Run Rate (RR) metrics.")

    # --- 1. Global Settings & Data Upload (Sidebar) ---
    st.sidebar.header("1. Global Configuration")
    
    # Shared Parameters (Crossover Settings)
    with st.sidebar.expander("⚙️ Calculation Parameters", expanded=True):
        global_run_interval = st.slider(
            "Run Interval Threshold (Hours)", 1.0, 24.0, 8.0, 0.5,
            help="Gaps longer than this define a new 'Production Run'. Applies to both CR and RR."
        )
        global_stop_gap = st.slider(
            "Stop Threshold / Gap (Seconds)", 0.0, 10.0, 2.0, 0.5,
            help="Idle time required to trigger a stop event. Applies to both CR and RR."
        )
        global_ct_tolerance = st.slider(
            "Cycle Time Tolerance (%)", 0.01, 0.50, 0.05, 0.01,
            help="Variation allowed around Mode CT before flagging as abnormal. Applies to both CR and RR."
        )

    # File Uploaders
    st.sidebar.header("2. Data Sources")
    cr_file = st.sidebar.file_uploader("Upload Capacity Risk Data (CSV/XLS)", type=["csv", "xlsx", "xls"], key="global_cr_upload")
    rr_file = st.sidebar.file_uploader("Upload Run Rate Data (XLSX)", type=["xlsx", "xls"], key="global_rr_upload")
    
    # --- 2. Data Loading & Date Filtering ---
    
    # Initialize placeholders
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
            # Wrap single file in list because load_all_data expects a list
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

    # Date Picker (Configurable at top of page as requested)
    if not cr_data.empty or not rr_data.empty:
        st.subheader("📅 Analysis Period")
        c1, c2 = st.columns([1, 3])
        with c1:
            # Default to full range
            date_range = st.date_input(
                "Select Date Range",
                value=(min_global_date, max_global_date),
                min_value=min_global_date,
                max_value=max_global_date
            )
        
        start_date, end_date = date_range if len(date_range) == 2 else (min_global_date, max_global_date)
        
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

    # --- 3. Calculations & Master Table ---
    
    metrics_list = []
    
    # -- CR Calculation --
    if not cr_filtered.empty:
        # We use the cached wrapper but pass our global settings
        # default_cavities=2, target_output=100% (optimal view)
        cr_results_df, _ = cr_utils.run_capacity_calculation_cached_v2(
            cr_filtered, False, 2, 100.0, 
            global_ct_tolerance, global_stop_gap, global_run_interval
        )
        
        if not cr_results_df.empty:
            total_optimal = cr_results_df['Optimal Output (parts)'].sum()
            total_actual = cr_results_df['Actual Output (parts)'].sum()
            total_loss = cr_results_df['Total Capacity Loss (parts)'].sum()
            
            metrics_list.append({"Category": "Capacity Risk", "Metric": "Optimal Output", "Value": f"{total_optimal:,.0f}", "Unit": "Parts"})
            metrics_list.append({"Category": "Capacity Risk", "Metric": "Actual Output", "Value": f"{total_actual:,.0f}", "Unit": "Parts"})
            metrics_list.append({"Category": "Capacity Risk", "Metric": "Total Capacity Loss", "Value": f"{total_loss:,.0f}", "Unit": "Parts"})
            
            # Determine color for CR
            cr_perf = (total_actual / total_optimal) * 100 if total_optimal > 0 else 0
            metrics_list.append({"Category": "Capacity Risk", "Metric": "Performance vs Optimal", "Value": f"{cr_perf:.1f}", "Unit": "%"})

    # -- RR Calculation --
    if not rr_filtered.empty:
        # Use the Calculator Class directly
        rr_calc = run_rate_utils.RunRateCalculator(
            rr_filtered, 
            global_ct_tolerance, 
            global_stop_gap, 
            analysis_mode='aggregate' # Aggregate for executive view
        )
        rr_res = rr_calc.results
        
        metrics_list.append({"Category": "Run Rate", "Metric": "Stability Index", "Value": f"{rr_res.get('stability_index', 0):.1f}", "Unit": "%"})
        metrics_list.append({"Category": "Run Rate", "Metric": "Efficiency", "Value": f"{rr_res.get('efficiency', 0)*100:.1f}", "Unit": "%"})
        metrics_list.append({"Category": "Run Rate", "Metric": "MTTR", "Value": f"{rr_res.get('mttr_min', 0):.1f}", "Unit": "Min"})
        metrics_list.append({"Category": "Run Rate", "Metric": "MTBF", "Value": f"{rr_res.get('mtbf_min', 0):.1f}", "Unit": "Min"})
        metrics_list.append({"Category": "Run Rate", "Metric": "Total Stop Events", "Value": f"{rr_res.get('stop_events', 0):,}", "Unit": "Count"})

    # -- Display Master Table --
    st.subheader("📊 Master Metrics Table")
    if metrics_list:
        df_master = pd.DataFrame(metrics_list)
        
        # Formatting for display
        st.dataframe(
            df_master.style.set_properties(**{'text-align': 'left'}).set_table_styles(
                [{'selector': 'th', 'props': [('text-align', 'left')]}]
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # -- KPI Cards Row --
        st.subheader("Key Performance Indicators")
        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        
        # Extract values safely
        def get_metric(name):
            row = df_master[df_master['Metric'] == name]
            return row.iloc[0]['Value'] if not row.empty else "N/A"

        with kpi_c1:
            st.metric("Actual Output (CR)", get_metric("Actual Output"))
        with kpi_c2:
            st.metric("Capacity Loss (CR)", get_metric("Total Capacity Loss"))
        with kpi_c3:
            st.metric("Stability Index (RR)", f"{get_metric('Stability Index')}%")
        with kpi_c4:
            st.metric("MTBF (RR)", f"{get_metric('MTBF')} min")

    else:
        st.warning("No data available for the selected period.")

    st.divider()

    # --- 4. Dashboard Graphs ---
    st.subheader("📈 Metric Dashboards")
    
    g1, g2 = st.columns(2)
    
    # Graph 1: Capacity Risk Waterfall (if data exists)
    with g1:
        if not cr_filtered.empty and not cr_results_df.empty:
            st.markdown("#### Capacity Loss Breakdown")
            
            # Aggregating totals for the waterfall
            tot_opt = cr_results_df['Optimal Output (parts)'].sum()
            tot_act = cr_results_df['Actual Output (parts)'].sum()
            loss_downtime = cr_results_df['Capacity Loss (downtime) (parts)'].sum()
            loss_speed = cr_results_df['Capacity Loss (slow cycle time) (parts)'].sum() - cr_results_df['Capacity Gain (fast cycle time) (parts)'].sum()
            
            fig_waterfall = go.Figure(go.Waterfall(
                name="Capacity", orientation="v",
                measure=["absolute", "relative", "relative", "total"],
                x=["Optimal", "Downtime Loss", "Speed Loss", "Actual"],
                y=[tot_opt, -loss_downtime, -loss_speed, tot_act],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#ff6961"}},
                increasing={"marker": {"color": "#2ca02c"}},
                totals={"marker": {"color": "#3498DB"}}
            ))
            fig_waterfall.update_layout(height=350, title="Production Waterfall (Parts)")
            st.plotly_chart(fig_waterfall, use_container_width=True)
        else:
            st.info("Waiting for Capacity Risk Data...")

    # Graph 2: Run Rate Stability Gauge (if data exists)
    with g2:
        if not rr_filtered.empty:
            st.markdown("#### Stability Performance")
            stab_val = float(get_metric("Stability Index")) if get_metric("Stability Index") != "N/A" else 0
            
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
        else:
            st.info("Waiting for Run Rate Data...")

# ==============================================================================
# --- MODULE 2 & 3: INDIVIDUAL APP LOADS ---
# ==============================================================================
elif app_mode == "Capacity Risk Details":
    cr_app_refactored.run_capacity_risk_ui()

elif app_mode == "Run Rate Analysis Details":
    run_rate_app_refactored.run_run_rate_ui()