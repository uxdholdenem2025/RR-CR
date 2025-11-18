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
def metric_with_badge(label, value, badge_text, badge_color="green"):
    """
    Custom component to display a metric with a colored badge below it.
    Colors: green (#77dd77), red (#ff6961), orange (#ffb347), gray (#d3d3d3)
    """
    color_map = {
        "green": "#77dd77",
        "red": "#ff6961",
        "orange": "#ffb347",
        "gray": "#d3d3d3"
    }
    bg_color = color_map.get(badge_color, "#d3d3d3")
    
    st.markdown(f"""
    <div style="margin-bottom: 10px;">
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

    # --- 3. Calculations & Data Prep ---
    
    # Dictionaries to store display values
    cr_metrics = {}
    rr_metrics = {}
    
    # -- CR Calculation --
    if not cr_filtered.empty:
        cr_results_df, _ = cr_utils.run_capacity_calculation_cached_v2(
            cr_filtered, False, 2, 100.0, 
            global_ct_tolerance, global_stop_gap, global_run_interval
        )
        
        if not cr_results_df.empty:
            total_optimal = cr_results_df['Optimal Output (parts)'].sum()
            total_actual = cr_results_df['Actual Output (parts)'].sum()
            total_loss = cr_results_df['Total Capacity Loss (parts)'].sum()
            
            cr_perf = (total_actual / total_optimal) * 100 if total_optimal > 0 else 0
            
            cr_metrics = {
                "optimal": total_optimal,
                "actual": total_actual,
                "loss": total_loss,
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
        
        # Extract raw values for badges
        rr_metrics = {
            "mttr": rr_res.get('mttr_min', 0),
            "mtbf": rr_res.get('mtbf_min', 0),
            "stability": rr_res.get('stability_index', 0),
            "efficiency": rr_res.get('efficiency', 0) * 100,
            "production_time_sec": rr_res.get('production_time_sec', 0),
            "downtime_sec": rr_res.get('downtime_sec', 0),
            "total_runtime_sec": rr_res.get('total_runtime_sec', 0),
            "normal_shots": rr_res.get('normal_shots', 0),
            "total_shots": rr_res.get('total_shots', 0),
            "stop_events": rr_res.get('stop_events', 0),
            "stopped_shots": rr_res.get('total_shots', 0) - rr_res.get('normal_shots', 0)
        }

    # --- 4. Display KPI Cards with Badges ---
    
    if cr_metrics or rr_metrics:
        st.subheader("Key Performance Indicators")
        
        # Row 1: Time & Stability
        kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
        
        with kpi_c1:
            if "mttr" in rr_metrics:
                val = run_rate_utils.format_minutes_to_dhm(rr_metrics["mttr"])
                st.metric("Run Rate MTTR", val, help="Mean Time To Repair")
            else:
                st.metric("Run Rate MTTR", "N/A")

        with kpi_c2:
            if "mtbf" in rr_metrics:
                val = run_rate_utils.format_minutes_to_dhm(rr_metrics["mtbf"])
                st.metric("Run Rate MTBF", val, help="Mean Time Between Failures")
            else:
                st.metric("Run Rate MTBF", "N/A")

        with kpi_c3:
            if "total_runtime_sec" in rr_metrics:
                val = run_rate_utils.format_duration(rr_metrics["total_runtime_sec"])
                st.metric("Total Run Duration", val)
            else:
                st.metric("Total Run Duration", "N/A")
        
        with kpi_c4:
            if "production_time_sec" in rr_metrics:
                val = run_rate_utils.format_duration(rr_metrics["production_time_sec"])
                pct = (rr_metrics["production_time_sec"] / rr_metrics["total_runtime_sec"] * 100) if rr_metrics["total_runtime_sec"] > 0 else 0
                metric_with_badge("Production Time", val, f"{pct:.1f}%", "green")
            else:
                st.metric("Production Time", "N/A")
        
        with kpi_c5:
            if "downtime_sec" in rr_metrics:
                val = run_rate_utils.format_duration(rr_metrics["downtime_sec"])
                pct = (rr_metrics["downtime_sec"] / rr_metrics["total_runtime_sec"] * 100) if rr_metrics["total_runtime_sec"] > 0 else 0
                metric_with_badge("Downtime", val, f"{pct:.1f}%", "red")
            else:
                st.metric("Downtime", "N/A")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 2: Output & Shots
        kpi_r2_c1, kpi_r2_c2, kpi_r2_c3, kpi_r2_c4 = st.columns(4)
        
        with kpi_r2_c1:
            if "actual" in cr_metrics:
                metric_with_badge("Actual Output (CR)", f"{cr_metrics['actual']:,.0f}", f"{cr_metrics['perf']:.1f}% of Optimal", "green" if cr_metrics['perf'] > 80 else "orange")
            else:
                st.metric("Actual Output", "N/A")

        with kpi_r2_c2:
            if "total_shots" in rr_metrics:
                st.metric("Total Shots", f"{rr_metrics['total_shots']:,}")
            else:
                st.metric("Total Shots", "N/A")

        with kpi_r2_c3:
            if "normal_shots" in rr_metrics:
                pct = (rr_metrics["normal_shots"] / rr_metrics["total_shots"] * 100) if rr_metrics["total_shots"] > 0 else 0
                metric_with_badge("Normal Shots", f"{rr_metrics['normal_shots']:,}", f"{pct:.1f}% of Total", "green")
            else:
                st.metric("Normal Shots", "N/A")

        with kpi_r2_c4:
            if "stop_events" in rr_metrics:
                pct = (rr_metrics["stopped_shots"] / rr_metrics["total_shots"] * 100) if rr_metrics["total_shots"] > 0 else 0
                metric_with_badge("Stop Events", f"{rr_metrics['stop_events']:,}", f"{pct:.1f}% Stopped Shots", "red")
            else:
                st.metric("Stop Events", "N/A")

    else:
        st.warning("No data available for the selected period.")

    st.divider()

    # --- 5. Dashboard Graphs ---
    st.subheader("📈 Metric Dashboards")
    
    g1, g2 = st.columns(2)
    
    # Graph 1: Capacity Risk Waterfall (if data exists)
    with g1:
        if "actual" in cr_metrics:
            st.markdown("#### Capacity Loss Breakdown")
            tot_opt = cr_metrics['optimal']
            tot_act = cr_metrics['actual']
            # Re-fetch these details only if needed for chart, or assume processed above
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
        else:
            st.info("Waiting for Run Rate Data...")

# ==============================================================================
# --- MODULE 2 & 3: INDIVIDUAL APP LOADS ---
# ==============================================================================
elif app_mode == "Capacity Risk Details":
    cr_app_refactored.run_capacity_risk_ui()

elif app_mode == "Run Rate Analysis Details":
    run_rate_app_refactored.run_run_rate_ui()