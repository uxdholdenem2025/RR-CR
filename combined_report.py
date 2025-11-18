import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date

# Import utils and refactored apps
import cr_utils
import run_rate_utils
import cr  # This imports your cr.py file
import run_rate_app_refactored # This imports your run_rate_app_refactored.py file

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
        global_cavities = st.number_input(
            "Default Working Cavities", min_value=1, value=2,
            help="Used if 'Working Cavities' column is missing."
        )
        global_exclude_maintenance = st.toggle(
            "Remove Maintenance/Warehouse Shots", value=False,
            help="Exclude shots where Plant Area is Maintenance or Warehouse."
        )

    # Cost Settings
    with st.sidebar.expander("💰 Cost Settings", expanded=True):
        machine_rate = st.number_input("Machine Rate ($/h)", value=170.0, step=10.0)
        labor_rate = st.number_input("Labor Cost ($/h)", value=10.0, step=5.0)

    # SINGLE GLOBAL FILE UPLOADER
    st.sidebar.header("2. Data Source")
    master_file = st.sidebar.file_uploader("Upload Master Data File (CSV/XLS)", type=["csv", "xlsx", "xls"], key="global_master_upload")
    
    # --- 2. Data Loading & Date Filtering ---
    
    df_master = pd.DataFrame()
    min_global_date = None
    max_global_date = None
    cavities_found = False

    if master_file:
        try:
            # Load using CR utils as base loader
            df_raw = cr_utils.load_data(master_file)
            if df_raw is not None and not df_raw.empty:
                # Preprocess using CR logic (standardizes SHOT TIME, etc)
                df_master, min_val, max_val = cr_utils.get_preprocessed_data(df_raw)
                
                if not df_master.empty:
                    # Add 'date_obj' for RR compatibility
                    df_master['date_obj'] = df_master['SHOT TIME'].dt.date
                    
                    # Check for cavities column
                    cav_cols = [c for c in df_master.columns if 'cavit' in str(c).lower()]
                    if cav_cols:
                        # Rename to standard
                        df_master.rename(columns={cav_cols[0]: 'Working Cavities'}, inplace=True)
                        cavities_found = True
                    else:
                        # Inject global default
                        df_master['Working Cavities'] = global_cavities
                        cavities_found = False 

                    # Set dates
                    min_global_date = min_val
                    max_global_date = max_val
                    
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

    # Date Picker
    if not df_master.empty:
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
        mask = (df_master['date'] >= start_date) & (df_master['date'] <= end_date)
        df_filtered = df_master[mask].copy()
    else:
        st.info("👈 Please upload a master data file in the sidebar to begin.")
        st.stop()

    st.divider()

    # --- 3. Calculations ---
    
    cr_metrics = {}
    rr_metrics = {}
    
    # -- CR Calculation --
    # Use cached wrapper
    cr_results_df, cr_all_shots_df = cr_utils.run_capacity_calculation_cached_v2(
        df_filtered, global_exclude_maintenance, global_cavities, 100.0, 
        global_ct_tolerance, global_stop_gap, global_run_interval
    )
    
    if not cr_results_df.empty and not cr_all_shots_df.empty:
        # --- CRITICAL FIX: AGGREGATE BY RUN (Matches Original App) ---
        run_summary_df = cr_utils.calculate_run_summaries(cr_all_shots_df, 100.0)
        
        if not run_summary_df.empty:
            total_optimal = run_summary_df['Optimal Output (parts)'].sum()
            total_actual = run_summary_df['Actual Output (parts)'].sum()
            
            loss_downtime_parts = run_summary_df['Capacity Loss (downtime) (parts)'].sum()
            loss_slow_parts = run_summary_df['Capacity Loss (slow cycle time) (parts)'].sum()
            gain_fast_parts = run_summary_df['Capacity Gain (fast cycle time) (parts)'].sum()
            net_efficiency_loss_parts = loss_slow_parts - gain_fast_parts
            
            total_loss_parts = run_summary_df['Total Capacity Loss (parts)'].sum()
            total_loss_sec = run_summary_df['Total Capacity Loss (sec)'].sum()
            
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
    rr_df_input = df_filtered.rename(columns={'SHOT TIME': 'shot_time', 'Actual CT': 'ACTUAL CT'})
    
    rr_calc = run_rate_utils.RunRateCalculator(
        rr_df_input, 
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

    # --- 4. INSIGHT TABLES ---
    
    st.subheader("📋 Tooling Performance Insights")
    
    if not cavities_found:
        st.warning("⚠️ 'Working Cavities' column not found. Used global default. Capacity (parts) metrics may be inaccurate if cavities vary per run.")
    
    # Data Preparation
    if cr_metrics:
        row_opp_lost = f"{cr_metrics['loss_total_parts']:,.0f} parts"
        loss_hours = cr_metrics['loss_total_sec'] / 3600.0
        total_cost = loss_hours * (machine_rate + labor_rate)
        row_loss_hrs = f"{loss_hours:,.1f} Hours"
        row_cost = f"${total_cost:,.0f} *"
        
        val_opt = f"{cr_metrics['optimal']:,.0f} parts"
        gap_total = cr_metrics['actual'] - cr_metrics['optimal']
        val_act = f"{cr_metrics['actual']:,.0f} ({gap_total:+,.0f} parts)"
        val_avail = f"-{cr_metrics['loss_availability_parts']:,.0f} parts"
        val_eff = f"-{cr_metrics['loss_efficiency_parts']:,.0f} parts"
    else:
        row_opp_lost = "N/A"
        row_loss_hrs = "N/A"
        row_cost = "N/A"
        val_opt = "N/A"
        val_act = "N/A"
        val_avail = "N/A"
        val_eff = "N/A"

    row_rr_eff = f"{rr_metrics.get('efficiency', 0):.1f}%"
    row_rr_mtbf = f"{rr_metrics.get('mtbf', 0):.0f} Minutes"
    row_rr_mttr = f"{rr_metrics.get('mttr', 0):.0f} Minutes"
    
    if cavities_found and cr_metrics:
        # Unified Table
        data_unified = {
            "KPI": [
                "Total Incurred Loss in Machine Hours",
                f"Total Incurred Costs (Machine Rate + Labor)*",
                "Parts Opportunity Lost",
                "Run Rate Efficiency",
                "Run Rate MTBF",
                "Run Rate MTTR",
                "Capacity Risk: Optimal Output (100% OEE)",
                "Capacity Risk: Actual Output",
                "Capacity Risk: Availability Loss",
                "Capacity Risk: Efficiency Loss"
            ],
            "Value": [
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
        st.table(pd.DataFrame(data_unified))
        st.caption(f"*Based on ${machine_rate}/h machine rate + ${labor_rate}/h labor")
    else:
        # Split Tables
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Run Rate Metrics")
            st.table(pd.DataFrame({
                "Metric": ["RR Efficiency", "RR MTBF", "RR MTTR"],
                "Value": [row_rr_eff, row_rr_mtbf, row_rr_mttr]
            }))
        with c2:
            if cr_metrics:
                st.markdown(f"##### Capacity Metrics (Est. {global_cavities} Cavities)")
                st.table(pd.DataFrame({
                    "Metric": ["Loss Hours", "Est. Cost", "Parts Opp. Lost", "Optimal", "Actual", "Availability Loss", "Efficiency Loss"],
                    "Value": [row_loss_hrs, row_cost, row_opp_lost, val_opt, val_act, val_avail, val_eff]
                }))
            else:
                st.info("Upload valid data to see Capacity Metrics.")

    st.divider()

    # --- 5. EXTENSIVE GRAPH FEED ---
    st.subheader("📈 Detailed Analysis Feed")
    st.markdown("*Review the following charts to identify specific performance patterns.*")

    if not cr_all_shots_df.empty:
        shots = cr_all_shots_df.copy()
        shots['Hour'] = shots['SHOT TIME'].dt.hour
        shots['Weekday'] = shots['SHOT TIME'].dt.day_name()
        shots['Date_Str'] = shots['SHOT TIME'].dt.strftime('%Y-%m-%d')
        
        # 1. SCATTER: Cycle Time
        fig1 = px.scatter(shots, x='SHOT TIME', y='Actual CT', color='Shot Type', title="1. Cycle Time Scatter Plot",
                          color_discrete_map={'Slow': '#ff6961', 'Fast': '#ffb347', 'On Target': '#77dd77', 'RR Downtime (Stop)': 'gray'})
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2. HISTOGRAM: Distribution
        fig2 = px.histogram(shots[shots['stop_flag']==0], x='Actual CT', nbins=50, title="2. Cycle Time Distribution",
                            color_discrete_sequence=['#3498DB'])
        st.plotly_chart(fig2, use_container_width=True)
        
        # 3. BAR: Daily Output
        daily_out = shots.groupby('Date_Str')['Working Cavities'].sum().reset_index()
        fig3 = px.bar(daily_out, x='Date_Str', y='Working Cavities', title="3. Total Daily Output",
                      color_discrete_sequence=['#2ECC71'])
        st.plotly_chart(fig3, use_container_width=True)
        
        # 4. HEATMAP
        heatmap_data = shots.groupby(['Date_Str', 'Hour'])['Working Cavities'].sum().reset_index()
        fig5 = px.density_heatmap(heatmap_data, x='Date_Str', y='Hour', z='Working Cavities', title="4. Output Heatmap (Day vs Hour)",
                                  color_continuous_scale='Viridis')
        st.plotly_chart(fig5, use_container_width=True)

        # 5. PIE: Breakdown
        fig7 = px.pie(shots, names='Shot Type', title="5. Shot Classification Breakdown",
                      color='Shot Type', color_discrete_map={'Slow': '#ff6961', 'Fast': '#ffb347', 'On Target': '#77dd77', 'RR Downtime (Stop)': 'gray'})
        st.plotly_chart(fig7, use_container_width=True)

        # 6. BAR: Longest Stops
        if 'adj_ct_sec' in shots.columns:
            stops = shots[shots['stop_flag']==1].copy()
            top_stops = stops.nlargest(10, 'adj_ct_sec')
            fig8 = px.bar(top_stops, x='SHOT TIME', y='adj_ct_sec', title="6. Top 10 Longest Stop Events (sec)",
                          color_discrete_sequence=['#E74C3C'])
            st.plotly_chart(fig8, use_container_width=True)
        
        # 7. SUNBURST: Loss
        if cr_metrics:
            loss_data = pd.DataFrame([
                {'Category': 'Loss', 'Subcat': 'Availability', 'Value': cr_metrics['loss_availability_parts']},
                {'Category': 'Loss', 'Subcat': 'Efficiency', 'Value': cr_metrics['loss_efficiency_parts']}
            ])
            loss_data['Value'] = loss_data['Value'].abs()
            fig9 = px.sunburst(loss_data, path=['Category', 'Subcat'], values='Value', title="7. Capacity Loss Hierarchy",
                               color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig9, use_container_width=True)
        
        # 8. GAUGE: Stability
        stab_val = rr_metrics.get('stability', 0)
        fig14 = go.Figure(go.Indicator(
            mode = "gauge+number", value = stab_val, title = {'text': "8. Overall Stability Index"},
            gauge = {'axis': {'range': [None, 100]}, 'steps' : [{'range': [0, 85], 'color': "lightgray"}], 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 85}}))
        st.plotly_chart(fig14, use_container_width=True)
        
        # 9. BAR: Shift Blocks
        shots['Shift_Block'] = (shots['Hour'] // 8) + 1
        shift_perf = shots.groupby('Shift_Block')['Working Cavities'].sum().reset_index()
        fig19 = px.bar(shift_perf, x='Shift_Block', y='Working Cavities', title="9. Output by 8-Hour Shift Block")
        st.plotly_chart(fig19, use_container_width=True)


# ==============================================================================
# --- MODULE 2 & 3: INDIVIDUAL APP LOADS ---
# ==============================================================================
elif app_mode == "Capacity Risk Details":
    cr.run_capacity_risk_ui()

elif app_mode == "Run Rate Analysis Details":
    run_rate_app_refactored.run_run_rate_ui()