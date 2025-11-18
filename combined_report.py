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
        global_cavities = st.number_input(
            "Default Working Cavities", min_value=1, value=2,
            help="Used if 'Working Cavities' column is missing."
        )

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
                    # The preprocessor doesn't rename 'Working Cavities' standardly, check flexible names
                    cav_cols = [c for c in df_master.columns if 'cavit' in str(c).lower()]
                    if cav_cols:
                        # Rename to standard
                        df_master.rename(columns={cav_cols[0]: 'Working Cavities'}, inplace=True)
                        cavities_found = True
                    else:
                        # Inject global default
                        df_master['Working Cavities'] = global_cavities
                        cavities_found = False # Mark as not originally found

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
        df_filtered, False, global_cavities, 100.0, 
        global_ct_tolerance, global_stop_gap, global_run_interval
    )
    
    if not cr_results_df.empty:
        total_optimal = cr_results_df['Optimal Output (parts)'].sum()
        total_actual = cr_results_df['Actual Output (parts)'].sum()
        
        loss_downtime_parts = cr_results_df['Capacity Loss (downtime) (parts)'].sum()
        loss_slow_parts = cr_results_df['Capacity Loss (slow cycle time) (parts)'].sum()
        gain_fast_parts = cr_results_df['Capacity Gain (fast cycle time) (parts)'].sum()
        net_efficiency_loss_parts = loss_slow_parts - gain_fast_parts
        
        total_loss_parts = cr_results_df['Total Capacity Loss (parts)'].sum()
        
        cr_perf = (total_actual / total_optimal) * 100 if total_optimal > 0 else 0
        
        cr_metrics = {
            "optimal": total_optimal,
            "actual": total_actual,
            "loss_total_parts": total_loss_parts,
            "loss_availability_parts": loss_downtime_parts,
            "loss_efficiency_parts": net_efficiency_loss_parts,
            "perf": cr_perf
        }

    # -- RR Calculation --
    # RR needs 'shot_time', CR prep creates 'SHOT TIME'. Map it.
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
        "total_shots": rr_res.get('total_shots', 0),
        "stop_events": rr_res.get('stop_events', 0)
    }

    # --- 4. INSIGHT TABLES ---
    
    st.subheader("📋 Tooling Performance Insights")
    
    # Check if cavities were present or defaulted
    if not cavities_found:
        st.warning("⚠️ 'Working Cavities' column not found. Used global default. Capacity (parts) metrics may be inaccurate if cavities vary per run.")
    
    # Data Preparation for Table
    row_opp_lost = f"{cr_metrics['loss_total_parts']:,.0f} parts"
    row_rr_eff = f"{rr_metrics.get('efficiency', 0):.1f}%"
    row_rr_mtbf = f"{rr_metrics.get('mtbf', 0):.0f} Minutes"
    row_rr_mttr = f"{rr_metrics.get('mttr', 0):.0f} Minutes"
    
    val_opt = f"{cr_metrics['optimal']:,.0f} parts"
    gap_total = cr_metrics['actual'] - cr_metrics['optimal']
    val_act = f"{cr_metrics['actual']:,.0f} ({gap_total:+,.0f} parts)"
    val_avail = f"-{cr_metrics['loss_availability_parts']:,.0f} parts"
    val_eff = f"-{cr_metrics['loss_efficiency_parts']:,.0f} parts"

    if cavities_found:
        # Unified Table
        data_unified = {
            "KPI": [
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
    else:
        # Split Tables (RR is safe, CR is estimated)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Run Rate Metrics (Time-Based)")
            st.table(pd.DataFrame({
                "Metric": ["RR Efficiency", "RR MTBF", "RR MTTR"],
                "Value": [row_rr_eff, row_rr_mtbf, row_rr_mttr]
            }))
        with c2:
            st.markdown(f"##### Capacity Metrics (Est. {global_cavities} Cavities)")
            st.table(pd.DataFrame({
                "Metric": ["Parts Opp. Lost", "Optimal Output", "Actual Output", "Availability Loss", "Efficiency Loss"],
                "Value": [row_opp_lost, val_opt, val_act, val_avail, val_eff]
            }))

    st.divider()

    # --- 5. EXTENSIVE GRAPH FEED (20+ Options) ---
    st.subheader("📈 Detailed Analysis Feed")
    st.markdown("*Review the following charts to identify specific performance patterns.*")

    # Prepare Data for Charting
    if not cr_all_shots_df.empty:
        shots = cr_all_shots_df.copy()
        shots['Hour'] = shots['SHOT TIME'].dt.hour
        shots['Weekday'] = shots['SHOT TIME'].dt.day_name()
        shots['Date_Str'] = shots['SHOT TIME'].dt.strftime('%Y-%m-%d')
        shots['Cycle Time (s)'] = shots['Actual CT']
        
        # 1. SCATTER: Cycle Time over Time (The classic)
        fig1 = px.scatter(shots, x='SHOT TIME', y='Actual CT', color='Shot Type', title="1. Cycle Time Scatter Plot",
                          color_discrete_map={'Slow': '#ff6961', 'Fast': '#ffb347', 'On Target': '#77dd77', 'RR Downtime (Stop)': 'gray'})
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2. HISTOGRAM: Cycle Time Distribution
        fig2 = px.histogram(shots[shots['stop_flag']==0], x='Actual CT', nbins=50, title="2. Cycle Time Distribution (Production Shots)",
                            color_discrete_sequence=['#3498DB'])
        st.plotly_chart(fig2, use_container_width=True)
        
        # 3. BAR: Total Output by Day
        daily_out = shots.groupby('Date_Str')['Working Cavities'].sum().reset_index()
        fig3 = px.bar(daily_out, x='Date_Str', y='Working Cavities', title="3. Total Daily Output (Parts)",
                      color_discrete_sequence=['#2ECC71'])
        st.plotly_chart(fig3, use_container_width=True)
        
        # 4. BOX: Cycle Time Variability by Hour
        fig4 = px.box(shots[shots['stop_flag']==0], x='Hour', y='Actual CT', title="4. Cycle Time Consistency by Hour of Day")
        st.plotly_chart(fig4, use_container_width=True)
        
        # 5. HEATMAP: Output Intensity (Day vs Hour)
        heatmap_data = shots.groupby(['Date_Str', 'Hour'])['Working Cavities'].sum().reset_index()
        fig5 = px.density_heatmap(heatmap_data, x='Date_Str', y='Hour', z='Working Cavities', title="5. Output Heatmap (Day vs Hour)",
                                  color_continuous_scale='Viridis')
        st.plotly_chart(fig5, use_container_width=True)

        # 6. LINE: Cumulative Production
        shots['Cumulative Output'] = shots['Working Cavities'].cumsum()
        fig6 = px.line(shots, x='SHOT TIME', y='Cumulative Output', title="6. Cumulative Production Trend")
        st.plotly_chart(fig6, use_container_width=True)

        # 7. PIE: Shot Type Breakdown
        fig7 = px.pie(shots, names='Shot Type', title="7. Shot Classification Breakdown",
                      color='Shot Type', color_discrete_map={'Slow': '#ff6961', 'Fast': '#ffb347', 'On Target': '#77dd77', 'RR Downtime (Stop)': 'gray'})
        st.plotly_chart(fig7, use_container_width=True)

        # 8. BAR: Top Longest Downtime Events
        # Need to identify distinct stop events and their duration
        stops = shots[shots['stop_flag']==1].copy()
        # Simple logic: large adj_ct_sec values are long stops
        top_stops = stops.nlargest(10, 'adj_ct_sec')
        fig8 = px.bar(top_stops, x='SHOT TIME', y='adj_ct_sec', title="8. Top 10 Longest Stop Events",
                      labels={'adj_ct_sec': 'Duration (sec)'}, color_discrete_sequence=['#E74C3C'])
        st.plotly_chart(fig8, use_container_width=True)
        
        # 9. SUNBURST: Loss Hierarchy
        # Create dummy hierarchy for visual
        loss_data = pd.DataFrame([
            {'Category': 'Loss', 'Subcat': 'Availability', 'Value': cr_metrics['loss_availability_parts']},
            {'Category': 'Loss', 'Subcat': 'Efficiency', 'Value': cr_metrics['loss_efficiency_parts']}
        ])
        # Only show if values are positive (losses)
        loss_data['Value'] = loss_data['Value'].abs()
        fig9 = px.sunburst(loss_data, path=['Category', 'Subcat'], values='Value', title="9. Capacity Loss Hierarchy (Parts)",
                           color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig9, use_container_width=True)
        
        # 10. SCATTER: Run Duration vs Output (if run_id exists)
        if 'run_id' in shots.columns:
            run_stats = shots.groupby('run_id').agg({'Working Cavities': 'sum', 'Actual CT': 'count'}).rename(columns={'Actual CT': 'Shot Count'})
            fig10 = px.scatter(run_stats, x='Shot Count', y='Working Cavities', title="10. Run Size Analysis: Output vs Shot Count",
                               trendline="ols")
            st.plotly_chart(fig10, use_container_width=True)
            
        # 11. LINE: Rolling Efficiency (Moving Average)
        # Calculate a rolling 100-shot efficiency
        shots['is_normal'] = (shots['stop_flag'] == 0).astype(int)
        shots['Rolling Eff'] = shots['is_normal'].rolling(100).mean()
        fig11 = px.line(shots, x='SHOT TIME', y='Rolling Eff', title="11. Rolling Efficiency (100-shot window)",
                        labels={'Rolling Eff': 'Efficiency'})
        st.plotly_chart(fig11, use_container_width=True)

        # 12. BAR: Stops per Hour of Day
        stops_hourly = shots[shots['stop_flag']==1].groupby('Hour').size().reset_index(name='Stop Count')
        fig12 = px.bar(stops_hourly, x='Hour', y='Stop Count', title="12. Frequency of Stops by Hour of Day")
        st.plotly_chart(fig12, use_container_width=True)

        # 13. AREA: Production vs Loss over Time (Daily)
        if not cr_results_df.empty:
            area_data = cr_results_df.reset_index()
            fig13 = go.Figure()
            fig13.add_trace(go.Scatter(x=area_data['Date'], y=area_data['Actual Output (parts)'], stackgroup='one', name='Actual'))
            fig13.add_trace(go.Scatter(x=area_data['Date'], y=area_data['Total Capacity Loss (parts)'], stackgroup='one', name='Loss'))
            fig13.update_layout(title="13. Daily Production vs Loss (Stacked Area)")
            st.plotly_chart(fig13, use_container_width=True)
            
        # 14. GAUGE: Overall Stability (Large)
        stab_val = rr_metrics.get('stability', 0)
        fig14 = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = stab_val,
            title = {'text': "14. Overall Stability Index"},
            delta = {'reference': 85},
            gauge = {'axis': {'range': [None, 100]},
                     'steps' : [
                         {'range': [0, 50], 'color': "lightgray"},
                         {'range': [50, 85], 'color': "gray"}],
                     'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 85}}))
        st.plotly_chart(fig14, use_container_width=True)
        
        # 15. VIOLIN: Cycle Time Distribution by Weekday
        fig15 = px.violin(shots[shots['stop_flag']==0], y="Actual CT", x="Weekday", box=True, title="15. Cycle Time Spread by Weekday")
        st.plotly_chart(fig15, use_container_width=True)

        # 16. BAR: Downtime Duration Distribution (Buckets)
        if 'adj_ct_sec' in shots.columns:
            downtime_shots = shots[shots['stop_flag']==1]
            fig16 = px.histogram(downtime_shots, x="adj_ct_sec", nbins=30, title="16. Distribution of Downtime Durations (Log Scale)",
                                 log_y=True, labels={'adj_ct_sec': 'Stop Duration (s)'})
            st.plotly_chart(fig16, use_container_width=True)

        # 17. SCATTER: CT vs Time (Filtered for Production Only)
        fig17 = px.scatter(shots[shots['stop_flag']==0], x='SHOT TIME', y='Actual CT', title="17. Production-Only Cycle Time Stability (Zoomed)",
                           opacity=0.5)
        st.plotly_chart(fig17, use_container_width=True)
        
        # 18. LINE: Cumulative Downtime
        shots['Cumulative Downtime'] = np.where(shots['stop_flag']==1, shots['adj_ct_sec'], 0).cumsum() / 3600 # in Hours
        fig18 = px.line(shots, x='SHOT TIME', y='Cumulative Downtime', title="18. Cumulative Downtime Accumulation (Hours)")
        st.plotly_chart(fig18, use_container_width=True)
        
        # 19. BAR: Shift Performance (Approximation 8h blocks)
        shots['Shift_Block'] = (shots['Hour'] // 8) + 1
        shift_perf = shots.groupby('Shift_Block')['Working Cavities'].sum().reset_index()
        fig19 = px.bar(shift_perf, x='Shift_Block', y='Working Cavities', title="19. Output by 8-Hour Shift Block (1=0-8, 2=8-16, 3=16-24)")
        st.plotly_chart(fig19, use_container_width=True)
        
        # 20. 3D SCATTER: Time vs CT vs Output (Experimental)
        fig20 = px.scatter_3d(shots.head(1000), x='Hour', y='Actual CT', z='Working Cavities', color='Shot Type', title="20. 3D Analysis (First 1000 shots)")
        st.plotly_chart(fig20, use_container_width=True)


# ==============================================================================
# --- MODULE 2 & 3: INDIVIDUAL APP LOADS ---
# ==============================================================================
elif app_mode == "Capacity Risk Details":
    cr_app_refactored.run_capacity_risk_ui()

elif app_mode == "Run Rate Analysis Details":
    run_rate_app_refactored.run_run_rate_ui()