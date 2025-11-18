import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import calendar

# Import all calculation functions from the helper file
import cr_utils

# ==================================================================
#                       MAIN APP UI FUNCTION
# ==================================================================

def run_capacity_risk_ui():
    """
    This function contains the main UI logic for the Capacity Risk app.
    It will be imported and called by combined_report.py.
    """
    
    # Get version from utils (or define it here)
    __version__ = "v9.2 (downtime bug fix 999.9)"

    st.title("Capacity Risk Report")
    st.markdown(f"**App Version:** `{__version__}` (RR-Downtime + CR-Inefficiency)")

    # --- Sidebar for Inputs ---
    st.sidebar.header("Capacity Risk Configuration") # Renamed for clarity

    uploaded_file = st.sidebar.file_uploader(
        "Upload Raw Data File (CSV or Excel)", 
        type=["csv", "xlsx", "xls"],
        key="cr_file_uploader" # Add a unique key
    )

    # --- Main Page Display ---
    if uploaded_file is not None:

        df_raw = cr_utils.load_data(uploaded_file)
        
        if df_raw is None or df_raw.empty:
            st.error("Uploaded file is empty or could not be loaded.")
            st.stop()
            
        # --- Pre-process data to get date ranges for filters ---
        df_processed, min_date, max_date = cr_utils.get_preprocessed_data(df_raw)
        
        if df_processed.empty or min_date is None:
            st.error("Could not parse 'SHOT TIME' column or find valid data.")
            st.stop()

        st.success(f"Successfully loaded file: **{uploaded_file.name}**")
        
        # --- 1. "Select Analysis Period" control ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("1. Select Analysis Period")
        analysis_period_selector = st.sidebar.radio(
            "Filter data by:",
            ["Entire File", "Daily", "Weekly", "Monthly", "Custom Period"],
            key="cr_analysis_period_selector" # Added key
        )
        
        df_to_process = pd.DataFrame()

        if analysis_period_selector == "Daily":
            available_dates = sorted(df_processed["date"].unique(), reverse=True)
            selected_date = st.sidebar.selectbox("Select Date", available_dates, format_func=lambda d: d.strftime('%Y-%m-%d'), key="cr_daily_select")
            df_to_process = df_processed[df_processed["date"] == selected_date].copy()
        
        elif analysis_period_selector == "Weekly":
            available_weeks = sorted(df_processed["week"].unique(), reverse=True)
            selected_week = st.sidebar.selectbox("Select Week", available_weeks, format_func=lambda w: f"Week {w.week}, {w.start_time.year}", key="cr_weekly_select")
            df_to_process = df_processed[df_processed["week"] == selected_week].copy()

        elif analysis_period_selector == "Monthly":
            available_months = sorted(df_processed["month"].unique(), reverse=True)
            selected_month = st.sidebar.selectbox("Select Month", available_months, format_func=lambda m: m.strftime('%B %Y'), key="cr_monthly_select")
            df_to_process = df_processed[df_processed["month"] == selected_month].copy()
        
        elif analysis_period_selector == "Custom Period":
            start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date, key="cr_custom_start")
            end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date, key="cr_custom_end")
            if start_date > end_date:
                st.sidebar.error("Start Date must be before End Date.")
                st.stop()
            else:
                mask = (df_processed['date'] >= start_date) & (df_processed['date'] <= end_date)
                df_to_process = df_processed[mask].copy()
        
        else: # "Entire File"
            df_to_process = df_processed.copy()

        # --- 2. "Select Grouping" ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("2. Select Grouping")
        display_frequency = st.sidebar.radio(
            "Display Frequency",
            ['Daily', 'Weekly', 'Monthly', 'by Run'],
            index=3, # Default to 'by Run'
            horizontal=True,
            key="cr_display_frequency"
        )
        
        # --- 3. "Set Calculation Logic" ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("3. Set Calculation Logic")
        
        mode_ct_tolerance = st.sidebar.slider(
            "Mode CT Tolerance (%)", 0.01, 0.50, 0.05, 0.01,
            key="cr_mode_ct_tolerance",
            help="Tolerance band (±) around the **Actual Mode CT**. Shots outside this band are flagged as 'Abnormal Cycle' (Downtime)."
        )
        
        rr_downtime_gap = st.sidebar.slider(
            "RR Downtime Gap (sec)", 0.0, 10.0, 2.0, 0.5, 
            key="cr_rr_downtime_gap",
            help="Minimum idle time between shots to be considered a stop."
        )
        
        run_interval_hours = st.sidebar.slider(
            "Run Interval Threshold (hours)", 1.0, 24.0, 8.0, 0.5,
            key="cr_run_interval_hours",
            help="Gaps between shots *longer* than this will be excluded from all calculations (e.g., weekends)."
        )

        toggle_filter = st.sidebar.toggle(
            "Remove Maintenance/Warehouse Shots",
            value=False,
            key="cr_toggle_filter",
            help="If ON, all calculations will exclude shots where 'Plant Area' is 'Maintenance' or 'Warehouse'."
        )
        
        default_cavities = st.sidebar.number_input(
            "Default Working Cavities",
            min_value=1,
            value=2,
            key="cr_default_cavities",
            help="This value will be used if the 'Working Cavities' column is not found in the file."
        )

        # --- 4. "Set Report Benchmark" ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("4. Set Report Benchmark")
        
        benchmark_view = st.sidebar.radio(
            "Select Report Benchmark",
            ['Optimal Output', 'Target Output'],
            index=0,
            horizontal=False,
            key="cr_benchmark_view",
            help="Select the benchmark to compare against (e.g., 'Total Capacity Loss' vs 'Optimal' or 'Target')."
        )

        if benchmark_view == "Target Output":
            target_output_perc = st.sidebar.slider(
                "Target Output % (of Optimal)",
                min_value=0.0, max_value=100.0,
                value=90.0,
                step=1.0,
                format="%.0f%%",
                key="cr_target_output_perc",
                help="Sets the 'Target Output (parts)' goal as a percentage of 'Optimal Output (parts)'."
            )
        else:
            target_output_perc = 100.0 
        
        st.sidebar.caption(f"App Version: **{__version__}**")

        # --- Run Calculation ---
        with st.spinner("Calculating Capacity Risk..."):
            
            # Create a unique cache key based on all inputs
            cache_key_parts = [
                __version__,
                uploaded_file.name,
                analysis_period_selector,
                toggle_filter,
                default_cavities,
                target_output_perc,
                mode_ct_tolerance,
                rr_downtime_gap,
                run_interval_hours
            ]
            if analysis_period_selector == "Custom Period":
                cache_key_parts.extend([start_date, end_date])
            elif analysis_period_selector in ["Daily", "Weekly", "Monthly"]:
                if 'selected_date' in locals(): cache_key_parts.append(selected_date)
                if 'selected_week' in locals(): cache_key_parts.append(selected_week)
                if 'selected_month' in locals(): cache_key_parts.append(selected_month)
                
            cache_key = "_".join(map(str, cache_key_parts))

            # --- Run the cached calculation (from cr_utils) ---
            results_df, all_shots_df = cr_utils.run_capacity_calculation_cached_v2(
                df_to_process,
                toggle_filter,
                default_cavities,
                target_output_perc,
                mode_ct_tolerance,  
                rr_downtime_gap,        
                run_interval_hours,      
                _cache_version=cache_key
            )

            if results_df is None or results_df.empty or all_shots_df.empty:
                st.error("No valid data found for the selected period. Cannot proceed.")
            else:
                
                # --- 1. Calculate all dataframes ONCE ---
                daily_summary_df = results_df.copy()
                # Call logic function from cr_utils
                run_summary_df = cr_utils.calculate_run_summaries(all_shots_df, target_output_perc)
                run_summary_df_for_total = run_summary_df.copy()
                
                # Initialize variables to avoid NameError
                report_table_1 = pd.DataFrame()
                report_table_optimal = pd.DataFrame()
                report_table_target = pd.DataFrame()
                
                # --- 2. Get All-Time totals ---
                all_time_totals = {}
                
                if run_summary_df_for_total.empty:
                    st.error("Failed to calculate 'by Run' summary for All-Time totals.")
                    # ... (set all_time_totals to 0) ...
                    all_time_totals = {
                        'total_produced': 0, 'total_downtime_loss_parts': 0,
                        'total_slow_loss_parts': 0, 'total_fast_gain_parts': 0,
                        'total_net_cycle_loss_parts': 0, 'total_optimal_100': 0,
                        'total_target': 0, 'total_downtime_loss_sec': 0,
                        'total_slow_loss_sec': 0, 'total_fast_gain_sec': 0,
                        'total_net_cycle_loss_sec': 0, 'run_time_sec_total': 0,
                        'run_time_dhm_total': "0m", 'total_actual_ct_sec': 0,
                        'total_actual_ct_dhm': "0m", 'total_true_net_loss_parts': 0,
                        'total_true_net_loss_sec': 0, 'total_calculated_net_loss_parts': 0,
                        'total_calculated_net_loss_sec': 0
                    }
                else:
                    total_produced = run_summary_df_for_total['Actual Output (parts)'].sum()
                    total_downtime_loss_parts = run_summary_df_for_total['Capacity Loss (downtime) (parts)'].sum()
                    total_slow_loss_parts = run_summary_df_for_total['Capacity Loss (slow cycle time) (parts)'].sum()
                    total_fast_gain_parts = run_summary_df_for_total['Capacity Gain (fast cycle time) (parts)'].sum()
                    total_net_cycle_loss_parts = total_slow_loss_parts - total_fast_gain_parts
                    
                    total_optimal_100 = run_summary_df_for_total['Optimal Output (parts)'].sum()
                    total_target = run_summary_df_for_total['Target Output (parts)'].sum()
                    
                    total_downtime_loss_sec = run_summary_df_for_total['Capacity Loss (downtime) (sec)'].sum()
                    total_slow_loss_sec = run_summary_df_for_total['Capacity Loss (slow cycle time) (sec)'].sum()
                    total_fast_gain_sec = run_summary_df_for_total['Capacity Gain (fast cycle time) (sec)'].sum()
                    total_net_cycle_loss_sec = total_slow_loss_sec - total_fast_gain_sec
                    
                    run_time_sec_total = run_summary_df_for_total['Filtered Run Time (sec)'].sum()
                    run_time_dhm_total = cr_utils.format_seconds_to_dhm(run_time_sec_total) # Use util function
                    
                    total_actual_ct_sec = run_summary_df_for_total['Actual Cycle Time Total (sec)'].sum()
                    total_actual_ct_dhm = cr_utils.format_seconds_to_dhm(total_actual_ct_sec) # Use util function
                    
                    total_calculated_net_loss_parts = total_downtime_loss_parts + total_net_cycle_loss_parts
                    total_calculated_net_loss_sec = total_downtime_loss_sec + total_net_cycle_loss_sec
                    
                    total_true_net_loss_parts = total_optimal_100 - total_produced
                    total_true_net_loss_sec = total_calculated_net_loss_sec
                    
                    all_time_totals = {
                        'total_produced': total_produced, 'total_downtime_loss_parts': total_downtime_loss_parts,
                        'total_slow_loss_parts': total_slow_loss_parts, 'total_fast_gain_parts': total_fast_gain_parts,
                        'total_net_cycle_loss_parts': total_net_cycle_loss_parts, 'total_optimal_100': total_optimal_100,
                        'total_target': total_target, 'total_downtime_loss_sec': total_downtime_loss_sec,
                        'total_slow_loss_sec': total_slow_loss_sec, 'total_fast_gain_sec': total_fast_gain_sec,
                        'total_net_cycle_loss_sec': total_net_cycle_loss_sec, 'run_time_sec_total': run_time_sec_total,
                        'run_time_dhm_total': run_time_dhm_total, 'total_actual_ct_sec': total_actual_ct_sec,
                        'total_actual_ct_dhm': total_actual_ct_dhm, 'total_true_net_loss_parts': total_true_net_loss_parts,
                        'total_true_net_loss_sec': total_true_net_loss_sec, 'total_calculated_net_loss_parts': total_calculated_net_loss_parts,
                        'total_calculated_net_loss_sec': total_calculated_net_loss_sec
                    }
                
                
                tab1, = st.tabs(["Capacity Risk Report"])

                with tab1:
                    st.header("Overall Summary for Selected Period")
                    
                    run_time_label = "Overall Run Time" if not toggle_filter else "Filtered Run Time"
                    actual_output_perc_val = (all_time_totals['total_produced'] / all_time_totals['total_optimal_100']) if all_time_totals['total_optimal_100'] > 0 else 0
                    benchmark_title = "Optimal Output"

                    # --- Box 1: Overall Summary ---
                    with st.container(border=True):
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric(run_time_label, all_time_totals['run_time_dhm_total'])
                        with c2:
                            if benchmark_view == "Target Output":
                                st.metric(f"Target Output ({target_output_perc:.0f}%)", f"{all_time_totals['total_target']:,.0f}")
                                st.caption(f"Optimal (100%): {all_time_totals['total_optimal_100']:,.0f}")
                            else:
                                st.metric("Optimal Output (100%)", f"{all_time_totals['total_optimal_100']:,.0f}")
                        with c3:
                            st.metric(f"Actual Output ({actual_output_perc_val:.1%})", f"{all_time_totals['total_produced']:,.0f} parts")
                            st.caption(f"Actual Production Time: {all_time_totals['total_actual_ct_dhm']}")
                            if benchmark_view == "Target Output":
                                gap_to_target = all_time_totals['total_produced'] - all_time_totals['total_target']
                                gap_perc = (gap_to_target / all_time_totals['total_target']) if all_time_totals['total_target'] > 0 else 0
                                gap_color = "green" if gap_to_target > 0 else "red"
                                st.caption(f"Gap to Target: <span style='color:{gap_color};'>{gap_to_target:+,.0f} ({gap_perc:+.1%})</span>", unsafe_allow_html=True)
                        with c4:
                            if benchmark_view == "Target Output":
                                total_loss_vs_target_parts = np.maximum(0, all_time_totals['total_target'] - all_time_totals['total_produced'])
                                total_loss_vs_target_sec = run_summary_df_for_total['Capacity Loss (vs Target) (sec)'].sum()
                                st.markdown(f"**Capacity Loss (vs Target)**")
                                st.markdown(f"<h3><span style='color:red;'>{total_loss_vs_target_parts:,.0f} parts</span></h3>", unsafe_allow_html=True) 
                                st.caption(f"Total Time Lost vs Target: {cr_utils.format_seconds_to_dhm(total_loss_vs_target_sec)}")
                            else:
                                st.markdown(f"**Total Capacity Loss (True)**")
                                st.markdown(f"<h3><span style='color:red;'>{all_time_totals['total_true_net_loss_parts']:,.0f} parts</span></h3>", unsafe_allow_html=True) 
                                st.caption(f"Total Time Lost: {cr_utils.format_seconds_to_dhm(all_time_totals['total_true_net_loss_sec'])}")

                    # --- Waterfall Chart Layout ---
                    st.subheader(f"Capacity Loss Breakdown (vs {benchmark_title})")
                    
                    c1, c2 = st.columns([1, 1])

                    with c1:
                        st.markdown("<h6 style='text-align: center;'>Overall Performance Breakdown</h6>", unsafe_allow_html=True)
                        
                        waterfall_x = [f"<b>Optimal Output (100%)</b>", "Loss (RR Downtime)"]
                        waterfall_y = [all_time_totals['total_optimal_100'], -all_time_totals['total_downtime_loss_parts']]
                        waterfall_measure = ["absolute", "relative"]
                        waterfall_text = [f"{all_time_totals['total_optimal_100']:,.0f}", f"{-all_time_totals['total_downtime_loss_parts']:,.0f}"]

                        if all_time_totals['total_net_cycle_loss_parts'] >= 0:
                            waterfall_x.append("Net Loss (Cycle Time)")
                            waterfall_y.append(-all_time_totals['total_net_cycle_loss_parts'])
                            waterfall_measure.append("relative")
                            waterfall_text.append(f"{-all_time_totals['total_net_cycle_loss_parts']:,.0f}")
                        else:
                            waterfall_x.append("Net Gain (Cycle Time)")
                            waterfall_y.append(abs(all_time_totals['total_net_cycle_loss_parts']))
                            waterfall_measure.append("relative")
                            waterfall_text.append(f"{abs(all_time_totals['total_net_cycle_loss_parts']):+,.0f}")
                        
                        waterfall_x.append("<b>Actual Output</b>")
                        waterfall_y.append(all_time_totals['total_produced'])
                        waterfall_measure.append("total")
                        waterfall_text.append(f"{all_time_totals['total_produced']:,.0f}")
                        
                        fig_waterfall = go.Figure(go.Waterfall(
                            name = "Breakdown", orientation = "v", measure = waterfall_measure,
                            x = waterfall_x, y = waterfall_y, text = waterfall_text,
                            textposition = "outside", connector = {"line":{"color":"rgb(63, 63, 63)"}},
                            increasing = {"marker":{"color":"#2ca02c"}}, decreasing = {"marker":{"color":"#ff6961"}},
                            totals = {"marker":{"color":"#1f77b4"}}
                        ))
                        
                        fig_waterfall.update_layout(
                            showlegend=False, margin=dict(t=0, b=0, l=0, r=0),
                            height=400, yaxis_title='Parts'
                        )
                        
                        if benchmark_view == "Target Output":
                            fig_waterfall.add_shape(
                                type='line', x0=-0.5, x1=len(waterfall_x)-0.5,
                                y0=all_time_totals['total_target'], y1=all_time_totals['total_target'],
                                line=dict(color='deepskyblue', dash='dash', width=2)
                            )
                            fig_waterfall.add_annotation(
                                x=0, y=all_time_totals['total_target'], text=f"Target: {all_time_totals['total_target']:,.0f}",
                                showarrow=True, arrowhead=1, ax=-40, ay=-20
                            )
                            fig_waterfall.add_annotation(
                                x=len(waterfall_x)-0.5, y=all_time_totals['total_optimal_100'],
                                text=f"Optimal (100%): {all_time_totals['total_optimal_100']:,.0f}",
                                showarrow=True, arrowhead=1, ax=40, ay=-20
                            )
                        
                        st.plotly_chart(fig_waterfall, use_container_width=True, config={'displayModeBar': False})
                        
                    with c2:
                        def get_color_css(val):
                            if val > 0: return "color: red;"
                            if val < 0: return "color: green;"
                            return "color: black;"

                        net_loss_val = all_time_totals['total_calculated_net_loss_parts']
                        net_loss_color = get_color_css(net_loss_val)
                        with st.container(border=True):
                            st.markdown(f"**Total Net Impact**")
                            st.markdown(f"<h3><span style='{net_loss_color}'>{net_loss_val:,.0f} parts</span></h3>", unsafe_allow_html=True)
                            st.caption(f"Net Time Lost: {cr_utils.format_seconds_to_dhm(all_time_totals['total_calculated_net_loss_sec'])}")
                        
                        table_data = {
                            "Metric": [
                                "Loss (RR Downtime)", 
                                "Net Loss (Cycle Time)", 
                                "\u00A0\u00A0\u00A0 └ Loss (Slow Cycles)", 
                                "\u00A0\u00A0\u00A0 └ Gain (Fast Cycles)"
                            ],
                            "Parts": [
                                all_time_totals['total_downtime_loss_parts'],
                                all_time_totals['total_net_cycle_loss_parts'],
                                all_time_totals['total_slow_loss_parts'],
                                all_time_totals['total_fast_gain_parts']
                            ],
                            "Time": [
                                cr_utils.format_seconds_to_dhm(all_time_totals['total_downtime_loss_sec']),
                                cr_utils.format_seconds_to_dhm(all_time_totals['total_net_cycle_loss_sec']),
                                cr_utils.format_seconds_to_dhm(all_time_totals['total_slow_loss_sec']),
                                cr_utils.format_seconds_to_dhm(all_time_totals['total_fast_gain_sec'])
                            ]
                        }
                        df_table = pd.DataFrame(table_data)

                        def style_parts_col(val, row_index):
                            if row_index == 0: color_style = get_color_css(val)
                            elif row_index == 1: color_style = get_color_css(val)
                            elif row_index == 2: color_style = get_color_css(val)
                            elif row_index == 3: color_style = get_color_css(val * -1)
                            else: color_style = "color: black;"
                            return color_style

                        styled_df = df_table.style.apply(
                            lambda row: [style_parts_col(row['Parts'], row.name) if col == 'Parts' else '' for col in row.index],
                            axis=1
                        ).format(
                            {"Parts": "{:,.0f}"}
                        ).set_properties(
                            **{'text-align': 'left'}, subset=['Metric', 'Time']
                        ).set_properties(
                            **{'text-align': 'right'}, subset=['Parts']
                        ).hide(axis='index')
                        
                        st.dataframe(styled_df, use_container_width=True)


                    # --- Collapsible Daily Summary Table ---
                    if analysis_period_selector == "Entire File" or analysis_period_selector == "Custom Period":
                        with st.expander("View Daily Summary Data"):
                            
                            daily_summary_df['Actual Cycle Time Total (time %)'] = np.where( daily_summary_df['Filtered Run Time (sec)'] > 0, daily_summary_df['Actual Cycle Time Total (sec)'] / daily_summary_df['Filtered Run Time (sec)'], 0 )
                            daily_summary_df['Actual Output (parts %)'] = np.where( results_df['Optimal Output (parts)'] > 0, daily_summary_df['Actual Output (parts)'] / results_df['Optimal Output (parts)'], 0 )
                            perc_base_parts = daily_summary_df['Optimal Output (parts)']
                            perc_base_sec = daily_summary_df['Filtered Run Time (sec)']
                            daily_summary_df['Total Capacity Loss (time %)'] = np.where( perc_base_sec > 0, daily_summary_df['Total Capacity Loss (sec)'] / perc_base_sec, 0 )
                            daily_summary_df['Total Capacity Loss (parts %)'] = np.where( perc_base_parts > 0, daily_summary_df['Total Capacity Loss (parts)'] / perc_base_parts, 0 )
                            daily_summary_df['Total Capacity Loss (d/h/m)'] = daily_summary_df['Total Capacity Loss (sec)'].apply(cr_utils.format_seconds_to_dhm)
                            daily_summary_df['Capacity Loss (vs Target) (parts %)'] = np.where( daily_summary_df['Target Output (parts)'] > 0, daily_summary_df['Capacity Loss (vs Target) (parts)'] / daily_summary_df['Target Output (parts)'], 0 )
                            daily_summary_df['Capacity Loss (vs Target) (time %)'] = np.where( daily_summary_df['Filtered Run Time (sec)'] > 0, daily_summary_df['Capacity Loss (vs Target) (sec)'] / daily_summary_df['Filtered Run Time (sec)'], 0 )
                            daily_summary_df['Capacity Loss (vs Target) (d/h/m)'] = daily_summary_df['Capacity Loss (vs Target) (sec)'].apply(cr_utils.format_seconds_to_dhm)
                            daily_summary_df['Filtered Run Time (d/h/m)'] = daily_summary_df['Filtered Run Time (sec)'].apply(cr_utils.format_seconds_to_dhm)
                            daily_summary_df['Actual Cycle Time Total (d/h/m)'] = daily_summary_df['Actual Cycle Time Total (sec)'].apply(cr_utils.format_seconds_to_dhm)

                            daily_kpi_table = pd.DataFrame(index=daily_summary_df.index)
                            daily_kpi_table[run_time_label] = daily_summary_df.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']} ({r['Filtered Run Time (sec)']:,.0f}s)", axis=1)
                            daily_kpi_table['Actual Production Time'] = daily_summary_df.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']} ({r['Actual Cycle Time Total (time %)']:.1%})", axis=1)
                            daily_kpi_table['Actual Output (parts)'] = daily_summary_df.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (parts %)']:.1%})", axis=1)

                            if benchmark_view == "Optimal Output":
                                daily_kpi_table['Total Capacity Loss (Time)'] = daily_summary_df.apply(lambda r: f"{r['Total Capacity Loss (d/h/m)']} ({r['Total Capacity Loss (time %)']:.1%})", axis=1)
                                daily_kpi_table['Total Capacity Loss (parts)'] = daily_summary_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                                st.dataframe(daily_kpi_table, use_container_width=True)
                            else: # Target Output
                                daily_summary_df['Gap to Target (parts)'] = pd.to_numeric(daily_summary_df['Gap to Target (parts)'], errors='coerce').fillna(0)
                                daily_kpi_table['Gap to Target (parts)'] = daily_summary_df['Gap to Target (parts)'].apply(lambda x: "{:+,.2f}".format(x) if pd.notna(x) else "N/A")
                                daily_kpi_table['Capacity Loss (vs Target) (Time)'] = daily_summary_df.apply(lambda r: f"{r['Capacity Loss (vs Target) (d/h/m)']} ({r['Capacity Loss (vs Target) (time %)']:.1%})", axis=1)
                                st.dataframe(daily_kpi_table.style.applymap(
                                    lambda x: 'color: green' if str(x).startswith('+') else 'color: red' if str(x).startswith('-') else None,
                                    subset=['Gap to Target (parts)']
                                ), use_container_width=True)

                    st.divider()

                    # --- 3. AGGREGATED REPORT (Chart & Table) ---
                    
                    if display_frequency == 'by Run':
                        agg_df = run_summary_df.copy()
                        chart_title_prefix = "Run-by-Run"
                    elif display_frequency == 'Weekly':
                        agg_df = daily_summary_df.resample('W').sum().replace([np.inf, -np.inf], np.nan).fillna(0)
                        chart_title_prefix = "Weekly"
                    elif display_frequency == 'Monthly':
                        agg_df = daily_summary_df.resample('ME').sum().replace([np.inf, -np.inf], np.nan).fillna(0)
                        chart_title_prefix = "Monthly"
                    else: # Daily
                        agg_df = daily_summary_df.copy()
                        chart_title_prefix = "Daily"
                    
                    display_df = agg_df
                    
                    if display_df.empty:
                        st.warning(f"No data to display for the '{display_frequency}' frequency.")
                    else:
                        display_df['Capacity Loss (vs Target) (parts)'] = np.maximum(0, -display_df['Gap to Target (parts)'])
                        perc_base_parts = display_df['Optimal Output (parts)']
                        chart_title = f"{chart_title_prefix} Capacity Report (vs Optimal)"
                        optimal_100_base = display_df['Optimal Output (parts)']
                        display_df['Actual Output (%)'] = np.where( optimal_100_base > 0, display_df['Actual Output (parts)'] / optimal_100_base, 0)
                        display_df['Production Shots (%)'] = np.where( display_df['Total Shots (all)'] > 0, display_df['Production Shots'] / display_df['Total Shots (all)'], 0)
                        display_df['Actual Cycle Time Total (time %)'] = np.where( display_df['Filtered Run Time (sec)'] > 0, display_df['Actual Cycle Time Total (sec)'] / display_df['Filtered Run Time (sec)'], 0)
                        display_df['Capacity Loss (downtime) (parts %)'] = np.where( perc_base_parts > 0, display_df['Capacity Loss (downtime) (parts)'] / perc_base_parts, 0)
                        display_df['Capacity Loss (slow cycle time) (parts %)'] = np.where( perc_base_parts > 0, display_df['Capacity Loss (slow cycle time) (parts)'] / perc_base_parts, 0)
                        display_df['Capacity Gain (fast cycle time) (parts %)'] = np.where( perc_base_parts > 0, display_df['Capacity Gain (fast cycle time) (parts)'] / perc_base_parts, 0)
                        display_df['Total Capacity Loss (parts %)'] = np.where( perc_base_parts > 0, display_df['Total Capacity Loss (parts)'] / perc_base_parts, 0)
                        display_df['Capacity Loss (vs Target) (parts %)'] = np.where( display_df['Target Output (parts)'] > 0, display_df['Capacity Loss (vs Target) (parts)'] / display_df['Target Output (parts)'], 0)
                        display_df['Total Capacity Loss (cycle time) (parts)'] = display_df['Capacity Loss (slow cycle time) (parts)'] - display_df['Capacity Gain (fast cycle time) (parts)']
                        display_df['(Ref) Net Loss (RR)'] = display_df['Capacity Loss (downtime) (parts)']
                        display_df['(Ref) Net Loss (Slow)'] = display_df['Capacity Loss (slow cycle time) (parts)']
                        display_df['(Ref) Net Gain (Fast)'] = display_df['Capacity Gain (fast cycle time) (parts)']
                        display_df['(Ref) Total Net Loss'] = display_df['(Ref) Net Loss (RR)'] + display_df['(Ref) Net Loss (Slow)'] - display_df['(Ref) Net Gain (Fast)']
                        display_df['loss_downtime_ratio'] = np.where(display_df['(Ref) Total Net Loss'] != 0, display_df['(Ref) Net Loss (RR)'] / display_df['(Ref) Total Net Loss'], 0)
                        display_df['loss_slow_ratio'] = np.where(display_df['(Ref) Total Net Loss'] != 0, display_df['(Ref) Net Loss (Slow)'] / display_df['(Ref) Total Net Loss'], 0)
                        display_df['gain_fast_ratio'] = np.where(display_df['(Ref) Total Net Loss'] != 0, -display_df['(Ref) Net Gain (Fast)'] / display_df['(Ref) Total Net Loss'], 0)
                        display_df['Allocated Loss (RR Downtime)'] = display_df['Capacity Loss (vs Target) (parts)'] * display_df['loss_downtime_ratio']
                        display_df['Allocated Loss (Slow Cycles)'] = display_df['Capacity Loss (vs Target) (parts)'] * display_df['loss_slow_ratio']
                        display_df['Allocated Gain (Fast Cycles)'] = display_df['Capacity Loss (vs Target) (parts)'] * display_df['gain_fast_ratio']
                        display_df['Filtered Run Time (d/h/m)'] = display_df['Filtered Run Time (sec)'].apply(cr_utils.format_seconds_to_dhm)
                        display_df['Actual Cycle Time Total (d/h/m)'] = display_df['Actual Cycle Time Total (sec)'].apply(cr_utils.format_seconds_to_dhm)
                        
                        if 'Start Time' in display_df.columns:
                            display_df['Start Time_str'] = pd.to_datetime(display_df['Start Time']).dt.strftime('%Y-%m-%d %H:%M')
                            
                        if display_frequency == 'Weekly': xaxis_title = "Week"
                        elif display_frequency == 'Monthly': xaxis_title = "Month"
                        elif display_frequency == 'by Run': xaxis_title = "Run ID"
                        else: xaxis_title = "Date"
                        
                        if display_frequency == 'by Run':
                            chart_df = display_df.reset_index().rename(columns={'run_id': 'X-Axis'})
                            # Add 1 to run_id for display
                            chart_df['X-Axis'] = 'Run ' + (chart_df['X-Axis'] + 1).astype(str)
                        else:
                            chart_df = display_df.reset_index().rename(columns={'Date': 'X-Axis'})
                        

                        # --- Unified Performance Breakdown Chart ---
                        st.header(f"{display_frequency} Performance Breakdown (vs {benchmark_title})")
                        fig_ts = go.Figure()
                        
                        fig_ts.add_trace(go.Bar(
                            x=chart_df['X-Axis'], y=chart_df['Actual Output (parts)'], name='Actual Output',
                            marker_color='#3498DB', customdata=chart_df['Actual Output (%)'],
                            hovertemplate='Actual Output: %{y:,.0f} (%{customdata:.1%})<extra></extra>'
                        ))
                        
                        chart_df['Net Cycle Time Loss (parts)'] = chart_df['Total Capacity Loss (cycle time) (parts)']
                        chart_df['Net Cycle Time Loss (positive)'] = np.maximum(0, chart_df['Net Cycle Time Loss (parts)'])

                        fig_ts.add_trace(go.Bar(
                            x=chart_df['X-Axis'], y=chart_df['Net Cycle Time Loss (positive)'], name='Capacity Loss (cycle time)',
                            marker_color='#ffb347',
                            customdata=np.stack((
                                chart_df['Net Cycle Time Loss (parts)'],
                                chart_df['Capacity Loss (slow cycle time) (parts)'],
                                chart_df['Capacity Gain (fast cycle time) (parts)']
                            ), axis=-1),
                            # --- v8.6: Fixed SyntaxError by removing stray underscore ---
                            hovertemplate=
                                '<b>Net Cycle Time Loss: %{customdata[0]:,.0f}</b><br>' +
                                'Slow Cycle Loss: %{customdata[1]:,.0f}<br>' +
                                'Fast Cycle Gain: -%{customdata[2]:,.0f}<br>' +
                                '<extra></extra>'
                        ))
                        
                        fig_ts.add_trace(go.Bar(
                            x=chart_df['X-Axis'], y=chart_df['Capacity Loss (downtime) (parts)'], name='Run Rate Downtime (Stops)',
                            marker_color='#808080', customdata=chart_df['Capacity Loss (downtime) (parts %)'],
                            hovertemplate='Run Rate Downtime (Stops): %{y:,.0f} (%{customdata:.1%})<extra></extra>'
                        ))
                        
                        fig_ts.update_layout(barmode='stack')

                        if benchmark_view == "Target Output":
                            fig_ts.add_trace(go.Scatter(
                                x=chart_df['X-Axis'], y=chart_df['Target Output (parts)'],
                                name=f'Target Output ({target_output_perc:.0f}%)', mode='lines',
                                line=dict(color='deepskyblue', dash='dash'),
                                hovertemplate=f'<b>Target Output ({target_output_perc:.0f}%)</b>: %{{y:,.0f}}<extra></extra>'
                            ))
                            
                        fig_ts.add_trace(go.Scatter(
                            x=chart_df['X-Axis'], y=chart_df['Optimal Output (parts)'],
                            name='Optimal Output (100%)',
                            mode='lines',
                            line=dict(color='darkblue', dash='dot'),
                            hovertemplate='Optimal Output (100%): %{y:,.0f}<extra></extra>'
                        ))

                        fig_ts.update_layout(
                            title=chart_title, xaxis_title=xaxis_title, yaxis_title='Parts (Output & Loss)',
                            legend_title='Metric', hovermode="x unified"
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)

                        # --- Full Data Table ---
                        display_df_totals = display_df
                        
                        st.header(f"Production Totals Report ({display_frequency})")
                        if display_frequency == 'by Run':
                            report_table_1_df = display_df_totals.reset_index().rename(columns={'run_id': 'Run ID'})
                            # Add 1 to run_id for display
                            report_table_1_df['Run ID'] = report_table_1_df['Run ID'] + 1
                            
                            # --- v8.3 FIX: Use the correct bottom-up downtime sum ---
                            report_table_1_df['Total Downtime (sec)'] = report_table_1_df['Capacity Loss (downtime) (sec)']
                            # --- End v8.3 Fix ---

                            report_table_1_df['Total Downtime (d/h/m)'] = report_table_1_df['Total Downtime (sec)'].apply(format_seconds_to_dhm)
                            report_table_1 = pd.DataFrame(index=report_table_1_df.index)
                            report_table_1['Run ID'] = report_table_1_df['Run ID']
                            report_table_1['Start Time'] = report_table_1_df['Start Time_str']
                            report_table_1['Overall Run Time'] = report_table_1_df.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']}", axis=1)
                            report_table_1['Actual Production Time'] = report_table_1_df.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']}", axis=1)
                            report_table_1['Total Downtime'] = report_table_1_df.apply(lambda r: f"{r['Total Downtime (d/h/m)']}", axis=1)
                            report_table_1['Total Shots'] = report_table_1_df['Total Shots (all)'].map('{:,.0f}'.format)
                            report_table_1['Production Shots'] = report_table_1_df['Production Shots'].map('{:,.0f}'.format)
                            report_table_1['Downtime Shots'] = report_table_1_df['Downtime Shots'].map('{:,.0f}'.format)
                            report_table_1['Mode CT'] = report_table_1_df['Mode CT'].map('{:.2f}s'.format)
                        else: # Daily, Weekly, Monthly
                            report_table_1 = pd.DataFrame(index=display_df_totals.index)
                            report_table_1_df = display_df_totals
                            report_table_1['Total Shots (all)'] = report_table_1_df['Total Shots (all)'].map('{:,.0f}'.format)
                            report_table_1['Production Shots'] = report_table_1_df.apply(lambda r: f"{r['Production Shots']:,.0f} ({r['Production Shots (%)']:.1%})", axis=1)
                            report_table_1['Downtime Shots'] = report_table_1_df['Downtime Shots'].map('{:,.0f}'.format)
                            report_table_1[run_time_label] = report_table_1_df.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']} ({r['Filtered Run Time (sec)']:,.0f}s)", axis=1)
                            report_table_1['Actual Production Time'] = report_table_1_df.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']} ({r['Actual Cycle Time Total (time %)']:.1%})", axis=1)

                        st.dataframe(report_table_1, use_container_width=True)

                        # --- Conditional Tables ---
                        st.header(f"Capacity Loss & Gain Report (vs Optimal) ({display_frequency})")
                        
                        display_df_optimal = display_df

                        if display_frequency == 'by Run':
                            report_table_optimal_df = display_df_optimal.reset_index().rename(columns={'run_id': 'Run ID'})
                            # Add 1 to run_id for display
                            report_table_optimal_df['Run ID'] = report_table_optimal_df['Run ID'] + 1
                            report_table_optimal = pd.DataFrame(index=report_table_optimal_df.index)
                            report_table_optimal['Run ID'] = report_table_optimal_df['Run ID']
                        else:
                            report_table_optimal = pd.DataFrame(index=display_df_optimal.index)
                            report_table_optimal_df = display_df_optimal

                        report_table_optimal['Optimal Output (parts)'] = report_table_optimal_df['Optimal Output (parts)'].map('{:,.2f}'.format)
                        report_table_optimal['Actual Output (parts)'] = report_table_optimal_df.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (%)']:.1%})", axis=1)
                        report_table_optimal['Loss (RR Downtime)'] = report_table_optimal_df.apply(lambda r: f"{r['Capacity Loss (downtime) (parts)']:,.2f} ({r['Capacity Loss (downtime) (parts %)']:.1%})", axis=1)
                        report_table_optimal['Loss (Slow Cycles)'] = report_table_optimal_df.apply(lambda r: f"{r['Capacity Loss (slow cycle time) (parts)']:,.2f} ({r['Capacity Loss (slow cycle time) (parts %)']:.1%})", axis=1)
                        report_table_optimal['Gain (Fast Cycles)'] = report_table_optimal_df.apply(lambda r: f"{r['Capacity Gain (fast cycle time) (parts)']:,.2f} ({r['Capacity Gain (fast cycle time) (parts %)']:.1%})", axis=1)
                        report_table_optimal['Total Net Loss'] = report_table_optimal_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                        
                        def style_loss_gain_table(col):
                            col_name = col.name
                            if col_name == 'Loss (RR Downtime)': return ['color: red'] * len(col)
                            if col_name == 'Loss (Slow Cycles)': return ['color: red'] * len(col)
                            if col_name == 'Gain (Fast Cycles)': return ['color: green'] * len(col)
                            if col_name == 'Total Net Loss':
                                return ['color: green' if v < 0 else 'color: red' for v in display_df_optimal['Total Capacity Loss (parts)']]
                            return [''] * len(col)

                        st.dataframe(
                            report_table_optimal.style.apply(style_loss_gain_table, axis=0),
                            use_container_width=True
                        )
                        
                        if benchmark_view == "Target Output": 
                            st.header(f"Target Report (90%) ({display_frequency})")
                            # ... (all your report_table_target logic) ...
                            st.dataframe(
                                report_table_target.style.apply(style_target_report_table, axis=0),
                                use_container_width=True
                            )

                    # --- 4. SHOT-BY-SHOT ANALYSIS ---
                    st.divider()
                    st.header("Shot-by-Shot Analysis (All Shots)")
                    # ... (all your shot-by-shot logic) ...
                    if all_shots_df.empty:
                        st.warning("No shots were found in the file to analyze.")
                    else:
                        # ... (all your shot-by-shot selectbox, slider, and chart logic) ...
                        st.plotly_chart(fig_ct, use_container_width=True)
                        # ... (all your shot-by-shot dataframe logic) ...
                        st.dataframe(
                            df_to_display[[
                                'SHOT TIME', 'Actual CT', 'Approved CT',
                                'Working Cavities', 'run_id', 'mode_ct', 
                                'Shot Type', 'stop_flag',
                                'rr_time_diff', 'adj_ct_sec',
                                'reference_ct', 'Mode CT Lower', 'Mode CT Upper'
                            ]].style.format({
                                'Actual CT': '{:.2f}',
                                # ... (rest of formatting) ...
                                'SHOT TIME': lambda t: t.strftime('%Y-%m-%d %H:%M:%S') if selected_date == "All Dates" else t.strftime('%H:%M:%S')
                            }),
                            use_container_width=True
                        )

    else:
        st.info("👈 Please upload a data file to begin.")

# This allows the file to be run standalone if needed
if __name__ == "__main__":
    # --- Page Config (Copied from combined_report.py) ---
    st.set_page_config(
        page_title=f"Capacity Risk Calculator (Standalone)",
        layout="wide"
    )
    
    # This is the main function we defined above
    run_capacity_risk_ui()