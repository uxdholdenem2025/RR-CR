import pandas as pd
import streamlit as st

# This file mocks the utility functions required by the main app.
# Replace the contents of this file with your actual cr_utils.py code.

def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

def get_preprocessed_data(df):
    # Mock preprocessing
    # Ensure a 'date' column exists for the main app filter
    if 'date' not in df.columns:
        # Try to find a date-like column or create a dummy one
        df['date'] = pd.to_datetime('today').date()
    else:
        df['date'] = pd.to_datetime(df['date']).dt.date
        
    min_date = df['date'].min()
    max_date = df['date'].max()
    return df, min_date, max_date

def run_capacity_calculation_cached_v2(df, is_group, shift, target, tolerance, stop_gap, run_int):
    # Mock calculation returning a dataframe with required columns
    # We create a dummy result so the main app doesn't crash on column access
    
    records = []
    # Create one dummy record per row to simulate calculation
    for i in range(min(len(df), 5)):
        records.append({
            'Optimal Output (parts)': 100,
            'Actual Output (parts)': 80,
            'Capacity Loss (downtime) (parts)': 10,
            'Capacity Loss (slow cycle time) (parts)': 15,
            'Capacity Gain (fast cycle time) (parts)': 5,
            'Total Capacity Loss (parts)': 20,
            'Total Capacity Loss (sec)': 7200
        })
    
    results_df = pd.DataFrame(records)
    summary_text = "Calculation Complete"
    return results_df, summary_text