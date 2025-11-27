import pandas as pd

# This file mocks the utility functions required by the main app.
# Replace the contents of this file with your actual run_rate_utils.py code.

def load_all_data(file_list):
    # Just read the first file
    if not file_list: return pd.DataFrame()
    f = file_list[0]
    if f.name.endswith('.csv'):
        df = pd.read_csv(f)
    else:
        df = pd.read_excel(f)
    
    # Ensure shot_time exists
    if 'shot_time' not in df.columns:
        df['shot_time'] = pd.to_datetime('today')
        
    return df

class RunRateCalculator:
    def __init__(self, df, tolerance, stop_gap, analysis_mode):
        self.results = {
            'mttr_min': 15,
            'mtbf_min': 120,
            'stability_index': 85.5,
            'efficiency': 0.92
        }