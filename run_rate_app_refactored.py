import streamlit as st
import pandas as pd
import run_rate_utils

def run_run_rate_ui():
    """
    This function wraps the entire UI logic.
    It is only executed when called by main.py, preventing import errors.
    """
    st.title("Run Rate Analysis Details")
    st.markdown("Detailed breakdown of Run Rate stability and efficiency.")

    uploaded_file = st.file_uploader("Upload Run Rate Data (Individual)", type=["xlsx"])
    
    if uploaded_file:
        try:
            df = run_rate_utils.load_all_data([uploaded_file])
            st.success("Run Rate Data Loaded")
            st.dataframe(df.head())
            
            # Placeholder for complex logic found in your original file
            st.info("Logic from your original `run_rate_app_refactored.py` would go here.")
            
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Please upload a file to view details.")

if __name__ == "__main__":
    run_run_rate_ui()