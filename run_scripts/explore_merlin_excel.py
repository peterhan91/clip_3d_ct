#!/usr/bin/env python3
"""
Explore Merlin dataset Excel file to understand structure.
"""

import pandas as pd
import os

def explore_merlin():
    excel_path = "/cbica/projects/CXR/data/Merlin/merlinabdominalctdataset/reports_final.xlsx"
    
    try:
        # Try to read Excel file
        print("Reading Excel sheets...")
        excel_file = pd.ExcelFile(excel_path)
        print(f"Available sheets: {excel_file.sheet_names}")
        
        # Read the first sheet to understand structure
        df = pd.read_excel(excel_path, sheet_name=0)
        print(f"\nFirst sheet shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Check if there are file ID columns that might map to CT files
        print(f"\nSample values from first few columns:")
        for col in df.columns[:5]:
            print(f"{col}: {df[col].head(3).tolist()}")
        
        return df
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

if __name__ == "__main__":
    explore_merlin()