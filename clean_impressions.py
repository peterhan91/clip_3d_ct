#!/usr/bin/env python3
"""
Clean up impression section in extracted reports CSV by removing numbered list formatting.
"""

import pandas as pd
import re
import argparse
from pathlib import Path


def clean_numbered_list(text):
    """
    Remove numbered list formatting from text and combine into single paragraph.
    Handles patterns like:
    - "1. ", "2. ", etc. at the beginning of lines
    - Also handles cases with extra spaces after the number
    - Removes extra spacing and newlines to create one paragraph
    """
    if pd.isna(text) or text == "Not given":
        return text
    
    # Pattern to match numbered lists at the beginning of lines
    # This handles: "1. ", "2. ", "10. ", etc.
    pattern = r'^(\d+)\.\s+|(?:\n)(\d+)\.\s+'
    
    # Replace the pattern with a space
    cleaned = re.sub(pattern, ' ', text)
    
    # Replace multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Replace newlines with spaces to create single paragraph
    cleaned = re.sub(r'\n+', ' ', cleaned)
    
    # Clean up any extra whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Clean numbered lists from impression column in CSV")
    parser.add_argument("--input", type=str, default="data/merlin/extracted_reports.csv",
                        help="Input CSV file path")
    parser.add_argument("--output", type=str, default="data/merlin/extracted_reports_cleaned.csv",
                        help="Output CSV file path")
    parser.add_argument("--preview", type=int, default=5,
                        help="Number of samples to preview (0 to skip preview)")
    
    args = parser.parse_args()
    
    # Read the CSV file
    print(f"Reading from: {args.input}")
    df = pd.read_csv(args.input)
    
    # Check if Impressions_EN column exists
    if 'Impressions_EN' not in df.columns:
        raise ValueError("Column 'Impressions_EN' not found in the CSV file")
    
    # Show some examples before cleaning
    if args.preview > 0:
        print("\n--- Preview of impressions before cleaning ---")
        sample_indices = df[df['Impressions_EN'] != 'Not given'].head(args.preview).index
        for idx in sample_indices:
            print(f"\nVolume: {df.loc[idx, 'VolumeName']}")
            print(f"Original impression:\n{df.loc[idx, 'Impressions_EN']}")
            print("-" * 50)
    
    # Apply cleaning to the Impressions_EN column
    print("\nCleaning impressions...")
    df['Impressions_EN'] = df['Impressions_EN'].apply(clean_numbered_list)
    
    # Show the same examples after cleaning
    if args.preview > 0:
        print("\n--- Preview of impressions after cleaning ---")
        for idx in sample_indices:
            print(f"\nVolume: {df.loc[idx, 'VolumeName']}")
            print(f"Cleaned impression:\n{df.loc[idx, 'Impressions_EN']}")
            print("-" * 50)
    
    # Save the cleaned data
    print(f"\nSaving cleaned data to: {args.output}")
    df.to_csv(args.output, index=False)
    
    # Print statistics
    total_rows = len(df)
    not_given = (df['Impressions_EN'] == 'Not given').sum()
    has_impressions = total_rows - not_given
    
    print(f"\nStatistics:")
    print(f"Total rows: {total_rows:,}")
    print(f"Rows with impressions: {has_impressions:,}")
    print(f"Rows with 'Not given': {not_given:,}")
    print(f"\nCleaning completed successfully!")


if __name__ == "__main__":
    main()