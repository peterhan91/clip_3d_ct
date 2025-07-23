import pandas as pd
import argparse
import re
from pathlib import Path

def clean_impression_text(impression_text):
    """
    Clean impression text by removing unwanted prefixes and sections.
    Adapted from the original preprocessing code for impression cleaning.
    """
    if pd.isna(impression_text) or impression_text.strip() == '':
        return impression_text
    
    # Convert to string and split into words
    text = str(impression_text).strip()
    words = text.split()
    
    # Remove "Impression:" prefix if present (case insensitive)
    if len(words) > 0 and words[0].lower() in ['impression:', 'impression']:
        words = words[1:]
    
    # Find start of cleaned impression
    begin = 0
    
    # Find end by looking for recommendation or notification sections
    end = None
    end_cand1 = None
    end_cand2 = None
    
    # Look for recommendation sections (case insensitive)
    for i, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in ["recommendation(s):", "recommendation:", "recommendations:"]:
            end_cand1 = i
            break
    
    # Look for notification sections (case insensitive)
    for i, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in ["notification:", "notifications:"]:
            end_cand2 = i
            break
    
    # Determine the actual end point
    if end_cand1 is not None and end_cand2 is not None:
        end = min(end_cand1, end_cand2)
    elif end_cand1 is not None:
        end = end_cand1
    elif end_cand2 is not None:
        end = end_cand2
    
    # Extract the cleaned impression
    if end is None:
        cleaned_words = words[begin:]
    else:
        cleaned_words = words[begin:end]
    
    # Join back to text and clean up whitespace
    cleaned_text = " ".join(cleaned_words).strip()
    
    # Remove multiple spaces and normalize whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text

def clean_reports_csv(input_csv_path, output_csv_path=None, backup=True):
    """
    Clean impression texts in a reports CSV file.
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path for output CSV (default: overwrites input)
        backup: Whether to create backup of original file
    """
    # Read the CSV file
    print(f"Loading CSV file: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # Create backup if requested
    if backup:
        backup_path = str(input_csv_path).replace('.csv', '_backup.csv')
        df.to_csv(backup_path, index=False)
        print(f"Backup created: {backup_path}")
    
    # Check if Impressions_EN column exists
    if 'Impressions_EN' not in df.columns:
        print("Warning: 'Impressions_EN' column not found in the CSV file")
        return
    
    # Count impressions that need cleaning
    needs_cleaning = 0
    for idx, impression in df['Impressions_EN'].items():
        if pd.notna(impression) and str(impression).strip().lower().startswith('impression'):
            needs_cleaning += 1
    
    print(f"Total rows: {len(df)}")
    print(f"Rows with 'Impression:' prefix: {needs_cleaning}")
    
    # Clean impressions
    print("Cleaning impression texts...")
    cleaned_impressions = []
    
    for idx, impression in df['Impressions_EN'].items():
        cleaned = clean_impression_text(impression)
        cleaned_impressions.append(cleaned)
    
    # Update the dataframe
    df['Impressions_EN'] = cleaned_impressions
    
    # Determine output path
    if output_csv_path is None:
        output_csv_path = input_csv_path
    
    # Save cleaned CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Cleaned CSV saved to: {output_csv_path}")
    
    # Show some examples of cleaning
    print("\nSample of cleaned impressions:")
    for idx in range(min(5, len(df))):
        original = str(df.iloc[idx]['Impressions_EN'])
        if len(original) > 100:
            print(f"Row {idx}: {original[:100]}...")
        else:
            print(f"Row {idx}: {original}")
    
    print(f"\nSuccessfully cleaned {needs_cleaning} impressions with prefixes")

def main():
    parser = argparse.ArgumentParser(description='Clean impression texts in CT reports CSV')
    parser.add_argument('--input_csv', type=str, required=True, 
                        help='Path to input CSV file with reports')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path for output CSV (default: overwrites input)')
    parser.add_argument('--no_backup', action='store_true',
                        help='Skip creating backup of original file')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_csv).exists():
        print(f"Error: Input file not found: {args.input_csv}")
        return
    
    # Clean the reports
    clean_reports_csv(
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        backup=not args.no_backup
    )
    
    print("Text cleaning completed successfully!")

if __name__ == "__main__":
    main()