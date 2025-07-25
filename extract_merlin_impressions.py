import pandas as pd
import re
from tqdm import tqdm
import argparse

def extract_findings_and_impression(text):
    """Extract findings and impression sections from combined text."""
    # Only look for explicit IMPRESSION: or Impression: markers
    match = re.search(r'(.*?)(IMPRESSION:|Impression:)\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    
    if match:
        findings = match.group(1).strip()
        impression = match.group(3).strip()
        
        # Clean up findings - remove "FINDINGS:" prefix if present
        if findings.startswith("FINDINGS:"):
            findings = findings[9:].strip()
        
        # Clean up impression - remove trailing notes
        impression = re.sub(r'\s*There are no substantial differences.*$', '', impression, flags=re.DOTALL).strip()
        impression = re.sub(r'\s*Results discussed with.*$', '', impression, flags=re.DOTALL).strip()
        
        return findings, impression
    else:
        # No IMPRESSION marker found - entire text is findings
        findings = text.strip()
        if findings.startswith("FINDINGS:"):
            findings = findings[9:].strip()
        return findings, None

def process_merlin_data(input_path, output_path):
    """Process Merlin data to extract findings and impressions into separate columns."""
    print(f"Reading Merlin data from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Total entries: {len(df)}")
    print("Extracting findings and impressions...")
    
    # Process each row
    extracted_data = []
    missing_impressions = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        findings, impression = extract_findings_and_impression(row['findings'])
        
        if impression is None or impression.strip() == '':
            missing_impressions += 1
            impression = "Not given"  # Match CT-RATE format for missing impressions
        
        extracted_data.append({
            'VolumeName': row['VolumeName'],
            'Findings_EN': findings,
            'Impressions_EN': impression
        })
    
    # Create new dataframe
    result_df = pd.DataFrame(extracted_data)
    
    # Save to CSV
    result_df.to_csv(output_path, index=False)
    
    print(f"\nProcessing complete!")
    print(f"Total entries: {len(df)}")
    print(f"Entries with impressions: {len(df) - missing_impressions} ({(len(df) - missing_impressions)/len(df)*100:.1f}%)")
    print(f"Entries missing impressions: {missing_impressions} ({missing_impressions/len(df)*100:.1f}%)")
    print(f"Output saved to: {output_path}")
    
    # Show sample of the output
    print("\nSample of extracted data:")
    print(result_df.head(3).to_string(max_colwidth=80))
    
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract findings and impressions from Merlin combined reports')
    parser.add_argument('--input', type=str, default='data/merlin/combined_reports.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, default='data/merlin/extracted_reports.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    process_merlin_data(args.input, args.output)