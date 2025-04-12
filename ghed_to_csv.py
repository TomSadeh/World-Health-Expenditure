"""
Script to convert GHED Excel data to optimized CSV format.
This script extracts only the necessary columns from the GHED Excel file,
performs basic preprocessing, and saves the result as a CSV file for faster loading.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import os

# Define paths
current_path = Path(".")
data_path = current_path / "data"
output_path = current_path / "data" / "processed"

# Create output directory if it doesn't exist
output_path.mkdir(parents=True, exist_ok=True)

def convert_ghed_to_csv():
    """
    Convert GHED Excel file to optimized CSV format.
    Only extracts and processes the necessary columns.
    """
    excel_file = "GHED_data_2025.xlsx"
    csv_file = "ghed_data_optimized.csv"
    
    print(f"Starting conversion of {excel_file} to CSV format...")
    start_time = time.time()
    
    try:
        # Check if the file exists
        excel_path = data_path / excel_file
        if not excel_path.exists():
            print(f"Error: {excel_file} not found in {data_path}")
            return False
        
        # Get file size
        file_size_mb = os.path.getsize(excel_path) / (1024 * 1024)
        print(f"Excel file size: {file_size_mb:.2f} MB")
        
        # Read only the necessary columns from the Excel file
        print("Reading GHED Excel file...")
        required_cols = ['location', 'code', 'year', 'che']
        optional_cols = ['gghed_che', 'pvtd_che']
        
        # First check if all required columns exist
        # This is a quick check without loading the full dataset
        xl = pd.ExcelFile(excel_path)
        sheet_names = xl.sheet_names
        
        if "Data" not in sheet_names:
            print(f"Error: 'Data' sheet not found in {excel_file}")
            print(f"Available sheets: {sheet_names}")
            return False
            
        # Read the header row to check columns
        header_df = pd.read_excel(excel_path, sheet_name="Data", nrows=0)
        
        # Check for required columns
        missing_cols = [col for col in required_cols if col not in header_df.columns]
        if missing_cols:
            print(f"Error: Required columns missing: {missing_cols}")
            print(f"Available columns: {header_df.columns.tolist()}")
            return False
        
        # Identify which optional columns are available
        available_optional_cols = [col for col in optional_cols if col in header_df.columns]
        if set(optional_cols) != set(available_optional_cols):
            missing_optional = set(optional_cols) - set(available_optional_cols)
            print(f"Note: Some optional columns are missing: {missing_optional}")
        
        # Combine required and available optional columns
        columns_to_read = required_cols + available_optional_cols
        
        # Now read only the necessary columns
        print(f"Reading only these columns: {columns_to_read}")
        read_start = time.time()
        ghed_data = pd.read_excel(excel_path, sheet_name="Data", usecols=columns_to_read)
        read_end = time.time()
        print(f"Excel read completed in {read_end - read_start:.2f} seconds")
        
        # Basic preprocessing
        print("Preprocessing data...")
        
        # Check for missing values in required columns
        for col in required_cols:
            missing_count = ghed_data[col].isna().sum()
            missing_percent = (missing_count / len(ghed_data)) * 100
            print(f"Column '{col}' has {missing_count} missing values ({missing_percent:.2f}%)")
        
        # Remove rows with missing values in required columns
        original_rows = len(ghed_data)
        ghed_data = ghed_data.dropna(subset=required_cols)
        retained_rows = len(ghed_data)
        print(f"Retained {retained_rows} out of {original_rows} rows after removing missing values")
        
        # Ensure 'year' is an integer
        ghed_data['year'] = ghed_data['year'].astype(int)
        
        # Calculate public and private expenditure
        if 'gghed_che' in ghed_data.columns:
            ghed_data['public_expenditure'] = ghed_data['che'] * (ghed_data['gghed_che'] / 100)
            print("Added 'public_expenditure' column")
        
        if 'pvtd_che' in ghed_data.columns:
            ghed_data['private_expenditure'] = ghed_data['che'] * (ghed_data['pvtd_che'] / 100)
            print("Added 'private_expenditure' column")
        
        # Save to CSV
        csv_path = output_path / csv_file
        print(f"Saving to CSV: {csv_path}")
        ghed_data.to_csv(csv_path, index=False)
        
        # Check the resulting file size
        csv_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"CSV file size: {csv_size_mb:.2f} MB")
        print(f"Size reduction: {((file_size_mb - csv_size_mb) / file_size_mb) * 100:.2f}%")
        
        end_time = time.time()
        print(f"Conversion completed in {end_time - start_time:.2f} seconds")
        print(f"Optimized GHED data saved to {csv_path}")
        
        # Test loading the CSV file
        test_load_time(csv_path)
        
        return True
        
    except Exception as e:
        print(f"Error converting GHED data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_load_time(csv_path):
    """Test the loading time of the CSV file compared to Excel."""
    print("\nTesting load times...")
    
    # Test loading the original Excel file
    excel_path = data_path / "GHED_data_2025.xlsx"
    
    try:
        print("Loading original Excel file...")
        excel_start = time.time()
        excel_data = pd.read_excel(excel_path, sheet_name="Data")
        excel_end = time.time()
        excel_load_time = excel_end - excel_start
        print(f"Excel load time: {excel_load_time:.2f} seconds")
        print(f"Excel columns: {excel_data.shape[1]}")
        print(f"Excel rows: {excel_data.shape[0]}")
    except Exception as e:
        print(f"Error loading Excel: {e}")
    
    try:
        # Test loading the CSV file
        print("Loading CSV file...")
        csv_start = time.time()
        csv_data = pd.read_csv(csv_path)
        csv_end = time.time()
        csv_load_time = csv_end - csv_start
        print(f"CSV load time: {csv_load_time:.2f} seconds")
        print(f"CSV columns: {csv_data.shape[1]}")
        print(f"CSV rows: {csv_data.shape[0]}")
        
        # Calculate speedup
        if 'excel_load_time' in locals():
            speedup = excel_load_time / csv_load_time
            print(f"Loading speedup: {speedup:.2f}x faster")
    except Exception as e:
        print(f"Error loading CSV: {e}")

def main():
    """Main function"""
    print("GHED Excel to CSV Converter")
    print("===========================")
    
    success = convert_ghed_to_csv()
    
    if success:
        print("\nTo use the optimized CSV file in your main script:")
        print("1. Replace the load_ghed_data function with the following code:")
        
        replacement_code = """
def load_ghed_data():
    \"\"\"
    Load and process GHED data from the optimized CSV file.
    \"\"\"
    print("Loading GHED data from optimized CSV...")
    try:
        # Read the optimized CSV data
        ghed_data = pd.read_csv(data_path / "processed" / "ghed_data_optimized.csv")
        
        # Adjust the numbers to be actual counts and not in millions (if not already done)
        ghed_data['che'] *= 10**6
        
        if 'public_expenditure' in ghed_data.columns:
            ghed_data['public_expenditure'] *= 10**6
            
        if 'private_expenditure' in ghed_data.columns:
            ghed_data['private_expenditure'] *= 10**6
        
        # Ensure year is an integer
        ghed_data['year'] = ghed_data['year'].astype(int)
        
        # Rename the ISO3 column to maintain consistency
        ghed_data = ghed_data.rename(columns={'code': 'ISO3'})
        
        print(f"Loaded GHED data with shape: {ghed_data.shape}")
        return ghed_data
    
    except Exception as e:
        print(f"Error loading GHED data: {e}")
        import traceback
        traceback.print_exc()
        raise
"""
        print(replacement_code)
        
        print("\n2. Make sure the data/processed directory exists and contains the ghed_data_optimized.csv file")
    else:
        print("\nConversion failed. Please check the error messages above.")

if __name__ == "__main__":
    main()