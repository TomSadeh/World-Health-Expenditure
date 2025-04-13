import pandas as pd
from pathlib import Path

input_path = Path('./data')
output_path = Path('./data/processed')
def process_wb_data(input_file, output_file):
    """
    Process World Bank data by removing the first 3 rows.
    
    Args:
        input_file (str): Path to the original WB data file
        output_file (str): Path where the processed file will be saved
    """
    # Read the CSV file starting from row 5 (skipping first 4 rows)
    data = pd.read_csv(input_path / input_file, skiprows=3)
    
    # Write to the output file
    data.to_csv(output_file, index=False)
    
    print(f"Successfully processed {input_file} and saved to {output_file}")
    
    
if __name__ == "__main__":
    for file, new_file_name in zip(["API_NY.GDP.MKTP.CN_DS2_en_csv_v2_26332.csv", 
                 "API_NY.GDP.MKTP.KN_DS2_en_csv_v2_13325.csv",
                 "API_PA.NUS.PPP_DS2_en_csv_v2_13721.csv"],
                                   ['gdp_current.csv',
                                    'gdp_constant.csv',
                                    'ppp.csv']):
        process_wb_data(file, output_path / new_file_name)