import pandas as pd
from pathlib import Path

# Define paths
input_path = Path('./data')
output_path = Path('./data/processed')

# Create output directory if it doesn't exist
output_path.mkdir(parents=True, exist_ok=True)

def process_population_file(input_file, output_file):
    """
    Process World Population Prospects data file by reading from Excel and saving to CSV.
    
    Args:
        input_file (str): Path to the original WPP Excel file
        output_file (str): Path where the processed CSV file will be saved
    """
    print(f"Processing population data from {input_file}...")
    
    # Read the Excel file starting from row 17 (header is on row 17)
    # Use low_memory=False to avoid DtypeWarning for mixed types
    data = pd.read_excel(input_path / input_file, sheet_name="Estimates", header=16)
    
    # Convert age columns (which might have space-separated thousands)
    age_columns = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', 
                   '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', 
                   '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+']
    
    # Only process columns that exist in the dataframe
    age_columns = [col for col in age_columns if col in data.columns]
    
    for col in age_columns:
        # First convert to string to handle any mixed types
        data[col] = data[col].astype(str)
        # Then remove spaces and convert to numeric
        data[col] = pd.to_numeric(data[col].str.replace(' ', ''), errors='coerce')
    
    # Write to the output file
    data.to_csv(output_path / output_file, index=False)
    
    print(f"Successfully processed {input_file} and saved to {output_file}")

if __name__ == "__main__":
    # Process male and female population files
    files_to_process = [
        ("WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx", "male_pop.csv"),
        ("WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx", "female_pop.csv")
    ]
    
    for input_file, output_file in files_to_process:
        process_population_file(input_file, output_file)
