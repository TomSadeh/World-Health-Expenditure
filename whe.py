"""
Script for calculating Health Expenditure per Standardized Capita with PPP adjustment.
This script processes GHED data, World Population Prospect data, and World Bank PPP data
to calculate health expenditure per standardized capita for each country in each year,
using the Israeli Capitation Formula and adjusting for purchasing power parity.

Required files:
- GHED_data_2025.xlsx: Contains health expenditure data
- male_pop.csv: Contains male population data by age groups
- female_pop.csv: Contains female population data by age groups
- cap.csv: Contains the Israeli capitation formula weights by age group
- API_PA.NUS.PPP_DS2_en_csv_v2_13721.csv: World Bank PPP conversion factors
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define paths
current_path = Path(".")
data_path = current_path / "data"  # Create path to data folder
export_path = Path("Standardized_Expenditure")
export_path.mkdir(parents=True, exist_ok=True)


# Israeli Capitation Formula weights by age group
# If you have a cap.csv file, we'll load it, otherwise we'll use these default values
# These are placeholder values - replace with actual Israeli capitation formula values
ISRAELI_CAPITATION = {
    "0 to 4": {"Men": 1.55, "Women": 1.26},
    "5 to 14": {"Men": 0.48, "Women": 0.38},
    "15 to 24": {"Men": 0.42, "Women": 0.63},
    "25 to 34": {"Men": 0.57, "Women": 1.07},
    "35 to 44": {"Men": 0.68, "Women": 0.91},
    "45 to 54": {"Men": 1.07, "Women": 1.32},
    "55 to 64": {"Men": 1.86, "Women": 1.79},
    "65 to 74": {"Men": 2.9, "Women": 2.36},
    "75 to 84": {"Men": 3.64, "Women": 3.23},
    "85 and over": {"Men": 3.64, "Women": 2.7}
}

# Reference year for constant price calculations and PPP adjustment
REFERENCE_YEAR = 2017
BASE_COUNTRY = "United States"  # Base country for PPP comparisons


def load_capitation_weights(formula='israeli'):
    """
    Load capitation weights from CSV file if it exists, otherwise use default values.
    
    Args:
        formula: The capitation formula to use ('israeli', 'ltc', or 'eu27')
    
    Returns:
        Dictionary of capitation weights
    """
    try:
        cap_df = pd.read_csv(data_path / "cap.csv", index_col="Age")
        
        if formula == 'israeli':
            # Convert to dictionary format for Israeli formula (men/women separate)
            cap_dict = {}
            for age_group in cap_df.index:
                cap_dict[age_group] = {
                    "Men": cap_df.loc[age_group, "Men"],
                    "Women": cap_df.loc[age_group, "Women"]
                }
            print(f"Loaded Israeli capitation weights from cap.csv")
            return cap_dict, 'israeli'
        
        elif formula == 'ltc':
            # Convert to dictionary format for LTC formula (combined)
            cap_dict = {}
            for age_group in cap_df.index:
                cap_dict[age_group] = {
                    "Combined": cap_df.loc[age_group, "LTC"]
                }
            print(f"Loaded LTC capitation weights from cap.csv")
            return cap_dict, 'ltc'
        
        elif formula == 'eu27':
            # Convert to dictionary format for EU27 formula (combined)
            cap_dict = {}
            for age_group in cap_df.index:
                cap_dict[age_group] = {
                    "Combined": cap_df.loc[age_group, "EU27"]
                }
            print(f"Loaded EU27 capitation weights from cap.csv")
            return cap_dict, 'eu27'
        
        else:
            print(f"Unknown formula '{formula}', using default Israeli capitation weights")
            return ISRAELI_CAPITATION, 'israeli'
    
    except FileNotFoundError:
        print("cap.csv not found, using default Israeli capitation weights")
        return ISRAELI_CAPITATION, 'israeli'
    
    except KeyError as e:
        print(f"Error loading capitation weights: Missing column {e} in cap.csv")
        print("Using default Israeli capitation weights")
        return ISRAELI_CAPITATION, 'israeli'


def load_ppp_data():
    """
    Load and process World Bank PPP data.
    
    Returns:
        DataFrame with columns: ISO3, Year, PPP_Factor
    """
    print("Loading PPP data...")
    
    try:
        # Read the PPP data file
        ppp_file = "API_PA.NUS.PPP_DS2_en_csv_v2_13721.csv"
        
        # Load the CSV file, skipping the metadata rows
        ppp_df = pd.read_csv(data_path / ppp_file)
        
        # Process to create year-country pairs with PPP values
        ppp_data = []
        
        # Get year columns (exclude metadata columns)
        year_columns = [col for col in ppp_df.columns if col.isdigit()]
        
        # Process each row
        for _, row in ppp_df.iterrows():
            country = row['Country Name']
            country_code = row['Country Code']  # This is the ISO3 code in World Bank data
            indicator = row['Indicator Name']
            
            # Check if this is a PPP row
            if "PPP conversion factor, GDP (LCU per international $)" not in indicator:
                continue
                
            # Skip aggregate regions
            if any(x in country.lower() for x in ['region', 'world', 'income', 'development']):
                continue
                
            # Extract PPP values for each year
            for year in year_columns:
                if pd.notna(row[year]) and row[year] != 0:
                    try:
                        ppp_value = float(row[year])
                        ppp_data.append({
                            "Country": country,  # Keep country name for reference
                            "ISO3": country_code,  # Use ISO3 code for merging
                            "Year": int(year),
                            "PPP_Factor": ppp_value
                        })
                    except (ValueError, TypeError):
                        # Skip values that can't be converted to float
                        pass
        
        # Convert to DataFrame
        result = pd.DataFrame(ppp_data)
        
        if result.empty:
            print("Warning: No PPP data was successfully parsed")
            return pd.DataFrame(columns=["ISO3", "Country", "Year", "PPP_Factor"])
        
        print(f"Loaded PPP data with shape: {result.shape}")
        
        return result
    
    except Exception as e:
        print(f"Error loading PPP data: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty DataFrame as fallback
        return pd.DataFrame(columns=["ISO3", "Country", "Year", "PPP_Factor"])


def load_ghed_data():
    """
    Load and process GHED data from the optimized CSV file.
    Falls back to Excel if the CSV file is not available.
    """
    print("Loading GHED data...")
    try:
        # First try to load the optimized CSV file
        csv_path = data_path / "processed" / "ghed_data_optimized.csv"
        
        if csv_path.exists():

            # Read the optimized CSV data
            ghed_data = pd.read_csv(csv_path)
            

        else:
            print("Optimized CSV not found, loading from Excel (slower)...")
            print("Consider running the GHED conversion script to create an optimized CSV for faster loading.")
            
            # Read the GHED data from Excel
            ghed_data = pd.read_excel(data_path / "GHED_data_2025.xlsx", sheet_name="Data")
            
            # Check if the required columns exist
            required_cols = ['location', 'code', 'year', 'che']
            recommended_cols = ['gghed_che', 'pvtd_che']
            
            # Verify required columns
            missing_required = [col for col in required_cols if col not in ghed_data.columns]
            if missing_required:
                raise ValueError(f"Required columns missing from GHED data: {missing_required}")
            
            # Check for recommended columns
            missing_recommended = [col for col in recommended_cols if col not in ghed_data.columns]
            if missing_recommended:
                print(f"Warning: Recommended columns missing: {missing_recommended}")
            
            # Calculate public and private expenditure if percentages are available
            if 'gghed_che' in ghed_data.columns:
                ghed_data['public_expenditure'] = ghed_data['che'] * (ghed_data['gghed_che'] / 100)
            else:
                ghed_data['public_expenditure'] = None
                
            if 'pvtd_che' in ghed_data.columns:
                ghed_data['private_expenditure'] = ghed_data['che'] * (ghed_data['pvtd_che'] / 100)
            else:
                ghed_data['private_expenditure'] = None
            
        
        # Common processing regardless of source
        
        # Adjust the numbers to be actual counts and not in millions
        ghed_data['che'] *= 10**6
        
        if 'public_expenditure' in ghed_data.columns:
            ghed_data['public_expenditure'] *= 10**6
            
        if 'private_expenditure' in ghed_data.columns:
            ghed_data['private_expenditure'] *= 10**6
        
        # Select relevant columns
        relevant_cols = ['location', 'code', 'year', 'che', 'public_expenditure', 'private_expenditure']
        ghed_data = ghed_data[[col for col in relevant_cols if col in ghed_data.columns]]
        
        # Remove rows with missing expenditure data
        ghed_data = ghed_data.dropna(subset=['che'])
        
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


def load_population_data():
    """
    Load and process population data from CSV files using vectorized operations.
    
    Returns:
        Tuple of (male_pop, female_pop) DataFrames with processed population data.
    """
    print("Loading population data...")
    try:
        # Load male and female population data from CSV files
        male_pop_raw = pd.read_csv(data_path / "male_pop.csv")
        female_pop_raw = pd.read_csv(data_path / "female_pop.csv")
        
        print(f"Male population raw data shape: {male_pop_raw.shape}")
        print(f"Female population raw data shape: {female_pop_raw.shape}")
        
        # Check if ISO3 code column exists
        if 'ISO3 Alpha-code' not in male_pop_raw.columns or 'ISO3 Alpha-code' not in female_pop_raw.columns:
            raise ValueError("ISO3 Alpha-code column not found in population data")
        
        # Process both datasets using the vectorized function
        male_pop = process_population_dataset(male_pop_raw, sex="Men")
        female_pop = process_population_dataset(female_pop_raw, sex="Women")
        
        # Create a standardized ISO3 to country mapping
        iso3_to_country = create_standardized_country_mapping(male_pop, female_pop)
        
        # Apply the standardized country names
        male_pop['Country'] = male_pop['ISO3'].map(iso3_to_country)
        female_pop['Country'] = female_pop['ISO3'].map(iso3_to_country)
        
        # Fill any missing country names with the ISO3 code
        male_pop['Country'] = male_pop['Country'].fillna(male_pop['ISO3'])
        female_pop['Country'] = female_pop['Country'].fillna(female_pop['ISO3'])
        
        print(f"Processed male population data with shape: {male_pop.shape}")
        print(f"Processed female population data with shape: {female_pop.shape}")
        
        # Print sample of processed data
        if not male_pop.empty:
            print("\nProcessed male population sample:")
            print(male_pop.head())
        
        if not female_pop.empty:
            print("\nProcessed female population sample:")
            print(female_pop.head())
        
        return male_pop, female_pop
    
    except Exception as e:
        print(f"Error loading population data: {e}")
        print("Detailed error information:")
        import traceback
        traceback.print_exc()
        
        # Create empty DataFrames with the correct structure as a fallback
        columns = ["ISO3", "Year", "Age_Group", "Sex", "Population", "Country"]
        empty_df = pd.DataFrame(columns=columns)
        return empty_df, empty_df

def create_standardized_country_mapping(male_pop, female_pop):
    """
    Create a standardized mapping from ISO3 codes to country names.
    Combines data from both male and female population datasets.
    
    Args:
        male_pop: Processed male population DataFrame
        female_pop: Processed female population DataFrame
        
    Returns:
        Dictionary mapping ISO3 codes to standardized country names
    """
    iso3_to_country = {}
    
    # First collect all country names from male population data
    if 'ISO3' in male_pop.columns and 'Country' in male_pop.columns:
        male_mapping = male_pop.drop_duplicates('ISO3').set_index('ISO3')['Country'].to_dict()
        iso3_to_country.update(male_mapping)
    
    # Then add any missing ones from female population data
    if 'ISO3' in female_pop.columns and 'Country' in female_pop.columns:
        # Only add countries not already in the mapping
        for iso3, group in female_pop.groupby('ISO3'):
            if iso3 not in iso3_to_country and not group.empty:
                iso3_to_country[iso3] = group['Country'].iloc[0]
    
    print(f"Created standardized country mapping with {len(iso3_to_country)} ISO3 codes")
    
    return iso3_to_country

def standardize_country_names_by_iso3(df, iso3_to_country):
    """
    Apply standardized country names based on ISO3 codes.
    
    Args:
        df: DataFrame with ISO3 and potentially inconsistent Country columns
        iso3_to_country: Dictionary mapping ISO3 codes to standardized country names
        
    Returns:
        DataFrame with standardized country names
    """
    if df.empty:
        return df
        
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Replace Country values with standardized names from the mapping
    result['Country'] = result['ISO3'].map(iso3_to_country)
    
    return result

def process_population_dataset(pop_df, sex):
    """
    Process a population dataset (either male or female) using vectorized operations.
    
    Args:
        pop_df: DataFrame containing raw population data
        sex: String indicating sex ("Men" or "Women")
    
    Returns:
        DataFrame with processed population data
    """
    print(f"Processing {sex.lower()} population data...")
    
    # Get the age group columns
    age_columns = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', 
                  '35-39', '40-44', '45-49', '50-54', '55-59', '60-64',
                  '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+']
    
    # Filter out non-country rows and rows with missing ISO3 or year
    valid_mask = (
        pop_df['ISO3 Alpha-code'].notna() & 
        pop_df['Year'].notna() & 
        pop_df['ISO3 Alpha-code'].astype(str).str.len() > 0
    )
    
    # Filter out regions and aggregates
    region_terms = ['region', 'world', 'income', 'development', 'more developed', 'less developed']
    region_pattern = '|'.join(region_terms)
    country_mask = ~pop_df['Region, subregion, country or area *'].str.lower().str.contains(region_pattern, na=False)
    
    # Combine masks
    filtered_df = pop_df[valid_mask & country_mask].copy()
    
    if filtered_df.empty:
        print(f"Warning: No valid {sex.lower()} population data after filtering")
        return pd.DataFrame(columns=["ISO3", "Year", "Age_Group", "Sex", "Population", "Country"])
    
    # Ensure ISO3 and Year are proper types
    filtered_df['ISO3'] = filtered_df['ISO3 Alpha-code'].astype(str)
    filtered_df['Year'] = filtered_df['Year'].astype(int)
    filtered_df['Country'] = filtered_df['Region, subregion, country or area *']
    
    # Create a list to store the processed data for each age group
    processed_dfs = []
    
    for age_col in age_columns:
        # Create a subset for this age group
        age_df = filtered_df[['ISO3', 'Year', 'Country', age_col]].copy()
        
        # Skip age groups with all missing values
        if age_df[age_col].isna().all():
            continue
        
        # Drop rows with missing population values
        age_df = age_df.dropna(subset=[age_col])
        
        # Convert population values to numeric
        age_df['Population'] = pd.to_numeric(age_df[age_col].astype(str).str.replace(' ', ''), errors='coerce')
        
        # Drop rows with invalid population values
        age_df = age_df[age_df['Population'] > 0]
        
        if age_df.empty:
            continue
        
        # Convert population from thousands to actual counts
        age_df['Population'] = age_df['Population'] * 1000
        
        # Map age group
        mapped_age_group = map_age_group(age_col)
        
        if mapped_age_group is None:
            continue
        
        # Add age group and sex columns
        age_df['Age_Group'] = mapped_age_group
        age_df['Sex'] = sex
        
        # Select only the needed columns
        age_df = age_df[['ISO3', 'Year', 'Age_Group', 'Sex', 'Population', 'Country']]
        
        # Add to the list of processed dataframes
        processed_dfs.append(age_df)
    
    # If no age groups were processed, return an empty DataFrame
    if not processed_dfs:
        print(f"Warning: No valid {sex.lower()} population data after processing age groups")
        return pd.DataFrame(columns=["ISO3", "Year", "Age_Group", "Sex", "Population", "Country"])
    
    # Combine all age group dataframes
    combined_df = pd.concat(processed_dfs, ignore_index=True)
    
    # Group by to sum populations for the same ISO3, Year, Age_Group, Sex
    result_df = combined_df.groupby(['ISO3', 'Year', 'Age_Group', 'Sex']).agg({
        'Population': 'sum',
        'Country': 'first'  # Take the first country name
    }).reset_index()
    
    print(f"Processed {len(result_df)} rows of {sex.lower()} population data")
    
    return result_df


def is_region_or_aggregate(country_name):
    """
    Check if a country name represents a region or aggregate.
    
    Args:
        country_name: String with country name to check
    
    Returns:
        Boolean indicating if the country is a region or aggregate
    """
    region_terms = ['region', 'world', 'income', 'development', 'more developed', 'less developed']
    return any(term in country_name.lower() for term in region_terms)


def extract_population_value(pop_value):
    """
    Extract a numeric population value from a cell value.
    
    Args:
        pop_value: Cell value from CSV (could be string or numeric)
    
    Returns:
        Numeric population value
    """
    try:
        if isinstance(pop_value, str):
            # Remove spaces and convert to numeric
            population_str = pop_value.replace(' ', '')
            return pd.to_numeric(population_str, errors='coerce')
        else:
            # Convert to numeric (it might be already numeric)
            return pd.to_numeric(pop_value, errors='coerce')
    except Exception:
        return np.nan

def map_age_group(un_age_group):
    """
    Map the UN age groups to the age groups used in the capitation formula.
    
    Args:
        un_age_group: Age group from UN data (e.g., "0-4", "5-9", etc.)
    
    Returns:
        Mapped age group or None if it couldn't be mapped
    """
    # Extract age range from the age group string
    if isinstance(un_age_group, str):
        un_age_group = un_age_group.strip().lower()
    else:
        return None
    
    # Map UN age groups to our capitation formula age groups
    mapping = {
        # Map 0-4 to the capitation's 0 to 4
        "0-4": "0 to 4",
        
        # Map 5-9 and 10-14 to the capitation's 5 to 14
        "5-9": "5 to 14",
        "10-14": "5 to 14",
        
        # Map 15-19 and 20-24 to the capitation's 15 to 24
        "15-19": "15 to 24",
        "20-24": "15 to 24",
        
        # Map 25-29 and 30-34 to the capitation's 25 to 34
        "25-29": "25 to 34",
        "30-34": "25 to 34",
        
        # Map 35-39 and 40-44 to the capitation's 35 to 44
        "35-39": "35 to 44",
        "40-44": "35 to 44",
        
        # Map 45-49 and 50-54 to the capitation's 45 to 54
        "45-49": "45 to 54",
        "50-54": "45 to 54",
        
        # Map 55-59 and 60-64 to the capitation's 55 to 64
        "55-59": "55 to 64",
        "60-64": "55 to 64",
        
        # Map 65-69 and 70-74 to the capitation's 65 to 74
        "65-69": "65 to 74",
        "70-74": "65 to 74",
        
        # Map 75-79 and 80-84 to the capitation's 75 to 84
        "75-79": "75 to 84",
        "80-84": "75 to 84",
        
        # Map 85-89, 90-94, 95-99, 100+ to the capitation's 85 and over
        "85-89": "85 and over",
        "90-94": "85 and over",
        "95-99": "85 and over",
        "100+": "85 and over"
    }
    
    # Try to match the age group
    for pattern, mapped_group in mapping.items():
        if pattern in un_age_group:
            return mapped_group
    
    return None

def preprocess_population_data(male_pop, female_pop, cap_dict, formula_type='israeli'):
    """
    Calculate standardized population for each country and year using the capitation formula.
    Uses vectorized operations for improved performance.
    
    Args:
        male_pop: DataFrame with male population data
        female_pop: DataFrame with female population data
        cap_dict: Dictionary with capitation weights
        formula_type: Type of capitation formula ('israeli', 'ltc', or 'eu27')
    
    Returns:
        DataFrame with standardized population by country and year
    """
    print(f"Calculating standardized population using {formula_type} formula...")
    
    # Check for empty dataframes
    if male_pop.empty or female_pop.empty:
        print("Warning: Empty population data provided")
        return pd.DataFrame(columns=['ISO3', 'Year', 'Country', 'Standardized_Population'])
    
    # Only select the necessary columns for the keys to avoid duplicate columns
    male_keys = male_pop[['ISO3', 'Year']].drop_duplicates()
    female_keys = female_pop[['ISO3', 'Year']].drop_duplicates()
    
    # Use outer join to include all country-year combinations
    all_keys = pd.merge(male_keys, female_keys, on=['ISO3', 'Year'], how='outer')
    
    # Create a standardized country mapping
    country_mapping = {}
    
    # Get country names from male dataset
    if 'Country' in male_pop.columns:
        male_countries = male_pop[['ISO3', 'Country']].drop_duplicates()
        for _, row in male_countries.iterrows():
            if pd.notna(row['Country']):
                country_mapping[row['ISO3']] = row['Country']
    
    # Add any missing country names from female dataset
    if 'Country' in female_pop.columns:
        female_countries = female_pop[['ISO3', 'Country']].drop_duplicates()
        for _, row in female_countries.iterrows():
            if row['ISO3'] not in country_mapping and pd.notna(row['Country']):
                country_mapping[row['ISO3']] = row['Country']
    
    # Apply country names to the results
    all_keys['Country'] = all_keys['ISO3'].map(country_mapping)
    
    # Fill any missing country names with the ISO3 code
    all_keys['Country'] = all_keys['Country'].fillna(all_keys['ISO3'])
    
    # Initialize standardized population
    all_keys['Standardized_Population'] = 0.0
    
    if formula_type == 'israeli':
        # Process using Israeli formula (separate weights for men and women)
        process_israeli_formula(all_keys, male_pop, female_pop, cap_dict)
    else:
        # Process using LTC or EU27 formula (combined weight for both sexes)
        process_combined_formula(all_keys, male_pop, female_pop, cap_dict)
    
    # Count countries with standardized population
    countries_with_std_pop = all_keys[all_keys['Standardized_Population'] > 0]['ISO3'].nunique()
    print(f"Calculated standardized population for {countries_with_std_pop} unique countries")
    
    return all_keys


def process_israeli_formula(result_df, male_pop, female_pop, cap_dict):
    """
    Process population data using the Israeli formula with separate weights for men and women.
    
    Args:
        result_df: DataFrame to store results
        male_pop: Male population data
        female_pop: Female population data
        cap_dict: Dictionary with capitation weights
    """
    # Create a pivot table of male population by ISO3, Year, and Age_Group
    if not male_pop.empty:
        male_pivot = male_pop.pivot_table(
            index=['ISO3', 'Year'],
            columns='Age_Group',
            values='Population',
            aggfunc='sum',
            fill_value=0
        )
        
        # Apply weights to each age group and sum
        for age_group in cap_dict:
            if age_group in male_pivot.columns:
                weight = cap_dict[age_group]['Men']
                # Create a Series of weighted population
                weighted_pop = male_pivot[age_group] * weight
                
                # Create a mapping to update the results
                weighted_dict = {idx: val for idx, val in weighted_pop.items()}
                
                # Update the standardized population
                for idx, row in result_df.iterrows():
                    key = (row['ISO3'], row['Year'])
                    if key in weighted_dict:
                        result_df.at[idx, 'Standardized_Population'] += weighted_dict[key]
    
    # Create a pivot table of female population by ISO3, Year, and Age_Group
    if not female_pop.empty:
        female_pivot = female_pop.pivot_table(
            index=['ISO3', 'Year'],
            columns='Age_Group',
            values='Population',
            aggfunc='sum',
            fill_value=0
        )
        
        # Apply weights to each age group and sum
        for age_group in cap_dict:
            if age_group in female_pivot.columns:
                weight = cap_dict[age_group]['Women']
                # Create a Series of weighted population
                weighted_pop = female_pivot[age_group] * weight
                
                # Create a mapping to update the results
                weighted_dict = {idx: val for idx, val in weighted_pop.items()}
                
                # Update the standardized population
                for idx, row in result_df.iterrows():
                    key = (row['ISO3'], row['Year'])
                    if key in weighted_dict:
                        result_df.at[idx, 'Standardized_Population'] += weighted_dict[key]


def process_combined_formula(result_df, male_pop, female_pop, cap_dict):
    """
    Process population data using a formula with combined weights for both sexes.
    
    Args:
        result_df: DataFrame to store results
        male_pop: Male population data
        female_pop: Female population data
        cap_dict: Dictionary with capitation weights
    """
    # Create pivot tables
    if not male_pop.empty:
        male_pivot = male_pop.pivot_table(
            index=['ISO3', 'Year'],
            columns='Age_Group',
            values='Population',
            aggfunc='sum',
            fill_value=0
        )
    else:
        male_pivot = pd.DataFrame()
        
    if not female_pop.empty:
        female_pivot = female_pop.pivot_table(
            index=['ISO3', 'Year'],
            columns='Age_Group',
            values='Population',
            aggfunc='sum',
            fill_value=0
        )
    else:
        female_pivot = pd.DataFrame()
    
    # Process each age group
    for age_group in cap_dict:
        weight = cap_dict[age_group]['Combined']
        
        # Process male population for this age group
        if not male_pivot.empty and age_group in male_pivot.columns:
            weighted_pop = male_pivot[age_group] * weight
            weighted_dict = {idx: val for idx, val in weighted_pop.items()}
            
            # Update the standardized population
            for idx, row in result_df.iterrows():
                key = (row['ISO3'], row['Year'])
                if key in weighted_dict:
                    result_df.at[idx, 'Standardized_Population'] += weighted_dict[key]
        
        # Process female population for this age group
        if not female_pivot.empty and age_group in female_pivot.columns:
            weighted_pop = female_pivot[age_group] * weight
            weighted_dict = {idx: val for idx, val in weighted_pop.items()}
            
            # Update the standardized population
            for idx, row in result_df.iterrows():
                key = (row['ISO3'], row['Year'])
                if key in weighted_dict:
                    result_df.at[idx, 'Standardized_Population'] += weighted_dict[key]

def load_gdp_data(reference_year=2017):
    """
    Load and process World Bank GDP data to calculate GDP deflators.
    
    Args:
        reference_year: Year to use as the reference for constant prices (default: 2017)
    
    Returns:
        DataFrame with columns: ISO3, Year, GDP_Deflator
    """
    print("Loading GDP data for deflator calculation...")
    
    try:
        # Read the GDP files (current LCU and constant LCU)
        gdp_current_file = "API_NY.GDP.MKTP.CN_DS2_en_csv_v2_26332.csv"
        gdp_constant_file = "API_NY.GDP.MKTP.KN_DS2_en_csv_v2_13325.csv"
        
        # Load the CSV files
        gdp_current_df = pd.read_csv(data_path / gdp_current_file)
        gdp_constant_df = pd.read_csv(data_path / gdp_constant_file)
        
        # Process both datasets to create year-country pairs with GDP values
        gdp_current_data = []
        gdp_constant_data = []
        
        # Get year columns (exclude metadata columns)
        year_columns = [col for col in gdp_current_df.columns if col.isdigit()]
        
        # Process current GDP data
        for _, row in gdp_current_df.iterrows():
            country = row['Country Name']
            country_code = row['Country Code']  # ISO3 code in World Bank data
            
            # Skip aggregate regions
            if any(x in country.lower() for x in ['region', 'world', 'income', 'development']):
                continue
                
            for year in year_columns:
                if pd.notna(row[year]) and row[year] != 0:
                    gdp_current_data.append({
                        "Country": country,  # Keep country name for reference
                        "ISO3": country_code,  # Use ISO3 code for merging
                        "Year": int(year),
                        "GDP_Current": float(row[year])
                    })
        
        # Process constant GDP data
        for _, row in gdp_constant_df.iterrows():
            country = row['Country Name']
            country_code = row['Country Code']  # ISO3 code in World Bank data
            
            # Skip aggregate regions
            if any(x in country.lower() for x in ['region', 'world', 'income', 'development']):
                continue
                
            for year in year_columns:
                if pd.notna(row[year]) and row[year] != 0:
                    gdp_constant_data.append({
                        "Country": country,  # Keep country name for reference
                        "ISO3": country_code,  # Use ISO3 code for merging
                        "Year": int(year),
                        "GDP_Constant": float(row[year])
                    })
        
        # Convert to DataFrames
        gdp_current_df = pd.DataFrame(gdp_current_data)
        gdp_constant_df = pd.DataFrame(gdp_constant_data)
        
        # Merge the datasets on ISO3 and Year
        gdp_merged = pd.merge(
            gdp_current_df,
            gdp_constant_df,
            on=["ISO3", "Year"],
            how="inner"
        )
        
        # Calculate GDP deflator (GDP_Current / GDP_Constant)
        gdp_merged["GDP_Deflator"] = gdp_merged["GDP_Current"] / gdp_merged["GDP_Constant"]
        
        # For each country, normalize the deflator by the reference year
        gdp_deflator = []
        for iso3 in gdp_merged["ISO3"].unique():
            country_data = gdp_merged[gdp_merged["ISO3"] == iso3].copy()
            
            # Check if the reference year exists for this country
            ref_year_data = country_data[country_data["Year"] == reference_year]
            
            if not ref_year_data.empty:
                # If reference year exists, normalize by it
                ref_deflator = ref_year_data["GDP_Deflator"].iloc[0]
                country_data["GDP_Deflator_Normalized"] = country_data["GDP_Deflator"] / ref_deflator
            else:
                # If reference year doesn't exist, find the closest year
                available_years = country_data["Year"].tolist()
                if available_years:
                    closest_year = min(available_years, key=lambda x: abs(x - reference_year))
                    ref_deflator = country_data[country_data["Year"] == closest_year]["GDP_Deflator"].iloc[0]
                    country_data["GDP_Deflator_Normalized"] = country_data["GDP_Deflator"] / ref_deflator
                    country_name = country_data["Country_x"].iloc[0]  # Use the first country name from the merged data
                    print(f"Using {closest_year} as reference year for {country_name} ({iso3}) (reference {reference_year} not available)")
                else:
                    # No data for this country
                    country_data["GDP_Deflator_Normalized"] = np.nan
            
            # Add to the result list
            gdp_deflator.append(country_data[["ISO3", "Country_x", "Year", "GDP_Deflator_Normalized"]])
        
        # Combine all countries
        result = pd.concat(gdp_deflator, ignore_index=True)
        result = result.rename(columns={"GDP_Deflator_Normalized": "GDP_Deflator", "Country_x": "Country"})
        
        print(f"Loaded GDP deflator data with shape: {result.shape}")
        
        return result
    
    except Exception as e:
        print(f"Error loading GDP data: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty DataFrame as fallback
        return pd.DataFrame(columns=["ISO3", "Country", "Year", "GDP_Deflator"])

def apply_gdp_deflator_adjustment(data, gdp_deflator, impute_missing=False):
    """
    Apply GDP deflator adjustment to health expenditure data to convert to constant prices.
    
    Args:
        data: DataFrame with health expenditure data (with ISO3 codes)
        gdp_deflator: DataFrame with GDP deflators (with ISO3 codes)
        impute_missing: Whether to impute missing GDP deflators (default: False)
    
    Returns:
        DataFrame with constant price adjusted health expenditure
    """
    print("Applying GDP deflator adjustment for constant prices...")
    
    # Make a copy to avoid modifying the original data
    adjusted_data = data.copy()
    
    # Merge GDP deflator data with health expenditure data using ISO3 code
    merged = pd.merge(
        adjusted_data,
        gdp_deflator[['ISO3', 'Year', 'GDP_Deflator']],
        on=['ISO3', 'Year'],
        how='left'
    )
    
    # Check if any GDP deflators are missing
    missing_deflator = merged['GDP_Deflator'].isna().sum()
    if missing_deflator > 0:
        print(f"Warning: Missing GDP deflators for {missing_deflator} out of {len(merged)} rows")
        
        # For countries/years with missing deflators, impute using nearest available year if impute_missing is True
        if impute_missing:
            print(f"Imputing missing GDP deflators...")
            countries_with_missing = merged[merged['GDP_Deflator'].isna()]['ISO3'].unique()
            
            for iso3 in countries_with_missing:
                country_data = merged[merged['ISO3'] == iso3]
                missing_years = country_data[country_data['GDP_Deflator'].isna()]['Year'].tolist()
                
                # If country has any deflator data, use nearest year
                if any(~country_data['GDP_Deflator'].isna()):
                    available_years = country_data[~country_data['GDP_Deflator'].isna()]['Year'].tolist()
                    
                    for missing_year in missing_years:
                        # Find closest available year
                        closest_year = min(available_years, key=lambda x: abs(x - missing_year))
                        closest_deflator = country_data[country_data['Year'] == closest_year]['GDP_Deflator'].iloc[0]
                        
                        # Impute the missing value
                        idx = merged[(merged['ISO3'] == iso3) & (merged['Year'] == missing_year)].index
                        merged.loc[idx, 'GDP_Deflator'] = closest_deflator
                        
                        # Display country name if available
                        if 'Country' in merged.columns:
                            country_name = merged.loc[merged['ISO3'] == iso3, 'Country'].iloc[0]
                            print(f"Imputed GDP deflator for {country_name} ({iso3}) in year {missing_year} using data from {closest_year}")
                        else:
                            print(f"Imputed GDP deflator for ISO3: {iso3} in year {missing_year} using data from {closest_year}")
        else:
            print(f"Skipping imputation for missing GDP deflators as requested")
    
    # Create a mask for rows with valid deflator values
    valid_deflator_mask = ~merged['GDP_Deflator'].isna()
    
    # Adjust expenditure values to constant prices (reference year) only for rows with valid deflators
    # formula: constant_price_value = current_price_value / deflator
    
    # Adjust total health expenditure
    if 'Total_Health_Expenditure' in merged.columns:
        merged['Total_Health_Expenditure_Constant'] = np.nan
        merged['Total_Health_Expenditure_per_Std_Capita_Constant'] = np.nan
        
        # Calculate only for valid rows
        merged.loc[valid_deflator_mask, 'Total_Health_Expenditure_Constant'] = merged.loc[valid_deflator_mask, 'Total_Health_Expenditure'] / merged.loc[valid_deflator_mask, 'GDP_Deflator']
        merged.loc[valid_deflator_mask, 'Total_Health_Expenditure_per_Std_Capita_Constant'] = merged.loc[valid_deflator_mask, 'Total_Health_Expenditure_Constant'] / merged.loc[valid_deflator_mask, 'Standardized_Population']
    
    # Adjust public health expenditure if available
    if 'Public_Health_Expenditure' in merged.columns:
        merged['Public_Health_Expenditure_Constant'] = np.nan
        merged['Public_Health_Expenditure_per_Std_Capita_Constant'] = np.nan
        
        # Calculate only for valid rows with non-null public health expenditure
        valid_public_mask = valid_deflator_mask & ~merged['Public_Health_Expenditure'].isna()
        
        merged.loc[valid_public_mask, 'Public_Health_Expenditure_Constant'] = merged.loc[valid_public_mask, 'Public_Health_Expenditure'] / merged.loc[valid_public_mask, 'GDP_Deflator']
        merged.loc[valid_public_mask, 'Public_Health_Expenditure_per_Std_Capita_Constant'] = merged.loc[valid_public_mask, 'Public_Health_Expenditure_Constant'] / merged.loc[valid_public_mask, 'Standardized_Population']
    
    # Adjust private health expenditure if available
    if 'Private_Health_Expenditure' in merged.columns:
        merged['Private_Health_Expenditure_Constant'] = np.nan
        merged['Private_Health_Expenditure_per_Std_Capita_Constant'] = np.nan
        
        # Calculate only for valid rows with non-null private health expenditure
        valid_private_mask = valid_deflator_mask & ~merged['Private_Health_Expenditure'].isna()
        
        merged.loc[valid_private_mask, 'Private_Health_Expenditure_Constant'] = merged.loc[valid_private_mask, 'Private_Health_Expenditure'] / merged.loc[valid_private_mask, 'GDP_Deflator']
        merged.loc[valid_private_mask, 'Private_Health_Expenditure_per_Std_Capita_Constant'] = merged.loc[valid_private_mask, 'Private_Health_Expenditure_Constant'] / merged.loc[valid_private_mask, 'Standardized_Population']
    
    return merged

def apply_ppp_adjustment(data, ppp_data, base_country_iso="USA", reference_year=2017, impute_missing=False):
    """
    Apply PPP adjustment to health expenditure data.
    
    Args:
        data: DataFrame with health expenditure data (with ISO3 codes)
        ppp_data: DataFrame with PPP conversion factors (with ISO3 codes)
        base_country_iso: Base country ISO3 code for PPP comparisons (default: "USA")
        reference_year: Reference year for PPP adjustment
        impute_missing: Whether to impute missing PPP factors (default: False)
    
    Returns:
        DataFrame with PPP-adjusted health expenditure
    """
    # Make a copy to avoid modifying the original data
    adjusted_data = data.copy()
    
    # Merge PPP data with health expenditure data using ISO3 code
    merged = pd.merge(
        adjusted_data,
        ppp_data[['ISO3', 'Year', 'PPP_Factor']],
        on=['ISO3', 'Year'],
        how='left'
    )
    
    # Check if any PPP factors are missing
    missing_ppp = merged['PPP_Factor'].isna().sum()
    if missing_ppp > 0:
        print(f"Warning: Missing PPP factors for {missing_ppp} out of {len(merged)} rows")
        
        # For countries/years with missing PPP factors, impute using nearest available year if impute_missing is True
        if impute_missing:
            print(f"Imputing missing PPP factors...")
            countries_with_missing = merged[merged['PPP_Factor'].isna()]['ISO3'].unique()
            
            for iso3 in countries_with_missing:
                country_data = merged[merged['ISO3'] == iso3]
                missing_years = country_data[country_data['PPP_Factor'].isna()]['Year'].tolist()
                
                # If country has any PPP data, use nearest year
                if any(~country_data['PPP_Factor'].isna()):
                    available_years = country_data[~country_data['PPP_Factor'].isna()]['Year'].tolist()
                    
                    for missing_year in missing_years:
                        # Find closest available year
                        closest_year = min(available_years, key=lambda x: abs(x - missing_year))
                        closest_ppp = country_data[country_data['Year'] == closest_year]['PPP_Factor'].iloc[0]
                        
                        # Impute the missing value
                        idx = merged[(merged['ISO3'] == iso3) & (merged['Year'] == missing_year)].index
                        merged.loc[idx, 'PPP_Factor'] = closest_ppp
                        
                        # Display country name if available
                        if 'Country' in merged.columns:
                            country_name = merged.loc[merged['ISO3'] == iso3, 'Country'].iloc[0]
                            print(f"Imputed PPP factor for {country_name} ({iso3}) in year {missing_year} using data from {closest_year}")
                        else:
                            print(f"Imputed PPP factor for ISO3: {iso3} in year {missing_year} using data from {closest_year}")
        else:
            print(f"Skipping imputation for missing PPP factors as requested")
    
    # Check if we have any PPP factors for the base country
    base_country_ppp = ppp_data[ppp_data['ISO3'] == base_country_iso]
    
    if base_country_ppp.empty:
        # Get the country name if available
        base_country_name = "Unknown"
        if 'Country' in ppp_data.columns:
            base_country_matches = ppp_data[ppp_data['ISO3'] == base_country_iso]
            if not base_country_matches.empty:
                base_country_name = base_country_matches.iloc[0]['Country']
        
        print(f"Warning: No PPP factors available for base country ({base_country_name}, ISO3: {base_country_iso})")
        print("Cannot perform proper PPP adjustment without base country data")
        # Return the original data if we can't adjust
        return adjusted_data
    
    # Calculate PPP-adjusted values only for rows with valid PPP factors
    valid_ppp_mask = ~merged['PPP_Factor'].isna()
    
    # Adjust total health expenditure
    if 'Total_Health_Expenditure' in merged.columns:
        # Initialize columns with NaN
        merged['Total_Health_Expenditure_PPP'] = np.nan
        merged['Total_Health_Expenditure_per_Std_Capita_PPP'] = np.nan
        
        # Calculate for valid rows
        merged.loc[valid_ppp_mask, 'Total_Health_Expenditure_PPP'] = merged.loc[valid_ppp_mask, 'Total_Health_Expenditure'] / merged.loc[valid_ppp_mask, 'PPP_Factor']
        merged.loc[valid_ppp_mask, 'Total_Health_Expenditure_per_Std_Capita_PPP'] = merged.loc[valid_ppp_mask, 'Total_Health_Expenditure_PPP'] / merged.loc[valid_ppp_mask, 'Standardized_Population']
        
        # If constant price data is available, also adjust it with PPP
        if 'Total_Health_Expenditure_Constant' in merged.columns:
            merged['Total_Health_Expenditure_Constant_PPP'] = np.nan
            merged['Total_Health_Expenditure_per_Std_Capita_Constant_PPP'] = np.nan
            
            merged.loc[valid_ppp_mask, 'Total_Health_Expenditure_Constant_PPP'] = merged.loc[valid_ppp_mask, 'Total_Health_Expenditure_Constant'] / merged.loc[valid_ppp_mask, 'PPP_Factor']
            merged.loc[valid_ppp_mask, 'Total_Health_Expenditure_per_Std_Capita_Constant_PPP'] = merged.loc[valid_ppp_mask, 'Total_Health_Expenditure_Constant_PPP'] / merged.loc[valid_ppp_mask, 'Standardized_Population']
    
    # Adjust public health expenditure if available
    if 'Public_Health_Expenditure' in merged.columns:
        merged['Public_Health_Expenditure_PPP'] = np.nan
        merged['Public_Health_Expenditure_per_Std_Capita_PPP'] = np.nan
        
        valid_public_mask = valid_ppp_mask & ~merged['Public_Health_Expenditure'].isna()
        
        merged.loc[valid_public_mask, 'Public_Health_Expenditure_PPP'] = merged.loc[valid_public_mask, 'Public_Health_Expenditure'] / merged.loc[valid_public_mask, 'PPP_Factor']
        merged.loc[valid_public_mask, 'Public_Health_Expenditure_per_Std_Capita_PPP'] = merged.loc[valid_public_mask, 'Public_Health_Expenditure_PPP'] / merged.loc[valid_public_mask, 'Standardized_Population']
        
        # If constant price data is available, also adjust it with PPP
        if 'Public_Health_Expenditure_Constant' in merged.columns:
            merged['Public_Health_Expenditure_Constant_PPP'] = np.nan
            merged['Public_Health_Expenditure_per_Std_Capita_Constant_PPP'] = np.nan
            
            valid_public_const_mask = valid_ppp_mask & ~merged['Public_Health_Expenditure_Constant'].isna()
            
            merged.loc[valid_public_const_mask, 'Public_Health_Expenditure_Constant_PPP'] = merged.loc[valid_public_const_mask, 'Public_Health_Expenditure_Constant'] / merged.loc[valid_public_const_mask, 'PPP_Factor']
            merged.loc[valid_public_const_mask, 'Public_Health_Expenditure_per_Std_Capita_Constant_PPP'] = merged.loc[valid_public_const_mask, 'Public_Health_Expenditure_Constant_PPP'] / merged.loc[valid_public_const_mask, 'Standardized_Population']
    
    # Adjust private health expenditure if available
    if 'Private_Health_Expenditure' in merged.columns:
        merged['Private_Health_Expenditure_PPP'] = np.nan
        merged['Private_Health_Expenditure_per_Std_Capita_PPP'] = np.nan
        
        valid_private_mask = valid_ppp_mask & ~merged['Private_Health_Expenditure'].isna()
        
        merged.loc[valid_private_mask, 'Private_Health_Expenditure_PPP'] = merged.loc[valid_private_mask, 'Private_Health_Expenditure'] / merged.loc[valid_private_mask, 'PPP_Factor']
        merged.loc[valid_private_mask, 'Private_Health_Expenditure_per_Std_Capita_PPP'] = merged.loc[valid_private_mask, 'Private_Health_Expenditure_PPP'] / merged.loc[valid_private_mask, 'Standardized_Population']
        
        # If constant price data is available, also adjust it with PPP
        if 'Private_Health_Expenditure_Constant' in merged.columns:
            merged['Private_Health_Expenditure_Constant_PPP'] = np.nan
            merged['Private_Health_Expenditure_per_Std_Capita_Constant_PPP'] = np.nan
            
            valid_private_const_mask = valid_ppp_mask & ~merged['Private_Health_Expenditure_Constant'].isna()
            
            merged.loc[valid_private_const_mask, 'Private_Health_Expenditure_Constant_PPP'] = merged.loc[valid_private_const_mask, 'Private_Health_Expenditure_Constant'] / merged.loc[valid_private_const_mask, 'PPP_Factor']
            merged.loc[valid_private_const_mask, 'Private_Health_Expenditure_per_Std_Capita_Constant_PPP'] = merged.loc[valid_private_const_mask, 'Private_Health_Expenditure_Constant_PPP'] / merged.loc[valid_private_const_mask, 'Standardized_Population']
    
    return merged

def apply_constant_ppp_adjustment(data, ppp_data, base_country_iso="USA", reference_year=2017, impute_missing=False):
    """
    Apply constant PPP adjustment to health expenditure data using only the reference year PPP factors.
    
    Args:
        data: DataFrame with health expenditure data (with ISO3 codes)
        ppp_data: DataFrame with PPP conversion factors (with ISO3 codes)
        base_country_iso: Base country ISO3 code for PPP comparisons (default: "USA")
        reference_year: Reference year for PPP factors (default: 2017)
        impute_missing: Whether to impute missing PPP factors (default: False)
    
    Returns:
        DataFrame with constant PPP-adjusted health expenditure
    """
    print(f"Applying constant PPP adjustment using {reference_year} as reference year...")
    
    # Make a copy to avoid modifying the original data
    adjusted_data = data.copy()
    
    # Filter PPP data to only include the reference year
    ref_ppp_data = ppp_data[ppp_data['Year'] == reference_year].copy()
    
    if ref_ppp_data.empty:
        print(f"Warning: No PPP data available for reference year {reference_year}")
        print("Looking for nearest available year...")
        
        # Find the nearest available year to the reference year
        all_years = sorted(ppp_data['Year'].unique())
        if not all_years:
            print("Error: No PPP data available at all")
            return adjusted_data
            
        nearest_year = min(all_years, key=lambda x: abs(x - reference_year))
        print(f"Using PPP data from {nearest_year} as reference (closest to {reference_year})")
        ref_ppp_data = ppp_data[ppp_data['Year'] == nearest_year].copy()
    
    # Create a mapping of ISO3 -> PPP_Factor from the reference year
    ppp_mapping = ref_ppp_data[['ISO3', 'PPP_Factor']].drop_duplicates('ISO3').set_index('ISO3')['PPP_Factor'].to_dict()
    
    # Check if the base country has a PPP factor
    if base_country_iso not in ppp_mapping:
        print(f"Warning: Base country {base_country_iso} does not have a PPP factor for reference year")
        return adjusted_data
        
    # Add a column with constant PPP factors to the adjusted data
    adjusted_data['Constant_PPP_Factor'] = adjusted_data['ISO3'].map(ppp_mapping)
    
    # Check if any PPP factors are missing
    missing_ppp = adjusted_data['Constant_PPP_Factor'].isna().sum()
    if missing_ppp > 0:
        print(f"Warning: Missing PPP factors for {missing_ppp} out of {len(adjusted_data)} rows")
        
        if impute_missing:
            print("Attempting to impute missing PPP factors...")
            
            # For each ISO3 with missing values, try to find the closest year with PPP data
            for iso3 in adjusted_data[adjusted_data['Constant_PPP_Factor'].isna()]['ISO3'].unique():
                # Get all PPP data for this country
                country_ppp = ppp_data[ppp_data['ISO3'] == iso3]
                
                if not country_ppp.empty:
                    # Get the closest year to reference year
                    closest_year = min(country_ppp['Year'].unique(), key=lambda x: abs(x - reference_year))
                    closest_ppp = country_ppp[country_ppp['Year'] == closest_year]['PPP_Factor'].iloc[0]
                    
                    # Apply this PPP factor to all rows with this ISO3
                    adjusted_data.loc[adjusted_data['ISO3'] == iso3, 'Constant_PPP_Factor'] = closest_ppp
                    
                    # Display country name if available
                    if 'Country' in adjusted_data.columns:
                        country_name = adjusted_data.loc[adjusted_data['ISO3'] == iso3, 'Country'].iloc[0]
                        print(f"Imputed constant PPP factor for {country_name} ({iso3}) using data from {closest_year}")
                    else:
                        print(f"Imputed constant PPP factor for ISO3: {iso3} using data from {closest_year}")
    
    # Create a mask for rows with valid PPP factors
    valid_ppp_mask = ~adjusted_data['Constant_PPP_Factor'].isna()
    
    # Adjust total health expenditure
    if 'Total_Health_Expenditure' in adjusted_data.columns:
        # Initialize columns with NaN
        adjusted_data['Total_Health_Expenditure_Constant_PPP'] = np.nan
        adjusted_data['Total_Health_Expenditure_per_Std_Capita_Constant_PPP'] = np.nan
        
        # Calculate for valid rows
        # If already have constant price data, use that, otherwise use current prices
        if 'Total_Health_Expenditure_Constant' in adjusted_data.columns:
            adjusted_data.loc[valid_ppp_mask, 'Total_Health_Expenditure_Constant_PPP'] = (
                adjusted_data.loc[valid_ppp_mask, 'Total_Health_Expenditure_Constant'] / 
                adjusted_data.loc[valid_ppp_mask, 'Constant_PPP_Factor']
            )
        else:
            # Warning about using current prices without GDP deflator adjustment
            print("Warning: GDP deflator data not available, using current prices with constant PPP")
            adjusted_data.loc[valid_ppp_mask, 'Total_Health_Expenditure_Constant_PPP'] = (
                adjusted_data.loc[valid_ppp_mask, 'Total_Health_Expenditure'] / 
                adjusted_data.loc[valid_ppp_mask, 'Constant_PPP_Factor']
            )
        
        # Calculate per standardized capita
        adjusted_data.loc[valid_ppp_mask, 'Total_Health_Expenditure_per_Std_Capita_Constant_PPP'] = (
            adjusted_data.loc[valid_ppp_mask, 'Total_Health_Expenditure_Constant_PPP'] / 
            adjusted_data.loc[valid_ppp_mask, 'Standardized_Population']
        )
    
    # Adjust public health expenditure if available
    if 'Public_Health_Expenditure' in adjusted_data.columns:
        adjusted_data['Public_Health_Expenditure_Constant_PPP'] = np.nan
        adjusted_data['Public_Health_Expenditure_per_Std_Capita_Constant_PPP'] = np.nan
        
        valid_public_mask = valid_ppp_mask & ~adjusted_data['Public_Health_Expenditure'].isna()
        
        # Use constant price data if available
        if 'Public_Health_Expenditure_Constant' in adjusted_data.columns:
            adjusted_data.loc[valid_public_mask, 'Public_Health_Expenditure_Constant_PPP'] = (
                adjusted_data.loc[valid_public_mask, 'Public_Health_Expenditure_Constant'] / 
                adjusted_data.loc[valid_public_mask, 'Constant_PPP_Factor']
            )
        else:
            adjusted_data.loc[valid_public_mask, 'Public_Health_Expenditure_Constant_PPP'] = (
                adjusted_data.loc[valid_public_mask, 'Public_Health_Expenditure'] / 
                adjusted_data.loc[valid_public_mask, 'Constant_PPP_Factor']
            )
        
        adjusted_data.loc[valid_public_mask, 'Public_Health_Expenditure_per_Std_Capita_Constant_PPP'] = (
            adjusted_data.loc[valid_public_mask, 'Public_Health_Expenditure_Constant_PPP'] / 
            adjusted_data.loc[valid_public_mask, 'Standardized_Population']
        )
    
    # Adjust private health expenditure if available
    if 'Private_Health_Expenditure' in adjusted_data.columns:
        adjusted_data['Private_Health_Expenditure_Constant_PPP'] = np.nan
        adjusted_data['Private_Health_Expenditure_per_Std_Capita_Constant_PPP'] = np.nan
        
        valid_private_mask = valid_ppp_mask & ~adjusted_data['Private_Health_Expenditure'].isna()
        
        # Use constant price data if available
        if 'Private_Health_Expenditure_Constant' in adjusted_data.columns:
            adjusted_data.loc[valid_private_mask, 'Private_Health_Expenditure_Constant_PPP'] = (
                adjusted_data.loc[valid_private_mask, 'Private_Health_Expenditure_Constant'] / 
                adjusted_data.loc[valid_private_mask, 'Constant_PPP_Factor']
            )
        else:
            adjusted_data.loc[valid_private_mask, 'Private_Health_Expenditure_Constant_PPP'] = (
                adjusted_data.loc[valid_private_mask, 'Private_Health_Expenditure'] / 
                adjusted_data.loc[valid_private_mask, 'Constant_PPP_Factor']
            )
        
        adjusted_data.loc[valid_private_mask, 'Private_Health_Expenditure_per_Std_Capita_Constant_PPP'] = (
            adjusted_data.loc[valid_private_mask, 'Private_Health_Expenditure_Constant_PPP'] / 
            adjusted_data.loc[valid_private_mask, 'Standardized_Population']
        )
    
    # Drop the temporary column used for calculations
    adjusted_data = adjusted_data.drop(columns=['Constant_PPP_Factor'])
    
    return adjusted_data

def calculate_all_expenditure_indicators(ghed_data, standardized_pop, ppp_data=None, gdp_deflator=None, impute_missing_ppp=False, impute_missing_gdp=False, reference_year=2017, formula_type='israeli'):
    """
    Calculate health expenditure per standardized capita with all combinations of price and PPP adjustments:
    - Current prices, Current PPP
    - Current prices, Constant PPP
    - Constant prices, Current PPP
    - Constant prices, Constant PPP
    
    Args:
        ghed_data: DataFrame with GHED data (with ISO3 codes)
        standardized_pop: DataFrame with standardized population data (with ISO3 codes)
        ppp_data: DataFrame with PPP conversion factors (optional, with ISO3 codes)
        gdp_deflator: DataFrame with GDP deflators (optional, with ISO3 codes)
        impute_missing_ppp: Whether to impute missing PPP factors
        impute_missing_gdp: Whether to impute missing GDP deflators
        reference_year: Reference year for constant prices and PPP
        formula_type: Type of capitation formula used ('israeli', 'ltc', or 'eu27')
    
    Returns:
        DataFrame with all health expenditure indicators
    """
    print("Calculating comprehensive health expenditure indicators...")
    
    # Rename columns for consistency
    ghed_renamed = ghed_data.rename(columns={
        'location': 'Country_GHED',
        'year': 'Year',
        'che': 'Total_Health_Expenditure',
        'public_expenditure': 'Public_Health_Expenditure',
        'private_expenditure': 'Private_Health_Expenditure'
    })
    
    # Select only necessary columns from standardized_pop
    std_pop_selected = standardized_pop[['ISO3', 'Year', 'Standardized_Population']].copy()
    if 'Country' in standardized_pop.columns:
        std_pop_selected['Country_StdPop'] = standardized_pop['Country']
    
    # Merge data using ISO3 code and Year
    merged_data = pd.merge(
        ghed_renamed, 
        std_pop_selected,
        on=["ISO3", "Year"],
        how="inner"
    )
    
    # Create a single Country column
    if 'Country_GHED' in merged_data.columns and 'Country_StdPop' in merged_data.columns:
        merged_data['Country'] = merged_data['Country_GHED'].combine_first(merged_data['Country_StdPop'])
        merged_data = merged_data.drop(columns=['Country_GHED', 'Country_StdPop'])
    elif 'Country_GHED' in merged_data.columns:
        merged_data = merged_data.rename(columns={'Country_GHED': 'Country'})
    elif 'Country_StdPop' in merged_data.columns:
        merged_data = merged_data.rename(columns={'Country_StdPop': 'Country'})
    
    # Record the number of countries and years
    num_countries = merged_data['ISO3'].nunique()
    num_years = merged_data['Year'].nunique()
    print(f"Working with data for {num_countries} countries over {num_years} years")
    
    # STEP 1: Calculate base indicators with current prices (no adjustments yet)
    print("Step 1: Calculating base indicators with current prices...")
    
    # Calculate total health expenditure per standardized capita
    merged_data["THE_per_Std_Capita_Current"] = merged_data["Total_Health_Expenditure"] / merged_data["Standardized_Population"]
    
    # Calculate public and private expenditure per standardized capita if available
    if 'Public_Health_Expenditure' in merged_data.columns and not merged_data['Public_Health_Expenditure'].isna().all():
        merged_data["PubHE_per_Std_Capita_Current"] = merged_data["Public_Health_Expenditure"] / merged_data["Standardized_Population"]
    
    if 'Private_Health_Expenditure' in merged_data.columns and not merged_data['Private_Health_Expenditure'].isna().all():
        merged_data["PvtHE_per_Std_Capita_Current"] = merged_data["Private_Health_Expenditure"] / merged_data["Standardized_Population"]
    

    # STEP 2: Apply GDP deflator adjustment for constant prices
    if gdp_deflator is not None and not gdp_deflator.empty:
        print("Step 2: Applying GDP deflator adjustment for constant prices...")
        
        # Merge with GDP deflator data
        merged_gdp = pd.merge(
            merged_data,
            gdp_deflator[['ISO3', 'Year', 'GDP_Deflator']],
            on=['ISO3', 'Year'],
            how='left'
        )
        
        # Check if any GDP deflators are missing
        missing_deflator = merged_gdp['GDP_Deflator'].isna().sum()
        if missing_deflator > 0:
            print(f"Warning: Missing GDP deflators for {missing_deflator} out of {len(merged_gdp)} rows")
            
            # Impute missing GDP deflators if enabled
            if impute_missing_gdp:
                print(f"Imputing missing GDP deflators...")
                countries_with_missing = merged_gdp[merged_gdp['GDP_Deflator'].isna()]['ISO3'].unique()
                
                for iso3 in countries_with_missing:
                    country_data = merged_gdp[merged_gdp['ISO3'] == iso3]
                    missing_years = country_data[country_data['GDP_Deflator'].isna()]['Year'].tolist()
                    
                    # If country has any deflator data, use nearest year
                    if any(~country_data['GDP_Deflator'].isna()):
                        available_years = country_data[~country_data['GDP_Deflator'].isna()]['Year'].tolist()
                        
                        for missing_year in missing_years:
                            # Find closest available year
                            closest_year = min(available_years, key=lambda x: abs(x - missing_year))
                            closest_deflator = country_data[country_data['Year'] == closest_year]['GDP_Deflator'].iloc[0]
                            
                            # Impute the missing value
                            idx = merged_gdp[(merged_gdp['ISO3'] == iso3) & (merged_gdp['Year'] == missing_year)].index
                            merged_gdp.loc[idx, 'GDP_Deflator'] = closest_deflator
                            
                            country_name = merged_gdp.loc[merged_gdp['ISO3'] == iso3, 'Country'].iloc[0]
                            print(f"Imputed GDP deflator for {country_name} ({iso3}) in year {missing_year} using data from {closest_year}")
        
        # Create mask for valid deflator values
        valid_deflator_mask = ~merged_gdp['GDP_Deflator'].isna()
        
        # Calculate constant price values for Total Health Expenditure
        merged_gdp['THE_Constant'] = np.nan
        merged_gdp['THE_per_Std_Capita_Constant'] = np.nan
        
        merged_gdp.loc[valid_deflator_mask, 'THE_Constant'] = (
            merged_gdp.loc[valid_deflator_mask, 'Total_Health_Expenditure'] / 
            merged_gdp.loc[valid_deflator_mask, 'GDP_Deflator']
        )
        
        merged_gdp.loc[valid_deflator_mask, 'THE_per_Std_Capita_Constant'] = (
            merged_gdp.loc[valid_deflator_mask, 'THE_Constant'] / 
            merged_gdp.loc[valid_deflator_mask, 'Standardized_Population']
        )
        
        # Calculate constant price values for Public Health Expenditure if available
        if 'Public_Health_Expenditure' in merged_gdp.columns:
            merged_gdp['PubHE_Constant'] = np.nan
            merged_gdp['PubHE_per_Std_Capita_Constant'] = np.nan
            
            valid_public_mask = valid_deflator_mask & ~merged_gdp['Public_Health_Expenditure'].isna()
            
            merged_gdp.loc[valid_public_mask, 'PubHE_Constant'] = (
                merged_gdp.loc[valid_public_mask, 'Public_Health_Expenditure'] / 
                merged_gdp.loc[valid_public_mask, 'GDP_Deflator']
            )
            
            merged_gdp.loc[valid_public_mask, 'PubHE_per_Std_Capita_Constant'] = (
                merged_gdp.loc[valid_public_mask, 'PubHE_Constant'] / 
                merged_gdp.loc[valid_public_mask, 'Standardized_Population']
            )
        
        # Calculate constant price values for Private Health Expenditure if available
        if 'Private_Health_Expenditure' in merged_gdp.columns:
            merged_gdp['PvtHE_Constant'] = np.nan
            merged_gdp['PvtHE_per_Std_Capita_Constant'] = np.nan
            
            valid_private_mask = valid_deflator_mask & ~merged_gdp['Private_Health_Expenditure'].isna()
            
            merged_gdp.loc[valid_private_mask, 'PvtHE_Constant'] = (
                merged_gdp.loc[valid_private_mask, 'Private_Health_Expenditure'] / 
                merged_gdp.loc[valid_private_mask, 'GDP_Deflator']
            )
            
            merged_gdp.loc[valid_private_mask, 'PvtHE_per_Std_Capita_Constant'] = (
                merged_gdp.loc[valid_private_mask, 'PvtHE_Constant'] / 
                merged_gdp.loc[valid_private_mask, 'Standardized_Population']
            )
        
        # Update merged_data with constant price calculations
        merged_data = merged_gdp
    else:
        print("Step 2: Skipping GDP deflator adjustment as no data is available")
    
    
    # STEP 3: Apply current-year PPP adjustment
    if ppp_data is not None and not ppp_data.empty:
        print("Step 3: Applying current-year PPP adjustment...")
        
        # Merge with PPP data
        merged_ppp = pd.merge(
            merged_data,
            ppp_data[['ISO3', 'Year', 'PPP_Factor']],
            on=['ISO3', 'Year'],
            how='left'
        )
        
        # Check if any PPP factors are missing
        missing_ppp = merged_ppp['PPP_Factor'].isna().sum()
        if missing_ppp > 0:
            print(f"Warning: Missing PPP factors for {missing_ppp} out of {len(merged_ppp)} rows")
            
            # Impute missing PPP factors if enabled
            if impute_missing_ppp:
                print(f"Imputing missing PPP factors...")
                countries_with_missing = merged_ppp[merged_ppp['PPP_Factor'].isna()]['ISO3'].unique()
                
                for iso3 in countries_with_missing:
                    country_data = merged_ppp[merged_ppp['ISO3'] == iso3]
                    missing_years = country_data[country_data['PPP_Factor'].isna()]['Year'].tolist()
                    
                    # If country has any PPP data, use nearest year
                    if any(~country_data['PPP_Factor'].isna()):
                        available_years = country_data[~country_data['PPP_Factor'].isna()]['Year'].tolist()
                        
                        for missing_year in missing_years:
                            # Find closest available year
                            closest_year = min(available_years, key=lambda x: abs(x - missing_year))
                            closest_ppp = country_data[country_data['Year'] == closest_year]['PPP_Factor'].iloc[0]
                            
                            # Impute the missing value
                            idx = merged_ppp[(merged_ppp['ISO3'] == iso3) & (merged_ppp['Year'] == missing_year)].index
                            merged_ppp.loc[idx, 'PPP_Factor'] = closest_ppp
                            
                            country_name = merged_ppp.loc[merged_ppp['ISO3'] == iso3, 'Country'].iloc[0]
                            print(f"Imputed PPP factor for {country_name} ({iso3}) in year {missing_year} using data from {closest_year}")
        
        # Create mask for valid PPP factors
        valid_ppp_mask = ~merged_ppp['PPP_Factor'].isna()
        
        # Calculate current PPP values for current price Total Health Expenditure
        merged_ppp['THE_CurrentPPP'] = np.nan
        merged_ppp['THE_per_Std_Capita_CurrentPPP'] = np.nan
        
        merged_ppp.loc[valid_ppp_mask, 'THE_CurrentPPP'] = (
            merged_ppp.loc[valid_ppp_mask, 'Total_Health_Expenditure'] / 
            merged_ppp.loc[valid_ppp_mask, 'PPP_Factor']
        )
        
        merged_ppp.loc[valid_ppp_mask, 'THE_per_Std_Capita_CurrentPPP'] = (
            merged_ppp.loc[valid_ppp_mask, 'THE_CurrentPPP'] / 
            merged_ppp.loc[valid_ppp_mask, 'Standardized_Population']
        )
        
        # Calculate current PPP values for current price Public Health Expenditure if available
        if 'Public_Health_Expenditure' in merged_ppp.columns:
            merged_ppp['PubHE_CurrentPPP'] = np.nan
            merged_ppp['PubHE_per_Std_Capita_CurrentPPP'] = np.nan
            
            valid_public_mask = valid_ppp_mask & ~merged_ppp['Public_Health_Expenditure'].isna()
            
            merged_ppp.loc[valid_public_mask, 'PubHE_CurrentPPP'] = (
                merged_ppp.loc[valid_public_mask, 'Public_Health_Expenditure'] / 
                merged_ppp.loc[valid_public_mask, 'PPP_Factor']
            )
            
            merged_ppp.loc[valid_public_mask, 'PubHE_per_Std_Capita_CurrentPPP'] = (
                merged_ppp.loc[valid_public_mask, 'PubHE_CurrentPPP'] / 
                merged_ppp.loc[valid_public_mask, 'Standardized_Population']
            )
        
        # Calculate current PPP values for current price Private Health Expenditure if available
        if 'Private_Health_Expenditure' in merged_ppp.columns:
            merged_ppp['PvtHE_CurrentPPP'] = np.nan
            merged_ppp['PvtHE_per_Std_Capita_CurrentPPP'] = np.nan
            
            valid_private_mask = valid_ppp_mask & ~merged_ppp['Private_Health_Expenditure'].isna()
            
            merged_ppp.loc[valid_private_mask, 'PvtHE_CurrentPPP'] = (
                merged_ppp.loc[valid_private_mask, 'Private_Health_Expenditure'] / 
                merged_ppp.loc[valid_private_mask, 'PPP_Factor']
            )
            
            merged_ppp.loc[valid_private_mask, 'PvtHE_per_Std_Capita_CurrentPPP'] = (
                merged_ppp.loc[valid_private_mask, 'PvtHE_CurrentPPP'] / 
                merged_ppp.loc[valid_private_mask, 'Standardized_Population']
            )
        
        # If constant price data is available, also apply current PPP
        if 'THE_Constant' in merged_ppp.columns:
            merged_ppp['THE_Constant_CurrentPPP'] = np.nan
            merged_ppp['THE_per_Std_Capita_Constant_CurrentPPP'] = np.nan
            
            merged_ppp.loc[valid_ppp_mask, 'THE_Constant_CurrentPPP'] = (
                merged_ppp.loc[valid_ppp_mask, 'THE_Constant'] / 
                merged_ppp.loc[valid_ppp_mask, 'PPP_Factor']
            )
            
            merged_ppp.loc[valid_ppp_mask, 'THE_per_Std_Capita_Constant_CurrentPPP'] = (
                merged_ppp.loc[valid_ppp_mask, 'THE_Constant_CurrentPPP'] / 
                merged_ppp.loc[valid_ppp_mask, 'Standardized_Population']
            )
        
        if 'PubHE_Constant' in merged_ppp.columns:
            merged_ppp['PubHE_Constant_CurrentPPP'] = np.nan
            merged_ppp['PubHE_per_Std_Capita_Constant_CurrentPPP'] = np.nan
            
            valid_public_const_mask = valid_ppp_mask & ~merged_ppp['PubHE_Constant'].isna()
            
            merged_ppp.loc[valid_public_const_mask, 'PubHE_Constant_CurrentPPP'] = (
                merged_ppp.loc[valid_public_const_mask, 'PubHE_Constant'] / 
                merged_ppp.loc[valid_public_const_mask, 'PPP_Factor']
            )
            
            merged_ppp.loc[valid_public_const_mask, 'PubHE_per_Std_Capita_Constant_CurrentPPP'] = (
                merged_ppp.loc[valid_public_const_mask, 'PubHE_Constant_CurrentPPP'] / 
                merged_ppp.loc[valid_public_const_mask, 'Standardized_Population']
            )
        
        if 'PvtHE_Constant' in merged_ppp.columns:
            merged_ppp['PvtHE_Constant_CurrentPPP'] = np.nan
            merged_ppp['PvtHE_per_Std_Capita_Constant_CurrentPPP'] = np.nan
            
            valid_private_const_mask = valid_ppp_mask & ~merged_ppp['PvtHE_Constant'].isna()
            
            merged_ppp.loc[valid_private_const_mask, 'PvtHE_Constant_CurrentPPP'] = (
                merged_ppp.loc[valid_private_const_mask, 'PvtHE_Constant'] / 
                merged_ppp.loc[valid_private_const_mask, 'PPP_Factor']
            )
            
            merged_ppp.loc[valid_private_const_mask, 'PvtHE_per_Std_Capita_Constant_CurrentPPP'] = (
                merged_ppp.loc[valid_private_const_mask, 'PvtHE_Constant_CurrentPPP'] / 
                merged_ppp.loc[valid_private_const_mask, 'Standardized_Population']
            )
        
        # Update merged_data with current PPP calculations
        merged_data = merged_ppp
        
    else:
        print("Step 3: Skipping current-year PPP adjustment as no data is available")
    
    
    # STEP 4: Apply constant (reference year) PPP adjustment
    if ppp_data is not None and not ppp_data.empty:
        print(f"Step 4: Applying constant PPP adjustment using reference year {reference_year}...")
        
        # Filter PPP data to only include the reference year
        ref_ppp_data = ppp_data[ppp_data['Year'] == reference_year].copy()
        
        if ref_ppp_data.empty:
            print(f"Warning: No PPP data available for reference year {reference_year}")
            print("Looking for nearest available year...")
            
            # Find the nearest available year to the reference year
            all_years = sorted(ppp_data['Year'].unique())
            if not all_years:
                print("Error: No PPP data available at all")
            else:
                nearest_year = min(all_years, key=lambda x: abs(x - reference_year))
                print(f"Using PPP data from {nearest_year} as reference (closest to {reference_year})")
                ref_ppp_data = ppp_data[ppp_data['Year'] == nearest_year].copy()
        
        if not ref_ppp_data.empty:
            # Create a mapping of ISO3 -> PPP_Factor from the reference year
            ppp_mapping = ref_ppp_data[['ISO3', 'PPP_Factor']].drop_duplicates('ISO3').set_index('ISO3')['PPP_Factor'].to_dict()
            
            # Add a column with constant PPP factors to the data
            merged_data['Constant_PPP_Factor'] = merged_data['ISO3'].map(ppp_mapping)
            
            # Check if any constant PPP factors are missing
            missing_const_ppp = merged_data['Constant_PPP_Factor'].isna().sum()
            if missing_const_ppp > 0:
                print(f"Warning: Missing constant PPP factors for {missing_const_ppp} out of {len(merged_data)} rows")
                
                if impute_missing_ppp:
                    print("Attempting to impute missing constant PPP factors...")
                    
                    # For each ISO3 with missing values, try to find PPP data for any year
                    for iso3 in merged_data[merged_data['Constant_PPP_Factor'].isna()]['ISO3'].unique():
                        # Get all PPP data for this country
                        country_ppp = ppp_data[ppp_data['ISO3'] == iso3]
                        
                        if not country_ppp.empty:
                            # Get the closest year to reference year
                            closest_year = min(country_ppp['Year'].unique(), key=lambda x: abs(x - reference_year))
                            closest_ppp = country_ppp[country_ppp['Year'] == closest_year]['PPP_Factor'].iloc[0]
                            
                            # Apply this PPP factor to all rows with this ISO3
                            merged_data.loc[merged_data['ISO3'] == iso3, 'Constant_PPP_Factor'] = closest_ppp
                            
                            country_name = merged_data.loc[merged_data['ISO3'] == iso3, 'Country'].iloc[0]
                            print(f"Imputed constant PPP factor for {country_name} ({iso3}) using data from {closest_year}")
            
            # Create a mask for rows with valid constant PPP factors
            valid_const_ppp_mask = ~merged_data['Constant_PPP_Factor'].isna()
            
            # Apply constant PPP adjustment to current price values
            merged_data['THE_ConstantPPP'] = np.nan
            merged_data['THE_per_Std_Capita_ConstantPPP'] = np.nan
            
            merged_data.loc[valid_const_ppp_mask, 'THE_ConstantPPP'] = (
                merged_data.loc[valid_const_ppp_mask, 'Total_Health_Expenditure'] / 
                merged_data.loc[valid_const_ppp_mask, 'Constant_PPP_Factor']
            )
            
            merged_data.loc[valid_const_ppp_mask, 'THE_per_Std_Capita_ConstantPPP'] = (
                merged_data.loc[valid_const_ppp_mask, 'THE_ConstantPPP'] / 
                merged_data.loc[valid_const_ppp_mask, 'Standardized_Population']
            )
            
            # Apply constant PPP adjustment to current price public health expenditure if available
            if 'Public_Health_Expenditure' in merged_data.columns:
                merged_data['PubHE_ConstantPPP'] = np.nan
                merged_data['PubHE_per_Std_Capita_ConstantPPP'] = np.nan
                
                valid_public_mask = valid_const_ppp_mask & ~merged_data['Public_Health_Expenditure'].isna()
                
                merged_data.loc[valid_public_mask, 'PubHE_ConstantPPP'] = (
                    merged_data.loc[valid_public_mask, 'Public_Health_Expenditure'] / 
                    merged_data.loc[valid_public_mask, 'Constant_PPP_Factor']
                )
                
                merged_data.loc[valid_public_mask, 'PubHE_per_Std_Capita_ConstantPPP'] = (
                    merged_data.loc[valid_public_mask, 'PubHE_ConstantPPP'] / 
                    merged_data.loc[valid_public_mask, 'Standardized_Population']
                )
            
            # Apply constant PPP adjustment to current price private health expenditure if available
            if 'Private_Health_Expenditure' in merged_data.columns:
                merged_data['PvtHE_ConstantPPP'] = np.nan
                merged_data['PvtHE_per_Std_Capita_ConstantPPP'] = np.nan
                
                valid_private_mask = valid_const_ppp_mask & ~merged_data['Private_Health_Expenditure'].isna()
                
                merged_data.loc[valid_private_mask, 'PvtHE_ConstantPPP'] = (
                    merged_data.loc[valid_private_mask, 'Private_Health_Expenditure'] / 
                    merged_data.loc[valid_private_mask, 'Constant_PPP_Factor']
                )
                
                merged_data.loc[valid_private_mask, 'PvtHE_per_Std_Capita_ConstantPPP'] = (
                    merged_data.loc[valid_private_mask, 'PvtHE_ConstantPPP'] / 
                    merged_data.loc[valid_private_mask, 'Standardized_Population']
                )
            
            # Apply constant PPP adjustment to constant price values if available
            if 'THE_Constant' in merged_data.columns:
                merged_data['THE_Constant_ConstantPPP'] = np.nan
                merged_data['THE_per_Std_Capita_Constant_ConstantPPP'] = np.nan
                
                valid_const_mask = valid_const_ppp_mask & ~merged_data['THE_Constant'].isna()
                
                merged_data.loc[valid_const_mask, 'THE_Constant_ConstantPPP'] = (
                    merged_data.loc[valid_const_mask, 'THE_Constant'] / 
                    merged_data.loc[valid_const_mask, 'Constant_PPP_Factor']
                )
                
                merged_data.loc[valid_const_mask, 'THE_per_Std_Capita_Constant_ConstantPPP'] = (
                    merged_data.loc[valid_const_mask, 'THE_Constant_ConstantPPP'] / 
                    merged_data.loc[valid_const_mask, 'Standardized_Population']
                )
            
            if 'PubHE_Constant' in merged_data.columns:
                merged_data['PubHE_Constant_ConstantPPP'] = np.nan
                merged_data['PubHE_per_Std_Capita_Constant_ConstantPPP'] = np.nan
                
                valid_public_const_mask = valid_const_ppp_mask & ~merged_data['PubHE_Constant'].isna()
                
                merged_data.loc[valid_public_const_mask, 'PubHE_Constant_ConstantPPP'] = (
                    merged_data.loc[valid_public_const_mask, 'PubHE_Constant'] / 
                    merged_data.loc[valid_public_const_mask, 'Constant_PPP_Factor']
                )
                
                merged_data.loc[valid_public_const_mask, 'PubHE_per_Std_Capita_Constant_ConstantPPP'] = (
                    merged_data.loc[valid_public_const_mask, 'PubHE_Constant_ConstantPPP'] / 
                    merged_data.loc[valid_public_const_mask, 'Standardized_Population']
                )
            
            if 'PvtHE_Constant' in merged_data.columns:
                merged_data['PvtHE_Constant_ConstantPPP'] = np.nan
                merged_data['PvtHE_per_Std_Capita_Constant_ConstantPPP'] = np.nan
                
                valid_private_const_mask = valid_const_ppp_mask & ~merged_data['PvtHE_Constant'].isna()
                
                merged_data.loc[valid_private_const_mask, 'PvtHE_Constant_ConstantPPP'] = (
                    merged_data.loc[valid_private_const_mask, 'PvtHE_Constant'] / 
                    merged_data.loc[valid_private_const_mask, 'Constant_PPP_Factor']
                )
                
                merged_data.loc[valid_private_const_mask, 'PvtHE_per_Std_Capita_Constant_ConstantPPP'] = (
                    merged_data.loc[valid_private_const_mask, 'PvtHE_Constant_ConstantPPP'] / 
                    merged_data.loc[valid_private_const_mask, 'Standardized_Population']
                )
            
            # Drop the temporary column used for calculations
            merged_data = merged_data.drop(columns=['Constant_PPP_Factor'])
        else:
            print("Step 4: Skipping constant PPP adjustment as no reference year data is available")
    else:
        print("Step 4: Skipping constant PPP adjustment as no PPP data is available")
    
    # List all columns that were calculated
    print("\nGenerated indicators:")
    indicator_columns = [col for col in merged_data.columns if any(x in col for x in ['per_Std_Capita', 'CurrentPPP', 'ConstantPPP'])]
    for col in sorted(indicator_columns):
        print(f"- {col}")
    
    return merged_data

def document_imputation(data, ppp_data, gdp_deflator, output_path, impute_missing_ppp=True, impute_missing_gdp=True):
    """
    Document where data imputation is used and generate a report.
    
    Args:
        data: DataFrame with health expenditure data (with ISO3 codes)
        ppp_data: DataFrame with PPP conversion factors (with ISO3 codes)
        gdp_deflator: DataFrame with GDP deflators (with ISO3 codes)
        output_path: Path where to save the imputation report
        impute_missing_ppp: Whether PPP imputation is enabled
        impute_missing_gdp: Whether GDP deflator imputation is enabled
    
    Returns:
        DataFrame with imputation documentation
    """
    print("Documenting data imputation...")
    
    # Initialize a list to store imputation records
    imputation_records = []
    
    # Check PPP imputation
    if impute_missing_ppp and ppp_data is not None and not ppp_data.empty:
        # Merge health expenditure data with PPP data to identify missing values using ISO3
        merged_ppp = pd.merge(
            data[['ISO3', 'Country', 'Year']],
            ppp_data[['ISO3', 'Year', 'PPP_Factor']],
            on=['ISO3', 'Year'],
            how='left'
        )
        
        # Identify rows with missing PPP factors
        missing_ppp = merged_ppp[merged_ppp['PPP_Factor'].isna()]
        
        # Document each case of PPP imputation
        for iso3 in missing_ppp['ISO3'].unique():
            country_missing = missing_ppp[missing_ppp['ISO3'] == iso3]
            missing_years = country_missing['Year'].tolist()
            country_name = country_missing['Country'].iloc[0] if not country_missing.empty and 'Country' in country_missing.columns else "Unknown"
            
            # Find available PPP data for this country using ISO3
            country_available = ppp_data[ppp_data['ISO3'] == iso3]
            
            if not country_available.empty:
                available_years = country_available['Year'].tolist()
                
                for missing_year in missing_years:
                    # Find closest available year for imputation
                    if available_years:
                        closest_year = min(available_years, key=lambda x: abs(x - missing_year))
                        imputation_records.append({
                            'ISO3': iso3,
                            'Country': country_name,
                            'Year': missing_year,
                            'Missing_Data': 'PPP Factor',
                            'Imputation_Method': f'Nearest year ({closest_year})',
                            'Value_Source': f'PPP Factor from {closest_year}'
                        })
                    else:
                        # No data available for this country
                        imputation_records.append({
                            'ISO3': iso3,
                            'Country': country_name,
                            'Year': missing_year,
                            'Missing_Data': 'PPP Factor',
                            'Imputation_Method': 'None - No data available',
                            'Value_Source': 'Missing'
                        })
    
    # Check GDP deflator imputation
    if impute_missing_gdp and gdp_deflator is not None and not gdp_deflator.empty:
        # Merge health expenditure data with GDP deflator data to identify missing values
        merged_gdp = pd.merge(
            data[['ISO3', 'Country', 'Year']],
            gdp_deflator[['ISO3', 'Year', 'GDP_Deflator']],
            on=['ISO3', 'Year'],
            how='left'
        )
        
        # Identify rows with missing GDP deflators
        missing_gdp = merged_gdp[merged_gdp['GDP_Deflator'].isna()]
        
        # Document each case of GDP deflator imputation
        for iso3 in missing_gdp['ISO3'].unique():
            country_missing = missing_gdp[missing_gdp['ISO3'] == iso3]
            missing_years = country_missing['Year'].tolist()
            country_name = country_missing['Country'].iloc[0] if not country_missing.empty and 'Country' in country_missing.columns else "Unknown"
            
            # Find available GDP deflator data for this country
            country_available = gdp_deflator[gdp_deflator['ISO3'] == iso3]
            
            if not country_available.empty:
                available_years = country_available['Year'].tolist()
                
                for missing_year in missing_years:
                    # Find closest available year for imputation
                    if available_years:
                        closest_year = min(available_years, key=lambda x: abs(x - missing_year))
                        imputation_records.append({
                            'ISO3': iso3,
                            'Country': country_name,
                            'Year': missing_year,
                            'Missing_Data': 'GDP Deflator',
                            'Imputation_Method': f'Nearest year ({closest_year})',
                            'Value_Source': f'GDP Deflator from {closest_year}'
                        })
                    else:
                        # No data available for this country
                        imputation_records.append({
                            'ISO3': iso3,
                            'Country': country_name,
                            'Year': missing_year,
                            'Missing_Data': 'GDP Deflator',
                            'Imputation_Method': 'None - No data available',
                            'Value_Source': 'Missing'
                        })
    
    # Convert to DataFrame
    imputation_df = pd.DataFrame(imputation_records)
    
    # Sort by ISO3, year, and type of missing data
    if not imputation_df.empty:
        imputation_df = imputation_df.sort_values(['ISO3', 'Year', 'Missing_Data'])
        
        # Save to CSV
        imputation_filename = "imputation_documentation.csv"
        if output_path is not None:
            imputation_df.to_csv(output_path / imputation_filename, index=False)
            print(f"Imputation documentation saved to {output_path / imputation_filename}")
        
        # Create summary statistics
        print("\nImputation Summary:")
        print(f"Total records with imputation: {len(imputation_df)}")
        
        # Count by missing data type
        if 'Missing_Data' in imputation_df.columns:
            type_counts = imputation_df['Missing_Data'].value_counts()
            print("\nImputation by type:")
            for data_type, count in type_counts.items():
                print(f"  {data_type}: {count} records")
        
        # Count by country
        country_counts = imputation_df.groupby(['ISO3', 'Country']).size().reset_index(name='Count')
        country_counts = country_counts.sort_values('Count', ascending=False)
        print("\nTop 10 countries with most imputations:")
        for _, row in country_counts.head(10).iterrows():
            print(f"  {row['Country']} ({row['ISO3']}): {row['Count']} records")
    else:
        print("No imputation was performed or documented.")
    
    return imputation_df

def create_iso3_to_country_mapping(iso3_data, iso3_column='ISO3', country_column='Country'):
    """
    Create a mapping dictionary from ISO3 codes to country names.
    
    Args:
        iso3_data: DataFrame containing ISO3 codes and country names
        iso3_column: Column name for ISO3 codes
        country_column: Column name for country names
    
    Returns:
        Dictionary mapping ISO3 codes to country names
    """
    if iso3_column not in iso3_data.columns or country_column not in iso3_data.columns:
        print(f"Warning: Cannot create ISO3 mapping, columns {iso3_column} or {country_column} not found")
        return {}
    
    # Remove rows with missing values
    valid_data = iso3_data.dropna(subset=[iso3_column, country_column])
    
    # Create the mapping dictionary
    iso3_to_country = {}
    
    for _, row in valid_data.iterrows():
        iso3 = row[iso3_column]
        country = row[country_column]
        
        # Only add if both values are not empty and ISO3 is a string
        if pd.notna(iso3) and pd.notna(country) and isinstance(iso3, str):
            iso3_to_country[iso3] = country
    
    print(f"Created mapping with {len(iso3_to_country)} ISO3 codes to country names")
    return iso3_to_country

def merge_using_iso3(left_df, right_df, left_on='ISO3', right_on='ISO3', how='inner'):
    """
    Merge two DataFrames using ISO3 codes, and add helpful diagnostics.
    
    Args:
        left_df: Left DataFrame
        right_df: Right DataFrame
        left_on: Column name in left DataFrame (default: 'ISO3')
        right_on: Column name in right DataFrame (default: 'ISO3')
        how: Type of merge to perform (default: 'inner')
    
    Returns:
        Merged DataFrame
    """
    # Count unique ISO3 codes before merge
    left_iso3_count = left_df[left_on].nunique()
    right_iso3_count = right_df[right_on].nunique()
    
    # Perform merge
    merged = pd.merge(left_df, right_df, left_on=left_on, right_on=right_on, how=how)
    
    # Count unique ISO3 codes after merge
    merged_iso3_count = merged[left_on].nunique()
    
    # Print diagnostics
    print(f"Merge using ISO3 codes:")
    print(f"  Left DataFrame: {left_iso3_count} unique ISO3 codes")
    print(f"  Right DataFrame: {right_iso3_count} unique ISO3 codes")
    print(f"  Merged DataFrame: {merged_iso3_count} unique ISO3 codes")
    
    # Calculate and report missing matches
    if how == 'inner':
        missing_left = left_iso3_count - merged_iso3_count
        missing_right = right_iso3_count - merged_iso3_count
        print(f"  Missing matches: {missing_left} from left, {missing_right} from right")
        
        # Show examples of unmatched ISO3 codes
        if missing_left > 0:
            left_only = set(left_df[left_on].unique()) - set(merged[left_on].unique())
            print(f"  Examples of unmatched ISO3 codes from left: {list(left_only)[:5]}")
        
        if missing_right > 0:
            right_only = set(right_df[right_on].unique()) - set(merged[right_on].unique())
            print(f"  Examples of unmatched ISO3 codes from right: {list(right_only)[:5]}")
    
    return merged

def main():
    """Main function to run the script."""
    print("Starting Comprehensive Health Expenditure calculation with all indicator combinations...")
    
    # Define reference year and adjustment settings
    reference_year = REFERENCE_YEAR  # Uses the global constant
    impute_ppp = False  # Set to True to enable PPP imputation
    impute_gdp = False  # Set to True to enable GDP deflator imputation
    
    # Capitation formula to use
    formula = 'ltc'  # Options: 'israeli', 'ltc', 'eu27'
    
    # Export path
    export_path = Path("Standardized_Expenditure")
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Load capitation weights - important to capture both return values
    cap_dict, formula_type = load_capitation_weights(formula=formula)
    
    # Load data
    try:
        # Load GHED data with split public/private expenditure
        ghed_data = load_ghed_data()
        print(f"\nGHED data sample:\n{ghed_data.head()}")
        
        # Load PPP data from World Bank
        ppp_data = load_ppp_data()
        if not ppp_data.empty:
            print(f"\nPPP data sample:\n{ppp_data.head()}")
        else:
            print("\nWarning: No PPP data loaded, will proceed without PPP adjustment")
        
        # Load GDP data for deflator calculation
        gdp_deflator = load_gdp_data(reference_year=reference_year)
        if not gdp_deflator.empty:
            print(f"\nGDP deflator data sample:\n{gdp_deflator.head()}")
            print(f"Reference year for constant prices: {reference_year}")
        else:
            print("\nWarning: No GDP deflator data loaded, will proceed without constant price adjustment")
        
        # Load and process population data
        male_pop_raw, female_pop_raw = load_population_data()
        
        # Create ISO3 to country mapping for reference
        iso3_to_country = {}
        if not male_pop_raw.empty and 'ISO3' in male_pop_raw.columns and 'Country' in male_pop_raw.columns:
            iso3_to_country = create_iso3_to_country_mapping(male_pop_raw)
            print(f"Created ISO3 to country mapping with {len(iso3_to_country)} entries")
        
        # Calculate standardized population - use formula_type from the loaded weights
        standardized_pop = preprocess_population_data(male_pop_raw, female_pop_raw, cap_dict, formula_type=formula_type)
        
        # Document imputation process before applying adjustments (if imputation is enabled)
        if impute_ppp or impute_gdp:
            # Create base data for imputation documentation
            base_data = ghed_data.rename(columns={
                'location': 'Country',
                'year': 'Year',
                'che': 'Total_Health_Expenditure',
                'public_expenditure': 'Public_Health_Expenditure',
                'private_expenditure': 'Private_Health_Expenditure'
            })
            
            base_data = merge_using_iso3(
                base_data, 
                standardized_pop,
                left_on="ISO3",
                right_on="ISO3",
                how="inner"
            )
            
            document_imputation(
                base_data,
                ppp_data,
                gdp_deflator,
                export_path,
                impute_missing_ppp=impute_ppp,
                impute_missing_gdp=impute_gdp
            )
        
        # Calculate all health expenditure indicators (comprehensive approach)
        results = calculate_all_expenditure_indicators(
            ghed_data, 
            standardized_pop, 
            ppp_data=ppp_data, 
            gdp_deflator=gdp_deflator,
            impute_missing_ppp=impute_ppp,
            impute_missing_gdp=impute_gdp,
            reference_year=reference_year,
            formula_type=formula_type 
        )       
        # Create a more descriptive filename
        filename = f"Health_Expenditure_Comprehensive_{formula_type}_ref{reference_year}"
        
        if impute_ppp or impute_gdp:
            # Add suffix to filename to indicate imputation settings
            imputation_status = []
            if impute_ppp:
                imputation_status.append("ppp_imputed")
            if impute_gdp:
                imputation_status.append("gdp_imputed")
            
            if imputation_status:
                filename += "_" + "_".join(imputation_status)
        
        filename += ".csv"
        
        # Create a data dictionary to document column meanings
        column_descriptions = {
            # Base metrics
            "ISO3": "ISO3 country code",
            "Country": "Country name",
            "Year": "Year of data",
            "Standardized_Population": "Population standardized using capitation formula",
            "Total_Health_Expenditure": "Total health expenditure in local currency units (LCU)",
            "Public_Health_Expenditure": "Public health expenditure in LCU",
            "Private_Health_Expenditure": "Private health expenditure in LCU",
            
            # Current prices, no PPP adjustment
            "THE_per_Std_Capita_Current": "Total health expenditure per standardized capita (current LCU)",
            "PubHE_per_Std_Capita_Current": "Public health expenditure per standardized capita (current LCU)",
            "PvtHE_per_Std_Capita_Current": "Private health expenditure per standardized capita (current LCU)",
            
            # Constant prices, no PPP adjustment
            "THE_Constant": f"Total health expenditure in constant {reference_year} LCU",
            "THE_per_Std_Capita_Constant": f"Total health expenditure per standardized capita (constant {reference_year} LCU)",
            "PubHE_Constant": f"Public health expenditure in constant {reference_year} LCU",
            "PubHE_per_Std_Capita_Constant": f"Public health expenditure per standardized capita (constant {reference_year} LCU)",
            "PvtHE_Constant": f"Private health expenditure in constant {reference_year} LCU", 
            "PvtHE_per_Std_Capita_Constant": f"Private health expenditure per standardized capita (constant {reference_year} LCU)",
            
            # Current prices, current PPP
            "THE_CurrentPPP": "Total health expenditure in current international $ (current PPP)",
            "THE_per_Std_Capita_CurrentPPP": "Total health expenditure per standardized capita (current international $, current PPP)",
            "PubHE_CurrentPPP": "Public health expenditure in current international $ (current PPP)",
            "PubHE_per_Std_Capita_CurrentPPP": "Public health expenditure per standardized capita (current international $, current PPP)",
            "PvtHE_CurrentPPP": "Private health expenditure in current international $ (current PPP)",
            "PvtHE_per_Std_Capita_CurrentPPP": "Private health expenditure per standardized capita (current international $, current PPP)",
            
            # Constant prices, current PPP
            "THE_Constant_CurrentPPP": f"Total health expenditure in constant {reference_year} international $ (current PPP)",
            "THE_per_Std_Capita_Constant_CurrentPPP": f"Total health expenditure per standardized capita (constant {reference_year} international $, current PPP)",
            "PubHE_Constant_CurrentPPP": f"Public health expenditure in constant {reference_year} international $ (current PPP)",
            "PubHE_per_Std_Capita_Constant_CurrentPPP": f"Public health expenditure per standardized capita (constant {reference_year} international $, current PPP)",
            "PvtHE_Constant_CurrentPPP": f"Private health expenditure in constant {reference_year} international $ (current PPP)",
            "PvtHE_per_Std_Capita_Constant_CurrentPPP": f"Private health expenditure per standardized capita (constant {reference_year} international $, current PPP)",
            
            # Current prices, constant PPP
            "THE_ConstantPPP": f"Total health expenditure in current LCU converted to international $ using {reference_year} PPP factors",
            "THE_per_Std_Capita_ConstantPPP": f"Total health expenditure per standardized capita (current LCU converted to international $ using {reference_year} PPP factors)",
            "PubHE_ConstantPPP": f"Public health expenditure in current LCU converted to international $ using {reference_year} PPP factors",
            "PubHE_per_Std_Capita_ConstantPPP": f"Public health expenditure per standardized capita (current LCU converted to international $ using {reference_year} PPP factors)",
            "PvtHE_ConstantPPP": f"Private health expenditure in current LCU converted to international $ using {reference_year} PPP factors",
            "PvtHE_per_Std_Capita_ConstantPPP": f"Private health expenditure per standardized capita (current LCU converted to international $ using {reference_year} PPP factors)",
            
            # Constant prices, constant PPP (RECOMMENDED FOR TIME SERIES COMPARISON)
            "THE_Constant_ConstantPPP": f"Total health expenditure in constant {reference_year} LCU converted to international $ using {reference_year} PPP factors",
            "THE_per_Std_Capita_Constant_ConstantPPP": f"Total health expenditure per standardized capita (constant {reference_year} LCU converted to international $ using {reference_year} PPP factors)",
            "PubHE_Constant_ConstantPPP": f"Public health expenditure in constant {reference_year} LCU converted to international $ using {reference_year} PPP factors",
            "PubHE_per_Std_Capita_Constant_ConstantPPP": f"Public health expenditure per standardized capita (constant {reference_year} LCU converted to international $ using {reference_year} PPP factors)",
            "PvtHE_Constant_ConstantPPP": f"Private health expenditure in constant {reference_year} LCU converted to international $ using {reference_year} PPP factors",
            "PvtHE_per_Std_Capita_Constant_ConstantPPP": f"Private health expenditure per standardized capita (constant {reference_year} LCU converted to international $ using {reference_year} PPP factors)",
            
            # Other columns
            "Indicator_Notes": "Additional information about the data and calculations",
            "GDP_Deflator": "GDP deflator factor (base: reference year)",
            "PPP_Factor": "Current PPP conversion factor (LCU per international $)"
        }
        
        # Save the data dictionary
        dict_filename = f"Health_Expenditure_Data_Dictionary.csv"
        dict_df = pd.DataFrame([{"Column": col, "Description": desc} for col, desc in column_descriptions.items()])
        dict_df.to_csv(export_path / dict_filename, index=False)
        print(f"\nData dictionary saved to {export_path / dict_filename}")
        
        # Save the results
        results.to_csv(export_path / filename, index=False)
        print(f"\nComprehensive results saved to {export_path / filename}")
        
        # Generate a mapping file for ISO3 to country names as a reference
        if iso3_to_country:
            mapping_df = pd.DataFrame([(iso3, name) for iso3, name in iso3_to_country.items()], 
                                     columns=['ISO3', 'Country_Name'])
            mapping_df.to_csv(export_path / "ISO3_country_mapping.csv", index=False)
            print(f"ISO3 to country mapping saved to {export_path / 'ISO3_country_mapping.csv'}")
        
        # Print merge success statistics
        total_countries = len(iso3_to_country) if iso3_to_country else 0
        matched_countries = results['ISO3'].nunique()
        if total_countries > 0:
            match_percentage = (matched_countries / total_countries) * 100
            print(f"\nMatched {matched_countries} out of {total_countries} countries ({match_percentage:.1f}%)")
            
        # Print recommended indicators for time series analysis
        print("\nRECOMMENDED INDICATORS FOR TIME SERIES ANALYSIS:")
        print("For comparing health expenditure across countries and time:")
        print("- THE_per_Std_Capita_Constant_ConstantPPP: Total health expenditure per standardized capita")
        print("- PubHE_per_Std_Capita_Constant_ConstantPPP: Public health expenditure per standardized capita")
        print("- PvtHE_per_Std_Capita_Constant_ConstantPPP: Private health expenditure per standardized capita")
        print(f"(All using constant {reference_year} prices and constant {reference_year} PPP factors)")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        raise
        
if __name__ == "__main__":
    main()