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
import regex as re
import logging
import sys
from datetime import datetime

# Define paths
current_path = Path(".")
data_path = current_path / "data"  # Create path to data folder
export_path = Path("Standardized_Expenditure")
export_path.mkdir(parents=True, exist_ok=True)
log_path = export_path / "logs"
log_path.mkdir(parents=True, exist_ok=True)

# Reference year for constant price calculations and PPP adjustment
REFERENCE_YEAR = 2017

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

# Configure logging
def setup_logging(console_level=logging.INFO, file_level=logging.DEBUG, log_file=None):
    """
    Set up logging to both console and file.
    
    Args:
        console_level: Logging level for console output (default: INFO)
        file_level: Logging level for file output (default: DEBUG)
        log_file: Path to log file (default: None, which creates a timestamped file)
    
    Returns:
        Logger object
    """
    # Create a logger
    logger = logging.getLogger("whe")
    
    # Clear any existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.DEBUG)  # Set logger to capture all levels
    
    # Create formatter
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(simple_formatter)
    
    # Create file handler if needed
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"whe_analysis_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Log the configuration
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.debug(f"Console log level: {logging.getLevelName(console_level)}")
    logger.debug(f"File log level: {logging.getLevelName(file_level)}")
    
    return logger
# Global logger
logger = setup_logging()

def load_capitation_weights(formula='israeli'):
    """
    Load capitation weights from CSV file if it exists, otherwise use default values.
    
    Args:
        formula: The capitation formula to use ('israeli', 'ltc', or 'eu27')
    
    Returns:
        Tuple of (capitation_weights_dictionary, formula_type_used)
    """
    # Define mapping of formula types to column names and structure
    formula_config = {
        'israeli': {'columns': ['Men', 'Women'], 'structure': 'separate'},
        'ltc': {'columns': ['LTC'], 'structure': 'combined'},
        'eu27': {'columns': ['EU27'], 'structure': 'combined'}
    }
    
    # Check if formula is valid
    if formula not in formula_config:
        logger.warning(f"Unknown formula '{formula}', using default Israeli capitation weights")
        return ISRAELI_CAPITATION, 'israeli'
    
    try:
        cap_df = pd.read_csv(data_path / "cap.csv", index_col="Age")
        
        # Get the configuration for the requested formula
        config = formula_config[formula]
        
        # Check if all required columns exist
        if not all(col in cap_df.columns for col in config['columns']):
            raise KeyError(f"Missing columns for {formula} formula")
        
        cap_dict = {}
        
        # Process based on structure type
        if config['structure'] == 'separate':
            for age_group in cap_df.index:
                cap_dict[age_group] = {
                    "Men": cap_df.loc[age_group, "Men"],
                    "Women": cap_df.loc[age_group, "Women"]
                }
        else:  # combined structure
            column_name = config['columns'][0]
            for age_group in cap_df.index:
                cap_dict[age_group] = {
                    "Combined": cap_df.loc[age_group, column_name]
                }
        
        logger.info(f"Loaded {formula} capitation weights from cap.csv")
        return cap_dict, formula
        
    except (FileNotFoundError, KeyError) as e:
        error_type = "File not found" if isinstance(e, FileNotFoundError) else f"Missing column: {e}"
        logger.warning(f"Error loading capitation weights: {error_type}")
        logger.warning("Using default Israeli capitation weights")
        return ISRAELI_CAPITATION, 'israeli'

def load_ppp_data():
    """
    Load and process World Bank PPP data.
    
    Returns:
        DataFrame with columns: ISO3, Country, Year, PPP_Factor
    """
    logger.info("Loading PPP data...")
    
    # Define empty result DataFrame for fallback
    empty_result = pd.DataFrame(columns=["ISO3", "Country", "Year", "PPP_Factor"])
    
    try:
        # First try to load the processed version if it exists
        processed_file = data_path / "processed" / "ppp.csv"
        if processed_file.exists():
            logger.info("Loading processed PPP data file")
            ppp_df = pd.read_csv(processed_file, dtype=None)
        else:
            # Fall back to the original file if processed version doesn't exist
            ppp_file = "API_PA.NUS.PPP_DS2_en_csv_v2_13721.csv"
            logger.info(f"Loading original PPP data file: {ppp_file}")
            
            # Load the CSV file - we need to set dtype=None to detect numeric columns correctly
            try:
                ppp_df = pd.read_csv(data_path / ppp_file, dtype=None)
                logger.debug(f"Successfully loaded PPP file with shape: {ppp_df.shape}")
            except Exception as e:
                logger.error(f"Error loading PPP file: {e}")
                return empty_result
        
        # Get year columns (exclude metadata columns)
        year_columns = [col for col in ppp_df.columns if str(col).isdigit()]
        
        if not year_columns:
            logger.warning("No year columns found in PPP data")
            return empty_result
            
        logger.debug(f"Found {len(year_columns)} year columns from {min(year_columns)} to {max(year_columns)}")
        
        # Filter for PPP conversion factor rows - only needed for original WB data, not processed data
        if "Indicator Name" in ppp_df.columns:
            ppp_indicator = "PPP conversion factor, GDP (LCU per international $)"
            filtered_df = ppp_df[ppp_df['Indicator Name'].str.contains(ppp_indicator, na=False, regex=False)]
            
            if filtered_df.empty:
                logger.warning(f"No rows with '{ppp_indicator}' found in the data")
                return empty_result
                
            logger.debug(f"Found {len(filtered_df)} rows with PPP conversion factor data")
            
            # Filter out aggregates (regions, income groups, etc.)
            region_terms = ['region', 'world', 'income', 'development']
            region_pattern = '|'.join(region_terms)
            filtered_df = filtered_df[~filtered_df['Country Name'].str.lower().str.contains(region_pattern, na=False)]
            
            logger.debug(f"After filtering out regions, {len(filtered_df)} country rows remain")
            
            # Convert to long format - need to handle both string and numeric columns
            ppp_data = []
            
            for _, row in filtered_df.iterrows():
                country = row['Country Name']
                country_code = row['Country Code']  # This is the ISO3 code in World Bank data
                
                # Process each year column
                for year in year_columns:
                    # Skip missing or zero values
                    if pd.isna(row[year]) or row[year] == 0:
                        continue
                        
                    try:
                        # Convert to numeric, handling both string and float columns
                        ppp_value = pd.to_numeric(row[year], errors='coerce')
                        if pd.isna(ppp_value) or ppp_value <= 0:
                            continue
                            
                        ppp_data.append({
                            "Country": country,  # Keep country name for reference
                            "ISO3": country_code,  # Use ISO3 code for merging
                            "Year": int(year),
                            "PPP_Factor": ppp_value
                        })
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error processing {country} ({country_code}) for year {year}: {e}")
                        # Skip values that can't be converted
                        continue
            
            # Convert to DataFrame
            result = pd.DataFrame(ppp_data)
        else:
            # If using processed data, it's already in the right format
            result = ppp_df
        
        if result.empty:
            logger.warning("No PPP data was successfully parsed")
            return empty_result
        
        # Sort by Country, Year for better organization
        result = result.sort_values(['ISO3', 'Year'])
        
        logger.info(f"Loaded PPP data with shape: {result.shape}")
        logger.info(f"Data covers {result['ISO3'].nunique()} countries and years {result['Year'].min()}-{result['Year'].max()}")
        
        # Log a sample for verification
        logger.debug("\nSample of processed PPP data:")
        logger.debug(result.head())
        
        return result
    
    except Exception as e:
        logger.error(f"Error loading PPP data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return empty DataFrame as fallback
        return empty_result

def load_ghed_data():
    """
    Load and process GHED data from the optimized CSV file.
    Falls back to Excel if the CSV file is not available.
    
    Returns:
        DataFrame with GHED data containing columns: location, ISO3, year, che,
        and potentially public_expenditure and private_expenditure
    """
    logger.info("Loading GHED data...")
    
    # Define column specifications
    required_cols = ['location', 'code', 'year', 'che']
    recommended_cols = ['gghed_che', 'pvtd_che']
    
    try:
        # First try to load the optimized CSV file
        csv_path = data_path / "processed" / "ghed_data_optimized.csv"
        
        if csv_path.exists():
            ghed_data = pd.read_csv(csv_path)
            logger.info("Loaded optimized CSV file")
        else:
            logger.info("Optimized CSV not found, loading from Excel (slower)...")
            logger.info("Consider running the GHED conversion script to create an optimized CSV for faster loading.")
            
            # Load from Excel and process
            ghed_data = _process_excel_ghed_data(required_cols, recommended_cols)
        
        # Apply common processing to the data
        return _process_ghed_data(ghed_data)
        
    except Exception as e:
        logger.error(f"Error loading GHED data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def _process_excel_ghed_data(required_cols, recommended_cols):
    """Helper function to load and process GHED data from Excel"""
    # Read the GHED data from Excel
    ghed_data = pd.read_excel(data_path / "GHED_data_2025.xlsx", sheet_name="Data")
    
    # Verify required columns
    missing_required = [col for col in required_cols if col not in ghed_data.columns]
    if missing_required:
        raise ValueError(f"Required columns missing from GHED data: {missing_required}")
    
    # Check for recommended columns
    missing_recommended = [col for col in recommended_cols if col not in ghed_data.columns]
    if missing_recommended:
        logger.warning(f"Recommended columns missing: {missing_recommended}")
    
    # Calculate derived columns from percentages
    if 'gghed_che' in ghed_data.columns:
        ghed_data['public_expenditure'] = ghed_data['che'] * (ghed_data['gghed_che'] / 100)
    else:
        ghed_data['public_expenditure'] = None
        
    if 'pvtd_che' in ghed_data.columns:
        ghed_data['private_expenditure'] = ghed_data['che'] * (ghed_data['pvtd_che'] / 100)
    else:
        ghed_data['private_expenditure'] = None
    
    return ghed_data

def _process_ghed_data(ghed_data):
    """Apply common processing to GHED data regardless of source"""
    # Convert values from millions to actual amounts
    for col in ['che', 'public_expenditure', 'private_expenditure']:
        if col in ghed_data.columns:
            ghed_data[col] = ghed_data[col] * (10**6)
    
    # Select relevant columns
    relevant_cols = ['location', 'code', 'year', 'che', 'public_expenditure', 'private_expenditure']
    existing_cols = [col for col in relevant_cols if col in ghed_data.columns]
    ghed_data = ghed_data[existing_cols]
    
    # Data cleaning
    ghed_data = (ghed_data
                .dropna(subset=['che'])           # Remove rows with missing expenditure
                .astype({'year': int})            # Ensure year is integer
                .rename(columns={'code': 'ISO3'}) # Standardize column names
               )
    
    logger.info(f"Loaded GHED data with shape: {ghed_data.shape}")
    return ghed_data

def load_population_data():
    """
    Load and process population data from CSV files using vectorized operations.
    
    Returns:
        Tuple of (male_pop, female_pop) DataFrames with processed population data.
    """
    logger.info("Loading population data...")
    # Define the fallback empty DataFrame structure
    columns = ["ISO3", "Year", "Age_Group", "Sex", "Population", "Country"]
    empty_df = pd.DataFrame(columns=columns)
    
    try:
        # Load male and female population data from CSV files
        files = {
            "Men": "male_pop.csv",
            "Women": "female_pop.csv"
        }
        
        # Load and validate data
        datasets = {}
        for sex, filename in files.items():
            try:
                df = pd.read_csv(data_path / 'processed' / filename)
                if 'ISO3 Alpha-code' not in df.columns:
                    logger.warning(f"ISO3 Alpha-code column not found in {filename}")
                    continue
                logger.debug(f"{sex} population raw data shape: {df.shape}")
                datasets[sex] = df
            except Exception as file_error:
                logger.error(f"Error loading {filename}: {file_error}")
        
        # Check if we have both datasets
        if "Men" not in datasets or "Women" not in datasets:
            logger.error("Could not load one or both population datasets")
            return empty_df, empty_df
        
        # Process datasets
        processed_data = {}
        for sex, df in datasets.items():
            processed_data[sex] = process_population_dataset(df, sex)
        
        # Create mapping and standardize country names
        iso3_to_country = create_standardized_country_mapping(
            processed_data["Men"], processed_data["Women"]
        )
        
        # Apply mapping to both datasets
        for sex, df in processed_data.items():
            df['Country'] = df['ISO3'].map(iso3_to_country).fillna(df['ISO3'])
    
        
        return processed_data["Men"], processed_data["Women"]
    
    except Exception as e:
        logger.error(f"Error loading population data: {e}")
        logger.error("Detailed error information:")
        import traceback
        logger.error(traceback.format_exc())
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
    
    # Process both datasets and merge mappings
    for df, label in [(male_pop, "male"), (female_pop, "female")]:
        if df.empty or 'ISO3' not in df.columns or 'Country' not in df.columns:
            logger.warning(f"Cannot extract country mapping from {label} dataset")
            continue
            
        # Extract unique ISO3 to country mappings
        mapping = df.drop_duplicates('ISO3').set_index('ISO3')['Country'].to_dict()
        
        # Add new mappings (don't overwrite existing ones)
        for iso3, country in mapping.items():
            if iso3 not in iso3_to_country and pd.notna(country):
                iso3_to_country[iso3] = country
    
    logger.info(f"Created standardized country mapping with {len(iso3_to_country)} ISO3 codes")
    return iso3_to_country

def process_population_dataset(pop_df, sex):
    """
    Process a population dataset (either male or female) using vectorized operations.
    
    Args:
        pop_df: DataFrame containing raw population data
        sex: String indicating sex ("Men" or "Women")
    
    Returns:
        DataFrame with processed population data
    """
    logger.info(f"Processing {sex.lower()} population data...")
    
    # Define the structure of the empty result DataFrame
    result_columns = ["ISO3", "Year", "Age_Group", "Sex", "Population", "Country"]
    empty_result = pd.DataFrame(columns=result_columns)
    
    # Check if the dataframe is empty
    if pop_df.empty:
        logger.warning(f"Empty {sex.lower()} population data provided")
        return empty_result
    
    # Get the age group columns - use exact column names from the provided CSV info
    age_columns = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', 
                   '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', 
                   '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+']
    
    # Make sure these columns exist in the dataframe
    age_columns = [col for col in age_columns if col in pop_df.columns]
    
    if not age_columns:
        logger.warning(f"No age group columns found in {sex} population data")
        return empty_result
    
    # Filter out non-country rows
    # 1. Must have valid ISO3 code
    # 2. Must have valid Year
    # 3. Must not be a region or aggregate
    region_terms = ['region', 'world', 'income', 'development', 'more developed', 'less developed']
    
    # Make sure the filter columns exist
    if 'ISO3 Alpha-code' not in pop_df.columns:
        logger.warning(f"'ISO3 Alpha-code' column not found in {sex} population data")
        return empty_result
    
    if 'Year' not in pop_df.columns:
        logger.warning(f"'Year' column not found in {sex} population data")
        return empty_result
    
    if 'Region, subregion, country or area *' not in pop_df.columns:
        logger.warning(f"'Region, subregion, country or area *' column not found in {sex} population data")
        return empty_result
    
    # Apply the filter
    filter_conditions = (
        pop_df['ISO3 Alpha-code'].notna() & 
        pop_df['Year'].notna() & 
        ~pop_df['Region, subregion, country or area *'].str.lower().str.contains('|'.join(region_terms), na=False, regex=True)
    )
    
    filtered_df = pop_df[filter_conditions].copy()
    
    if filtered_df.empty:
        logger.warning(f"No valid {sex.lower()} population data after filtering")
        return empty_result
    
    # Standardize column names
    filtered_df = filtered_df.rename(columns={
        'ISO3 Alpha-code': 'ISO3',
        'Region, subregion, country or area *': 'Country'
    })
    
    # Ensure ISO3 and Year are proper types
    filtered_df['ISO3'] = filtered_df['ISO3'].astype(str)
    filtered_df['Year'] = pd.to_numeric(filtered_df['Year'], errors='coerce').fillna(0).astype(int)
    
    # Process age groups using a more efficient approach
    age_group_dfs = []
    
    for age_col in age_columns:
        # Map the UN age group to our standardized age group
        mapped_age_group = map_age_group(age_col)
        
        if mapped_age_group is None:
            logger.debug(f"Could not map age group '{age_col}' to a capitation age group")
            continue
            
        # Create a subset with only the needed columns for this age group
        age_df = filtered_df[['ISO3', 'Year', 'Country', age_col]].copy()
        
        # Check if the age column exists and has valid data
        if age_col not in age_df.columns:
            logger.debug(f"Age column '{age_col}' not found in dataframe")
            continue
        
        # Drop rows with missing/invalid population values 
        age_df = age_df.dropna(subset=[age_col])
        
        # Convert population values to numeric, handling different formats
        try:
            # Try to convert directly first
            age_df['Population'] = pd.to_numeric(age_df[age_col], errors='coerce')
        except:
            # If that fails, try removing spaces and then converting
            age_df['Population'] = pd.to_numeric(
                age_df[age_col], 
                errors='coerce'
            )
        
        # Keep only rows with positive population values
        age_df = age_df[age_df['Population'] > 0]
        
        if age_df.empty:
            continue
        
        # Convert from thousands to actual counts (if necessary)
        age_df['Population'] *= 1000
        
        # Add age group and sex columns
        age_df['Age_Group'] = mapped_age_group
        age_df['Sex'] = sex
        
        # Select only the needed columns
        age_df = age_df[['ISO3', 'Year', 'Age_Group', 'Sex', 'Population', 'Country']]
        
        # Add to the list
        age_group_dfs.append(age_df)
    
    # Combine all age group dataframes
    if not age_group_dfs:
        logger.warning(f"No valid {sex.lower()} population data after processing age groups")
        return empty_result
        
    combined_df = pd.concat(age_group_dfs, ignore_index=True)
    
    # Group by to sum populations for the same ISO3, Year, Age_Group, Sex
    result_df = combined_df.groupby(['ISO3', 'Year', 'Age_Group', 'Sex'], as_index=False).agg({
        'Population': 'sum',
        'Country': 'first'  # Take the first country name
    })
    
    logger.info(f"Processed {len(result_df)} rows of {sex.lower()} population data")
    logger.info(f"Data covers {result_df['ISO3'].nunique()} countries and years {result_df['Year'].min()}-{result_df['Year'].max()}")
    
    return result_df

def map_age_group(un_age_group):
    """
    Map the UN age groups to the age groups used in the capitation formula.
    
    Args:
        un_age_group: Age group from UN data (e.g., "0-4", "5-9", etc.)
    
    Returns:
        Mapped age group or None if it couldn't be mapped
    """
    # Define the mapping from UN age groups to capitation age groups
    mapping = {
        # 0-4 -> 0 to 4
        "0-4": "0 to 4",
        
        # 5-9, 10-14 -> 5 to 14
        "5-9": "5 to 14", "10-14": "5 to 14",
        
        # 15-19, 20-24 -> 15 to 24
        "15-19": "15 to 24", "20-24": "15 to 24",
        
        # 25-29, 30-34 -> 25 to 34
        "25-29": "25 to 34", "30-34": "25 to 34",
        
        # 35-39, 40-44 -> 35 to 44
        "35-39": "35 to 44", "40-44": "35 to 44",
        
        # 45-49, 50-54 -> 45 to 54
        "45-49": "45 to 54", "50-54": "45 to 54",
        
        # 55-59, 60-64 -> 55 to 64
        "55-59": "55 to 64", "60-64": "55 to 64",
        
        # 65-69, 70-74 -> 65 to 74
        "65-69": "65 to 74", "70-74": "65 to 74",
        
        # 75-79, 80-84 -> 75 to 84
        "75-79": "75 to 84", "80-84": "75 to 84",
        
        # 85-89, 90-94, 95-99, 100+ -> 85 and over
        "85-89": "85 and over", "90-94": "85 and over", 
        "95-99": "85 and over", "100+": "85 and over"
    }
    
    # Standardize the input age group
    if not isinstance(un_age_group, str):
        return None
    
    un_age_group = un_age_group.strip().lower()
    
    # Direct lookup
    if un_age_group in mapping:
        return mapping[un_age_group]
    
    # Try pattern matching if direct lookup fails
    for pattern, mapped_group in mapping.items():
        if pattern.lower() in un_age_group.lower():
            return mapped_group
    
    return None

def process_population_with_weights(pop_df, cap_dict, weight_key, result_index):
    """
    Process population data by applying weights and updating the standardized population.
    
    Args:
        pop_df: Population DataFrame (male or female)
        cap_dict: Dictionary with capitation weights
        weight_key: Key to use in cap_dict ("Men", "Women", or "Combined")
        result_index: Indexed result DataFrame to update
    """
    # Check if we have the necessary columns
    if not {'ISO3', 'Year', 'Age_Group', 'Population'}.issubset(pop_df.columns):
        logger.warning(f"Population DataFrame missing required columns for {weight_key}")
        return
    
    # Create pivot table to efficiently process by age group
    try:
        pop_pivot = pop_df.pivot_table(
            index=['ISO3', 'Year'],
            columns='Age_Group',
            values='Population',
            aggfunc='sum',
            fill_value=0
        )
        
        # Process each age group in the capitation formula
        for age_group, weights in cap_dict.items():
            if age_group in pop_pivot.columns and weight_key in weights:
                # Get the weight for this age group and sex
                weight = weights[weight_key]
                
                # Calculate weighted population
                weighted_pop = pop_pivot[age_group] * weight
                
                # Add to the standardized population in the result DataFrame
                for idx, value in weighted_pop.items():
                    if idx in result_index.index:
                        # Add to existing value (important for combined formulas)
                        result_index.at[idx, 'Standardized_Population'] += value
        
        # Verify some results were calculated
        non_zero_count = (result_index['Standardized_Population'] > 0).sum()
        if non_zero_count == 0:
            logger.warning("No standardized population values were calculated")
    
    except Exception as e:
        logger.error(f"Error processing population with weights for {weight_key}: {e}")
        
def _create_consolidated_keys(male_pop, female_pop):
    """
    Create a consolidated set of ISO3-Year keys from both datasets.
    
    Args:
        male_pop: DataFrame with male population data
        female_pop: DataFrame with female population data
        
    Returns:
        DataFrame with consolidated keys
    """
    # Print incoming data info
    logger.debug("Creating consolidated keys:")
    logger.debug(f"  Male data shape: {male_pop.shape if not male_pop.empty else 'Empty'}")
    logger.debug(f"  Female data shape: {female_pop.shape if not female_pop.empty else 'Empty'}")
    
    # Initialize with an empty DataFrame
    all_keys = pd.DataFrame(columns=['ISO3', 'Year'])
    
    # Add keys from male population if available
    if not male_pop.empty and 'ISO3' in male_pop.columns and 'Year' in male_pop.columns:
        male_keys = male_pop[['ISO3', 'Year']].drop_duplicates()
        all_keys = pd.concat([all_keys, male_keys], ignore_index=True)
        logger.debug(f"  Added {len(male_keys)} unique ISO3-Year pairs from male data")
    
    # Add keys from female population if available
    if not female_pop.empty and 'ISO3' in female_pop.columns and 'Year' in female_pop.columns:
        female_keys = female_pop[['ISO3', 'Year']].drop_duplicates()
        all_keys = pd.concat([all_keys, female_keys], ignore_index=True)
        logger.debug(f"  Added {len(female_keys)} unique ISO3-Year pairs from female data")
    
    # Remove duplicates after combining both
    all_keys = all_keys.drop_duplicates(['ISO3', 'Year'])
    logger.debug(f"  Final consolidated keys: {len(all_keys)} unique ISO3-Year pairs")
    
    # Check for any potential invalid values
    if not all_keys.empty:
        na_iso3 = all_keys['ISO3'].isna().sum()
        na_year = all_keys['Year'].isna().sum()
        if na_iso3 > 0 or na_year > 0:
            logger.warning(f"  Found {na_iso3} missing ISO3 values and {na_year} missing Year values")
            # Remove these problematic rows
            all_keys = all_keys.dropna(subset=['ISO3', 'Year'])
            logger.debug(f"  After removing missing values: {len(all_keys)} keys")
    
    # Initialize standardized population column
    all_keys['Standardized_Population'] = 0.0
    
    # Print a sample of the keys
    if not all_keys.empty:
        logger.debug("  Sample of consolidated keys:")
        logger.debug(all_keys.head())
    
    return all_keys

def _apply_country_mapping(keys_df, male_pop, female_pop):
    """
    Apply country names from population datasets to the keys.
    
    Args:
        keys_df: DataFrame with ISO3-Year keys
        male_pop: DataFrame with male population data
        female_pop: DataFrame with female population data
        
    Returns:
        DataFrame with country names added
    """
    # Create a unified country mapping
    country_mapping = {}
    
    # Extract mappings from both datasets
    for dataset in [male_pop, female_pop]:
        if not dataset.empty and 'ISO3' in dataset.columns and 'Country' in dataset.columns:
            # Extract unique ISO3-Country pairs
            mapping = dataset.drop_duplicates('ISO3')[['ISO3', 'Country']]
            for _, row in mapping.iterrows():
                if pd.notna(row['Country']) and (row['ISO3'] not in country_mapping or pd.isna(country_mapping[row['ISO3']])):
                    country_mapping[row['ISO3']] = row['Country']
    
    # Apply mapping to keys dataframe
    keys_df['Country'] = keys_df['ISO3'].map(country_mapping)
    
    # Fill missing country names with ISO3 code
    keys_df['Country'] = keys_df['Country'].fillna(keys_df['ISO3'])
    
    return keys_df

def calculate_standardized_population_israeli(result_df, male_pop, female_pop, cap_dict):
    """
    Calculate standardized population using the Israeli formula with separate weights for men and women.
    Updates the result_df in place.
    
    Args:
        result_df: DataFrame to store results (will be modified in place)
        male_pop: Male population data
        female_pop: Female population data
        cap_dict: Dictionary with capitation weights
    """
    # Create efficient index for the results DataFrame
    result_index = result_df.set_index(['ISO3', 'Year'])
    
    # Process male population data
    if not male_pop.empty:
        process_population_with_weights(male_pop, cap_dict, "Men", result_index)
    
    # Process female population data
    if not female_pop.empty:
        process_population_with_weights(female_pop, cap_dict, "Women", result_index)

def calculate_standardized_population_combined(result_df, male_pop, female_pop, cap_dict):
    """
    Calculate standardized population using a formula with combined weights for both sexes.
    Updates the result_df in place.
    
    Args:
        result_df: DataFrame to store results (will be modified in place)
        male_pop: Male population data
        female_pop: Female population data
        cap_dict: Dictionary with capitation weights
    """
    # Create efficient index for the results DataFrame
    result_index = result_df.set_index(['ISO3', 'Year'])
    
    # Process male population data with combined weights
    if not male_pop.empty:
        process_population_with_weights(male_pop, cap_dict, "Combined", result_index)
    
    # Process female population data with combined weights
    if not female_pop.empty:
        process_population_with_weights(female_pop, cap_dict, "Combined", result_index)

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
    logger.info(f"Calculating standardized population using {formula_type} formula...")
    
    # Check if both population dataframes are empty
    if male_pop.empty and female_pop.empty:
        logger.warning("Both male and female population data are empty")
        return pd.DataFrame(columns=['ISO3', 'Year', 'Country', 'Standardized_Population'])
    
    # Print shape information
    logger.debug(f"Male population data shape: {male_pop.shape if not male_pop.empty else 'Empty'}")
    logger.debug(f"Female population data shape: {female_pop.shape if not female_pop.empty else 'Empty'}")
    
    # Check for required columns
    required_cols = ['ISO3', 'Year', 'Age_Group', 'Population']
    for df_name, df in [('Male', male_pop), ('Female', female_pop)]:
        if not df.empty:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"{df_name} population data missing columns: {missing_cols}")
    
    # Create a consolidated set of ISO3-Year keys
    result_df = _create_consolidated_keys(male_pop, female_pop)
    
    # Apply country mapping
    result_df = _apply_country_mapping(result_df, male_pop, female_pop)
    
    # Make sure we have some rows in the result
    if result_df.empty:
        logger.error("Could not create keys for standardized population calculation")
        return result_df
    
    logger.debug(f"Created result dataframe with {len(result_df)} rows")
    
    # Create index for faster lookups
    result_index = result_df.set_index(['ISO3', 'Year'])
    logger.debug(f"Created indexed dataframe with {len(result_index)} rows")
    
    # Apply the appropriate formula calculation
    if formula_type == 'israeli':
        # Process using Israeli formula (separate weights for men and women)
        logger.info("Using Israeli formula with separate weights for men and women")
        if not male_pop.empty:
            process_population_with_weights(male_pop, cap_dict, "Men", result_index)
        if not female_pop.empty:
            process_population_with_weights(female_pop, cap_dict, "Women", result_index)
    else:
        # Process using LTC or EU27 formula (combined weight for both sexes)
        logger.info(f"Using {formula_type} formula with combined weights")
        if not male_pop.empty:
            process_population_with_weights(male_pop, cap_dict, "Combined", result_index)
        if not female_pop.empty:
            process_population_with_weights(female_pop, cap_dict, "Combined", result_index)
    
    # Reset index to get back to a standard DataFrame
    result_df = result_index.reset_index()
    
    # Count non-zero standardized population
    non_zero_count = (result_df['Standardized_Population'] > 0).sum()
    countries_with_std_pop = result_df[result_df['Standardized_Population'] > 0]['ISO3'].nunique()
    
    logger.info(f"Calculated non-zero standardized population for {countries_with_std_pop} unique countries")
    logger.info(f"Non-zero standardized population values: {non_zero_count} out of {len(result_df)} rows")
    
    if non_zero_count == 0:
        logger.warning("All standardized population values are zero!")
    
    return result_df

def load_gdp_data(reference_year=2017):
    """
    Load and process World Bank GDP data to calculate GDP deflators.
    
    Args:
        reference_year: Year to use as the reference for constant prices (default: 2017)
    
    Returns:
        DataFrame with columns: ISO3, Year, GDP_Deflator
    """
    logger.info("Loading GDP data for deflator calculation...")
    
    # Define empty result structure for fallback
    empty_result = pd.DataFrame(columns=["ISO3", "Country", "Year", "GDP_Deflator"])
    
    try:
        # First try to load processed files
        processed_files = {
            "current": data_path / "processed" / "gdp_current.csv",
            "constant": data_path / "processed" / "gdp_constant.csv"
        }
        
        original_files = {
            "current": "API_NY.GDP.MKTP.CN_DS2_en_csv_v2_26332.csv",
            "constant": "API_NY.GDP.MKTP.KN_DS2_en_csv_v2_13325.csv"
        }
        
        # Load both datasets
        gdp_data = {}
        for gdp_type, filepath in processed_files.items():
            if filepath.exists():
                try:
                    logger.info(f"Loading processed {gdp_type} GDP data")
                    gdp_data[gdp_type] = pd.read_csv(filepath)
                    logger.debug(f"Loaded {gdp_type} GDP data: {gdp_data[gdp_type].shape}")
                except Exception as file_error:
                    logger.error(f"Error loading processed {gdp_type} GDP data: {file_error}")
                    logger.info(f"Falling back to original {gdp_type} GDP data file")
                    # Fall back to original file
                    try:
                        gdp_data[gdp_type] = pd.read_csv(data_path / original_files[gdp_type])
                    except Exception as orig_error:
                        logger.error(f"Error loading original {gdp_type} GDP data: {orig_error}")
                        return empty_result
            else:
                logger.info(f"Processed {gdp_type} GDP file not found, using original file")
                try:
                    gdp_data[gdp_type] = pd.read_csv(data_path / original_files[gdp_type])
                    logger.debug(f"Loaded {gdp_type} GDP data: {gdp_data[gdp_type].shape}")
                except Exception as file_error:
                    logger.error(f"Error loading {gdp_type} GDP data: {file_error}")
                    return empty_result
        
        # Check if we have both datasets
        if not all(key in gdp_data for key in ["current", "constant"]):
            logger.error("Could not load all required GDP datasets")
            return empty_result
            
        # Process the data
        long_dfs = []
        for gdp_type, df in gdp_data.items():
            # Check if we're dealing with original WB data (which needs processing)
            # or already processed data
            if "Country Name" in df.columns and "Country Code" in df.columns:
                long_df = process_wb_data_to_long_format(
                    df, 
                    value_col_name=f"GDP_{gdp_type.capitalize()}",
                    skip_aggregates=True
                )
            else:
                # For processed data, rename columns if needed
                if "value" in df.columns:
                    df = df.rename(columns={"value": f"GDP_{gdp_type.capitalize()}"})
                else:
                    # Assume the value column already has the right name
                    pass
                long_df = df
                
            if not long_df.empty:
                long_dfs.append(long_df)
            else:
                logger.error(f"Could not process {gdp_type} GDP data")
                return empty_result
        
        # Merge the datasets
        gdp_merged = pd.merge(
            long_dfs[0], long_dfs[1],
            on=["ISO3", "Year", "Country"],
            how="inner"
        )
        
        # Calculate GDP deflator
        gdp_merged["GDP_Deflator"] = gdp_merged["GDP_Current"] / gdp_merged["GDP_Constant"]
        
        # Normalize by reference year
        result = normalize_by_reference_year(
            gdp_merged, 
            reference_year=reference_year,
            value_column="GDP_Deflator"
        )
        
        logger.info(f"Loaded GDP deflator data with shape: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"Error loading GDP data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return empty_result

def process_wb_data_to_long_format(df, value_col_name, skip_aggregates=True):
    """
    Process World Bank data from wide to long format.
    
    Args:
        df: World Bank data in wide format
        value_col_name: Name for the value column
        skip_aggregates: Whether to skip regional aggregates
        
    Returns:
        DataFrame in long format with ISO3, Year, Country, and value column
    """
    # Get year columns (numeric columns)
    year_columns = [col for col in df.columns if col.isdigit()]
    
    # Skip if no year columns found
    if not year_columns:
        logger.error("No year columns found in World Bank data")
        return pd.DataFrame()
    
    # Convert to long format
    long_df = pd.melt(
        df,
        id_vars=['Country Name', 'Country Code'],
        value_vars=year_columns,
        var_name='Year',
        value_name=value_col_name
    )
    
    # Clean up the data
    long_df = (long_df
               .rename(columns={'Country Name': 'Country', 'Country Code': 'ISO3'})
               .dropna(subset=[value_col_name])  # Remove missing values
               .query(f'{value_col_name} != 0')  # Remove zeros
               .astype({'Year': int})            # Convert year to integer
              )
    
    # Filter out aggregates if requested
    if skip_aggregates:
        region_terms = ['region', 'world', 'income', 'development']
        region_pattern = '|'.join(region_terms)
        long_df = long_df[~long_df['Country'].str.lower().str.contains(region_pattern, na=False)]
    
    return long_df

def normalize_by_reference_year(df, reference_year, value_column, id_columns=["ISO3"]):
    """
    Normalize values by a reference year for each entity.
    
    Args:
        df: DataFrame with data to normalize
        reference_year: Year to use as reference
        value_column: Column with values to normalize
        id_columns: List of columns that uniquely identify each entity
        
    Returns:
        DataFrame with normalized values
    """
    # Create a copy to avoid modifying the input
    result_df = df.copy()
    
    # Create a new column for normalized values
    norm_col = f"{value_column}_Normalized"
    result_df[norm_col] = np.nan
    
    # Process each entity separately
    entities = df.drop_duplicates(id_columns)[id_columns].values
    
    for entity_values in entities:
        # Create a filter for this entity
        entity_filter = np.ones(len(df), dtype=bool)
        for i, col in enumerate(id_columns):
            entity_filter &= (df[col] == entity_values[i])
        
        entity_data = df[entity_filter].copy()
        
        # Check if reference year exists for this entity
        ref_year_data = entity_data[entity_data["Year"] == reference_year]
        
        if not ref_year_data.empty:
            # Use reference year value
            ref_value = ref_year_data[value_column].iloc[0]
        else:
            # Find closest year
            available_years = entity_data["Year"].unique()
            if len(available_years) > 0:
                closest_year = min(available_years, key=lambda x: abs(x - reference_year))
                ref_value = entity_data[entity_data["Year"] == closest_year][value_column].iloc[0]
                
                # Log the substitution
                entity_name = ", ".join([f"{col}={entity_values[i]}" for i, col in enumerate(id_columns)])
                logger.debug(f"Using {closest_year} as reference year for {entity_name} (reference {reference_year} not available)")
            else:
                # No years available
                continue
        
        # Normalize by reference value
        if ref_value != 0:
            result_df.loc[entity_filter, norm_col] = result_df.loc[entity_filter, value_column] / ref_value
    
    # Return only the necessary columns
    result_cols = id_columns + ["Year", "Country", norm_col]
    result = result_df[result_cols].rename(columns={norm_col: value_column})
    
    return result

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
    logger.info("Applying GDP deflator adjustment for constant prices...")
    
    # Make a copy to avoid modifying the original data
    adjusted_data = data.copy()
    
    # Check if gdp_deflator is valid
    if gdp_deflator is None or gdp_deflator.empty:
        logger.warning("GDP deflator data is empty, skipping adjustment")
        return adjusted_data
    
    # Merge GDP deflator data with health expenditure data using ISO3 code
    merged = pd.merge(
        adjusted_data,
        gdp_deflator[['ISO3', 'Year', 'GDP_Deflator']],
        on=['ISO3', 'Year'],
        how='left'
    )
    
    # Handle missing GDP deflators
    merged = _handle_missing_values(
        merged, 
        value_column='GDP_Deflator',
        impute=impute_missing,
        value_label='GDP deflator'
    )
    
    # Create a mask for rows with valid deflator values
    valid_deflator_mask = ~merged['GDP_Deflator'].isna()
    
    # Define expenditure types to adjust (total, public, private)
    expenditure_types = {
        'Total': 'Total_Health_Expenditure',
        'Public': 'Public_Health_Expenditure',
        'Private': 'Private_Health_Expenditure'
    }
    
    # Process each expenditure type
    for type_label, column_name in expenditure_types.items():
        if column_name in merged.columns:
            # Apply constant price adjustment for this expenditure type
            merged = _apply_deflator_to_expenditure(
                merged,
                expenditure_column=column_name,
                deflator_mask=valid_deflator_mask,
                type_prefix=type_label.lower()
            )
    
    return merged

def _handle_missing_values(df, value_column, impute=False, value_label='value'):
    """
    Handle missing values in a DataFrame by either imputing or reporting.
    
    Args:
        df: DataFrame with potentially missing values
        value_column: Column name that may have missing values
        impute: Whether to impute missing values using nearest year
        value_label: Description of the value for messages
        
    Returns:
        DataFrame with potentially imputed values
    """
    # Check if any values are missing
    missing_count = df[value_column].isna().sum()
    if missing_count == 0:
        return df
    
    logger.warning(f"Missing {value_label}s for {missing_count} out of {len(df)} rows")
    
    # Skip imputation if not requested
    if not impute:
        logger.info(f"Skipping imputation for missing {value_label}s as requested")
        return df
    
    # Make a copy to avoid modifying the input
    result = df.copy()
    logger.info(f"Imputing missing {value_label}s...")
    
    # Get list of entities with missing values
    id_cols = ['ISO3']
    entities_with_missing = df[df[value_column].isna()][id_cols].drop_duplicates()
    
    # Process each entity
    for _, entity in entities_with_missing.iterrows():
        entity_filter = True
        entity_label_parts = []
        
        # Create filter and label for this entity
        for col in id_cols:
            entity_filter &= (df[col] == entity[col])
            entity_label_parts.append(f"{entity[col]}")
            
        entity_data = df[entity_filter]
        entity_label = ", ".join(entity_label_parts)
        
        # Get missing years for this entity
        missing_years = entity_data[entity_data[value_column].isna()]['Year'].tolist()
        
        # Check if entity has any non-missing values
        if any(~entity_data[value_column].isna()):
            available_years = entity_data[~entity_data[value_column].isna()]['Year'].tolist()
            
            for missing_year in missing_years:
                # Find closest available year
                closest_year = min(available_years, key=lambda x: abs(x - missing_year))
                closest_value = entity_data[entity_data['Year'] == closest_year][value_column].iloc[0]
                
                # Impute the missing value
                impute_filter = entity_filter & (df['Year'] == missing_year)
                result.loc[impute_filter, value_column] = closest_value
                
                # Display imputation information with country name if available
                if 'Country' in df.columns:
                    country_name = df.loc[entity_filter, 'Country'].iloc[0]
                    logger.debug(f"Imputed {value_label} for {country_name} ({entity_label}) in year {missing_year} using data from {closest_year}")
                else:
                    logger.debug(f"Imputed {value_label} for {entity_label} in year {missing_year} using data from {closest_year}")
    
    return result

def _apply_deflator_to_expenditure(df, expenditure_column, deflator_mask, type_prefix):
    """
    Apply GDP deflator to convert health expenditure to constant prices.
    
    Args:
        df: DataFrame with health expenditure and GDP deflator data
        expenditure_column: Column name for the expenditure to adjust
        deflator_mask: Boolean mask for rows with valid deflators
        type_prefix: Prefix for output column names (total, public, private)
    
    Returns:
        DataFrame with added constant price columns
    """
    # Create a copy to avoid modifying the input
    result = df.copy()
    
    # Create mask for valid expenditure rows (valid deflator and non-null expenditure)
    valid_exp_mask = deflator_mask & ~df[expenditure_column].isna()
    
    # Define output column names
    constant_col = f"{expenditure_column}_Constant"
    per_capita_col = f"{expenditure_column}_per_Std_Capita_Constant"
    
    # Initialize output columns
    result[constant_col] = np.nan
    result[per_capita_col] = np.nan
    
    # Calculate constant price values
    # Formula: constant_price = current_price / deflator
    result.loc[valid_exp_mask, constant_col] = (
        result.loc[valid_exp_mask, expenditure_column] / 
        result.loc[valid_exp_mask, 'GDP_Deflator']
    )
    
    # Calculate per standardized capita values if standardized population is available
    if 'Standardized_Population' in result.columns:
        valid_capita_mask = valid_exp_mask & (result['Standardized_Population'] > 0)
        
        result.loc[valid_capita_mask, per_capita_col] = (
            result.loc[valid_capita_mask, constant_col] / 
            result.loc[valid_capita_mask, 'Standardized_Population']
        )
    
    return result

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
    logger.info("Applying PPP adjustment...")
    
    # Check if ppp_data is valid
    if ppp_data is None or ppp_data.empty:
        logger.warning("PPP data is empty, skipping adjustment")
        return data.copy()
    
    # Make a copy to avoid modifying the original data
    adjusted_data = data.copy()
    
    # Merge PPP data with health expenditure data using ISO3 code
    merged = pd.merge(
        adjusted_data,
        ppp_data[['ISO3', 'Year', 'PPP_Factor']],
        on=['ISO3', 'Year'],
        how='left'
    )
    
    # Handle missing PPP factors
    merged = _handle_missing_values(
        merged, 
        value_column='PPP_Factor',
        impute=impute_missing,
        value_label='PPP factor'
    )
    
    # Check if we have PPP factors for the base country
    if not _validate_base_country(ppp_data, base_country_iso):
        logger.warning("Cannot perform proper PPP adjustment without base country data")
        return adjusted_data
    
    # Create a mask for rows with valid PPP factors
    valid_ppp_mask = ~merged['PPP_Factor'].isna()
    
    # Define expenditure types to adjust
    expenditure_types = {
        'Total': 'Total_Health_Expenditure',
        'Public': 'Public_Health_Expenditure',
        'Private': 'Private_Health_Expenditure'
    }
    
    # Process each expenditure type
    for type_label, column_name in expenditure_types.items():
        if column_name in merged.columns:
            # Apply PPP adjustment to current prices
            merged = _apply_ppp_to_expenditure(
                merged,
                expenditure_column=column_name,
                ppp_mask=valid_ppp_mask,
                suffix='PPP'
            )
            
            # Apply PPP adjustment to constant prices if available
            constant_column = f"{column_name}_Constant"
            if constant_column in merged.columns:
                merged = _apply_ppp_to_expenditure(
                    merged,
                    expenditure_column=constant_column,
                    ppp_mask=valid_ppp_mask,
                    suffix='PPP'
                )
    
    return merged

def _validate_base_country(ppp_data, base_country_iso):
    """
    Validate that PPP data is available for the base country.
    
    Args:
        ppp_data: DataFrame with PPP data
        base_country_iso: ISO3 code for the base country
        
    Returns:
        Boolean indicating if the base country has valid PPP data
    """
    # Check if we have any PPP factors for the base country
    base_country_ppp = ppp_data[ppp_data['ISO3'] == base_country_iso]
    
    if base_country_ppp.empty:
        # Get the country name if available
        base_country_name = "Unknown"
        if 'Country' in ppp_data.columns:
            base_country_matches = ppp_data[ppp_data['ISO3'] == base_country_iso]
            if not base_country_matches.empty:
                base_country_name = base_country_matches.iloc[0]['Country']
        
        logger.warning(f"No PPP factors available for base country ({base_country_name}, ISO3: {base_country_iso})")
        return False
    
    return True

def _apply_ppp_to_expenditure(df, expenditure_column, ppp_mask, suffix='PPP'):
    """
    Apply PPP adjustment to health expenditure values.
    
    Args:
        df: DataFrame with health expenditure and PPP data
        expenditure_column: Column name for the expenditure to adjust
        ppp_mask: Boolean mask for rows with valid PPP factors
        suffix: Suffix for output column names
        
    Returns:
        DataFrame with added PPP-adjusted columns
    """
    # Create a copy to avoid modifying the input
    result = df.copy()
    
    # Create mask for valid expenditure rows (valid PPP and non-null expenditure)
    valid_exp_mask = ppp_mask & ~df[expenditure_column].isna()
    
    # Define output column names
    ppp_col = f"{expenditure_column}_{suffix}"
    per_capita_col = f"{expenditure_column}_per_Std_Capita_{suffix}"
    
    # Initialize output columns
    result[ppp_col] = np.nan
    result[per_capita_col] = np.nan
    
    # Calculate PPP-adjusted values
    # Formula: ppp_adjusted = expenditure / ppp_factor
    result.loc[valid_exp_mask, ppp_col] = (
        result.loc[valid_exp_mask, expenditure_column] / 
        result.loc[valid_exp_mask, 'PPP_Factor']
    )
    
    # Calculate per standardized capita values if standardized population is available
    if 'Standardized_Population' in result.columns:
        valid_capita_mask = valid_exp_mask & (result['Standardized_Population'] > 0)
        
        result.loc[valid_capita_mask, per_capita_col] = (
            result.loc[valid_capita_mask, ppp_col] / 
            result.loc[valid_capita_mask, 'Standardized_Population']
        )
    
    return result

def calculate_all_expenditure_indicators(ghed_data, standardized_pop, ppp_data=None, gdp_deflator=None, 
                               impute_missing_ppp=False, impute_missing_gdp=False, reference_year=2017, formula_type='israeli'):
    """
    Calculate health expenditure per standardized capita with all combinations of price and PPP adjustments.
    
    Args:
        ghed_data (DataFrame): GHED data with ISO3 codes
        standardized_pop (DataFrame): Standardized population data with ISO3 codes
        ppp_data (DataFrame, optional): PPP conversion factors. Defaults to None.
        gdp_deflator (DataFrame, optional): GDP deflators. Defaults to None.
        impute_missing_ppp (bool, optional): Whether to impute missing PPP factors. Defaults to False.
        impute_missing_gdp (bool, optional): Whether to impute missing GDP deflators. Defaults to False.
        reference_year (int, optional): Reference year for constant prices and PPP. Defaults to 2017.
        formula_type (str, optional): Type of capitation formula used. Defaults to 'israeli'.
    
    Returns:
        DataFrame: Health expenditure indicators with various adjustments
    """
    logger.info("Calculating comprehensive health expenditure indicators...")
    
    # Step 1: Prepare and merge the data
    merged_data = _prepare_merged_dataset(ghed_data, standardized_pop)
    
    # Record dataset statistics
    num_countries = merged_data['ISO3'].nunique()
    num_years = merged_data['Year'].nunique()
    logger.info(f"Working with data for {num_countries} countries over {num_years} years")
    
    # Step 2: Calculate base metrics with current prices (no adjustments)
    merged_data = _calculate_base_indicators(merged_data)
    
    # Step 3: Apply GDP deflator for constant prices (if data available)
    if gdp_deflator is not None and not gdp_deflator.empty:
        merged_data = _apply_constant_price_adjustment(
            merged_data, 
            gdp_deflator, 
            reference_year,
            impute_missing=impute_missing_gdp
        )
    else:
        logger.info("Skipping GDP deflator adjustment as no data is available")
    
    # Step 4: Apply current-year PPP adjustment (if data available)
    if ppp_data is not None and not ppp_data.empty:
        merged_data = _apply_current_ppp_adjustment(
            merged_data, 
            ppp_data,
            impute_missing=impute_missing_ppp
        )
    else:
        logger.info("Skipping current-year PPP adjustment as no data is available")
    
    # Step 5: Apply constant (reference year) PPP adjustment (if data available)
    if ppp_data is not None and not ppp_data.empty:
        merged_data = _apply_constant_ppp_adjustment(
            merged_data, 
            ppp_data, 
            reference_year,
            impute_missing=impute_missing_ppp
        )
    else:
        logger.info("Skipping constant PPP adjustment as no data is available")
    
    return merged_data

def _prepare_merged_dataset(ghed_data, standardized_pop):
    """
    Prepare and merge GHED data with standardized population data.
    
    Args:
        ghed_data (DataFrame): GHED data
        standardized_pop (DataFrame): Standardized population data
    
    Returns:
        DataFrame: Merged dataset with standardized column names
    """
    logger.info("Preparing and merging datasets...")
    
    # Rename GHED columns for consistency
    ghed_renamed = ghed_data.rename(columns={
        'location': 'Country_GHED',
        'year': 'Year',
        'che': 'Total_Health_Expenditure',
        'public_expenditure': 'Public_Health_Expenditure',
        'private_expenditure': 'Private_Health_Expenditure'
    })
    
    # Select necessary columns from standardized population data
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
    
    # Create a single Country column from available sources
    if 'Country_GHED' in merged_data.columns and 'Country_StdPop' in merged_data.columns:
        merged_data['Country'] = merged_data['Country_GHED'].combine_first(merged_data['Country_StdPop'])
        merged_data = merged_data.drop(columns=['Country_GHED', 'Country_StdPop'])
    elif 'Country_GHED' in merged_data.columns:
        merged_data = merged_data.rename(columns={'Country_GHED': 'Country'})
    elif 'Country_StdPop' in merged_data.columns:
        merged_data = merged_data.rename(columns={'Country_StdPop': 'Country'})
    
    logger.debug(f"Merged dataset created with shape: {merged_data.shape}")
    return merged_data

def _calculate_base_indicators(data):
    """
    Calculate base health expenditure indicators with current prices.
    
    Args:
        data (DataFrame): Merged dataset with health expenditure and standardized population
    
    Returns:
        DataFrame: Dataset with base indicators added
    """
    logger.info("Calculating base indicators with current prices...")
    result = data.copy()
    
    # Calculate total health expenditure per standardized capita
    result["THE_per_Std_Capita_Current"] = result["Total_Health_Expenditure"] / result["Standardized_Population"]
    
    # Calculate public health expenditure per standardized capita if available
    if 'Public_Health_Expenditure' in result.columns and not result['Public_Health_Expenditure'].isna().all():
        result["PubHE_per_Std_Capita_Current"] = result["Public_Health_Expenditure"] / result["Standardized_Population"]
    
    # Calculate private health expenditure per standardized capita if available
    if 'Private_Health_Expenditure' in result.columns and not result['Private_Health_Expenditure'].isna().all():
        result["PvtHE_per_Std_Capita_Current"] = result["Private_Health_Expenditure"] / result["Standardized_Population"]
    
    return result

def _apply_constant_price_adjustment(data, gdp_deflator, reference_year, impute_missing=False):
    """
    Apply GDP deflator adjustment to get constant price indicators.
    
    Args:
        data (DataFrame): Dataset with base indicators
        gdp_deflator (DataFrame): GDP deflator data
        reference_year (int): Reference year for constant prices
        impute_missing (bool, optional): Whether to impute missing GDP deflators. Defaults to False.
    
    Returns:
        DataFrame: Dataset with constant price indicators added
    """
    logger.info(f"Applying GDP deflator adjustment for constant prices (reference year: {reference_year})...")
    result = data.copy()
    
    # Merge with GDP deflator data
    result = pd.merge(
        result,
        gdp_deflator[['ISO3', 'Year', 'GDP_Deflator']],
        on=['ISO3', 'Year'],
        how='left'
    )
    
    # Handle missing GDP deflators
    missing_deflator_count = result['GDP_Deflator'].isna().sum()
    if missing_deflator_count > 0:
        logger.warning(f"Missing GDP deflators for {missing_deflator_count} out of {len(result)} rows")
        
        if impute_missing:
            result = _impute_missing_values(
                result, 
                'GDP_Deflator', 
                id_columns=['ISO3'],
                value_label='GDP deflator'
            )
    
    # Create mask for valid deflator values
    valid_deflator_mask = ~result['GDP_Deflator'].isna()
    
    # Define expenditure types to process
    expenditure_types = [
        ('Total_Health_Expenditure', 'THE'),
        ('Public_Health_Expenditure', 'PubHE'),
        ('Private_Health_Expenditure', 'PvtHE')
    ]
    
    # Apply constant price adjustment to each expenditure type
    for source_col, prefix in expenditure_types:
        if source_col in result.columns and not result[source_col].isna().all():
            # Define output column names
            constant_col = f"{prefix}_Constant"
            per_capita_col = f"{prefix}_per_Std_Capita_Constant"
            
            # Initialize output columns
            result[constant_col] = np.nan
            result[per_capita_col] = np.nan
            
            # Calculate constant price values
            valid_mask = valid_deflator_mask & ~result[source_col].isna()
            
            result.loc[valid_mask, constant_col] = (
                result.loc[valid_mask, source_col] / 
                result.loc[valid_mask, 'GDP_Deflator']
            )
            
            result.loc[valid_mask, per_capita_col] = (
                result.loc[valid_mask, constant_col] / 
                result.loc[valid_mask, 'Standardized_Population']
            )
    
    return result

def _apply_current_ppp_adjustment(data, ppp_data, impute_missing=False):
    """
    Apply current-year PPP adjustment to health expenditure indicators.
    
    Args:
        data (DataFrame): Dataset with current and constant price indicators
        ppp_data (DataFrame): PPP conversion factor data
        impute_missing (bool, optional): Whether to impute missing PPP factors. Defaults to False.
    
    Returns:
        DataFrame: Dataset with current PPP indicators added
    """
    logger.info("Applying current-year PPP adjustment...")
    result = data.copy()
    
    # Merge with PPP data
    result = pd.merge(
        result,
        ppp_data[['ISO3', 'Year', 'PPP_Factor']],
        on=['ISO3', 'Year'],
        how='left'
    )
    
    # Handle missing PPP factors
    missing_ppp_count = result['PPP_Factor'].isna().sum()
    if missing_ppp_count > 0:
        logger.warning(f"Missing PPP factors for {missing_ppp_count} out of {len(result)} rows")
        
        if impute_missing:
            result = _impute_missing_values(
                result, 
                'PPP_Factor', 
                id_columns=['ISO3'],
                value_label='PPP factor'
            )
    
    # Create mask for valid PPP factors
    valid_ppp_mask = ~result['PPP_Factor'].isna()
    
    # Define expenditure types and adjustments to process
    adjustments = [
        # Current prices with current PPP
        ('Total_Health_Expenditure', 'THE', 'CurrentPPP'),
        ('Public_Health_Expenditure', 'PubHE', 'CurrentPPP'),
        ('Private_Health_Expenditure', 'PvtHE', 'CurrentPPP'),
        
        # Constant prices with current PPP
        ('THE_Constant', 'THE', 'Constant_CurrentPPP'),
        ('PubHE_Constant', 'PubHE', 'Constant_CurrentPPP'),
        ('PvtHE_Constant', 'PvtHE', 'Constant_CurrentPPP')
    ]
    
    # Apply PPP adjustment to each combination
    for source_col, prefix, suffix in adjustments:
        if source_col in result.columns and not result[source_col].isna().all():
            # Define output column names
            ppp_col = f"{prefix}_{suffix}"
            per_capita_col = f"{prefix}_per_Std_Capita_{suffix}"
            
            # Initialize output columns
            result[ppp_col] = np.nan
            result[per_capita_col] = np.nan
            
            # Calculate PPP-adjusted values
            valid_mask = valid_ppp_mask & ~result[source_col].isna()
            
            result.loc[valid_mask, ppp_col] = (
                result.loc[valid_mask, source_col] / 
                result.loc[valid_mask, 'PPP_Factor']
            )
            
            result.loc[valid_mask, per_capita_col] = (
                result.loc[valid_mask, ppp_col] / 
                result.loc[valid_mask, 'Standardized_Population']
            )
    
    return result

def _apply_constant_ppp_adjustment(data, ppp_data, reference_year, impute_missing=False):
    """
    Apply constant (reference year) PPP adjustment to health expenditure indicators.
    
    Args:
        data (DataFrame): Dataset with current and constant price indicators
        ppp_data (DataFrame): PPP conversion factor data
        reference_year (int): Reference year for PPP factors
        impute_missing (bool, optional): Whether to impute missing PPP factors. Defaults to False.
    
    Returns:
        DataFrame: Dataset with constant PPP indicators added
    """
    logger.info(f"Applying constant PPP adjustment using reference year {reference_year}...")
    result = data.copy()
    
    # Filter PPP data to only include the reference year
    ref_ppp_data = ppp_data[ppp_data['Year'] == reference_year].copy()
    
    if ref_ppp_data.empty:
        logger.warning(f"No PPP data available for reference year {reference_year}")
        logger.info("Looking for nearest available year...")
        
        # Find the nearest available year to the reference year
        all_years = sorted(ppp_data['Year'].unique())
        if not all_years:
            logger.error("No PPP data available at all")
            return result
        else:
            nearest_year = min(all_years, key=lambda x: abs(x - reference_year))
            logger.info(f"Using PPP data from {nearest_year} as reference (closest to {reference_year})")
            ref_ppp_data = ppp_data[ppp_data['Year'] == nearest_year].copy()
    
    # Create a mapping of ISO3 -> PPP_Factor from the reference year
    ppp_mapping = ref_ppp_data[['ISO3', 'PPP_Factor']].drop_duplicates('ISO3').set_index('ISO3')['PPP_Factor'].to_dict()
    
    # Add a column with constant PPP factors to the data
    result['Constant_PPP_Factor'] = result['ISO3'].map(ppp_mapping)
    
    # Handle missing constant PPP factors
    missing_const_ppp_count = result['Constant_PPP_Factor'].isna().sum()
    if missing_const_ppp_count > 0:
        logger.warning(f"Missing constant PPP factors for {missing_const_ppp_count} out of {len(result)} rows")
        
        if impute_missing:
            # For each ISO3 with missing values, try to find PPP data for any year
            iso3_missing = result[result['Constant_PPP_Factor'].isna()]['ISO3'].unique()
            
            for iso3 in iso3_missing:
                # Get all PPP data for this country
                country_ppp = ppp_data[ppp_data['ISO3'] == iso3]
                
                if not country_ppp.empty:
                    # Get the closest year to reference year
                    closest_year = min(country_ppp['Year'].unique(), key=lambda x: abs(x - reference_year))
                    closest_ppp = country_ppp[country_ppp['Year'] == closest_year]['PPP_Factor'].iloc[0]
                    
                    # Apply this PPP factor to all rows with this ISO3
                    result.loc[result['ISO3'] == iso3, 'Constant_PPP_Factor'] = closest_ppp
                    
                    country_name = result.loc[result['ISO3'] == iso3, 'Country'].iloc[0]
                    logger.debug(f"Imputed constant PPP factor for {country_name} ({iso3}) using data from {closest_year}")
    
    # Create a mask for rows with valid constant PPP factors
    valid_const_ppp_mask = ~result['Constant_PPP_Factor'].isna()
    
    # Define expenditure types and adjustments to process
    adjustments = [
        # Current prices with constant PPP
        ('Total_Health_Expenditure', 'THE', 'ConstantPPP'),
        ('Public_Health_Expenditure', 'PubHE', 'ConstantPPP'),
        ('Private_Health_Expenditure', 'PvtHE', 'ConstantPPP'),
        
        # Constant prices with constant PPP (RECOMMENDED FOR TIME SERIES COMPARISON)
        ('THE_Constant', 'THE', 'Constant_ConstantPPP'),
        ('PubHE_Constant', 'PubHE', 'Constant_ConstantPPP'),
        ('PvtHE_Constant', 'PvtHE', 'Constant_ConstantPPP')
    ]
    
    # Apply constant PPP adjustment to each combination
    for source_col, prefix, suffix in adjustments:
        if source_col in result.columns and not result[source_col].isna().all():
            # Define output column names
            ppp_col = f"{prefix}_{suffix}"
            per_capita_col = f"{prefix}_per_Std_Capita_{suffix}"
            
            # Initialize output columns
            result[ppp_col] = np.nan
            result[per_capita_col] = np.nan
            
            # Calculate constant PPP-adjusted values
            valid_mask = valid_const_ppp_mask & ~result[source_col].isna()
            
            result.loc[valid_mask, ppp_col] = (
                result.loc[valid_mask, source_col] / 
                result.loc[valid_mask, 'Constant_PPP_Factor']
            )
            
            result.loc[valid_mask, per_capita_col] = (
                result.loc[valid_mask, ppp_col] / 
                result.loc[valid_mask, 'Standardized_Population']
            )
    
    # Drop the temporary column used for calculations
    result = result.drop(columns=['Constant_PPP_Factor'])
    
    return result

def _impute_missing_values(df, value_column, id_columns=['ISO3'], value_label='value'):
    """
    Impute missing values using the nearest year available for each entity.
    
    Args:
        df (DataFrame): DataFrame with potentially missing values
        value_column (str): Column name that may have missing values
        id_columns (list, optional): Columns that uniquely identify each entity. Defaults to ['ISO3'].
        value_label (str, optional): Description of the value for messages. Defaults to 'value'.
    
    Returns:
        DataFrame: DataFrame with imputed values
    """
    result = df.copy()
    logger.info(f"Imputing missing {value_label}s...")
    
    # Get list of entities with missing values
    missing_filter = df[value_column].isna()
    entities_with_missing = df[missing_filter][id_columns].drop_duplicates()
    
    imputed_count = 0
    
    # Process each entity
    for idx, entity in entities_with_missing.iterrows():
        entity_filter = np.ones(len(df), dtype=bool)
        entity_label_parts = []
        
        # Create filter and label for this entity
        for col in id_columns:
            entity_filter &= (df[col] == entity[col])
            entity_label_parts.append(f"{entity[col]}")
            
        entity_data = df[entity_filter]
        entity_label = ", ".join(entity_label_parts)
        
        # Get missing years for this entity
        missing_years = entity_data[entity_data[value_column].isna()]['Year'].tolist()
        
        # Check if entity has any non-missing values
        if any(~entity_data[value_column].isna()):
            available_years = entity_data[~entity_data[value_column].isna()]['Year'].tolist()
            
            for missing_year in missing_years:
                # Find closest available year
                closest_year = min(available_years, key=lambda x: abs(x - missing_year))
                closest_value = entity_data[entity_data['Year'] == closest_year][value_column].iloc[0]
                
                # Impute the missing value
                impute_filter = entity_filter & (df['Year'] == missing_year)
                result.loc[impute_filter, value_column] = closest_value
                imputed_count += 1
                
                # Display imputation information with country name if available
                if 'Country' in df.columns:
                    country_name = df.loc[entity_filter, 'Country'].iloc[0]
                    logger.debug(f"Imputed {value_label} for {country_name} ({entity_label}) in year {missing_year} using data from {closest_year}")
                else:
                    logger.debug(f"Imputed {value_label} for {entity_label} in year {missing_year} using data from {closest_year}")
    
    logger.info(f"Imputed {imputed_count} missing {value_label} values")
    return result

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
    logger.info("Documenting data imputation...")
    
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
            logger.info(f"Imputation documentation saved to {output_path / imputation_filename}")
        
        # Create summary statistics
        logger.info("\nImputation Summary:")
        logger.info(f"Total records with imputation: {len(imputation_df)}")
        
        # Count by missing data type
        if 'Missing_Data' in imputation_df.columns:
            type_counts = imputation_df['Missing_Data'].value_counts()
            logger.info("\nImputation by type:")
            for data_type, count in type_counts.items():
                logger.info(f"  {data_type}: {count} records")
        
        # Count by country
        country_counts = imputation_df.groupby(['ISO3', 'Country']).size().reset_index(name='Count')
        country_counts = country_counts.sort_values('Count', ascending=False)
        
    else:
        logger.info("No imputation was performed or documented.")
    
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
        logger.warning(f"Cannot create ISO3 mapping, columns {iso3_column} or {country_column} not found")
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
    
    logger.info(f"Created mapping with {len(iso3_to_country)} ISO3 codes to country names")
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
    logger.info(f"Merge using ISO3 codes:")
    logger.info(f"  Left DataFrame: {left_iso3_count} unique ISO3 codes")
    logger.info(f"  Right DataFrame: {right_iso3_count} unique ISO3 codes")
    logger.info(f"  Merged DataFrame: {merged_iso3_count} unique ISO3 codes")
    
    # Calculate and report missing matches
    if how == 'inner':
        missing_left = left_iso3_count - merged_iso3_count
        missing_right = right_iso3_count - merged_iso3_count
        logger.info(f"  Missing matches: {missing_left} from left, {missing_right} from right")
        
        # Show examples of unmatched ISO3 codes
        if missing_left > 0:
            left_only = set(left_df[left_on].unique()) - set(merged[left_on].unique())
            logger.debug(f"  Examples of unmatched ISO3 codes from left: {list(left_only)[:5]}")
        
        if missing_right > 0:
            right_only = set(right_df[right_on].unique()) - set(merged[right_on].unique())
            logger.debug(f"  Examples of unmatched ISO3 codes from right: {list(right_only)[:5]}")
    
    return merged

def main():
    """Main function to run the script."""
    # Set up command-line argument parser
    import argparse
    parser = argparse.ArgumentParser(description="Calculate Health Expenditure per Standardized Capita with PPP adjustment")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Set console logging level (default: INFO)")
    parser.add_argument("--file-log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="DEBUG",
                        help="Set file logging level (default: DEBUG)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Specify log file (default: auto-generated with timestamp)")
    parser.add_argument("--formula", type=str, choices=["israeli", "ltc", "eu27"], default="israeli",
                        help="Capitation formula to use (default: israeli)")
    parser.add_argument("--reference-year", type=int, default=REFERENCE_YEAR,
                        help=f"Reference year for constant prices and PPP (default: {REFERENCE_YEAR})")
    parser.add_argument("--impute-ppp", action="store_true",
                        help="Enable PPP imputation")
    parser.add_argument("--impute-gdp", action="store_true",
                        help="Enable GDP deflator imputation")
    
    args = parser.parse_args()
    
    # Set up logging with specified levels
    global logger
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    console_level = log_levels[args.log_level]
    file_level = log_levels[args.file_log_level]
    logger = setup_logging(console_level=console_level, file_level=file_level, log_file=args.log_file)
    
    logger.info("Starting Comprehensive Health Expenditure calculation with all indicator combinations...")
    
    # Define reference year and adjustment settings
    reference_year = args.reference_year
    impute_ppp = args.impute_ppp
    impute_gdp = args.impute_gdp
    
    # Capitation formula to use
    formula = args.formula
    
    # Export path
    export_path = Path("Standardized_Expenditure")
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Capitation formula: {formula}")
    logger.info(f"  Reference year: {reference_year}")
    logger.info(f"  PPP imputation: {'Enabled' if impute_ppp else 'Disabled'}")
    logger.info(f"  GDP deflator imputation: {'Enabled' if impute_gdp else 'Disabled'}")
    
    # Load capitation weights - important to capture both return values
    cap_dict, formula_type = load_capitation_weights(formula=formula)
    
    # Load data
    try:
        # Load GHED data with split public/private expenditure
        ghed_data = load_ghed_data()
        logger.debug(f"\nGHED data sample:\n{ghed_data.head()}")
        
        # Load PPP data from World Bank
        ppp_data = load_ppp_data()
        if not ppp_data.empty:
            logger.debug(f"\nPPP data sample:\n{ppp_data.head()}")
        else:
            logger.warning("\nNo PPP data loaded, will proceed without PPP adjustment")
        
        # Load GDP data for deflator calculation
        gdp_deflator = load_gdp_data(reference_year=reference_year)
        if not gdp_deflator.empty:
            logger.debug(f"\nGDP deflator data sample:\n{gdp_deflator.head()}")
            logger.info(f"Reference year for constant prices: {reference_year}")
        else:
            logger.warning("\nNo GDP deflator data loaded, will proceed without constant price adjustment")
        
        # Load and process population data
        male_pop_raw, female_pop_raw = load_population_data()
        
        # Create ISO3 to country mapping for reference
        iso3_to_country = {}
        if not male_pop_raw.empty and 'ISO3' in male_pop_raw.columns and 'Country' in male_pop_raw.columns:
            iso3_to_country = create_iso3_to_country_mapping(male_pop_raw)
            logger.info(f"Created ISO3 to country mapping with {len(iso3_to_country)} entries")
        
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
        logger.info("Starting calculation of health expenditure indicators...")
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
            "GDP_Deflator": "GDP deflator factor (base: reference year)",
            "PPP_Factor": "Current PPP conversion factor (LCU per international $)"
        }
        
        # Save the data dictionary
        dict_filename = f"Health_Expenditure_Data_Dictionary.csv"
        dict_df = pd.DataFrame([{"Column": col, "Description": desc} for col, desc in column_descriptions.items()])
        dict_df.to_csv(export_path / dict_filename, index=False)
        logger.info(f"Data dictionary saved to {export_path / dict_filename}")
        
        # Save the results
        results.to_csv(export_path / filename, index=False)
        logger.info(f"Comprehensive results saved to {export_path / filename}")
        
        # Generate a mapping file for ISO3 to country names as a reference
        if iso3_to_country:
            mapping_df = pd.DataFrame([(iso3, name) for iso3, name in iso3_to_country.items()], 
                                     columns=['ISO3', 'Country_Name'])
            mapping_df.to_csv(export_path / "ISO3_country_mapping.csv", index=False)
            logger.info(f"ISO3 to country mapping saved to {export_path / 'ISO3_country_mapping.csv'}")
        
        # Print merge success statistics
        total_countries = len(iso3_to_country) if iso3_to_country else 0
        matched_countries = results['ISO3'].nunique()
        if total_countries > 0:
            match_percentage = (matched_countries / total_countries) * 100
            logger.info(f"Matched {matched_countries} out of {total_countries} countries ({match_percentage:.1f}%)")
            
        logger.info("Processing completed successfully")
            
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
        
if __name__ == "__main__":
    main()