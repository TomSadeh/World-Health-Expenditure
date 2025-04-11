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

def load_capitation_weights():
    """Load capitation weights from CSV file if it exists, otherwise use default values."""
    try:
        cap_df = pd.read_csv(current_path / "cap.csv", index_col="Age")
        # Convert to dictionary format
        cap_dict = {}
        for age_group in cap_df.index:
            cap_dict[age_group] = {
                "Men": cap_df.loc[age_group, "Men"],
                "Women": cap_df.loc[age_group, "Women"]
            }
        print("Loaded capitation weights from cap.csv")
        return cap_dict
    
    except FileNotFoundError:
        print("cap.csv not found, using default capitation weights")
        return ISRAELI_CAPITATION

def load_ppp_data():
    """
    Load and process World Bank PPP data.
    
    Returns:
        DataFrame with columns: Country, Year, PPP_Factor
    """
    print("Loading PPP data...")
    
    try:
        # Read the PPP data file
        ppp_file = "API_PA.NUS.PPP_DS2_en_csv_v2_13721.csv"
        
        # Load the CSV file, skipping the metadata rows
        ppp_df = pd.read_csv(ppp_file)
        
        # Process to create year-country pairs with PPP values
        ppp_data = []
        
        # Get year columns (exclude metadata columns)
        year_columns = [col for col in ppp_df.columns if col.isdigit()]
        
        # Process each row
        for _, row in ppp_df.iterrows():
            country = row['Country Name']
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
                            "Country": country,
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
            return pd.DataFrame(columns=["Country", "Year", "PPP_Factor"])
        
        print(f"Loaded PPP data with shape: {result.shape}")
        
        return result
    
    except Exception as e:
        print(f"Error loading PPP data: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty DataFrame as fallback
        return pd.DataFrame(columns=["Country", "Year", "PPP_Factor"])

def load_ghed_data():
    """
    Load and process GHED data, splitting expenditure into public and private components.
    Uses total health expenditure (che) and percentages for public (gghed_che) and private (pvtd_che).
    """
    print("Loading GHED data...")
    try:
        # Read the GHED data
        ghed_data = pd.read_excel(current_path / "GHED_data_2025.xlsx", sheet_name="Data")
        
        # Check if the required columns exist
        required_cols = ['location', 'year', 'che']
        recommended_cols = ['gghed_che', 'pvtd_che']
        
        # Verify required columns
        missing_required = [col for col in required_cols if col not in ghed_data.columns]
        if missing_required:
            raise ValueError(f"Required columns missing from GHED data: {missing_required}")
        
        # Check for recommended columns
        missing_recommended = [col for col in recommended_cols if col not in ghed_data.columns]
        if missing_recommended:
            print(f"Warning: Recommended columns missing: {missing_recommended}")
            # If percentages are missing, we'll just use total expenditure
        
        # Select relevant columns
        relevant_cols = ['location', 'year', 'che']
        
        # Add percentage columns if available
        if 'gghed_che' in ghed_data.columns:
            relevant_cols.append('gghed_che')
        if 'pvtd_che' in ghed_data.columns:
            relevant_cols.append('pvtd_che')
        
        # Filter to relevant columns
        ghed_data = ghed_data[relevant_cols]
        
        # Remove rows with missing expenditure data
        ghed_data = ghed_data.dropna(subset=['che'])
        
        # Adjust the numbers to be actual counts and not in millions
        ghed_data['che'] *= 10**6
        
        # Calculate public and private expenditure if percentages are available
        if 'gghed_che' in ghed_data.columns:
            ghed_data['public_expenditure'] = ghed_data['che'] * (ghed_data['gghed_che'] / 100)
        else:
            ghed_data['public_expenditure'] = None
            
        if 'pvtd_che' in ghed_data.columns:
            ghed_data['private_expenditure'] = ghed_data['che'] * (ghed_data['pvtd_che'] / 100)
        else:
            ghed_data['private_expenditure'] = None
        
        # Ensure year is an integer
        ghed_data['year'] = ghed_data['year'].astype(int)
        
        print(f"Loaded GHED data with shape: {ghed_data.shape}")
        return ghed_data
    
    except Exception as e:
        print(f"Error loading GHED data: {e}")
        import traceback
        traceback.print_exc()
        raise

def load_population_data():
    """Load and process population data from CSV files."""
    print("Loading population data...")
    try:
        # Load male and female population data from CSV files
        male_pop_raw = pd.read_csv(current_path / "male_pop.csv")
        female_pop_raw = pd.read_csv(current_path / "female_pop.csv")
        
        print(f"Male population raw data shape: {male_pop_raw.shape}")
        print(f"Female population raw data shape: {female_pop_raw.shape}")
        
        # Print column names to verify
        print(f"Male population columns: {male_pop_raw.columns.tolist()}")
        
        # Process male population data
        male_pop_processed = []
        
        # Get the age group columns
        age_columns = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', 
                       '35-39', '40-44', '45-49', '50-54', '55-59', '60-64',
                       '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+']
        
        # Process each row in the male population data
        for _, row in male_pop_raw.iterrows():
            # Get country and year
            country = row['Region, subregion, country or area *']
            year = row['Year']
            
            # Skip if not a country or year is missing
            if pd.isna(country) or pd.isna(year) or not isinstance(country, str):
                continue
                
            # Skip if this is a region or aggregate
            if any(x in country.lower() for x in ['region', 'world', 'income', 'development', 'more developed', 'less developed']):
                continue
            
             
            # Process each age group
            for age_col in age_columns:
                # Skip if population value is missing
                if pd.isna(row[age_col]):
                    continue
                    
                try:
                    if isinstance(row[age_col], str):
                        # Remove spaces and convert to numeric
                        population_str = row[age_col].replace(' ', '')
                        population = pd.to_numeric(population_str, errors='coerce')
                    else:
                        # Convert population to numeric (it might be already numeric)
                        population = pd.to_numeric(row[age_col], errors='coerce')   
                    if pd.isna(population) or population == 0:
                        continue
                        
                    # Convert population from thousands to actual counts
                    population = population * 1000
                        
                    # Map the age group to our capitation formula age groups
                    mapped_age_group = map_age_group(age_col)
                    if mapped_age_group is None:
                        continue
                        
                    # Add to processed data
                    male_pop_processed.append({
                        "Country": country,
                        "Year": int(year),
                        "Age_Group": mapped_age_group,
                        "Sex": "Men",
                        "Population": population
                    })
                except Exception as e:
                    print(f"Error processing male data for country {country}, year {year}, age {age_col}: {e}")
        
        # Similarly process female data
        female_pop_processed = []
        
        for _, row in female_pop_raw.iterrows():
            country = row['Region, subregion, country or area *']
            year = row['Year']
            
            if pd.isna(country) or pd.isna(year) or not isinstance(country, str):
                continue
                
            if any(x in country.lower() for x in ['region', 'world', 'income', 'development', 'more developed', 'less developed']):
                continue
                
            for age_col in age_columns:
                if pd.isna(row[age_col]):
                    continue
                    
                try:
                    population = pd.to_numeric(row[age_col], errors='coerce')
                    if pd.isna(population) or population == 0:
                        continue
                    
                    # Convert population from thousands to actual counts
                    population = population * 1000    
                        
                    mapped_age_group = map_age_group(age_col)
                    if mapped_age_group is None:
                        continue
                        
                    female_pop_processed.append({
                        "Country": country,
                        "Year": int(year),
                        "Age_Group": mapped_age_group,
                        "Sex": "Women",
                        "Population": population
                    })
                except Exception as e:
                    print(f"Error processing female data for country {country}, year {year}, age {age_col}: {e}")
        
        # Convert lists to DataFrames
        male_pop = pd.DataFrame(male_pop_processed)
        female_pop = pd.DataFrame(female_pop_processed)
        
        # Optionally, handle the case where age groups appear multiple times by aggregating
        male_pop = male_pop.groupby(['Country', 'Year', 'Age_Group', 'Sex']).sum().reset_index()
        female_pop = female_pop.groupby(['Country', 'Year', 'Age_Group', 'Sex']).sum().reset_index()
        
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
        columns = ["Country", "Year", "Age_Group", "Sex", "Population"]
        empty_df = pd.DataFrame(columns=columns)
        return empty_df, empty_df

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

def preprocess_population_data(male_pop, female_pop, cap_dict):
    """
    Calculate standardized population for each country and year using the capitation formula.
    Vectorized version for better performance.
    """
    print("Calculating standardized population (vectorized)...")
    
    # Create a merged dataframe with all combinations of country and year
    country_year_combinations = pd.merge(
        male_pop[['Country', 'Year']].drop_duplicates(),
        female_pop[['Country', 'Year']].drop_duplicates(),
        on=['Country', 'Year'],
        how='inner'
    )
    
    # Initialize standardized population column
    country_year_combinations['Standardized_Population'] = 0.0
    
    # Process each age group with vectorized operations
    for age_group in cap_dict:
        # Filter male and female population for this age group
        male_in_group = male_pop[male_pop['Age_Group'] == age_group]
        female_in_group = female_pop[female_pop['Age_Group'] == age_group]
        
        # Get weights
        male_weight = cap_dict[age_group]['Men']
        female_weight = cap_dict[age_group]['Women']
        
        # Sum male population by country and year, then multiply by weight
        male_weighted = male_in_group.groupby(['Country', 'Year'])['Population'].sum() * male_weight
        
        # Sum female population by country and year, then multiply by weight
        female_weighted = female_in_group.groupby(['Country', 'Year'])['Population'].sum() * female_weight
        
        # Add to standardized population
        for idx, row in country_year_combinations.iterrows():
            country, year = row['Country'], row['Year']
            try:
                # Add male weighted population if available
                if (country, year) in male_weighted.index:
                    country_year_combinations.at[idx, 'Standardized_Population'] += male_weighted.get((country, year), 0)
                
                # Add female weighted population if available
                if (country, year) in female_weighted.index:
                    country_year_combinations.at[idx, 'Standardized_Population'] += female_weighted.get((country, year), 0)
            except Exception as e:
                print(f"Error adding weighted population for {country}, {year}, {age_group}: {e}")
    
    return country_year_combinations

def load_gdp_data(reference_year=2017):
    """
    Load and process World Bank GDP data to calculate GDP deflators.
    
    Args:
        reference_year: Year to use as the reference for constant prices (default: 2017)
    
    Returns:
        DataFrame with columns: Country, Year, GDP_Deflator
    """
    print("Loading GDP data for deflator calculation...")
    
    try:
        # Read the GDP files (current LCU and constant LCU)
        gdp_current_file = "API_NY.GDP.MKTP.CN_DS2_en_csv_v2_26332.csv"
        gdp_constant_file = "API_NY.GDP.MKTP.KN_DS2_en_csv_v2_13325.csv"
        
        # Load the CSV files
        gdp_current_df = pd.read_csv(gdp_current_file)
        gdp_constant_df = pd.read_csv(gdp_constant_file)
        
        # Process both datasets to create year-country pairs with GDP values
        gdp_current_data = []
        gdp_constant_data = []
        
        # Get year columns (exclude metadata columns)
        year_columns = [col for col in gdp_current_df.columns if col.isdigit()]
        
        # Process current GDP data
        for _, row in gdp_current_df.iterrows():
            country = row['Country Name']
            
            # Skip aggregate regions
            if any(x in country.lower() for x in ['region', 'world', 'income', 'development']):
                continue
                
            for year in year_columns:
                if pd.notna(row[year]) and row[year] != 0:
                    gdp_current_data.append({
                        "Country": country,
                        "Year": int(year),
                        "GDP_Current": float(row[year])
                    })
        
        # Process constant GDP data
        for _, row in gdp_constant_df.iterrows():
            country = row['Country Name']
            
            # Skip aggregate regions
            if any(x in country.lower() for x in ['region', 'world', 'income', 'development']):
                continue
                
            for year in year_columns:
                if pd.notna(row[year]) and row[year] != 0:
                    gdp_constant_data.append({
                        "Country": country,
                        "Year": int(year),
                        "GDP_Constant": float(row[year])
                    })
        
        # Convert to DataFrames
        gdp_current_df = pd.DataFrame(gdp_current_data)
        gdp_constant_df = pd.DataFrame(gdp_constant_data)
        
        # Merge the datasets
        gdp_merged = pd.merge(
            gdp_current_df,
            gdp_constant_df,
            on=["Country", "Year"],
            how="inner"
        )
        
        # Calculate GDP deflator (GDP_Current / GDP_Constant)
        gdp_merged["GDP_Deflator"] = gdp_merged["GDP_Current"] / gdp_merged["GDP_Constant"]
        
        # For each country, normalize the deflator by the reference year
        gdp_deflator = []
        for country in gdp_merged["Country"].unique():
            country_data = gdp_merged[gdp_merged["Country"] == country].copy()
            
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
                    print(f"Using {closest_year} as reference year for {country} (reference {reference_year} not available)")
                else:
                    # No data for this country
                    country_data["GDP_Deflator_Normalized"] = np.nan
            
            # Add to the result list
            gdp_deflator.append(country_data[["Country", "Year", "GDP_Deflator_Normalized"]])
        
        # Combine all countries
        result = pd.concat(gdp_deflator, ignore_index=True)
        result = result.rename(columns={"GDP_Deflator_Normalized": "GDP_Deflator"})
        
        print(f"Loaded GDP deflator data with shape: {result.shape}")
        
        return result
    
    except Exception as e:
        print(f"Error loading GDP data: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty DataFrame as fallback
        return pd.DataFrame(columns=["Country", "Year", "GDP_Deflator"])

def apply_gdp_deflator_adjustment(data, gdp_deflator, impute_missing=False):
    """
    Apply GDP deflator adjustment to health expenditure data to convert to constant prices.
    
    Args:
        data: DataFrame with health expenditure data
        gdp_deflator: DataFrame with GDP deflators
        impute_missing: Whether to impute missing GDP deflators (default: False)
    
    Returns:
        DataFrame with constant price adjusted health expenditure
    """
    print("Applying GDP deflator adjustment for constant prices...")
    
    # Make a copy to avoid modifying the original data
    adjusted_data = data.copy()
    
    # Merge GDP deflator data with health expenditure data
    merged = pd.merge(
        adjusted_data,
        gdp_deflator[['Country', 'Year', 'GDP_Deflator']],
        on=['Country', 'Year'],
        how='left'
    )
    
    # Check if any GDP deflators are missing
    missing_deflator = merged['GDP_Deflator'].isna().sum()
    if missing_deflator > 0:
        print(f"Warning: Missing GDP deflators for {missing_deflator} out of {len(merged)} rows")
        
        # For countries/years with missing deflators, impute using nearest available year if impute_missing is True
        if impute_missing:
            print(f"Imputing missing GDP deflators...")
            countries_with_missing = merged[merged['GDP_Deflator'].isna()]['Country'].unique()
            
            for country in countries_with_missing:
                country_data = merged[merged['Country'] == country]
                missing_years = country_data[country_data['GDP_Deflator'].isna()]['Year'].tolist()
                
                # If country has any deflator data, use nearest year
                if any(~country_data['GDP_Deflator'].isna()):
                    available_years = country_data[~country_data['GDP_Deflator'].isna()]['Year'].tolist()
                    
                    for missing_year in missing_years:
                        # Find closest available year
                        closest_year = min(available_years, key=lambda x: abs(x - missing_year))
                        closest_deflator = country_data[country_data['Year'] == closest_year]['GDP_Deflator'].iloc[0]
                        
                        # Impute the missing value
                        idx = merged[(merged['Country'] == country) & (merged['Year'] == missing_year)].index
                        merged.loc[idx, 'GDP_Deflator'] = closest_deflator
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

def apply_ppp_adjustment(data, ppp_data, base_country="United States", reference_year=2017, impute_missing=False):
    """
    Apply PPP adjustment to health expenditure data.
    
    Args:
        data: DataFrame with health expenditure data
        ppp_data: DataFrame with PPP conversion factors
        base_country: Base country for PPP comparisons
        reference_year: Reference year for PPP adjustment
        impute_missing: Whether to impute missing PPP factors (default: False)
    
    Returns:
        DataFrame with PPP-adjusted health expenditure
    """
    # Make a copy to avoid modifying the original data
    adjusted_data = data.copy()
    
    # Merge PPP data with health expenditure data
    merged = pd.merge(
        adjusted_data,
        ppp_data[['Country', 'Year', 'PPP_Factor']],
        on=['Country', 'Year'],
        how='left'
    )
    
    # Check if any PPP factors are missing
    missing_ppp = merged['PPP_Factor'].isna().sum()
    if missing_ppp > 0:
        print(f"Warning: Missing PPP factors for {missing_ppp} out of {len(merged)} rows")
        
        # For countries/years with missing PPP factors, impute using nearest available year if impute_missing is True
        if impute_missing:
            print(f"Imputing missing PPP factors...")
            countries_with_missing = merged[merged['PPP_Factor'].isna()]['Country'].unique()
            
            for country in countries_with_missing:
                country_data = merged[merged['Country'] == country]
                missing_years = country_data[country_data['PPP_Factor'].isna()]['Year'].tolist()
                
                # If country has any PPP data, use nearest year
                if any(~country_data['PPP_Factor'].isna()):
                    available_years = country_data[~country_data['PPP_Factor'].isna()]['Year'].tolist()
                    
                    for missing_year in missing_years:
                        # Find closest available year
                        closest_year = min(available_years, key=lambda x: abs(x - missing_year))
                        closest_ppp = country_data[country_data['Year'] == closest_year]['PPP_Factor'].iloc[0]
                        
                        # Impute the missing value
                        idx = merged[(merged['Country'] == country) & (merged['Year'] == missing_year)].index
                        merged.loc[idx, 'PPP_Factor'] = closest_ppp
        else:
            print(f"Skipping imputation for missing PPP factors as requested")
    
    # Check if we have any PPP factors for the base country
    base_country_ppp = ppp_data[ppp_data['Country'] == base_country]
    
    if base_country_ppp.empty:
        print(f"Warning: No PPP factors available for base country ({base_country})")
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

def calculate_expenditure_per_std_capita(ghed_data, standardized_pop, ppp_data=None, gdp_deflator=None, impute_missing_ppp=False, impute_missing_gdp=False):
    """
    Calculate health expenditure per standardized capita, for total, public, and private expenditure.
    Optionally applies PPP adjustment if PPP data is provided and constant price adjustment if GDP deflator is provided.
    
    Args:
        ghed_data: DataFrame with GHED data
        standardized_pop: DataFrame with standardized population data
        ppp_data: DataFrame with PPP conversion factors (optional)
        gdp_deflator: DataFrame with GDP deflators (optional)
        impute_missing_ppp: Whether to impute missing PPP factors (default: False)
        impute_missing_gdp: Whether to impute missing GDP deflators (default: False)
    
    Returns:
        DataFrame with health expenditure per standardized capita
    """
    print("Calculating health expenditure per standardized capita...")
    
    # Rename columns for consistency
    ghed_renamed = ghed_data.rename(columns={
        'location': 'Country',
        'year': 'Year',
        'che': 'Total_Health_Expenditure',
        'public_expenditure': 'Public_Health_Expenditure',
        'private_expenditure': 'Private_Health_Expenditure'
    })
    
    # Merge data
    merged_data = pd.merge(
        ghed_renamed, 
        standardized_pop,
        on=["Country", "Year"],
        how="inner"
    )
    
    # Calculate expenditure per standardized capita for total, public, and private
    merged_data["Total_Health_Expenditure_per_Std_Capita"] = merged_data["Total_Health_Expenditure"] / merged_data["Standardized_Population"]
    
    # Calculate public and private expenditure per standardized capita if available
    if 'Public_Health_Expenditure' in merged_data.columns and not merged_data['Public_Health_Expenditure'].isna().all():
        merged_data["Public_Health_Expenditure_per_Std_Capita"] = merged_data["Public_Health_Expenditure"] / merged_data["Standardized_Population"]
    
    if 'Private_Health_Expenditure' in merged_data.columns and not merged_data['Private_Health_Expenditure'].isna().all():
        merged_data["Private_Health_Expenditure_per_Std_Capita"] = merged_data["Private_Health_Expenditure"] / merged_data["Standardized_Population"]
    
    # Apply GDP deflator adjustment for constant prices if data is provided
    if gdp_deflator is not None and not gdp_deflator.empty:
        print("Applying GDP deflator adjustment for constant prices...")
        merged_data = apply_gdp_deflator_adjustment(merged_data, gdp_deflator, impute_missing=impute_missing_gdp)
    
    # Apply PPP adjustment if data is provided
    if ppp_data is not None and not ppp_data.empty:
        print("Applying PPP adjustment to expenditure data...")
        merged_data = apply_ppp_adjustment(merged_data, ppp_data, impute_missing=impute_missing_ppp)
    
    return merged_data

def document_imputation(data, ppp_data, gdp_deflator, output_path, impute_missing_ppp=True, impute_missing_gdp=True):
    """
    Document where data imputation is used and generate a report.
    
    Args:
        data: DataFrame with health expenditure data
        ppp_data: DataFrame with PPP conversion factors
        gdp_deflator: DataFrame with GDP deflators
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
        # Merge health expenditure data with PPP data to identify missing values
        merged_ppp = pd.merge(
            data[['Country', 'Year']],
            ppp_data[['Country', 'Year', 'PPP_Factor']],
            on=['Country', 'Year'],
            how='left'
        )
        
        # Identify rows with missing PPP factors
        missing_ppp = merged_ppp[merged_ppp['PPP_Factor'].isna()]
        
        # Document each case of PPP imputation
        for country in missing_ppp['Country'].unique():
            country_missing = missing_ppp[missing_ppp['Country'] == country]
            missing_years = country_missing['Year'].tolist()
            
            # Find available PPP data for this country
            country_available = ppp_data[ppp_data['Country'] == country]
            
            if not country_available.empty:
                available_years = country_available['Year'].tolist()
                
                for missing_year in missing_years:
                    # Find closest available year for imputation
                    if available_years:
                        closest_year = min(available_years, key=lambda x: abs(x - missing_year))
                        imputation_records.append({
                            'Country': country,
                            'Year': missing_year,
                            'Missing_Data': 'PPP Factor',
                            'Imputation_Method': f'Nearest year ({closest_year})',
                            'Value_Source': f'PPP Factor from {closest_year}'
                        })
                    else:
                        # No data available for this country
                        imputation_records.append({
                            'Country': country,
                            'Year': missing_year,
                            'Missing_Data': 'PPP Factor',
                            'Imputation_Method': 'None - No data available',
                            'Value_Source': 'Missing'
                        })
    
    # Check GDP deflator imputation
    if impute_missing_gdp and gdp_deflator is not None and not gdp_deflator.empty:
        # Merge health expenditure data with GDP deflator data to identify missing values
        merged_gdp = pd.merge(
            data[['Country', 'Year']],
            gdp_deflator[['Country', 'Year', 'GDP_Deflator']],
            on=['Country', 'Year'],
            how='left'
        )
        
        # Identify rows with missing GDP deflators
        missing_gdp = merged_gdp[merged_gdp['GDP_Deflator'].isna()]
        
        # Document each case of GDP deflator imputation
        for country in missing_gdp['Country'].unique():
            country_missing = missing_gdp[missing_gdp['Country'] == country]
            missing_years = country_missing['Year'].tolist()
            
            # Find available GDP deflator data for this country
            country_available = gdp_deflator[gdp_deflator['Country'] == country]
            
            if not country_available.empty:
                available_years = country_available['Year'].tolist()
                
                for missing_year in missing_years:
                    # Find closest available year for imputation
                    if available_years:
                        closest_year = min(available_years, key=lambda x: abs(x - missing_year))
                        imputation_records.append({
                            'Country': country,
                            'Year': missing_year,
                            'Missing_Data': 'GDP Deflator',
                            'Imputation_Method': f'Nearest year ({closest_year})',
                            'Value_Source': f'GDP Deflator from {closest_year}'
                        })
                    else:
                        # No data available for this country
                        imputation_records.append({
                            'Country': country,
                            'Year': missing_year,
                            'Missing_Data': 'GDP Deflator',
                            'Imputation_Method': 'None - No data available',
                            'Value_Source': 'Missing'
                        })
    
    # Convert to DataFrame
    imputation_df = pd.DataFrame(imputation_records)
    
    # Sort by country, year, and type of missing data
    if not imputation_df.empty:
        imputation_df = imputation_df.sort_values(['Country', 'Year', 'Missing_Data'])
        
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
        country_counts = imputation_df['Country'].value_counts()
        print("\nTop 10 countries with most imputations:")
        for country, count in country_counts.head(10).items():
            print(f"  {country}: {count} records")
    else:
        print("No imputation was performed or documented.")
    
    return imputation_df

def standardize_country_names(data, country_column='Country'):
    """
    Standardize country names to ensure consistency across datasets.
    
    Args:
        data: DataFrame containing country names
        country_column: Name of the column containing country names
    
    Returns:
        DataFrame with standardized country names
    """
    # Create a mapping dictionary for country name variations
    country_mapping = {
        # Common variations in spellings
        "United States of America": "United States",
        "U.S.A.": "United States",
        "USA": "United States",
        "US": "United States",
        "United States (USA)": "United States",
        
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "UK": "United Kingdom",
        "Britain": "United Kingdom",
        "Great Britain": "United Kingdom",
        
        "Russian Federation": "Russia",
        
        "China, People's Republic of": "China",
        "China (Mainland)": "China",
        "People's Republic of China": "China",
        
        "Iran (Islamic Republic of)": "Iran",
        "Iran, Islamic Republic of": "Iran",
        
        "Korea, Republic of": "South Korea",
        "Republic of Korea": "South Korea",
        "Korea, South": "South Korea",
        
        "Korea, Democratic People's Republic of": "North Korea",
        "Democratic People's Republic of Korea": "North Korea",
        "Korea, North": "North Korea",
        
        "Venezuela, Bolivarian Republic of": "Venezuela",
        "Venezuela (Bolivarian Republic of)": "Venezuela",
        
        "Viet Nam": "Vietnam",
        
        "Bolivia (Plurinational State of)": "Bolivia",
        "Bolivia, Plurinational State of": "Bolivia",
        
        "Tanzania, United Republic of": "Tanzania",
        
        "Syrian Arab Republic": "Syria",
        
        "Republic of Moldova": "Moldova",
        "Moldova, Republic of": "Moldova",
        
        "Democratic Republic of the Congo": "Congo, Dem. Rep.",
        "Congo, Democratic Republic of the": "Congo, Dem. Rep.",
        "DR Congo": "Congo, Dem. Rep.",
        "Congo-Kinshasa": "Congo, Dem. Rep.",
        
        "Congo, Republic of the": "Congo, Rep.",
        "Republic of Congo": "Congo, Rep.",
        "Congo-Brazzaville": "Congo, Rep.",
        
        "Lao People's Democratic Republic": "Lao PDR",
        "Laos": "Lao PDR",
        
        "Brunei Darussalam": "Brunei",
        
        "Türkiye": "Turkey",
        
        "Czechia": "Czech Republic",
        
        "Myanmar": "Burma",
        
        "North Macedonia": "Macedonia, FYR",
        "Macedonia, The Former Yugoslav Republic of": "Macedonia, FYR",
        
        "Cabo Verde": "Cape Verde",
        
        "The Gambia": "Gambia, The",
        "Gambia": "Gambia, The",
        
        "Kyrgyz Republic": "Kyrgyzstan",
        
        "Timor-Leste": "East Timor",
        "Timor Leste": "East Timor",
        
        "Micronesia, Federated States of": "Micronesia, Fed. Sts.",
        "Micronesia": "Micronesia, Fed. Sts.",
        
        "Saint Kitts and Nevis": "St. Kitts and Nevis",
        
        "Saint Lucia": "St. Lucia",
        
        "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
        
        "Eswatini": "Swaziland",
        
        "Taiwan, Province of China": "Taiwan",
        "Taiwan, China": "Taiwan",
        "Chinese Taipei": "Taiwan",
        
        "Hong Kong, China": "Hong Kong SAR, China",
        "Hong Kong": "Hong Kong SAR, China",
        
        "Macao": "Macao SAR, China",
        "Macau": "Macao SAR, China",
        "Macao SAR": "Macao SAR, China",
        
        "Palestinian Authority": "West Bank and Gaza",
        "Palestine, State of": "West Bank and Gaza",
        "Palestinian Territories": "West Bank and Gaza",
        "Occupied Palestinian Territory": "West Bank and Gaza",
        
        # Add any other mappings you encounter in your data
    }
    
    # Create a copy of the data to avoid modifying the original
    data_copy = data.copy()
    
    # Apply the mapping to standardize country names
    if country_column in data_copy.columns:
        # First, check if there are countries in the data that need standardization
        countries_to_standardize = [country for country in data_copy[country_column].unique() 
                                  if country in country_mapping]
        
        if countries_to_standardize:
            print(f"Standardizing the following country names: {countries_to_standardize}")
            
            # Apply the mapping
            data_copy[country_column] = data_copy[country_column].replace(country_mapping)
    
    return data_copy

def main():
    """Main function to run the script."""
    print("Starting Health Expenditure per Standardized Capita calculation with PPP and GDP deflator adjustments...")
    
    # Define reference year and base country
    impute_ppp = False  # Set this to True to enable PPP imputation
    impute_gdp = False  # Set this to True to enable GDP deflator imputation
    
    # Export
    export_path = Path("Standardized_Expenditure")
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Load capitation weights
    cap_dict = load_capitation_weights()
    
    # Load data
    try:
        # Load GHED data with split public/private expenditure
        ghed_data = load_ghed_data()
        ghed_data = standardize_country_names(ghed_data, country_column='location')
        print(f"\nGHED data sample:\n{ghed_data.head()}")
        
        # Load PPP data from World Bank
        ppp_data = load_ppp_data()
        if not ppp_data.empty:
            print(f"\nPPP data sample:\n{ppp_data.head()}")
            ppp_data = standardize_country_names(ppp_data)
        else:
            print("\nWarning: No PPP data loaded, will proceed without PPP adjustment")
        
        # Load GDP data for deflator calculation
        gdp_deflator = load_gdp_data(reference_year=REFERENCE_YEAR)
        if not gdp_deflator.empty:
            print(f"\nGDP deflator data sample:\n{gdp_deflator.head()}")
            print(f"Reference year for constant prices: {REFERENCE_YEAR}")
            gdp_deflator = standardize_country_names(gdp_deflator)
        else:
            print("\nWarning: No GDP deflator data loaded, will proceed without constant price adjustment")
        
        # Load and process population data
        male_pop_raw, female_pop_raw = load_population_data()
        male_pop = standardize_country_names(male_pop_raw, country_column='Region, subregion, country or area *')
        female_pop = standardize_country_names(female_pop_raw, country_column='Region, subregion, country or area *')
        # Calculate standardized population
        standardized_pop = preprocess_population_data(male_pop, female_pop, cap_dict)
        
        # Create base data for imputation documentation
        # Rename GHED columns for consistency with the rest of the data
        base_data = ghed_data.rename(columns={
            'location': 'Country',
            'year': 'Year',
            'che': 'Total_Health_Expenditure',
            'public_expenditure': 'Public_Health_Expenditure',
            'private_expenditure': 'Private_Health_Expenditure'
        })
        
        # Merge with standardized population to get the complete dataset
        base_data = pd.merge(
            base_data, 
            standardized_pop,
            on=["Country", "Year"],
            how="inner"
        )
        
        # Document imputation process before applying adjustments
        if impute_ppp or impute_gdp:
            document_imputation(
                base_data,
                ppp_data,
                gdp_deflator,
                export_path,
                impute_missing_ppp=impute_ppp,
                impute_missing_gdp=impute_gdp
            )
        
        # Calculate health expenditure per standardized capita with PPP and GDP deflator adjustments
        results = calculate_expenditure_per_std_capita(
            ghed_data, 
            standardized_pop, 
            ppp_data=ppp_data, 
            gdp_deflator=gdp_deflator,
            impute_missing_ppp=impute_ppp,
            impute_missing_gdp=impute_gdp
        )
        
        
        
        # Add imputation info to the filename if imputation is disabled
        filename = "Health_Expenditure_per_Std_Capita"
        if not impute_ppp or not impute_gdp:
            # Add suffix to filename to indicate imputation settings
            imputation_status = []
            if not impute_ppp:
                imputation_status.append("no_ppp_imputed")
            if not impute_gdp:
                imputation_status.append("no_gdp_imputed")
            
            filename += "_" + "_".join(imputation_status)
        
        filename += ".csv"
        results.to_csv(export_path / filename, index=False)
        
        print(f"\nResults saved to {export_path / filename}")
        
        """
        # Create summary with averages by country
        create_summary_reports(results, export_path)
        
        print(f"\nResults saved to {export_path} directory")
        """
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        raise
        
if __name__ == "__main__":
    main()