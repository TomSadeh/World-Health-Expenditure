# World Health Expenditure

This tool calculates Health Expenditure per Standardized Capita with Purchasing Power Parity (PPP) adjustment by processing data from multiple international sources:

- [Global Health Expenditure Database (GHED)](https://apps.who.int/nha/database) from WHO
- [World Population Prospects (WPP)](https://population.un.org/wpp/) from the United Nations 
- [World Bank PPP conversion factors](https://data.worldbank.org/indicator/PA.NUS.PPP)

The script standardizes health expenditure measurement across countries by applying selectable capitation formulas to account for demographic differences and adjusts for purchasing power parity to enable meaningful cross-country comparisons.

## Features
- **Multiple capitation formulas**: Choose between Israeli, LTC, or EU27 capitation formulas
- **Demographic standardization**: Uses capitation formula weights to adjust for population age and gender distributions
- **Expenditure component analysis**: Separates total health expenditure into public and private components
- **Comprehensive adjustment combinations**: Calculates all combinations of current/constant prices and current/constant PPP
- **Constant price conversion**: Applies GDP deflators to convert nominal values to constant prices
- **Purchasing power parity adjustment**: Normalizes expenditure across countries using PPP conversion factors
- **Time series consistency**: Option to use constant PPP factors from reference year for better time series comparisons
- **Missing data handling**: Optional imputation for missing PPP and GDP deflator values
- **Country name standardization**: Harmonizes country names across different datasets
- **Comprehensive reporting**: Generates detailed output with imputation documentation and data dictionary
- **Detailed logging**: Configurable logging to both console and file for better tracking and debugging

## Requirements

### Python Dependencies
The required Python packages are listed in the `requirements.txt` file. Main dependencies include:
- pandas
- numpy
- pathlib
- regex
- logging

### Required Data Files
- [`GHED_data_2025.xlsx`](https://apps.who.int/nha/database/Select/Indicators/en): Health expenditure data from the Global Health Expenditure Database
  - *Note: Can be converted to optimized CSV format using the included `ghed_to_csv.py` script*
- World Population Prospects (WPP) data:
  - `WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`: Male population data by age groups
  - `WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`: Female population data by age groups
  - *Note: These files are processed into CSV format using the included `pop_data_processor.py` script*
- `cap.csv`: Capitation formula weights by age group (contains [Israeli, LTC, and EU27](https://www.vanleer.org.il/publication/%D7%A4%D7%A8%D7%95%D7%A4%D7%99%D7%9C-%D7%94%D7%94%D7%95%D7%A6%D7%90%D7%94-%D7%A2%D7%9C-%D7%91%D7%A8%D7%99%D7%90%D7%95%D7%AA-%D7%9C%D7%A4%D7%99-%D7%92%D7%99%D7%9C-%D7%91%D7%99%D7%A9%D7%A8%D7%90%D7%9C) formulas)
- World Bank data files:
  - `API_PA.NUS.PPP_DS2_en_csv_v2_13721.csv`: PPP conversion factors
  - `API_NY.GDP.MKTP.CN_DS2_en_csv_v2_26332.csv`: GDP in current LCU
  - `API_NY.GDP.MKTP.KN_DS2_en_csv_v2_13325.csv`: GDP in constant LCU
  - *Note: These files are processed using the included `wb_data_processor.py` script*

### Generated Files
- `data/processed/ghed_data_optimized.csv`: Optimized GHED data (created by `ghed_to_csv.py`)
- `data/processed/male_pop.csv`: Processed male population data (created by `pop_data_processor.py`)
- `data/processed/female_pop.csv`: Processed female population data (created by `pop_data_processor.py`)
- `data/processed/gdp_current.csv`: Processed GDP current price data (created by `wb_data_processor.py`)
- `data/processed/gdp_constant.csv`: Processed GDP constant price data (created by `wb_data_processor.py`)
- `data/processed/ppp.csv`: Processed PPP data (created by `wb_data_processor.py`)
- `Standardized_Expenditure/logs/`: Directory containing detailed log files

## Installation

1. Clone this repository
2. Install required dependencies using the requirements file:
   ```
   pip install -r requirements.txt
   ```
3. Place the required data files in the "data" directory if they are not in it
4. Process the input data files using the preprocessing scripts (see next section)

## Data Preprocessing

Before running the main analysis script, you should preprocess the input data files. This ensures consistent formatting and improves performance. The repository includes a master preprocessing script that runs all preprocessing steps in sequence:

```
python preprocess_all.py
```

This script will run all three preprocessing scripts in the correct order:
1. GHED Excel to CSV conversion
2. World Bank data processing 
3. Population data processing

The script also verifies that all required processed files were created successfully.

Alternatively, you can run each preprocessing script individually:

### 1. GHED Excel to CSV Conversion

Convert the GHED Excel file to an optimized CSV format:

```
python ghed_to_csv.py
```

This preprocessing step offers several advantages:
- Significantly faster loading times (often 10-20x faster)
- Reduced memory usage
- Only extracts the necessary columns
- Pre-calculates public and private expenditure components

The converter will create a file called `ghed_data_optimized.csv` in the `data/processed` directory.

### 2. World Bank Data Processing

Process the World Bank data files to remove metadata rows and standardize formatting:

```
python wb_data_processor.py
```

This script:
- Removes the first 4 rows from the original World Bank data files (metadata rows)
- Saves the processed files to the `data/processed` directory with simplified names:
  - `gdp_current.csv` (from `API_NY.GDP.MKTP.CN_DS2_en_csv_v2_26332.csv`)
  - `gdp_constant.csv` (from `API_NY.GDP.MKTP.KN_DS2_en_csv_v2_13325.csv`)
  - `ppp.csv` (from `API_PA.NUS.PPP_DS2_en_csv_v2_13721.csv`)

### 3. Population Data Processing

Process the World Population Prospects Excel files into CSV format:

```
python pop_data_processor.py
```

This script:
- Reads the WPP data from the original Excel files (headers on row 17, from the "Estimates" worksheet)
- Converts numeric data with space-separated thousands to standard numeric format
- Saves the processed files to the `data/processed` directory as:
  - `male_pop.csv` (from `WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`)
  - `female_pop.csv` (from `WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`)

The main script (`whe.py`) automatically uses these processed files when available, or falls back to the original files if necessary.
## Running the Main Script

After preprocessing the data, run the main script:

```
python whe.py
```

With command-line options:

```
python whe.py --log-level INFO --file-log-level DEBUG --formula ltc --reference-year 2017 --impute-ppp --impute-gdp
```

### Command Line Arguments

The script supports the following command line arguments:

- `--log-level`: Set console logging level (DEBUG, INFO, WARNING, ERROR; default: INFO)
- `--file-log-level`: Set file logging level (DEBUG, INFO, WARNING, ERROR; default: DEBUG)
- `--log-file`: Specify log file path (default: auto-generated with timestamp)
- `--formula`: Capitation formula to use (israeli, ltc, eu27; default: israeli)
- `--reference-year`: Reference year for constant prices and PPP (default: 2017)
- `--impute-ppp`: Enable PPP imputation (default: disabled)
- `--impute-gdp`: Enable GDP deflator imputation (default: disabled)

To see minimal console output but capture detailed logs in a file:

```
python whe.py --log-level WARNING --file-log-level DEBUG
```

## Output

The script creates a directory named `Standardized_Expenditure` containing:

- `Health_Expenditure_Comprehensive_{formula_type}_ref{reference_year}.csv`: Main output file with all calculated metrics and adjustment combinations
- `Health_Expenditure_Data_Dictionary.csv`: Detailed description of each column in the output file
- `ISO3_country_mapping.csv`: Mapping between ISO3 codes and country names
- `imputation_documentation.csv`: Records of data imputation if enabled
- `logs/`: Directory containing detailed log files with timestamps

Filename suffixes indicate formula type, reference year, and imputation settings used.

## Methodology

### Capitation Formulas

The script supports multiple capitation formulas:

1. **Israeli Capitation Formula**: Assigns different weights to men and women in various age groups
2. **LTC Formula**: Adjusted weights that account for long-term care needs
3. **EU27 Formula**: Based on average utilization patterns across EU member states

The default Israeli Capitation Formula assigns weights to different age-gender groups to adjust for varying healthcare needs:

| Age Group | Men | Women |
|-----------|-----|-------|
| 0 to 4 | 1.55 | 1.26 |
| 5 to 14 | 0.48 | 0.38 |
| 15 to 24 | 0.42 | 0.63 |
| 25 to 34 | 0.57 | 1.07 |
| 35 to 44 | 0.68 | 0.91 |
| 45 to 54 | 1.07 | 1.32 |
| 55 to 64 | 1.86 | 1.79 |
| 65 to 74 | 2.90 | 2.36 |
| 75 to 84 | 3.64 | 3.23 |
| 85 and over | 3.64 | 2.70 |

These weights can be customized by providing a cap.csv file.

### Adjustment Combinations

The tool now calculates health expenditure with four different adjustment combinations:

1. **Current prices, current PPP**
   - Uses prices from each year
   - Uses PPP factors from each year
   - Good for: Point-in-time comparisons within a single year

2. **Current prices, constant PPP**
   - Uses prices from each year
   - Uses PPP factors only from reference year
   - Good for: Removing the effect of changing PPP factors

3. **Constant prices, current PPP**
   - Adjusts all prices to reference year (removes inflation)
   - Uses PPP factors from each year
   - Good for: Removing the effect of inflation

4. **Constant prices, constant PPP**
   - Adjusts all prices to reference year (removes inflation)
   - Uses PPP factors only from reference year
   - Good for: Time series analysis and cross-country comparisons
   - **Most methodologically sound for longitudinal comparisons**

### Data Processing Steps

### Pre-processing (recommended)
1. **GHED Data Optimization**: Convert Excel to optimized CSV format using `ghed_to_csv.py`
2. **World Bank Data Processing**: Process World Bank files using `wb_data_processor.py`
3. **Population Data Processing**: Convert WPP Excel files to CSV using `pop_data_processor.py`

### Main Processing
1. **Data Loading**: Reads the processed data files (or falls back to original files if needed)
2. **Country Name Standardization**: Harmonizes country names across datasets
3. **Population Standardization**: Applies capitation weights to demographic data
4. **Base Indicator Calculation**: Computes per standardized capita metrics with current prices
5. **GDP Deflator Adjustment**: Converts to constant prices using GDP deflators (if available)
6. **Current PPP Adjustment**: Applies current-year PPP factors to both current and constant prices
7. **Constant PPP Adjustment**: Applies reference-year PPP factors to both current and constant prices
8. **Imputation Documentation**: Records any data imputations performed
9. **Output Generation**: Creates comprehensive CSV with all indicator combinations and data dictionary

### Age Group Mapping

The script maps WPP age groups to capitation formula age groups as follows:

| WPP Age Groups | Capitation Age Group |
|----------------|---------------------|
| 0-4 | 0 to 4 |
| 5-9, 10-14 | 5 to 14 |
| 15-19, 20-24 | 15 to 24 |
| 25-29, 30-34 | 25 to 34 |
| 35-39, 40-44 | 35 to 44 |
| 45-49, 50-54 | 45 to 54 |
| 55-59, 60-64 | 55 to 64 |
| 65-69, 70-74 | 65 to 74 |
| 75-79, 80-84 | 75 to 84 |
| 85-89, 90-94, 95-99, 100+ | 85 and over |

## Data Limitations
### World Population Prospects (WPP) Data

It's important to note that the [World Population Prospects (WPP)](https://population.un.org/wpp/) data used in this script are not direct census counts but rather estimates produced through various demographic methods. The United Nations Population Division generates these estimates using:

- Multiple data sources including censuses, surveys, and administrative records
- Demographic modeling techniques to estimate and project population distributions
- Statistical methods to impute missing values and reconcile inconsistencies
- Standardized approaches to handle varying data quality across countries

These population estimates undergo rigorous validation but inherently contain some uncertainty, especially for countries with limited data collection infrastructure. Users should be aware that the standardized population calculations in this tool reflect these WPP estimation methodologies.

### Global Health Expenditure Database (GHED) Data

The [Global Health Expenditure Database (GHED)](https://apps.who.int/nha/database) data used in this script are compiled by the World Health Organization (WHO) and represent the most comprehensive source of health spending information. This script uses only highly aggregated GHED indicators, which should consist of non-imputed data reported directly by countries. It's worth noting that:

- Data are collected through National Health Accounts (NHA) frameworks that countries report to WHO
- The aggregated indicators used in this script represent the most reliable portion of GHED data
- By focusing on top-level expenditure categories, the script avoids many of the data quality issues present in more granular GHED components
- Even at the aggregate level, methodological differences exist across countries in how health expenditures are categorized and reported
- Revisions to historical data occur as countries improve their health accounting systems
- Data quality varies by country, with high-income countries generally having more reliable estimates

While GHED represents the global standard for health expenditure data, users should understand these underlying limitations when interpreting results.

### World Bank PPP and GDP Data

The [World Bank data](https://data.worldbank.org/) on Purchasing Power Parity (PPP) conversion factors and GDP used in this script come with several considerations:

- PPP conversion factors are derived from the [International Comparison Program (ICP)](https://www.worldbank.org/en/programs/icp), which conducts comprehensive price surveys only periodically (typically every 6 years)
- Values for non-benchmark years are estimated through extrapolation and modeling
- GDP deflators reflect countries' own national accounting practices, which may vary in methodology
- Data revisions are common as countries update their national accounts
- Coverage varies by country and year, with some developing economies having less reliable data

These datasets undergo extensive quality assurance by the World Bank but necessarily contain estimation uncertainty that carries through to the final calculated indicators.

## Output Variables

The generated dataset includes the following key variables:

### Base Metrics
- `ISO3`: ISO3 country code
- `Country`: Country name
- `Year`: Year of data
- `Standardized_Population`: Population standardized using capitation formula
- `Total_Health_Expenditure`: Total health expenditure in local currency units (LCU)
- `Public_Health_Expenditure`: Public health expenditure in LCU
- `Private_Health_Expenditure`: Private health expenditure in LCU

### Current Prices Indicators (No PPP)
- `THE_per_Std_Capita_Current`: Total health expenditure per standardized capita (current LCU)
- `PubHE_per_Std_Capita_Current`: Public health expenditure per standardized capita (current LCU)
- `PvtHE_per_Std_Capita_Current`: Private health expenditure per standardized capita (current LCU)

### Constant Prices Indicators (No PPP)
- `THE_Constant`: Total health expenditure in constant reference year LCU
- `THE_per_Std_Capita_Constant`: Total health expenditure per standardized capita (constant reference year LCU)
- `PubHE_Constant`: Public health expenditure in constant reference year LCU
- `PubHE_per_Std_Capita_Constant`: Public health expenditure per standardized capita (constant reference year LCU)
- `PvtHE_Constant`: Private health expenditure in constant reference year LCU
- `PvtHE_per_Std_Capita_Constant`: Private health expenditure per standardized capita (constant reference year LCU)

### Current Prices with Current PPP
- `THE_CurrentPPP`: Total health expenditure in current international $ (current PPP)
- `THE_per_Std_Capita_CurrentPPP`: Total health expenditure per standardized capita (current international $, current PPP)
- `PubHE_CurrentPPP`: Public health expenditure in current international $ (current PPP)
- `PubHE_per_Std_Capita_CurrentPPP`: Public health expenditure per standardized capita (current international $, current PPP)
- `PvtHE_CurrentPPP`: Private health expenditure in current international $ (current PPP)
- `PvtHE_per_Std_Capita_CurrentPPP`: Private health expenditure per standardized capita (current international $, current PPP)

### Constant Prices with Current PPP
- `THE_Constant_CurrentPPP`: Total health expenditure in constant reference year international $ (current PPP)
- `THE_per_Std_Capita_Constant_CurrentPPP`: Total health expenditure per standardized capita (constant reference year international $, current PPP)
- `PubHE_Constant_CurrentPPP`: Public health expenditure in constant reference year international $ (current PPP)
- `PubHE_per_Std_Capita_Constant_CurrentPPP`: Public health expenditure per standardized capita (constant reference year international $, current PPP)
- `PvtHE_Constant_CurrentPPP`: Private health expenditure in constant reference year international $ (current PPP)
- `PvtHE_per_Std_Capita_Constant_CurrentPPP`: Private health expenditure per standardized capita (constant reference year international $, current PPP)

### Current Prices with Constant PPP
- `THE_ConstantPPP`: Total health expenditure in current LCU converted to international $ using reference year PPP factors
- `THE_per_Std_Capita_ConstantPPP`: Total health expenditure per standardized capita (current LCU converted to international $ using reference year PPP factors)
- `PubHE_ConstantPPP`: Public health expenditure in current LCU converted to international $ using reference year PPP factors
- `PubHE_per_Std_Capita_ConstantPPP`: Public health expenditure per standardized capita (current LCU converted to international $ using reference year PPP factors)
- `PvtHE_ConstantPPP`: Private health expenditure in current LCU converted to international $ using reference year PPP factors
- `PvtHE_per_Std_Capita_ConstantPPP`: Private health expenditure per standardized capita (current LCU converted to international $ using reference year PPP factors)

### Constant Prices with Constant PPP (Recommended for Time Series)
- `THE_Constant_ConstantPPP`: Total health expenditure in constant reference year LCU converted to international $ using reference year PPP factors
- `THE_per_Std_Capita_Constant_ConstantPPP`: Total health expenditure per standardized capita (constant reference year LCU converted to international $ using reference year PPP factors)
- `PubHE_Constant_ConstantPPP`: Public health expenditure in constant reference year LCU converted to international $ using reference year PPP factors
- `PubHE_per_Std_Capita_Constant_ConstantPPP`: Public health expenditure per standardized capita (constant reference year LCU converted to international $ using reference year PPP factors)
- `PvtHE_Constant_ConstantPPP`: Private health expenditure in constant reference year LCU converted to international $ using reference year PPP factors
- `PvtHE_per_Std_Capita_Constant_ConstantPPP`: Private health expenditure per standardized capita (constant reference year LCU converted to international $ using reference year PPP factors)

### Other Metrics
- `GDP_Deflator`: GDP deflator factor (base: reference year)
- `PPP_Factor`: Current PPP conversion factor (LCU per international $)

## License



## Contributing



## Citation

If you use this tool in your research, please cite:



## Contact

Contact me at [dtsj89@gmail.com](mailto:dtsj89@gmail.com) if you have any questions or problems with the script or data used in it.
