# Health Expenditure per Standardized Capita Calculator

This tool calculates Health Expenditure per Standardized Capita with Purchasing Power Parity (PPP) adjustment by processing data from multiple international sources:

- Global Health Expenditure Database (GHED)
- World Population Prospects (WPP) 
- World Bank PPP conversion factors

The script standardizes health expenditure measurement across countries by applying selectable capitation formulas to account for demographic differences and adjusts for purchasing power parity to enable meaningful cross-country comparisons.

## Features
- **Multiple capitation formulas**: Choose between Israeli, LTC, or EU27 capitation formulas
- **Demographic standardization**: Uses Israeli Capitation Formula weights to adjust for population age and gender distributions
- **Expenditure component analysis**: Separates total health expenditure into public and private components
- **Constant price conversion**: Applies GDP deflators to convert nominal values to constant prices
- **Purchasing power parity adjustment**: Normalizes expenditure across countries using PPP conversion factors
- **Missing data handling**: Optional imputation for missing PPP and GDP deflator values
- **Country name standardization**: Harmonizes country names across different datasets
- **Comprehensive reporting**: Generates detailed output with imputation documentation

## Requirements

### Python Dependencies
- pandas
- numpy
- pathlib

### Required Data Files
- `GHED_data_2025.xlsx`: Health expenditure data from the Global Health Expenditure Database
- `male_pop.csv`: Male population data by age groups from World Population Prospects
- `female_pop.csv`: Female population data by age groups from World Population Prospects
- `cap.csv`: Capitation formula weights by age group (contains Israeli, LTC, and EU27 formulas)
- `API_PA.NUS.PPP_DS2_en_csv_v2_13721.csv`: World Bank PPP conversion factors
- GDP data files (optional):
  - `API_NY.GDP.MKTP.CN_DS2_en_csv_v2_26332.csv`: GDP in current LCU
  - `API_NY.GDP.MKTP.KN_DS2_en_csv_v2_13325.csv`: GDP in constant LCU

## Installation

1. Clone this repository
2. Install required dependencies:
   ```
   pip install pandas numpy
   ```
3. Place the required data files in the "data" directory if they are not in it

## Usage

Run the script with:

```
python whe.py
```

### Configuration

The main script parameters can be adjusted at the top of the file:

- `REFERENCE_YEAR`: Year for constant price calculations and PPP adjustment (default: 2017)
- `BASE_COUNTRY`: Base country for PPP comparisons (default: "United States")
- `impute_ppp`: Enable/disable PPP imputation (default: False)
- `impute_gdp`: Enable/disable GDP deflator imputation (default: False)

If desired, you can modify the Israeli Capitation Formula weights defined in the `ISRAELI_CAPITATION` dictionary.

## Output

The script creates a directory named `Standardized_Expenditure` containing:

- `Health_Expenditure_per_Std_Capita.csv`: Main output file with all calculated metrics
- `imputation_documentation.csv`: Records of data imputation if enabled

Filename suffixes like `_no_ppp_imputed_no_gdp_imputed` indicate imputation settings used.

## Methodology

### Israeli Capitation Formula

The script uses the Israeli Capitation Formula to standardize population demographics across countries. This formula assigns weights to different age-gender groups to adjust for varying healthcare needs:

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

These weights can be customized by providing a cap.csv file. The cap file contains capitation formulae as well, including EU27 and OECD with long-term care adjustment. Future versions may incorporate additional formulations.

### Data Processing Steps

1. **Data Loading**: Reads GHED, WPP, and World Bank data files
2. **Country Name Standardization**: Harmonizes country names across datasets
3. **Population Standardization**: Applies capitation weights to demographic data
4. **Price Adjustment**: Converts to constant prices using GDP deflators
5. **PPP Adjustment**: Normalizes using purchasing power parity conversion factors
6. **Per Capita Calculation**: Computes expenditure per standardized capita
7. **Imputation Documentation**: Records any data imputations performed

### Age Group Mapping

The script maps WPP age groups to Israeli Capitation Formula age groups as follows:

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

It's important to note that the World Population Prospects (WPP) data used in this script are not direct census counts but rather estimates produced through various demographic methods. The United Nations Population Division generates these estimates using:

- Multiple data sources including censuses, surveys, and administrative records
- Demographic modeling techniques to estimate and project population distributions
- Statistical methods to impute missing values and reconcile inconsistencies
- Standardized approaches to handle varying data quality across countries

These population estimates undergo rigorous validation but inherently contain some uncertainty, especially for countries with limited data collection infrastructure. Users should be aware that the standardized population calculations in this tool reflect these WPP estimation methodologies.

### Global Health Expenditure Database (GHED) Data

The Global Health Expenditure Database (GHED) data used in this script are compiled by the World Health Organization (WHO) and represent the most comprehensive source of health spending information. This script uses only highly aggregated GHED indicators, which should consist of non-imputed data reported directly by countries. It's worth noting that:

- Data are collected through National Health Accounts (NHA) frameworks that countries report to WHO
- The aggregated indicators used in this script represent the most reliable portion of GHED data
- By focusing on top-level expenditure categories, the script avoids many of the data quality issues present in more granular GHED components
- Even at the aggregate level, methodological differences exist across countries in how health expenditures are categorized and reported
- Revisions to historical data occur as countries improve their health accounting systems
- Data quality varies by country, with high-income countries generally having more reliable estimates

While GHED represents the global standard for health expenditure data, users should understand these underlying limitations when interpreting results.

### World Bank PPP and GDP Data

The World Bank data on Purchasing Power Parity (PPP) conversion factors and GDP used in this script come with several considerations:

- PPP conversion factors are derived from the International Comparison Program (ICP), which conducts comprehensive price surveys only periodically (typically every 6 years)
- Values for non-benchmark years are estimated through extrapolation and modeling
- GDP deflators reflect countries' own national accounting practices, which may vary in methodology
- Data revisions are common as countries update their national accounts
- Coverage varies by country and year, with some developing economies having less reliable data

These datasets undergo extensive quality assurance by the World Bank but necessarily contain estimation uncertainty that carries through to the final calculated indicators.

## Output Variables

The generated dataset includes the following key variables:

- `Country`, `Year`: Country and year identifiers
- `Standardized_Population`: Population adjusted using capitation weights
- `Total_Health_Expenditure`: Raw health expenditure in local currency units
- `Public_Health_Expenditure`, `Private_Health_Expenditure`: Public and private components
- `Total_Health_Expenditure_per_Std_Capita`: Expenditure divided by standardized population
- `*_Constant`: Variables adjusted to constant prices using GDP deflators
- `*_PPP`: Variables adjusted for purchasing power parity
- `*_Constant_PPP`: Variables with both constant price and PPP adjustments

## License



## Contributing



## Citation

If you use this tool in your research, please cite:



## Contact

Contact me at dtsj89@gmail.com if you have any questions or problems with the script or data used in it.
