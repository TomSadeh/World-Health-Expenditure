"""
Master preprocessing script that runs all three data preprocessing steps:
1. GHED data conversion to CSV
2. World Bank data processing
3. Population data processing

This script simplifies the preprocessing workflow by automating all the necessary data preparation steps.
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def run_script(script_name):
    """
    Run a preprocessing script and report results.
    
    Args:
        script_name: Name of the Python script to run
    
    Returns:
        bool: True if script ran successfully, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script using the same Python interpreter that is running this script
        result = subprocess.run([sys.executable, script_name], check=True)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Successfully completed {script_name} in {elapsed_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_name}: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_name}")
        return False

def verify_processed_files():
    """
    Verify that the expected processed files were created.
    """
    processed_dir = Path("./data/processed")
    
    expected_files = [
        "ghed_data_optimized.csv",
        "male_pop.csv",
        "female_pop.csv",
        "gdp_current.csv",
        "gdp_constant.csv",
        "ppp.csv"
    ]
    
    print("\nVerifying processed files:")
    
    for filename in expected_files:
        file_path = processed_dir / filename
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)  # Convert bytes to MB
            print(f"  ✓ {filename} ({file_size:.2f} MB)")
        else:
            print(f"  ✗ {filename} - NOT FOUND")

def main():
    """
    Run all preprocessing scripts in the correct order.
    """
    print("Starting data preprocessing workflow...")
    
    # Ensure the processed directory exists
    processed_dir = Path("./data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # List of scripts to run in order
    scripts = [
        "ghed_to_csv.py",
        "wb_data_processor.py",
        "pop_data_processor.py"
    ]
    
    # Track overall success
    success_count = 0
    
    # Run each script
    for script in scripts:
        if run_script(script):
            success_count += 1
    
    # Report overall results
    print(f"\n{'='*60}")
    print(f"Preprocessing summary: {success_count}/{len(scripts)} scripts completed successfully")
    print(f"{'='*60}")
    
    # Verify the processed files
    verify_processed_files()
    
    if success_count == len(scripts):
        print("\nAll preprocessing completed successfully! You can now run the main analysis script:")
        print("  python whe.py")
    else:
        print("\nSome preprocessing steps failed. Please check the errors above and try again.")

if __name__ == "__main__":
    main()