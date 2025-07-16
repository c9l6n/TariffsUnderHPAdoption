import os
import h5py
import pandas as pd
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from src import dataProcessing as dp
from src import PFA_LV
from pathlib import Path

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent

# Assume the project directory is the parent of the script directory
project_dir = script_dir.parent


def load_input_data(project_dir):
    """Load all required input data"""

    # Construct paths relative to the project directory
    hdf5_path = os.path.join(project_dir, "data", "2019_data_15min.hdf5")
    csv_path = os.path.join(project_dir, "data", "2019_data_household_information.csv")
    units_per_house_path = os.path.join(project_dir, "data", "units_per_house.csv")
    people_per_unit_path = os.path.join(project_dir, "data", "people_per_unit.csv")

    # Load HDF5 file
    f = h5py.File(hdf5_path, 'r')
    h = pd.read_csv(csv_path, delimiter=";")
    
    # Prepare data
    load_data, household_info = dp.h5py_prep(f, h)
    
    # Load probability information
    units_per_house_probs = pd.read_csv(units_per_house_path, sep=',', index_col=0)
    people_per_unit_probs = pd.read_csv(people_per_unit_path, sep=',', index_col=0)
    
    return load_data, household_info, units_per_house_probs, people_per_unit_probs


def process_single_grid(grid_code, load_data, household_info, units_per_house_probs, 
                       people_per_unit_probs, random_seeds, HP_percentages):
    """Process a single grid configuration"""
    script_dir = os.getcwd()
    results_dir = os.path.join(script_dir, "data", "results", "LVGridResults")
    os.makedirs(results_dir, exist_ok=True)

    grid_name = grid_code.split('--')[0].replace('1-LV-', '').replace('-', '_')
    filename = f"lv_{grid_name}.csv"
    file_path = os.path.join(results_dir, filename)

    # Run calculation
    results_df = PFA_LV.run_scenarios(load_data, household_info, 
                                    units_per_house_probs, people_per_unit_probs, 
                                    [grid_code], random_seeds, HP_percentages)

    # Handle file saving
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        updated_data = pd.concat([existing_data, results_df], ignore_index=True)
    else:
        updated_data = results_df

    updated_data.to_csv(file_path, index=False)
    print(f"Results for {grid_code} appended to {filename}")
    
    return grid_code


def run_parallel_analysis(random_seeds, HP_percentages):
    """Main function to run parallel grid analysis"""
    # Grid configurations
    grid_codes = [
        '1-LV-rural1--0-no_sw',
        '1-LV-rural2--0-no_sw',
        '1-LV-rural3--0-no_sw',
        '1-LV-semiurb4--0-no_sw',
        '1-LV-semiurb5--0-no_sw',
        '1-LV-urban6--0-no_sw'
    ]
    
    # Determine the project directory
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    
    # Load input data once
    load_data, household_info, units_per_house_probs, people_per_unit_probs = load_input_data(project_dir)
    
    # Setup parallel processing
    num_cores = mp.cpu_count()
    print(f"Running on {num_cores} cores")
    
    try:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit all grid processes
            futures = [
                executor.submit(
                    process_single_grid,
                    grid_code,
                    load_data,
                    household_info,
                    units_per_house_probs,
                    people_per_unit_probs,
                    random_seeds,
                    HP_percentages
                ) for grid_code in grid_codes
            ]
            
            # Gather results
            results = [future.result() for future in futures]
            print("All grids processed successfully")
            
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        raise
        
    return results

if __name__ == '__main__':
    # Test run
    random_seeds = np.arange(0, 2)
    HP_percentages = np.arange(0, 101, 5)
    results = run_parallel_analysis(random_seeds, HP_percentages)