import os
import pandas as pd
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from src import PFA_MV


def process_single_grid(grid_code, random_seeds, HP_percentages):
    """Process a single MV grid configuration"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(base_dir, "data", "results", "MVGridResults")
    os.makedirs(results_dir, exist_ok=True)

    grid_name = grid_code.split('--')[0].replace('1-MV-', '').replace('-', '_')
    filename = f"mv_{grid_name}.csv"
    file_path = os.path.join(results_dir, filename)

    # Run calculation
    results_df = PFA_MV.run_scenarios([grid_code], random_seeds, HP_percentages)

    # Handle file operations
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
    grid_codes = [
        '1-MV-rural--0-no_sw',
        '1-MV-semiurb--0-no_sw',
        '1-MV-urban--0-no_sw',
        '1-MV-comm--0-no_sw'
    ]
    
    num_cores = mp.cpu_count()
    print(f"Running on {num_cores} cores")
    
    try:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [
                executor.submit(
                    process_single_grid,
                    grid_code,
                    random_seeds,
                    HP_percentages
                ) for grid_code in grid_codes
            ]
            
            results = [future.result() for future in futures]
            print("All MV grids processed successfully")
            
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        raise
        
    return results

if __name__ == '__main__':
    random_seeds = np.arange(0, 2)
    HP_percentages = np.arange(0, 101, 5)
    results = run_parallel_analysis(random_seeds, HP_percentages)