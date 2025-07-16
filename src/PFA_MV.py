# Import required packages
import pandas as pd
import os
import simbench as sb
import numpy as np
from statistics import mean

from src import reinforceGrid as rg
    
script_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.dirname(script_dir)
directory_lv_grid_results = os.path.join(base_dir, "data", "results", "LVGridResults")


# Function to load the LV grid CSV files into a dictionary
def load_lv_grid_results(directory):
    # List all CSV files in the directory
    lv_grid_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Load each CSV file into a DataFrame and store it in a list
    lv_grid_dfs = [pd.read_csv(os.path.join(directory, file)) for file in lv_grid_files]

    # Concatenate all DataFrames into one
    combined_df = pd.concat(lv_grid_dfs, ignore_index=True)

    return combined_df


def run_scenarios(grid_codes, random_seeds, HP_percentages):
    
    # Sort HP percentages to ensure correct order
    HP_percentages = sorted(HP_percentages)

    # Define the columns for output df
    columns = ['grid_code', 'random_seed', 'HP_percentage', 'total_households', 'total_households_HP',
               'LV_total_reinforcement_cost', 'LV_total_trafo_reinforcement_cost', 'LV_total_line_reinforcement_cost',
               'MV_total_reinforcement_cost', 'MV_total_trafo_reinforcement_cost', 'MV_total_line_reinforcement_cost',
               'base_load', 'HP_percentage_load', 'max_trafo_load', 'HH_total_kWh_consumed', 'HP_total_kWh_consumed',
               'LV_base_#_of_transformers', 'LV_HP_avg_trafo_load', 'LV_HP_#_of_crit_transformers', 'LV_HP_#_of_transformers',
               'LV_base_km_of_lines', 'LV_HP_avg_line_load', 'LV_HP_km_of_crit_lines', 'LV_HP_new_km_of_lines',
               'MV_base_#_of_transformers', 'MV_HP_avg_trafo_load', 'MV_HP_#_of_crit_transformers', 'MV_HP_#_of_transformers',
               'MV_base_km_of_lines', 'MV_HP_avg_line_load', 'MV_HP_km_of_crit_lines', 'MV_HP_new_km_of_lines']

    # Create a list to collect results
    results = []

    # Instantiate no_HP_load
    no_HP_load = 0

    # Function to load the LV grid CSV files into a dictionary
    lv_grid_results = load_lv_grid_results(directory_lv_grid_results)

    # For-Loop
    for seed in random_seeds:
        for code in grid_codes:

            # Reload grid to start new
            net = sb.get_simbench_net(code)
            net.sgen.drop(net.sgen.index, inplace=True)
            net.storage.drop(net.storage.index, inplace=True)

            # Clearly assign one random seed to each of the profiles to be added in the future
            random_seeds_per_grid_code = lv_grid_results.groupby('grid_code')['random_seed'].apply(set)
            common_random_seeds = set.intersection(*random_seeds_per_grid_code)
            common_random_seeds_array = np.array(list(common_random_seeds))
            load_seed = net.load[['name', 'profile']].copy()
            np.random.seed(seed)
            load_seed['random_seed'] = np.random.choice(common_random_seeds_array, size=len(net.load))

            # Instantiate grid
            (trafo_cost, line_cost, LV_simulation_results, total_load,
             max_trafo_load, max_crit_trafo, max_crit_lines_km,
             max_avg_trafo_load, max_avg_line_load) = rg.reinforce_MV_grid(net, code, lv_grid_results,
                                                                           0, load_seed, seed)

            base_line = net.line.copy()
            base_trafo = net.trafo.copy()
            base_load = net.load.copy()

            for percentage in HP_percentages:

                # Store pre-reinforcement state
                pre_trafo_count = len(net.trafo)
                pre_line_km = net.line["length_km"].sum()

                # Calculate results for scenario
                (trafo_cost, line_cost, LV_simulation_results, total_load,
                 max_trafo_load, max_crit_trafo, max_crit_lines_km,
                 max_avg_trafo_load, max_avg_line_load) = rg.reinforce_MV_grid(net, code, lv_grid_results,
                                                                               percentage, load_seed, seed)

                if percentage == 0:
                    no_HP_load = total_load

                # Collect results in a list
                results.append({
                    'grid_code': code,
                    'random_seed': seed,
                    'HP_percentage': percentage,
                    'total_households': sum(row['total_households'] for row in LV_simulation_results),
                    'total_households_HP': sum(row['total_households_HP'] for row in LV_simulation_results),
                    'LV_total_reinforcement_cost': sum(row['total_reinforcement_cost'] for row in LV_simulation_results),
                    'LV_total_trafo_reinforcement_cost':  sum(row['total_trafo_reinforcement_cost'] for row in LV_simulation_results),
                    'LV_total_line_reinforcement_cost':  sum(row['total_line_reinforcement_cost'] for row in LV_simulation_results),
                    'MV_total_reinforcement_cost': trafo_cost + line_cost,
                    'MV_total_trafo_reinforcement_cost': trafo_cost,
                    'MV_total_line_reinforcement_cost': line_cost,
                    'base_load': no_HP_load,
                    'HP_percentage_load': total_load,
                    'max_trafo_load': max_trafo_load,
                    'HH_total_kWh_consumed': sum(row['HH_total_kWh_consumed'] for row in LV_simulation_results),
                    'HP_total_kWh_consumed': sum(row['HP_total_kWh_consumed'] for row in LV_simulation_results),
                    'LV_base_#_of_transformers': sum(row['base_#_of_transformers'] for row in LV_simulation_results),
                    'LV_HP_avg_trafo_load': mean(row['HP_avg_trafo_load'] for row in LV_simulation_results),
                    'LV_HP_#_of_crit_transformers': sum(row['HP_#_of_crit_transformers'] for row in LV_simulation_results),
                    'LV_HP_#_of_transformers': sum(row['HP_#_of_transformers'] for row in LV_simulation_results),
                    'LV_base_km_of_lines': sum(row['base_km_of_lines'] for row in LV_simulation_results),
                    'LV_HP_avg_line_load': mean(row['HP_avg_line_load'] for row in LV_simulation_results),
                    'LV_HP_km_of_crit_lines': sum(row['HP_km_of_crit_lines'] for row in LV_simulation_results),
                    'LV_HP_new_km_of_lines': sum(row['HP_new_km_of_lines'] for row in LV_simulation_results),
                    'MV_base_#_of_transformers': pre_trafo_count,
                    'MV_HP_avg_trafo_load': max_avg_trafo_load,
                    'MV_HP_#_of_crit_transformers': max_crit_trafo,
                    'MV_HP_#_of_transformers': len(net.trafo),
                    'MV_base_km_of_lines': pre_line_km,
                    'MV_HP_avg_line_load': max_avg_line_load,
                    'MV_HP_km_of_crit_lines': max_crit_lines_km,
                    'MV_HP_new_km_of_lines': net.line["length_km"].sum()
                })

                print(f"\nFinished calculation for seed {seed}, grid code {code} and HP percentage of {percentage}%.")

    # Create DataFrame from collected results
    results_df = pd.DataFrame(results, columns=columns)

    return results_df