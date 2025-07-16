import pandas as pd
import simbench as sb
from src import generateProfiles as gp
from src import reinforceGrid as rg


def run_scenarios(load_data, household_info, units_per_house_probs, people_per_unit_probs,
                  grid_codes, random_seeds, HP_percentages):
    
    # Sort HP percentages to ensure correct order
    HP_percentages = sorted(HP_percentages)

    # Define the columns for output df
    columns = ['grid_code', 'random_seed', 'HP_percentage', 'total_households', 'total_households_HP',
               'total_reinforcement_cost', 'total_trafo_reinforcement_cost', 'total_line_reinforcement_cost',
               'base_load', 'HP_percentage_load', 'max_trafo_load', 'HH_total_kWh_consumed', 'HP_total_kWh_consumed',
               'base_#_of_transformers', 'HP_avg_trafo_load', 'HP_#_of_crit_transformers', 'HP_#_of_transformers',
               'base_km_of_lines', 'HP_avg_line_load', 'HP_km_of_crit_lines', 'HP_new_km_of_lines']

    # Create a list to collect results
    results = []

    # Instantiate no_HP_load
    no_HP_load = 0

    # For-Loop
    for seed in random_seeds:
        for code in grid_codes:

            # Generate load profiles for the specific grid
            load_profiles, household_profiles = gp.generate_load_profiles(code, load_data, household_info,
                                                                          units_per_house_probs, people_per_unit_probs,
                                                                          random_seed=seed)

            # Reload grid to start new
            net = sb.get_simbench_net(code)
            net.sgen.drop(net.sgen.index, inplace=True)
            net.storage.drop(net.storage.index, inplace=True)

            # Instantiate grid
            (trafo_cost, line_cost, topMW, consumption_data,
             max_trafo_load, max_crit_trafo, max_crit_lines_km,
             max_avg_trafo_load, max_avg_line_load) = rg.reinforce_LV_grid(net, code, load_profiles,
                                                                           0, 10,
                                                                           household_profiles, seed)
            base_line = net.line.copy()
            base_trafo = net.trafo.copy()
            base_load = net.load.copy()

            for percentage in HP_percentages:

                # Store pre-reinforcement state
                pre_trafo_count = len(net.trafo)
                pre_line_km = net.line["length_km"].sum()

                # Calculate results for scenario
                (trafo_cost, line_cost, topMW, consumption_data,
                 max_trafo_load, max_crit_trafo, max_crit_lines_km,
                 max_avg_trafo_load, max_avg_line_load) = rg.reinforce_LV_grid(net, code, load_profiles,
                                                                               percentage, 10,
                                                                               household_profiles, seed)

                if percentage == 0:
                    no_HP_load = topMW

                # Collect results in a list
                results.append({
                    'grid_code': code,
                    'random_seed': seed,
                    'HP_percentage': percentage,
                    'total_households': consumption_data["num_units"].sum(),
                    'total_households_HP': consumption_data["include_pump"].sum(),
                    'total_reinforcement_cost': trafo_cost + line_cost,
                    'total_trafo_reinforcement_cost': trafo_cost,
                    'total_line_reinforcement_cost': line_cost,
                    'base_load': no_HP_load,
                    'HP_percentage_load': topMW,
                    'max_trafo_load': max_trafo_load,
                    'HH_total_kWh_consumed': consumption_data["P_Load"].sum(),
                    'HP_total_kWh_consumed': consumption_data["P_Pump"].sum(),
                    'base_#_of_transformers': pre_trafo_count,
                    'HP_avg_trafo_load': max_avg_trafo_load,
                    'HP_#_of_crit_transformers': max_crit_trafo,
                    'HP_#_of_transformers': len(net.trafo),
                    'base_km_of_lines': pre_line_km,
                    'HP_avg_line_load': max_avg_line_load,
                    'HP_km_of_crit_lines': max_crit_lines_km,
                    'HP_new_km_of_lines': net.line["length_km"].sum()
                })

                print(f"\nFinished calculation for seed {seed}, grid code {code} and HP percentage of {percentage}%.")

    # Create DataFrame from collected results
    results_df = pd.DataFrame(results, columns=columns)

    return results_df
