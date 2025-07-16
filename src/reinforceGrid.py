import pandas as pd
import numpy as np
import random
from src import energyFlow as ef


def set_random_seed(seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def calculate_hp_factor(household_profiles, percentage):
    num_buses = household_profiles["bus_id"].nunique()
    num_with_pump = int(num_buses * (percentage / 100))
    num_without_pump = num_buses - num_with_pump

    # Create an indicator column DataFrame
    include_pump_df = pd.DataFrame({'include_pump': [1] * num_with_pump + [0] * num_without_pump})

    # Shuffle the DataFrame rows to randomize the distribution of 'include_pump'
    include_pump_df = include_pump_df.sample(frac=1).reset_index(drop=True)

    # Add bus names as the index to keep track
    include_pump_df["bus_id"] = household_profiles["bus_id"].unique()

    return include_pump_df


def find_top_consumption_times(load_profiles, top_n):

    # Adjust the P_Pump based on include_pump
    load_profiles['P_Pump'] = load_profiles['P_Pump'] * load_profiles['include_pump']

    # Group by 'time' and calculate the sum of 'P_Load' and adjusted 'P_Pump'
    load_profiles['total_load'] = load_profiles['P_Load'] + load_profiles['P_Pump']
    time_sums = load_profiles.groupby('time')['total_load'].sum()

    # Get the top N times with the highest total consumption
    top_times = time_sums.nlargest(top_n).index

    # Prepare df for model
    filtered_df = load_profiles[load_profiles['time'].isin(top_times)]

    return top_times, filtered_df  # Return sorted times from highest to lowest as well as df only with top time consumptions


def aggregate_household_profiles(household_profiles):

    # Aggregate by bus_id to get the number of units and total number of people per bus
    aggregated_profiles = household_profiles.groupby('bus_id').agg(
        num_units=('household_id', 'count'),  # Count the number of units (households)
        num_people=('num_people', 'sum')  # Sum the total number of people per bus
    ).reset_index()

    return aggregated_profiles


def assign_loads_to_network(net, load_profiles, specific_time, aggregated_profiles):

    # Filter load profiles for rows with specific time
    load_profiles_time = load_profiles[load_profiles["time"] == specific_time]

    # Get buses that have newly created loads
    buses_with_new_loads = aggregated_profiles['bus_id'].unique()

    # Copy net.load to avoid modifying the original DataFrame
    netload_new = net.load.copy()

    # Extract the subnet value once, assuming all buses share the same subnet
    subnet = netload_new['subnet'].iloc[0] if 'subnet' in netload_new.columns and not netload_new.empty else 'Unknown'

    # Extract the subnet value once, assuming all buses share the same subnet
    voltLvl = netload_new['voltLvl'].iloc[0] if 'voltLvl' in netload_new.columns and not netload_new.empty else 'Unknown'

    # Delete rows in netload_new where the bus_id is in buses_with_new_loads
    netload_new = netload_new[~netload_new['bus'].isin(buses_with_new_loads)]

    # Now create new rows for the buses in buses_with_new_loads
    new_loads = []

    for index, row in load_profiles_time.iterrows():
        if row['bus_id'] not in buses_with_new_loads:
            continue
        bus_id = row['bus_id']

        # Extract data
        p_mw = row["total_load"] / 1000 # convert kWh to MW
        power_factor = 0.95
        q_mvar = p_mw * np.tan(np.arccos(power_factor))
        sn_mva = np.sqrt(p_mw ** 2 + q_mvar ** 2)

        # Establish profile name
        num_units = row["num_units"]
        num_people = row["num_people"]
        profile_name = f"{num_units}_{num_people}_noHP"
        if row["include_pump"] == 1:
            profile_name = f"{num_units}_{num_people}_HP"

        # Create a new load entry
        load_entry = {
            'name': f'{subnet} Load {bus_id}',
            'bus': row["bus_id"],
            'p_mw': p_mw,
            'q_mvar': q_mvar,
            'const_z_percent': 0.0,
            'const_i_percent': 0.0,
            'sn_mva': sn_mva,
            'scaling': 1.0,
            'in_service': True,
            'type': np.nan,
            'subnet': subnet,
            'max_p_mw': np.nan,
            'min_p_mw': np.nan,
            'profile': profile_name,
            'min_q_mvar': np.nan,
            'max_q_mvar': np.nan,
            'voltLvl': voltLvl
        }
        new_loads.append(load_entry)

    # Append the new loads to netload_new using pd.concat
    netload_new = pd.concat([netload_new, pd.DataFrame(new_loads)], ignore_index=True)

    return netload_new


# Function to replace load values in the MV grid based on LV grid results and return a copy of the updated net.load DataFrame
def replace_mv_loads(net, lv_grid_results, hp_percentage, load_seed):
    netload_new = net.load.copy()  # Make a copy of the net.load DataFrame

    # Define the columns for output df
    columns = ['grid_code', 'random_seed', 'HP_percentage', 'total_households', 'total_households_HP',
               'total_reinforcement_cost', 'total_trafo_reinforcement_cost', 'total_line_reinforcement_cost',
               'base_load', 'HP_percentage_load', 'max_trafo_load', 'HH_total_kWh_consumed', 'HP_total_kWh_consumed',
               'base_#_of_transformers', 'HP_avg_trafo_load', 'HP_#_of_crit_transformers', 'HP_#_of_transformers',
               'base_km_of_lines', 'HP_avg_line_load', 'HP_km_of_crit_lines', 'HP_new_km_of_lines']

    LV_simulation_results = []

    lv_grid_results_HP = lv_grid_results[lv_grid_results['HP_percentage'] == hp_percentage]

    for i, row in netload_new.iterrows():
        profile = row['profile']
        if profile.startswith('lv_'):  # Check if the profile corresponds to an LV grid
            profile_test = profile[3:].split('_')[0]
            selected_seed = load_seed.iloc[i]["random_seed"]
            lv_df = lv_grid_results_HP[(lv_grid_results_HP["grid_code"].str.contains(profile_test, case=False, na=False))
                                    & (lv_grid_results_HP["random_seed"] == selected_seed)]
            if not lv_df.empty:
                selected_row = lv_df.iloc[0]  # Select the row
                new_p_mw = selected_row['HP_percentage_load']
                random_seed = selected_row['random_seed']  # Get the random_seed from the selected row
                netload_new.at[i, 'p_mw'] = new_p_mw
                power_factor = 0.95
                q_mvar = new_p_mw * np.tan(np.arccos(power_factor))
                sn_mva = np.sqrt(new_p_mw ** 2 + q_mvar ** 2)
                netload_new.at[i, 'q_mvar'] = q_mvar
                netload_new.at[i, 'sn_mva'] = sn_mva
                netload_new.at[i, 'profile'] = f"{profile}_{hp_percentage}_seed{random_seed}"  # Update profile name

                # Prepare the row to append, keeping only relevant columns
                result_row = selected_row.copy()
                result_row["grid_code"] = f"{profile}_{hp_percentage}_seed{random_seed}"
                LV_simulation_results.append(result_row[columns])

    return netload_new, LV_simulation_results


# Function to replace load values in the HV grid based on MV grid results and return a copy of the updated net.load DataFrame
def replace_hv_loads(net, mv_grid_results, hp_percentage, load_seed):
    netload_new = net.load.copy()  # Make a copy of the net.load DataFrame

    # Define the columns for output df
    columns = ['grid_code', 'random_seed', 'HP_percentage', 'total_households', 'total_households_HP',
               'LV_total_reinforcement_cost', 'LV_total_trafo_reinforcement_cost', 'LV_total_line_reinforcement_cost',
               'MV_total_reinforcement_cost', 'MV_total_trafo_reinforcement_cost', 'MV_total_line_reinforcement_cost',
               'base_load', 'HP_percentage_load', 'max_trafo_load', 'HH_total_kWh_consumed', 'HP_total_kWh_consumed',
               'LV_base_#_of_transformers', 'LV_HP_avg_trafo_load', 'LV_HP_#_of_crit_transformers', 'LV_HP_#_of_transformers',
               'LV_base_km_of_lines', 'LV_HP_avg_line_load', 'LV_HP_km_of_crit_lines', 'LV_HP_new_km_of_lines',
               'MV_base_#_of_transformers', 'MV_HP_avg_trafo_load', 'MV_HP_#_of_crit_transformers', 'MV_HP_#_of_transformers',
               'MV_base_km_of_lines', 'MV_HP_avg_line_load', 'MV_HP_km_of_crit_lines', 'MV_HP_new_km_of_lines']

    MV_simulation_results = []

    mv_grid_results_HP = mv_grid_results[mv_grid_results['HP_percentage'] == hp_percentage]

    for i, row in netload_new.iterrows():
        profile = row['profile']
        if profile.startswith('mv_'):  # Check if the profile corresponds to an MV grid
            profile_test = profile[3:].split('_')[0]
            selected_seed = load_seed.iloc[i]["random_seed"]
            mv_df = mv_grid_results_HP[(mv_grid_results_HP["grid_code"].str.contains(profile_test, case=False, na=False))
                                    & (mv_grid_results_HP["random_seed"] == selected_seed)]
            if not mv_df.empty:
                selected_row = mv_df.iloc[0]  # Select the row
                new_p_mw = selected_row['HP_percentage_load']
                random_seed = selected_row['random_seed']  # Get the random_seed from the selected row
                netload_new.at[i, 'p_mw'] = new_p_mw
                power_factor = 0.95
                q_mvar = new_p_mw * np.tan(np.arccos(power_factor))
                sn_mva = np.sqrt(new_p_mw ** 2 + q_mvar ** 2)
                netload_new.at[i, 'q_mvar'] = q_mvar
                netload_new.at[i, 'sn_mva'] = sn_mva
                netload_new.at[i, 'profile'] = f"{profile}_{hp_percentage}_seed{random_seed}"  # Update profile name

                # Prepare the row to append, keeping only relevant columns
                result_row = selected_row.copy()
                result_row["grid_code"] = f"{profile}_{hp_percentage}_seed{random_seed}"
                MV_simulation_results.append(result_row[columns])

    return netload_new, MV_simulation_results


def calculate_LV_consumption_data(load_profiles):
    # Calculate the time interval in minutes using the first load profile
    time_difference = (load_profiles["time"].sort_values().unique()[1] - load_profiles["time"].sort_values().unique()[0]).total_seconds() / 60

    # Determine the conversion factor based on the calculated interval
    conversion_factor = 60 / time_difference

    consumption_data = load_profiles.groupby('bus_id').agg({
        'num_units': 'mean',
        'include_pump': 'sum',
        'P_Load': 'sum',
        'P_Pump': 'sum'
    }).reset_index()

    # Divide 'P_Load' and 'P_Pump' by the conversion factor
    consumption_data['P_Load'] = consumption_data['P_Load'] / conversion_factor
    consumption_data['P_Pump'] = consumption_data['P_Pump'] / conversion_factor

    # Set 'include_pump' to 'num_units' if 'include_pump' is not 0
    consumption_data['include_pump'] = consumption_data.apply(
        lambda row: row['num_units'] if row['include_pump'] != 0 else row['include_pump'], axis=1
    )

    return consumption_data


def reinforce_LV_grid(net, code, load_profiles, hp_percentage, topX, household_profiles, random_seed=None):
    set_random_seed(random_seed)
    trafo_cost = 0
    line_cost = 0
    max_trafo_load = 0
    max_crit_trafos = 0
    max_crit_lines_km = 0
    max_avg_trafo_load = 0
    max_avg_line_load = 0

    include_pump_df = calculate_hp_factor(household_profiles, hp_percentage)

    # Merge load_profiles with pump_df
    load_profiles = load_profiles.merge(include_pump_df[['bus_id', 'include_pump']], on='bus_id', how='left')

    # Aggregate household profiles to get num_units and num_people per bus
    aggregated_profiles = aggregate_household_profiles(household_profiles)

    top_times, top_time_load_profiles = find_top_consumption_times(load_profiles, topX)
    topMW = 0

    for rank, specific_time in enumerate(top_times, start=1):
        netload_new = assign_loads_to_network(net, top_time_load_profiles, specific_time, aggregated_profiles)
        total_load = netload_new['p_mw'].sum()
        scenario_name = f"Top-{rank} peak load ({total_load:.2f} MW on {specific_time})"
        # print(f"{scenario_name}")

        if rank == 1:  # Capture the highest load from the first (highest) peak
            topMW = total_load

        # Run reinforcement calculations
        (t_cost, l_cost, trafo_load, crit_trafos, crit_lines_km,
         avg_trafo_load, avg_line_load) = ef.analyze_scenario(net, code, netload_new, "LV")

        trafo_cost += t_cost
        line_cost += l_cost
        max_trafo_load = max(max_trafo_load, trafo_load)
        max_crit_trafos = max(max_crit_trafos, crit_trafos)
        max_crit_lines_km = max(max_crit_lines_km, crit_lines_km)
        max_avg_trafo_load = max(max_avg_trafo_load, avg_trafo_load)
        max_avg_line_load = max(max_avg_line_load, avg_line_load)

    # HP_total_kWh_consumed
    consumption_data = calculate_LV_consumption_data(load_profiles)

    return trafo_cost, line_cost, topMW, consumption_data, max_trafo_load, max_crit_trafos, max_crit_lines_km, max_avg_trafo_load, max_avg_line_load


def reinforce_MV_grid(net, code, lv_grid_results, percentage, load_seed, seed):
    set_random_seed(seed)
    trafo_cost = 0
    line_cost = 0
    max_trafo_load = 0
    max_crit_trafos = 0
    max_crit_lines_km = 0
    max_avg_trafo_load = 0
    max_avg_line_load = 0

    # Calculate load at HP level
    netload_new, LV_simulation_results = replace_mv_loads(net, lv_grid_results, percentage, load_seed)

    total_load = netload_new['p_mw'].sum()
    scenario_name = f" peak load ({total_load:.2f} MW at {percentage}% HP level, seed = {seed})"
    # print(f"{scenario_name}")

    # Run reinforcement calculations
    (trafo_cost, line_cost, max_trafo_load, max_crit_trafos, max_crit_lines_km,
     max_avg_trafo_load, max_avg_line_load) = ef.analyze_scenario(net, code, netload_new, "MV")

    return trafo_cost, line_cost, LV_simulation_results, total_load, max_trafo_load, max_crit_trafos, max_crit_lines_km, max_avg_trafo_load, max_avg_line_load


def reinforce_HV_grid(net, code, mv_grid_results, percentage, load_seed, seed):
    set_random_seed(seed)
    trafo_cost = 0
    line_cost = 0
    max_crit_lines_km = 0
    max_avg_trafo_load = 0
    max_avg_line_load = 0

    # Calculate load at HP level
    netload_new, MV_simulation_results = replace_hv_loads(net, mv_grid_results, percentage, load_seed)

    total_load = netload_new['p_mw'].sum()
    scenario_name = f" peak load ({total_load:.2f} MW at {percentage}% HP level, seed = {seed})"
    # print(f"{scenario_name}")

    # Run reinforcement calculations
    (trafo_cost, line_cost, max_trafo_load, max_crit_trafos, max_crit_lines_km,
     max_avg_trafo_load, max_avg_line_load) = ef.analyze_scenario(net, code, netload_new, "HV")

    return trafo_cost, line_cost, MV_simulation_results, total_load, max_crit_lines_km, max_avg_trafo_load, max_avg_line_load