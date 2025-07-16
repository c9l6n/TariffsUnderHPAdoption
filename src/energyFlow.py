import pandas as pd
import pandapower as pp
import numpy as np
import re
from src import generateProfiles as gp
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(os.path.dirname(script_dir), 'data')  # Go up one level from src to v2

standard_lines = pd.read_csv(os.path.join(data_folder, 'standardLines.csv'), delimiter=';')
standard_trafos = pd.read_csv(os.path.join(data_folder, 'standardTrafos.csv'), delimiter=';')

# Function to run power flow and check for convergence
def run_power_flow(net):
    try:
        pp.runpp(net, max_iteration=50)
        return True
    except pp.LoadflowNotConverged:
        return False


# Function to reduce the load by a given factor
def adjust_load(net, factor):
    net.load['p_mw'] *= factor
    net.load['q_mvar'] *= factor


# Function to restore the load to original factor
def restore_load(net, original_load):
    net.load['p_mw'] = original_load['p_mw']
    net.load['q_mvar'] = original_load['q_mvar']


# Function to incrementally reduce load until power flow converges
def incremental_load_reduction(net, load_df, reduction_step=0.05):
    net.load.update(load_df[['p_mw', 'q_mvar', 'sn_mva']])  # Overwrite net.load with load_df
    original_load = net.load[['p_mw', 'q_mvar']].copy()  # Save the original load
    load_factor = 1.0
    converged = False

    while not converged and load_factor > 0:
        adjust_load(net, load_factor)
        converged = run_power_flow(net)
        if not converged:
            total_load = net.load['p_mw'].sum()
            restore_load(net, original_load)
            load_factor -= reduction_step

    if converged:
        final_load = net.load['p_mw'].sum()
        final_load_factor = final_load / original_load['p_mw'].sum()
        return converged, final_load_factor

    return converged, load_factor


# Function to check line loadings and identify critical lines
def check_critical_lines(net, threshold):
    line_loading = net.res_line[['loading_percent']]
    critical_lines = line_loading[line_loading['loading_percent'] > threshold].merge(net.line, left_index=True,
                                                                                     right_index=True)
    return critical_lines[
        ['from_bus', 'to_bus', 'length_km', 'loading_percent', 'std_type', 'name', 'max_loading_percent']]


# Function to check transformer loadings and identify critical transformers
def check_critical_transformers(net, threshold):
    # Check two-winding transformers
    trafo_loading = net.res_trafo[['loading_percent']]
    critical_trafos = trafo_loading[trafo_loading['loading_percent'] > threshold].merge(net.trafo, left_index=True,
                                                                                        right_index=True)

    # Check three-winding transformers
    trafo3w_loading = net.res_trafo3w[['loading_percent']]
    critical_trafo3w = trafo3w_loading[trafo3w_loading['loading_percent'] > threshold].merge(net.trafo3w,
                                                                                             left_index=True,
                                                                                             right_index=True)

    return critical_trafos[['hv_bus', 'lv_bus', 'loading_percent', 'max_loading_percent']], critical_trafo3w[
        ['hv_bus', 'lv_bus', 'loading_percent', 'max_loading_percent']]


# Function to extract the original name
def extract_original_name(line_name):
    match = re.match(r"^(.*?)(?=_[^_]*$|$)", line_name)
    if match:
        return match.group(1)
    else:
        return line_name


def reinforce_most_overloaded_line(net, code, critical_lines, iteration, load_level, grid_level):
    total_cost = 0
    target_loading_percent = 100

    if grid_level == "MV":
        target_loading_percent = 50

    if grid_level == "HV":
        target_loading_percent = 40

    if not critical_lines.empty:
        # Get the most overloaded line
        row = critical_lines.loc[critical_lines['loading_percent'].idxmax()]
        from_bus, to_bus, length_km, std_type, original_name, max_loading_percent, loading_percent = row[
            ['from_bus', 'to_bus', 'length_km', 'std_type', 'name', 'max_loading_percent', 'loading_percent']]

        # Get the current line's grid level and determine the maximum capacity line for that level
        current_line_data = standard_lines[standard_lines['line'] == std_type].iloc[0]
        grid_level = current_line_data['grid_level']
        line_type = current_line_data['type']
        max_capacity_line = standard_lines[(standard_lines['grid_level'] == grid_level) &
                                           (standard_lines['type'] == line_type)].sort_values(by='max_i_ka',
                                                                                              ascending=False).iloc[0]

        if std_type != max_capacity_line['line']:
            # If the current line is not the highest capacity, overwrite it with the highest capacity line
            net.line.at[row.name, 'name'] = f"{original_name}_upgraded_{load_level:.2f}_{iteration}"
            net.line.at[row.name, 'r_ohm_per_km'] = max_capacity_line['r_ohm_per_km']
            net.line.at[row.name, 'x_ohm_per_km'] = max_capacity_line['x_ohm_per_km']
            net.line.at[row.name, 'c_nf_per_km'] = max_capacity_line['c_nf_per_km']
            net.line.at[row.name, 'max_i_ka'] = max_capacity_line['max_i_ka']
            net.line.at[row.name, 'max_loading_percent'] = max_loading_percent
            net.line.at[row.name, 'std_type'] = max_capacity_line['line']
            net.line.at[row.name, 'type'] = max_capacity_line['type']

            grid_type = gp.get_grid_type(code)
            line_cost_per_km = max_capacity_line['line_cost_per_km']
            earthwork_cost_per_km = get_earthwork_cost(grid_type, max_capacity_line)

            # Calculate the cost of replacing with the highest capacity line
            cost = (line_cost_per_km + earthwork_cost_per_km) * length_km
            total_cost += cost

        else:
            # If it's already the highest capacity line, add parallel lines as before
            required_additional_lines = np.ceil(loading_percent / (target_loading_percent)) - 1
            for i in range(int(required_additional_lines)):
                new_line_name = f"{original_name}_parallel_{i + 1}_{load_level:.2f}_{iteration}"
                new_line = pp.create_line_from_parameters(
                    net, from_bus, to_bus, length_km, current_line_data['r_ohm_per_km'],
                    current_line_data['x_ohm_per_km'], current_line_data['c_nf_per_km'],
                    current_line_data['max_i_ka'], name=new_line_name)
                net.line.at[new_line, 'max_loading_percent'] = max_loading_percent
                net.line.at[new_line, 'std_type'] = current_line_data['line']
                net.line.at[new_line, 'type'] = current_line_data['type']

            grid_type = gp.get_grid_type(code)
            line_cost_per_km = current_line_data['line_cost_per_km']
            earthwork_cost_per_km = get_earthwork_cost(grid_type, current_line_data)

            # Calculate the cost
            cost = (required_additional_lines * line_cost_per_km +
                    (required_additional_lines // 3 + 1) * earthwork_cost_per_km) * length_km

            # Check if 'upgraded' is in the original_name string
            if "upgraded" in original_name:
                cost = (required_additional_lines * line_cost_per_km) * length_km

            total_cost += cost

    return net, total_cost

def get_earthwork_cost(grid_type, line_data):
    if grid_type == 'urban':
        return line_data['urban_earthwork_cost_per_km']
    elif grid_type == 'semiurb':
        return line_data['semiurb_earthwork_cost_per_km']
    elif grid_type == 'rural':
        return line_data['rural_earthwork_cost_per_km']
    return line_data['rural_earthwork_cost_per_km']  # Default to rural if type is unknown


def reinforce_most_overloaded_transformer(net, critical_trafos, iteration, load_level, grid_level):
    total_cost = 0
    target_loading_percent = 100

    if grid_level == "MV":
        target_loading_percent = 50

    if grid_level == "HV":
        target_loading_percent = 40

    if not critical_trafos.empty:
        # Get the most overloaded transformer
        row = critical_trafos.loc[critical_trafos['loading_percent'].idxmax()]
        hv_bus, lv_bus, max_loading_percent, loading_percent = row[['hv_bus', 'lv_bus', 'max_loading_percent', 'loading_percent']]
        original_name = net.trafo.at[row.name, 'name']
        std_type = net.trafo.at[row.name, 'std_type']

        # Get the current transformer's grid level and determine the maximum capacity transformer for that level
        current_trafo_data = standard_trafos[standard_trafos['trafo'] == std_type]
        grid_level = current_trafo_data['grid_level'].iloc[0]

        # Determine the maximum capacity transformer for the grid level
        max_capacity_trafo = \
        standard_trafos[standard_trafos['grid_level'] == grid_level].sort_values(by='sn_mva', ascending=False).iloc[0]

        # Additional conditions for "MV" grid level with specific voltage levels
        if grid_level == "MV" and "20 kV" in std_type:
            max_capacity_trafo = standard_trafos[(standard_trafos['grid_level'] == grid_level) & (
                standard_trafos['trafo'].str.contains("20 kV"))].sort_values(by='sn_mva', ascending=False).iloc[0]

        if grid_level == "MV" and "10 kV" in std_type:
            max_capacity_trafo = standard_trafos[(standard_trafos['grid_level'] == grid_level) & (
                standard_trafos['trafo'].str.contains("10 kV"))].sort_values(by='sn_mva', ascending=False).iloc[0]

        if std_type != max_capacity_trafo['trafo']:
            # If the current transformer is not the highest capacity, overwrite it with the highest capacity transformer
            new_trafo_name = f"{original_name}_upgraded_{load_level:.2f}_{iteration}"

            # Overwrite the existing transformer's parameters
            net.trafo.at[row.name, 'name'] = new_trafo_name
            net.trafo.at[row.name, 'sn_mva'] = max_capacity_trafo['sn_mva']
            net.trafo.at[row.name, 'vn_lv_kv'] = max_capacity_trafo['vn_lv_kv']
            net.trafo.at[row.name, 'vn_hv_kv'] = max_capacity_trafo['vn_hv_kv']
            net.trafo.at[row.name, 'vk_percent'] = max_capacity_trafo['vk_percent']
            net.trafo.at[row.name, 'vkr_percent'] = max_capacity_trafo['vkr_percent']
            net.trafo.at[row.name, 'pfe_kw'] = max_capacity_trafo['pfe_kw']
            net.trafo.at[row.name, 'i0_percent'] = max_capacity_trafo['i0_percent']
            net.trafo.at[row.name, 'shift_degree'] = max_capacity_trafo['shift_degree']
            net.trafo.at[row.name, 'max_loading_percent'] = max_loading_percent
            net.trafo.at[row.name, 'std_type'] = max_capacity_trafo['trafo']

            # Calculate the cost
            cost = max_capacity_trafo['transformer_cost']
            total_cost += cost

        else:
            # If it's already the highest capacity transformer, add parallel transformers as before
            required_additional_transformers = np.ceil(loading_percent / target_loading_percent) - 1
            for i in range(int(required_additional_transformers)):
                new_transformer_name = f"{original_name}_parallel_{i + 1}_{load_level:.2f}_{iteration}"
                new_trafo = pp.create_transformer(net, hv_bus, lv_bus, std_type, name=new_transformer_name)
                # Set the max loading percent for the new transformer
                net.trafo.at[new_trafo, 'max_loading_percent'] = max_loading_percent

            cost = required_additional_transformers * current_trafo_data['transformer_cost'].iloc[0]
            total_cost += cost

    return net, total_cost


def iterative_reinforcement(net, code, load_level, threshold, grid_level, iteration_start=1):
    trafo_cost = 0
    line_cost = 0
    trafo_load = 0
    crit_trafos = 0
    crit_lines_km = 0
    avg_trafo_load = 0
    avg_line_load = 0
    iteration_trafo = 1
    converged = True

    while converged:
        converged = run_power_flow(net)

        if not converged:
            break

        if iteration_trafo == 1:

                # Record average trafo and line load
                if net.res_trafo["loading_percent"].mean() > avg_trafo_load:
                    avg_trafo_load = net.res_trafo["loading_percent"].mean()
                if net.res_line["loading_percent"].mean() > avg_line_load:
                    avg_line_load = net.res_line["loading_percent"].mean()

        # Break if HV - no trafo to upgrade
        if grid_level == "HV":
            break

        # First, check for overloaded transformers
        critical_trafos, critical_trafo3w = check_critical_transformers(net, threshold)

        if iteration_trafo == 1:
            # Record critical trafos
            if len(critical_trafos)+len(critical_trafo3w) > crit_trafos:
                crit_trafos = len(critical_trafos) + len(critical_trafo3w)

        if not critical_trafos.empty:

            net, cost = (
                reinforce_most_overloaded_transformer(net, critical_trafos, iteration_trafo, load_level, grid_level))
            trafo_cost += cost
            iteration_trafo += 1

        else:
            break

    iteration_line = 1
    # Now, reinforce lines if no further transformer reinforcements are needed
    while converged:
        converged = run_power_flow(net)
        if not converged:
            break

        # Check for overloaded lines
        critical_lines = check_critical_lines(net, threshold)
        if iteration_line == 1:
            # Record critical lines
            if critical_lines["length_km"].sum() > crit_lines_km:
                crit_lines_km = critical_lines["length_km"].sum()

        if not critical_lines.empty:
            net, cost = reinforce_most_overloaded_line(net, code, critical_lines, iteration_line, load_level, grid_level)
            line_cost += cost
            iteration_line += 1

        else:
            if grid_level != "HV":
                trafo_load = net.res_trafo["p_hv_mw"].sum()
            break

    return net, trafo_cost, line_cost, trafo_load, crit_trafos, crit_lines_km, avg_trafo_load, avg_line_load


def analyze_scenario(net, code, load_df, grid_level):
    trafo_cost = 0
    line_cost = 0
    max_trafo_load = 0
    max_crit_trafos = 0
    max_crit_lines_km = 0
    max_avg_trafo_load = 0
    max_avg_line_load = 0
    net.load = load_df
    iteration = 0
    load_level = 1.0

    threshold = 100

    if grid_level == "MV":
        threshold = 50  # Quelle: Verteilnetzstudie NRW - 100% theshold with n-1 criterion

    if grid_level == "HV":
        threshold = 40  # Quelle: Verteilnetzstudie NRW - 40% threshold

    while load_level > 0:
        iteration += 1
        converged, load_level = incremental_load_reduction(net, load_df)
        if not converged:
            break

        (net, t_cost, l_cost, trafo_load, crit_trafos, crit_lines_km,
         avg_trafo_load, avg_line_load) = iterative_reinforcement(net, code, load_level, threshold, grid_level, iteration_start=iteration)

        trafo_cost += t_cost
        line_cost += l_cost
        max_trafo_load = max(max_trafo_load, trafo_load)
        max_crit_trafos = max(max_crit_trafos, crit_trafos)
        max_crit_lines_km = max(max_crit_lines_km, crit_lines_km)
        max_avg_trafo_load = max(max_avg_trafo_load, avg_trafo_load)
        max_avg_line_load = max(max_avg_line_load, avg_line_load)

        iteration = 0
        if load_level == 1:
            break

    return trafo_cost, line_cost, max_trafo_load, max_crit_trafos, max_crit_lines_km, max_avg_trafo_load, max_avg_line_load
