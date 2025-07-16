import os
import pandas as pd
import numpy as np
import simbench as sb


def load_grid_data(base_dir, voltage_level):
    """Load grid data for specified voltage level."""
    results_dir = f"{voltage_level}GridResults"
    pattern = f"{voltage_level.lower()}_*.csv"
    
    data_path = os.path.join(base_dir, "data", "results", results_dir)
    all_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    
    dfs = []
    for file in all_files:
        df = pd.read_csv(os.path.join(data_path, file))
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def get_base_filtered_load(net, level):
    """Get non-household load for a specific grid level."""
    if level == 'LV':
        return net.load[~net.load['profile'].str.startswith('H', na=False)]['p_mw'].sum()
    elif level == 'MV':
        return net.load[~net.load['profile'].str.startswith('lv_', na=False)]['p_mw'].sum()
    elif level == 'HV':
        return net.load[net.load['profile'].str.startswith('mv_add', na=False)]['p_mw'].sum()
    return 0


def get_grid_code_from_profile(profile):
    """Convert profile string to SimBench grid code."""
    if not isinstance(profile, str):
        return None
        
    if profile.startswith('mv_'):
        # Convert mv_rural to 1-MV-rural--0-no_sw
        return f"1-MV-{profile[3:]}--0-no_sw"
    elif profile.startswith('lv_'):
        # Convert lv_rural2 to 1-LV-rural2--0-no_sw
        return f"1-LV-{profile[3:]}--0-no_sw"
    return None


def get_filtered_loads_hierarchical():
    """Calculate filtered loads for all grids hierarchically."""
    # Storage for filtered loads
    lv_loads = {}
    mv_loads = {}
    hv_loads = {}

    # Step 1: Process LV grids
    lv_codes = ['1-LV-rural1--0-no_sw', '1-LV-rural2--0-no_sw', '1-LV-rural3--0-no_sw',
                '1-LV-semiurb4--0-no_sw', '1-LV-semiurb5--0-no_sw', '1-LV-urban6--0-no_sw']
    
    for code in lv_codes:
        net = sb.get_simbench_net(code)
        lv_loads[code] = {
            'total_load': net.load['p_mw'].sum(),
            'filtered_load': get_base_filtered_load(net, 'LV')
        }

    # Step 2: Process MV grids
    mv_codes = ['1-MV-rural--0-no_sw', '1-MV-semiurb--0-no_sw', 
                '1-MV-urban--0-no_sw', '1-MV-comm--0-no_sw']

    for code in mv_codes:
        net = sb.get_simbench_net(code)
        base_filtered = get_base_filtered_load(net, 'MV')
        
        # Add filtered loads from LV subgrids
        lv_subgrids = net.load[net.load['profile'].str.startswith('lv_', na=False)]['profile']
        lv_filtered_sum = sum(lv_loads.get(get_grid_code_from_profile(grid), {'filtered_load': 0})['filtered_load'] 
                            for grid in lv_subgrids)
        
        mv_loads[code] = {
            'total_load': net.load['p_mw'].sum(),
            'filtered_load': base_filtered + lv_filtered_sum
        }

    # Step 3: Process HV grids
    hv_codes = ['1-HV-mixed--0-no_sw', '1-HV-urban--0-no_sw']

    for code in hv_codes:
        net = sb.get_simbench_net(code)
        base_filtered = get_base_filtered_load(net, 'HV')
        
        # Get MV subgrids from profiles
        mv_subgrids = [get_grid_code_from_profile(p) for p in net.load['profile'] 
                      if not str(p).startswith('mv_add')]
        mv_filtered_sum = sum(mv_loads.get(grid, {'filtered_load': 0})['filtered_load'] 
                            for grid in mv_subgrids if grid)
        
        hv_loads[code] = {
            'total_load': net.load['p_mw'].sum(),
            'filtered_load': base_filtered + mv_filtered_sum
        }

    return {'LV': lv_loads, 'MV': mv_loads, 'HV': hv_loads}


def rename_voltage_columns(df, prefix):
    """Rename columns with voltage level prefix."""
    rename_dict = {
        'total_reinforcement_cost': f'{prefix}_total_reinforcement_cost',
        'total_trafo_reinforcement_cost': f'{prefix}_total_trafo_reinforcement_cost',
        'total_line_reinforcement_cost': f'{prefix}_total_line_reinforcement_cost',
        'base_#_of_transformers': f'{prefix}_base_#_of_transformers',
        'HP_avg_trafo_load': f'{prefix}_HP_avg_trafo_load',
        'HP_#_of_crit_transformers': f'{prefix}_HP_#_of_crit_transformers',
        'HP_#_of_transformers': f'{prefix}_HP_#_of_transformers',
        'base_km_of_lines': f'{prefix}_base_km_of_lines',
        'HP_avg_line_load': f'{prefix}_HP_avg_line_load',
        'HP_km_of_crit_lines': f'{prefix}_HP_km_of_crit_lines',
        'HP_new_km_of_lines': f'{prefix}_HP_new_km_of_lines'
    }
    return df.rename(columns=rename_dict)


def process_columns(df, voltage_level):
    """Add voltage level and region info to DataFrame."""
    df['grid_level'] = voltage_level
    df['grid_region'] = df['grid_code'].apply(lambda x: 
        'rural' if 'rural' in x or 'mixed' in x 
        else 'semiurb' if 'semiurb' in x 
        else 'urban' if 'urban' in x 
        else 'comm' if 'comm' in x 
        else 'unknown')
    return df


def calculate_cumulative_costs(df):
    """Calculate cumulative costs within groups."""
    cumsum_columns = [
        'total_reinforcement_cost',
        'LV_total_reinforcement_cost', 'LV_total_trafo_reinforcement_cost', 'LV_total_line_reinforcement_cost',
        'MV_total_reinforcement_cost', 'MV_total_trafo_reinforcement_cost', 'MV_total_line_reinforcement_cost',
        'HV_total_reinforcement_cost', 'HV_total_trafo_reinforcement_cost', 'HV_total_line_reinforcement_cost'
    ]
    
    df_sorted = df.sort_values(['grid_code', 'random_seed', 'HP_percentage'])
    
    # Create incremental columns before applying cumsum
    for column in cumsum_columns:
        inc_column = f"{column.replace('total', 'inc')}"
        df_sorted[inc_column] = df_sorted[column].copy()
    
    # Apply cumsum to original columns
    for column in cumsum_columns:
        df_sorted[column] = df_sorted.groupby(['grid_code', 'random_seed'])[column].cumsum()
    
    return df_sorted


def pv_rab_return(row, payback_period, discount_rate, equity_rate, equity_percentage, debt_rate):
    """Calculate present value of regulatory asset base return."""
    investment_volume = (row['LV_total_reinforcement_cost'] + 
                        row['MV_total_reinforcement_cost'] + 
                        row['HV_total_reinforcement_cost'])
    annual_depreciation = investment_volume / payback_period
    present_value_total_return = 0
    
    for year in range(1, payback_period + 1):
        beginning_rab = investment_volume - (year - 1) * annual_depreciation
        end_rab = beginning_rab - annual_depreciation
        avg_rab = (beginning_rab + end_rab) / 2
        
        equity_rab = avg_rab * equity_percentage
        debt_rab = avg_rab * (1 - equity_percentage)
        
        equity_return = equity_rab * equity_rate
        debt_return = debt_rab * debt_rate
        annual_return = equity_return + debt_return
        
        present_value_total_return += annual_return / (1 + discount_rate) ** year
    
    return present_value_total_return


def pv_depreciation(row, payback_period, discount_rate):
    """Calculate present value of depreciation."""
    total_reinforcement_cost = (row['LV_total_reinforcement_cost'] + 
                              row['MV_total_reinforcement_cost'] + 
                              row['HV_total_reinforcement_cost'])
    annual_depreciation = total_reinforcement_cost / payback_period
    return sum(annual_depreciation / (1 + discount_rate) ** t 
              for t in range(1, payback_period + 1))


def pv_maintenance(row, payback_period, discount_rate, maintenance_percentage_trafo, maintenance_percentage_line):
    """Calculate present value of maintenance costs."""
    maintenance_costs = sum(
        row[f'{level}_total_trafo_reinforcement_cost'] * maintenance_percentage_trafo +
        row[f'{level}_total_line_reinforcement_cost'] * maintenance_percentage_line
        for level in ['LV', 'MV', 'HV']
    )
    return sum(maintenance_costs / (1 + discount_rate) ** t 
              for t in range(1, payback_period + 1))


def calculate_network_tariff(row, pv_results, payback_period, discount_rate, hp_tariff_discount, sales_tax):
    """Calculate required network tariff with PV values passed explicitly."""
    try:
        total_annual_consumption = row['HP_total_kWh_consumed'] + row['HH_total_kWh_consumed']
        if total_annual_consumption == 0:
            return 0.0
        
        adj_consumption = (row['HH_total_kWh_consumed'] + 
                         row['HP_total_kWh_consumed'] * (1 - hp_tariff_discount))
        pv_factor = (1 - (1 + discount_rate)**-payback_period)
        
        total_cost = (pv_results['PV_RAB_Return'] + 
                     pv_results['PV_Depreciation'] + 
                     pv_results['PV_Maintenance'])
        
        tariff = (total_cost * discount_rate) / (adj_consumption * pv_factor)
        return tariff * 100 * (1 + sales_tax)
    except Exception as e:
        print(f"Error calculating network tariff: {str(e)}")
        return 0.0


def calculate_financial_metrics(df, **kwargs):
    """Calculate financial metrics using keyword arguments."""
    results = df.copy()
    
    # First calculate all PV values
    for index, row in results.iterrows():
        # Store PV calculations in temporary dict
        pv_results = {
            'PV_RAB_Return': pv_rab_return(
                row=row, 
                payback_period=kwargs['payback_period'],
                discount_rate=kwargs['discount_rate'],
                equity_rate=kwargs['equity_rate'],
                equity_percentage=kwargs['equity_percentage'],
                debt_rate=kwargs['debt_rate']
            ),
            
            'PV_Depreciation': pv_depreciation(
                row=row,
                payback_period=kwargs['payback_period'],
                discount_rate=kwargs['discount_rate']
            ),
            
            'PV_Maintenance': pv_maintenance(
                row=row,
                payback_period=kwargs['payback_period'],
                discount_rate=kwargs['discount_rate'],
                maintenance_percentage_trafo=kwargs['maintenance_percentage_trafo'],
                maintenance_percentage_line=kwargs['maintenance_percentage_line']
            )
        }
        
        # Store PV values in DataFrame
        for key, value in pv_results.items():
            results.loc[index, key] = value
            
        # Calculate network tariff using stored PV values
        results.loc[index, 'variable_network_tariff_ct_per_kWh'] = calculate_network_tariff(
            row=row,
            pv_results=pv_results,
            payback_period=kwargs['payback_period'],
            discount_rate=kwargs['discount_rate'],
            hp_tariff_discount=kwargs['hp_tariff_discount'],
            sales_tax=kwargs['sales_tax']
        )
    
    return results


def prepare_results(notebook_dir, payback_period, discount_rate, equity_rate, 
                   equity_percentage, debt_rate, maintenance_percentage_trafo=0.015,
                   maintenance_percentage_line=0.01, hp_tariff_discount=0.0, 
                   sales_tax=0.19):
    """Main function to prepare results DataFrame."""
    try:
    # Load and process data
        lv_df = load_grid_data(notebook_dir, "LV")
        mv_df = load_grid_data(notebook_dir, "MV")
        hv_df = load_grid_data(notebook_dir, "HV")

        # Get hierarchical filtered loads for all grids
        print("Calculating hierarchical filtered loads...")
        hierarchical_loads = get_filtered_loads_hierarchical()
        print(hierarchical_loads)

        # Map loads to dataframes
        for df, level in [(lv_df, 'LV'), (mv_df, 'MV'), (hv_df, 'HV')]:
            level_loads = hierarchical_loads[level]
            
            # Calculate HH-only loads using hierarchical filtered loads
            filtered_loads = df['grid_code'].map(lambda x: level_loads[x]['filtered_load'])
            df['base_load_HH_only'] = df['base_load'] - filtered_loads
            df['HP_percentage_load_HH_only'] = df['HP_percentage_load'] - filtered_loads

        # Clean grid codes
        for df in [lv_df, mv_df, hv_df]:
            df['grid_code'] = df['grid_code'].str.slice(2, -9)
        
        # Rename columns with voltage level prefixes
        lv_df = rename_voltage_columns(lv_df, "LV")
        
        # Process columns
        lv_df = process_columns(lv_df, "LV")
        mv_df = process_columns(mv_df, "MV")
        hv_df = process_columns(hv_df, "HV")
        
        # Merge DataFrames
        merged_df = pd.concat([lv_df, mv_df, hv_df], ignore_index=True)

        # Loop over each grid and adjust the reinforcement costs based on both grid_code and random_seed
        columns_to_adjust=["LV_total_reinforcement_cost", "LV_total_trafo_reinforcement_cost", "LV_total_line_reinforcement_cost", 
                        "MV_total_reinforcement_cost", "MV_total_trafo_reinforcement_cost", "MV_total_line_reinforcement_cost",
                        "HV_total_reinforcement_cost", "HV_total_trafo_reinforcement_cost", "HV_total_line_reinforcement_cost"]

        # Replace NaN with 0 in the specified columns
        merged_df[columns_to_adjust] = merged_df[columns_to_adjust].fillna(0)

        # Clip negative values to 0 for the specified columns
        merged_df[columns_to_adjust] = merged_df[columns_to_adjust].clip(lower=0)

        # Calculate total reinforcement cost
        merged_df['total_reinforcement_cost'] = merged_df['LV_total_reinforcement_cost'] + merged_df['MV_total_reinforcement_cost'] + merged_df['HV_total_reinforcement_cost']
        
        # # Calculate cumulative costs
        results_df = calculate_cumulative_costs(merged_df)
        
        # Prepare financial parameters
        params = {
            'payback_period': payback_period,
            'discount_rate': discount_rate,
            'equity_rate': equity_rate,
            'equity_percentage': equity_percentage,
            'debt_rate': debt_rate,
            'maintenance_percentage_trafo': maintenance_percentage_trafo,
            'maintenance_percentage_line': maintenance_percentage_line,
            'hp_tariff_discount': hp_tariff_discount,
            'sales_tax': sales_tax
        }
        
        # Calculate financial metrics
        results_df = calculate_financial_metrics(df=results_df, **params)

        # Check df
        results_df = results_df[['grid_code', 'grid_level', 'grid_region', 'random_seed', 'HP_percentage', 
                        'total_households', 'total_households_HP', 'HH_total_kWh_consumed', 'HP_total_kWh_consumed',
                        'base_load', 'HP_percentage_load', 'base_load_HH_only', 'HP_percentage_load_HH_only',
                        'total_reinforcement_cost', 'inc_reinforcement_cost',
                        'LV_total_reinforcement_cost', 'LV_inc_reinforcement_cost', 
                        'LV_total_trafo_reinforcement_cost', 'LV_inc_trafo_reinforcement_cost',
                        'LV_total_line_reinforcement_cost', 'LV_inc_line_reinforcement_cost',
                        'MV_total_reinforcement_cost', 'MV_inc_reinforcement_cost',
                        'MV_total_trafo_reinforcement_cost', 'MV_inc_trafo_reinforcement_cost',
                        'MV_total_line_reinforcement_cost', 'MV_inc_line_reinforcement_cost',
                        'HV_total_reinforcement_cost', 'HV_inc_reinforcement_cost',
                        'HV_total_trafo_reinforcement_cost', 'HV_inc_trafo_reinforcement_cost',
                        'HV_total_line_reinforcement_cost', 'HV_inc_line_reinforcement_cost',
                        'LV_base_#_of_transformers', 'LV_HP_avg_trafo_load', 'LV_HP_#_of_crit_transformers', 'LV_HP_#_of_transformers',
                        'LV_base_km_of_lines', 'LV_HP_avg_line_load', 'LV_HP_km_of_crit_lines', 'LV_HP_new_km_of_lines',
                        'MV_base_#_of_transformers', 'MV_HP_avg_trafo_load', 'MV_HP_#_of_crit_transformers', 'MV_HP_#_of_transformers',
                        'MV_base_km_of_lines', 'MV_HP_avg_line_load', 'MV_HP_km_of_crit_lines', 'MV_HP_new_km_of_lines',
                        'HV_base_#_of_transformers', 'HV_HP_avg_trafo_load', 'HV_HP_#_of_crit_transformers', 'HV_HP_#_of_transformers',
                        'HV_base_km_of_lines', 'HV_HP_avg_line_load', 'HV_HP_km_of_crit_lines', 'HV_HP_new_km_of_lines',                        
                        'variable_network_tariff_ct_per_kWh', 'PV_Depreciation', 'PV_Maintenance', 'PV_RAB_Return']]
        results_df.fillna(0, inplace=True)
        
        return results_df
        
    except Exception as e:
        print(f"Error in prepare_results: {str(e)}")
        raise

if __name__ == "__main__":
    # Test parameters
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        results = prepare_results(
            notebook_dir=base_dir,
            payback_period=20,
            discount_rate=0.04,
            equity_rate=0.08,
            equity_percentage=0.3,
            debt_rate=0.03,
            maintenance_percentage_trafo=0.015,
            maintenance_percentage_line=0.01,
            hp_tariff_discount=0.0,
            sales_tax=0.19
        )
        
        print("\nResults summary:")
        print(f"Total rows: {len(results)}")
        print(f"Unique grid codes: {results['grid_code'].nunique()}")
        print(f"Average network tariff: {results['variable_network_tariff_ct_per_kWh'].mean():.2f} ct/kWh")
        
        # Save results
        results.to_csv(os.path.join(base_dir, 'data', 'test_results.csv'), index=False)
        
    except Exception as e:
        print(f"Error in test run: {str(e)}")
        raise