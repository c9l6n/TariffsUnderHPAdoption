import pandas as pd
import numpy as np
import simbench as sb
import random


def get_grid_type(grid_code):
    if 'urban' in grid_code:
        return 'urban'
    elif 'semiurb' in grid_code:
        return 'semiurb'
    elif 'rural' in grid_code:
        return 'rural'
    elif 'comm' in grid_code:
        return 'semiurb'
    elif 'mixed' in grid_code:
        return 'semiurb'
    else:
        raise ValueError("Unknown grid type in grid code.")


def count_number_of_profiles(net):
    num_houses = len(net.bus[net.bus['vn_kv'] == 0.4])  # Number of houses with vn_kv of 0.4
    return num_houses


def assign_units_to_houses(grid_type, num_houses, units_per_house_probs):
    units_distribution = []
    for _ in range(num_houses):
        units = np.random.choice(
            [1, 2, random.randint(3, 6), random.randint(7, 12), random.randint(13, 20)],
            p=units_per_house_probs.loc[grid_type].values
        )
        units_distribution.append(units)
    return units_distribution


def assign_people_to_units(grid_type, units_per_house, people_per_unit_probs):
    people_distribution = []
    for units in units_per_house:
        for _ in range(units):
            people = np.random.choice(
                [1, 2, 3, 4],
                p=people_per_unit_probs.loc[grid_type].values
            )
            people_distribution.append(people)
    return people_distribution


def create_household_profiles_from_net_load(net, grid_type, units_per_house_probs, people_per_unit_probs, spacing_in_minutes):
    
    # Filter net.load for rows where the "profile" column starts with "H"
    filtered_loads = net.load[net.load['profile'].str.startswith('H', na=False)]

    # Get distinct bus IDs
    distinct_buses = filtered_loads['bus'].unique()

    # Prepare to assign household profiles
    num_houses = len(distinct_buses)
    units_per_house = assign_units_to_houses(grid_type, num_houses, units_per_house_probs)
    people_per_household = assign_people_to_units(grid_type, units_per_house, people_per_unit_probs)

    household_profiles = []

    for bus_id, bus in enumerate(distinct_buses):
        num_units = units_per_house[bus_id]
        people_per_unit = people_per_household[:num_units]
        people_per_household = people_per_household[num_units:]  # Update remaining people

        for index, people in enumerate(people_per_unit):
            shift_minutes = 0
            if spacing_in_minutes == 1:
                shift_minutes = np.random.randint(-15, 16)  # for file with 1 minute intervals
            elif spacing_in_minutes == 15:
                shift_minutes = np.random.randint(-1, 2) * 15  # for file with 15 min intervals
            else:
                raise ValueError("Spacing not 1 or 15 min")  # Use raise to stop execution on error

            household_profile = {
                'bus_id': bus,
                'household_id': f"{bus}_{index + 1}",  # Combine bus ID with a running index
                'num_people': people,
                'input_household_id': None,  # Placeholder for future input_household_id assignment
                'shift_minutes': shift_minutes
            }
            household_profiles.append(household_profile)

    return pd.DataFrame(household_profiles)



def match_households(household_profiles, household_info):
    def get_matching_household(num_people):
        matching_households = household_info[household_info['Number of inhabitants'] == num_people]
        if matching_households.empty:
            print(f"No matching households found for {num_people} inhabitants")
        return matching_households.sample(n=1).iloc[0]

    matched_households = household_profiles['num_people'].apply(get_matching_household).reset_index(drop=True)
    household_profiles = household_profiles.reset_index(drop=True)
    household_profiles['input_household_id'] = matched_households['household id'].values
    return household_profiles


def shift_profile(df, shift_minutes):
    df_shifted = df.copy()
    df_shifted['time'] = df_shifted['time'] + pd.to_timedelta(shift_minutes, unit='m')

    if len(df_shifted) >= 365 * 24 * 60:
        # Create a complete time range and merge with the shifted dataframe
        full_time_range = pd.date_range(start='2019-01-01 00:00', end='2019-12-31 23:59', freq='min')
        df_full_time = pd.DataFrame({'time': full_time_range})
        df_shifted = pd.merge(df_full_time, df_shifted, on='time', how='left')

        # Use NumPy to fill NaN values more efficiently
        for col in ['P_Load', 'P_Pump']:
            data = df_shifted[col].values
            mask = np.isnan(data)
            data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
            df_shifted[col] = data

    elif len(df_shifted) == 365 * 24 * 4:
        full_time_range = pd.date_range(start='2019-01-01 00:00', end='2019-12-31 23:45', freq='15min')
        df_full_time = pd.DataFrame({'time': full_time_range})
        df_shifted = pd.merge(df_full_time, df_shifted, on='time', how='left')

        # Use NumPy to fill NaN values more efficiently
        for col in ['P_Load', 'P_Pump']:
            data = df_shifted[col].values
            mask = np.isnan(data)
            data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
            df_shifted[col] = data

    return df_shifted


def create_load_profiles(household_profiles, load_data, spacing_in_minutes):
    
    # Initialize an empty list to collect all profiles
    all_profiles = []

    for _, household in household_profiles.iterrows():
        input_household_id = household['input_household_id']

        # Filter and reset index for the specific household
        df_household = load_data[load_data['household_id'] == input_household_id].drop(
            columns=['household_id']).reset_index(drop=True)

        # Shift the profile based on shift_minutes
        shift_minutes = household['shift_minutes']
        df_shifted = shift_profile(df_household, shift_minutes)

        # Add identifying columns
        df_shifted['bus_id'] = household['bus_id']
        df_shifted['input_household_id'] = input_household_id

        # Append to the all_profiles list
        all_profiles.append(df_shifted)

    # Concatenate all profiles into a single DataFrame
    load_profiles_df = pd.concat(all_profiles, ignore_index=True)

    return load_profiles_df


def aggregate_profiles(load_profiles_df):
    
    # Ensure 'time' is in datetime format for accurate grouping
    load_profiles_df['time'] = pd.to_datetime(load_profiles_df['time'])

    # Group by both 'bus_id' and 'time', then sum all load values for each group
    aggregated_profiles_df = load_profiles_df.groupby(['bus_id', 'time'], as_index=False).sum()

    return aggregated_profiles_df


def generate_load_profiles(grid_code, load_data, household_info, units_per_house_probs, people_per_unit_probs,
                           random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    net = sb.get_simbench_net(grid_code)
    grid_type = get_grid_type(grid_code)
    spacing_in_seconds = (load_data["time"][1] - load_data["time"][0]).total_seconds()
    spacing_in_minutes = spacing_in_seconds / 60  # Convert seconds to minutes

    household_profiles = create_household_profiles_from_net_load(net, grid_type, units_per_house_probs, people_per_unit_probs, spacing_in_minutes)
    household_profiles = match_households(household_profiles, household_info)

    # Aggregate by bus_id to get the number of units and total number of people per bus
    agg_profiles = household_profiles.groupby('bus_id').agg(
        num_units=('household_id', 'count'),  # Count the number of units (households)
        num_people=('num_people', 'sum')  # Sum the total number of people per bus
    ).reset_index()

    load_profiles = create_load_profiles(household_profiles, load_data, spacing_in_minutes)
    aggregated_profiles = aggregate_profiles(load_profiles)

    # Merge load_profiles_time and aggregated_profiles
    aggregated_profiles = aggregated_profiles.merge(agg_profiles[['bus_id', 'num_units', 'num_people']],
                                                    on='bus_id', how='left')

    return aggregated_profiles, household_profiles