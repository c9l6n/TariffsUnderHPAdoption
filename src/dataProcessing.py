import pandas as pd
import numpy as np

def h5py_prep(f, household_info):
    """
    Processes household load and heat pump data from HDF5 and merges with household metadata.

    Parameters:
    - f: HDF5 file object
    - household_info: DataFrame with household metadata

    Returns:
    - dfMerged: Processed load data with 'P_Load' and 'P_Pump' per household
    - household_info_filtered: Filtered metadata for valid households
    """
    # Replace NaNs with 0 and convert 'Ventilation system' to boolean
    household_info.fillna(0, inplace=True)
    household_info['Ventilation system'] = household_info['Ventilation system'].apply(
        lambda x: True if x == 'Yes' else False
    )

    def process_data(data_dict, system_type):
        df = pd.concat([
            pd.DataFrame(np.array(data_dict[key][system_type]["table"]))[["index", "P_TOT"]]
            .assign(building=key)
            for key in data_dict.keys()
        ], ignore_index=True)
        df['time'] = pd.to_datetime('1970-01-01') + pd.to_timedelta(df['index'], unit='s')
        df = df[["building", "time", "P_TOT"]]
        return df

    # Process household and heat pump data
    dfLoads = process_data(f["NO_PV"], "HOUSEHOLD")
    dfPumps = process_data(f["NO_PV"], "HEATPUMP")

    # Merge and rename columns
    dfLoads.rename(columns={"P_TOT": "P_Load"}, inplace=True)
    dfPumps.rename(columns={"P_TOT": "P_Pump"}, inplace=True)
    dfMerged = pd.merge(dfLoads, dfPumps, on=["building", "time"], how="outer")

    # Drop buildings with poor data quality
    droppedBuildings = ["SFH6", "SFH13", "SFH17", "SFH24", "SFH25", "SFH31", "SFH34", "SFH37", "SFH40"]
    dfMerged = dfMerged[~dfMerged['building'].isin(droppedBuildings)]

    # Standardize household ID format
    dfMerged.rename(columns={"building": "household_id"}, inplace=True)
    dfMerged['household_id'] = dfMerged['household_id'].str.extract('(\d+)').astype(int)

    # Filter household metadata to only those present in dfMerged
    valid_household_ids = dfMerged['household_id'].unique()
    household_info_filtered = household_info[household_info['household id'].isin(valid_household_ids)]

    # Convert power to kW and clip heat pump loads to physical maximum
    dfMerged['P_Load'] = dfMerged['P_Load'] / 1000
    dfMerged['P_Pump'] = dfMerged['P_Pump'] / 1000
    dfMerged['P_Pump'] = dfMerged['P_Pump'].clip(upper=4.2)

    return dfMerged, household_info_filtered