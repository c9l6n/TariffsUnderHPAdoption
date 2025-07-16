# Heat Pump Grid Reinforcement Simulation

This repository includes all code related to the analyses performed as part of the manuscript "Impact of Widespread Heat Pump Adoption on Grid Reinforcement Costs and Network Tariffs". It contains a modular, parallelized simulation framework to assess the impact of increasing heat pump (HP) penetration on electricity distribution networks (LV, MV, HV). It supports empirical household load modeling, automated reinforcement planning, and cost/tariff analysis.

---

## 📁 Repository Structure

   ```bash
   project-root/
   │
   ├── main.py # Main script to launch all grid simulations in parallel
   │
   ├── data/ # Input datasets and result files
   │ ├── 2019_data_15min.hdf5 # File to be downloaded here: https://doi.org/10.5281/zenodo.5642902, file 2019_data_15min.hdf5
   │ ├── 2019_data_household_information.csv
   │ ├── people_per_unit.csv
   │ ├── units_per_house.csv
   │ ├── standardLines.csv
   │ ├── standardTrafos.csv
   │ └── results/ # Created once results are generated
   │ ├── LVGridResults/ # Intermediate results per LV grid
   │ ├── MVGridResults/ # Intermediate results per MV grid
   │ └── HVGridResults/ # Intermediate results per HV grid
   │
   ├── src/ # Core model logic
   │ ├── PFA_LV.py
   │ ├── PFA_MV.py
   │ ├── PFA_HV.py
   │ ├── LV_parallel_processing.py
   │ ├── MV_parallel_processing.py
   │ ├── HV_parallel_processing.py
   │ ├── dataProcessing.py
   │ ├── energyFlow.py
   │ ├── generateProfiles.py
   │ ├── reinforceGrid.py
   │ ├── results_prep.py
   │
   ├── ipynb/ # Post-processing notebooks
   │ ├── overload_and_cost_analysis_v1.ipynb
   │ └── tariff_analysis_v1.ipynb
```
---

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy simbench pandapower h5py

2. **Execute the Full Simulation**:
   ```bash
   python main.py

3. **Analyze Results**:

   Use notebooks in ipynb/ for:
   - Overload and cost visualization.
   - Tariff projection using PV-based asset depreciation models.
  
--

## 📊 Output

The model will generate:
- CSV summaries per grid archetype in /data/results/
- Optional plots and tables via notebooks.
