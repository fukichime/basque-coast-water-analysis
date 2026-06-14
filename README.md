# Coastal Water Quality Analysis: Clustering, Association Rules, and Spatial Networks for Hypoxia Drivers in the Basque Country (1995–2023)

**Purpose:** To mine 30 years of coastal water data to identify the chemical regimes, geographic hotspots, and predictive thresholds that trigger hypoxia (ecological stress).

### Folder Structure
* `data/` - Raw datasets (`WATER.csv`, `METADATA.csv`)
* `data/processed/` - Cleaned, merged, and clustered datasets
* `outputs/figures/` - EDA and evaluation visualizations
* `outputs/models/` - Saved models, Apriori rules, and Graph edge lists
* `source/` - Python execution scripts

### Methods Used
1. **Unsupervised Clustering:** K-Means (Regime detection)
2. **Association Rule Mining:** Apriori (Trigger combinations)
3. **Supervised Classification:** Decision Trees (Threshold prediction)
4. **Spatial Network Analysis:** K-Nearest Neighbors Graph Mining (Topological vulnerability)

### How to Reproduce
1. Clone this repository and install dependencies: `pip install -r requirements.txt`
2. Place the raw `WATER.csv` and `METADATA.csv` datasets directly into the `data/` folder.
3. Execute the pipeline in strict order:
   * **Step 1:** Clean data and generate EDA: `python source/data_load.py`
   * **Step 2:** Run ML models and save objects: `python source/analysis.py`
   * **Step 3:** Generate evaluation metrics and plots: `python source/eval.py`

### Summary & Findings
* **Chemical Regimes:** K-Means successfully segmented the coastal waters into 3 distinct regimes, isolating a massive dominant baseline (4,376 samples) and a highly anomalous pollution/stress regime (61 samples).
* **Stress Triggers:** Apriori rules revealed that ecological stress is not random. The combination of Summer heat and Medium Salinity (estuarine mixing zones) makes the waters 3.54 times more likely to experience a hypoxic crash.
* **Predictive Thresholds:** The Decision Tree successfully learned to predict Stressed events with a 79% recall rate, establishing a transparent, actionable threshold for early-warning monitoring.
* **Geographic Contagion:** Spatial Graph analysis built a 51-node network demonstrating that hypoxic stress clusters into tight, geographically contiguous communities rather than appearing randomly along the coast.

### Reference
* Borja Á, Adarraga I, Bald J, Belzunce-Segarra MJ, Cruz I, Franco J, Garmendia JM, Larreta J, Laza-Martínez A, Manzanos A, Marquiegui MA, Martín I, Martínez J, Menchaca I, Pouso S, Revilla M, Rodríguez JG, Ruiz JM, Sagarmínaga Y, Solaun O, Uriarte A, Zorita I and Muxika I (2024). Marine Biodiversity and Environmental Data: An AI-Ready, Open Dataset from the long term (1995–2023) Basque Country Monitoring Network. *Front. Ocean Sustain.* 2:1528837. doi: 10.3389/focsu.2024.1528837 
* Data preprocessing, ML execution, and network graphing performed via `scikit-learn`, `mlxtend`, and `networkx`.