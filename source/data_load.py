import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_RAW = "data"
DATA_PROCESSED = "data/processed"
OUTPUT_FIGURES = "outputs/figures"

os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(OUTPUT_FIGURES, exist_ok=True)

sns.set_theme(style="whitegrid")

def get_season(month):
    if pd.isna(month): return None
    if month in [3, 4, 5]: return 'Spring'
    if month in [6, 7, 8]: return 'Summer'
    if month in [9, 10, 11]: return 'Autumn'
    return 'Winter'

def print_diagnostics(df, name):
    print(f"\n{name} LOOKUP -->")
    print(f"Shape: {df.shape}")
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nMissing Values %:")
    print((df.isnull().sum() / len(df) * 100).round(2).astype(str) + '%')


water = pd.read_csv(os.path.join(DATA_RAW, "WATER.csv"), low_memory=False)
metadata = pd.read_csv(os.path.join(DATA_RAW, "METADATA.csv"), low_memory=False)

#numeric data
water['parameter_value'] = pd.to_numeric(water['parameter_value'], errors='coerce')
water = water.dropna(subset=['parameter_value', 'datecollected'])

#standardize siteid 
water['siteid'] = water['siteid'].str.replace(r'[HLBS]+$', '', regex=True)

required_params = ['Temperature', 'Salinity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Ammonium', 'Nitrate', 'Silicate']
water_filtered = water[water['parameter'].isin(required_params)]

#long --> wide
master = water_filtered.pivot_table(
    index=['siteid', 'datecollected'],
    columns='parameter',
    values='parameter_value',
    aggfunc='mean'
).reset_index()

#merge with meta
coords = metadata[['siteid', 'decimallatitude', 'decimallongitude']].drop_duplicates(subset=['siteid'])
master = pd.merge(master, coords, on='siteid', how='inner')

master = master.dropna(subset=['Dissolved oxygen'])

#categorical bins

#temp cat
master['datecollected'] = pd.to_datetime(master['datecollected'])
master['year'] = master['datecollected'].dt.year
master['month'] = master['datecollected'].dt.month
master['season'] = master['month'].apply(get_season)

#Target Category --> Hypoxia Status
do_threshold = master['Dissolved oxygen'].quantile(0.25)
print(f"Setting Stress Threshold at DO < {do_threshold:.2f} mg/L")

master['oxygen_status'] = master['Dissolved oxygen'].apply(
    lambda x: 'Low (Stressed)' if x < do_threshold else 'Normal'
)

#chem cat Quartiles --> Apriori
#High/Medium/Low bins
chem_features = ['Temperature', 'Salinity', 'Nitrate', 'Ammonium']
for chem in chem_features:
    if chem in master.columns:
        p33 = master[chem].quantile(0.33)
        p66 = master[chem].quantile(0.66)
        
        master[f'{chem}_cat'] = master[chem].apply(
            lambda x: 'High' if pd.notnull(x) and x > p66 
            else ('Medium' if pd.notnull(x) and x > p33 
            else ('Low' if pd.notnull(x) else None))
        )

print_diagnostics(master, "FINAL MASTER DATASET")

output_path = os.path.join(DATA_PROCESSED, "clean_water_spatial.csv")
master.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

#EDA
print("\nEDA Visuals")

#Chemical Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_cols = [p for p in required_params if p in master.columns]
sns.heatmap(master[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title("Chemical Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIGURES, "eda_01_correlation.png"))
plt.close()

#Geospatial Hotspots For Graph
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=master, x='decimallongitude', y='decimallatitude',
    hue='Dissolved oxygen', palette='viridis_r', s=50, alpha=0.7
)
plt.title("Geospatial Distribution of Dissolved Oxygen")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="DO (mg/L)", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIGURES, "eda_02_spatial_hotspots.png"))
plt.close()

#Target Class Balance for DT and Assos
plt.figure(figsize=(8, 5))
sns.countplot(
    data=master, 
    x='oxygen_status', 
    order=['Normal', 'Low (Stressed)'], 
    hue='oxygen_status',
    palette=['#3498db', '#e74c3c'],
    legend=False
)
plt.title(f"Distribution of Oxygen Status (Stress Threshold: < {do_threshold:.2f} mg/L)")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIGURES, "eda_03_target_balance.png"))
plt.close()

#Box plot
features = ['Temperature', 'Salinity', 'Nitrate', 'Ammonium']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, var in enumerate(features):
    if var in master.columns:
        sns.boxplot(
            data=master, x='oxygen_status', y=var, 
            order=['Normal', 'Low (Stressed)'], 
            palette={'Normal': '#3498db', 'Low (Stressed)': '#e74c3c'}, ax=axes[i], 
            showfliers=False, boxprops={'alpha': 0.5} # Hide outliers, lower opacity
        )
        
        sns.stripplot(
            data=master, x='oxygen_status', y=var, 
            order=['Normal', 'Low (Stressed)'], 
            color='black', alpha=0.15, size=3, jitter=True, ax=axes[i]
        )
        
        axes[i].set_title(f'{var} by Oxygen Status', fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel(var)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIGURES, "eda_04_scatter_boxplots.png"), dpi=300)


#double-check
print("DOUBLE-CHECK --> VERIFICATION")
print(master.info(verbose=False))
print("\nSample Preview (Numerical + Categorical Alignment):")
preview_cols = ['siteid', 'datecollected', 'Temperature', 'Dissolved oxygen', 'oxygen_status', 'Temperature_cat', 'season']
print(master[preview_cols].head(3).to_string(index=False))