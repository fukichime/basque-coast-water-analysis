import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import joblib
import community as community_louvain
from sklearn.metrics import adjusted_rand_score
from sklearn.tree import plot_tree

sns.set_theme(style="whitegrid", context="talk")
OUTPUT_FIGURES = "outputs/figures"

master = pd.read_csv("data/processed/clean_water_spatial_clustered.csv")
rules = pd.read_csv("outputs/models/apriori_rules.csv")
edges = pd.read_csv("outputs/models/spatial_edges.csv")
dt_model = joblib.load("outputs/models/decision_tree_model.pkl")

num_features = ['Temperature', 'Salinity', 'Nitrate', 'Ammonium']
short_features = ['Temp', 'Salinity', 'Nitrate', 'Ammonium']

cluster_palette = sns.color_palette("Set1", n_colors=3)



#K-MEANS DUAL SCATTER
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

counts = master['cluster_label'].value_counts()

#Temperature vs Salinity
sns.scatterplot(data=master, x='Temperature', y='Salinity', hue='cluster_label', 
                palette=cluster_palette, alpha=0.6, s=50, ax=ax1)
ax1.set_title("Regimes: Temperature vs. Salinity", fontweight='bold')

#Nutrients
sns.scatterplot(data=master, x='Nitrate', y='Ammonium', hue='cluster_label', 
                palette=cluster_palette, alpha=0.6, s=50, ax=ax2)
ax2.set_title("Regimes: Nutrients (Nitrate vs. Ammonium)", fontweight='bold')

#legend
handles, labels = ax1.get_legend_handles_labels()
new_labels = [f"Regime {int(label)} (n={counts[int(label)]})" for label in labels]
ax1.legend(handles=handles, labels=new_labels, title="Chemical Regimes")
ax2.get_legend().remove()

plt.tight_layout()
plt.savefig(f"{OUTPUT_FIGURES}/pres_01_kmeans_dual_scatter.png", dpi=300)
plt.show()


#TOP STRESS RULES
def clean_rule_with_consequent(ante, cons):
    ant_str = str(ante).replace("frozenset({'", "").replace("'})", "").replace("', '", " + ").replace("_cat", "")
    cons_str = str(cons).replace("frozenset({'", "").replace("'})", "").replace("_cat", "").replace("oxygen_status_", "")
    return f"{ant_str} → {cons_str}"

rules['Rule'] = rules.apply(lambda r: clean_rule_with_consequent(r['antecedents'], r['consequents']), axis=1)
top_rules = rules.head(10).sort_values('lift', ascending=True)

plt.figure(figsize=(14, 7)) 
plt.barh(top_rules['Rule'], top_rules['lift'], color='#e74c3c')
plt.axvline(x=1.0, color='black', linestyle='--', label='Baseline (Lift = 1)')

plt.title("Top Environmental Triggers for Ecological Stress", fontweight='bold')
plt.xlabel("Lift (Likelihood multiplier over baseline)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_FIGURES}/pres_02_rules.png", dpi=300)
plt.show()


#DECISION TREE
plt.figure(figsize=(26, 12))

plot_tree(
    dt_model, 
    feature_names=short_features, 
    class_names=dt_model.classes_, 
    filled=True, 
    rounded=True,
    fontsize=11,
    proportion=True
)
plt.title("Decision Tree: Predictive Thresholds for Ecological Stress", fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_FIGURES}/pres_03_decision_tree.png", dpi=300)
plt.show()


#DECISION BOUNDARY
root_idx = dt_model.tree_.feature[0]
root_var = num_features[root_idx]
root_thresh = dt_model.tree_.threshold[0]

if root_var == 'Temperature':
    x_col, y_col = 'Temperature', 'Salinity'
elif root_var == 'Salinity':
    x_col, y_col = 'Salinity', 'Temperature'
elif root_var == 'Nitrate':
    x_col, y_col = 'Nitrate', 'Salinity'
else:
    x_col, y_col = 'Ammonium', 'Salinity'

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=master, 
    x=x_col, 
    y=y_col, 
    hue='oxygen_status', 
    palette={'Normal': '#3498db', 'Low (Stressed)': '#e74c3c'}, 
    alpha=0.5
)

plt.axvline(x=root_thresh, color='black', linestyle='--', linewidth=2.5, 
            label=f'Root split: {root_var} = {root_thresh:.2f}')

plt.title(f"Data Spread & Primary Tree Boundary", fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{OUTPUT_FIGURES}/pres_04_tree_boundary.png", dpi=300)
plt.show()




#SPATIAL NETWORK
plt.figure(figsize=(14, 9))
G = nx.Graph()

for _, row in edges.iterrows():
    G.add_edge(row['Source'], row['Target'])

node_positions = {}
node_colors = []
unique_stations = master.drop_duplicates('siteid')

for _, row in unique_stations.iterrows():
    site = row['siteid']
    if site in G.nodes():
        node_positions[site] = (row['decimallongitude'], row['decimallatitude'])
        cluster = int(row['cluster_label'])
        node_colors.append(cluster_palette[cluster] if cluster < len(cluster_palette) else 'gray')

nx.draw_networkx_nodes(G, node_positions, node_color=node_colors, node_size=150, edgecolors='white')
nx.draw_networkx_edges(G, node_positions, alpha=0.3, edge_color='gray')

import matplotlib.patches as mpatches

legend_patches = [mpatches.Patch(color=cluster_palette[i], label=f'Regime {i}') for i in range(len(counts))]
plt.legend(handles=legend_patches, title="K-Means Regimes", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title("Spatial Vulnerability Network (Basque Coast)", fontweight='bold')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FIGURES}/pres_05_spatial_network_clean.png", dpi=300)
plt.show()

#TEMPORAL TRENDS (1994 - 2023)
plt.figure(figsize=(14, 6))

#only the stressed
stressed_data = master[master['oxygen_status'] == 'Low (Stressed)']

#stress events per year
yearly_stress = stressed_data.groupby('year').size().reset_index(name='stress_events')

sns.lineplot(
    data=yearly_stress, 
    x='year', 
    y='stress_events', 
    marker='o', 
    linewidth=2.5, 
    color='#e74c3c',
    markersize=8
)

sns.regplot(
    data=yearly_stress, 
    x='year', 
    y='stress_events', 
    scatter=False, 
    color='black', 
    line_kws={"linestyle": "--", "alpha": 0.5},
    label="Overall Trend"
)

plt.title("Frequency of Ecological Stress Events Over Time (30 Years)", fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Number of Hypoxic Events")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FIGURES}/pres_06_temporal_trend.png", dpi=300)
plt.show()