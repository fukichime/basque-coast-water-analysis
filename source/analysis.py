import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors

from mlxtend.frequent_patterns import apriori, association_rules

DATA_PROCESSED = "data/processed"
OUTPUT_MODELS = "outputs/models"
os.makedirs(OUTPUT_MODELS, exist_ok=True)

master = pd.read_csv(os.path.join(DATA_PROCESSED, "clean_water_spatial.csv"))

num_features = ['Temperature', 'Salinity', 'Nitrate', 'Ammonium']
cat_features = ['season', 'oxygen_status', 'Temperature_cat', 'Salinity_cat', 'Nitrate_cat', 'Ammonium_cat']

print("K-MEANS CLUSTERING")
#filling  NaNs with median
X_cluster = master[num_features].fillna(master[num_features].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
master['cluster_label'] = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, master['cluster_label'])

print(f"Optimal Clusters Formed: 3")
print(f"Silhouette Score (Cohesion): {sil_score:.3f}")
print("Cluster distribution:")
print(master['cluster_label'].value_counts().to_string())


print("ASSOCIATION RULE MINING(APRIORI)")
#droping NaNs
df_apriori = master[cat_features].dropna()

#One-Hot
df_dummies = pd.get_dummies(df_apriori).astype(bool)

itemsets = apriori(df_dummies, min_support=0.05, use_colnames=True)
rules = association_rules(itemsets, metric="lift", min_threshold=1.2)

#predict --> oxygen_status_Low (Stressed)
target_rules = rules[rules['consequents'].apply(lambda x: 'oxygen_status_Low (Stressed)' in x)]
target_rules = target_rules.sort_values(by='lift', ascending=False)

rules_path = os.path.join(OUTPUT_MODELS, "apriori_rules.csv")
target_rules.to_csv(rules_path, index=False)

print(f"Generated {len(target_rules)} rules pointing to Ecological Stress.")
print("Top 2 Rules by Lift:")
for idx, row in target_rules.head(2).iterrows():
    ants = list(row['antecedents'])
    cons = list(row['consequents'])
    print(f"IF {ants} THEN {cons} (Lift: {row['lift']:.2f}, Confidence: {row['confidence']:.2f})")


print("DECISION TREE")
X_tree = master[num_features].fillna(master[num_features].median())
y_tree = master['oxygen_status']

#train/test split(stratified for 75/25 balance)
X_train, X_test, y_train, y_test = train_test_split(X_tree, y_tree, test_size=0.2, stratify=y_tree, random_state=42)

#DT depth 4
dt = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
print("Decision Tree Classification Report (Test Set):")
print(classification_report(y_test, y_pred))

joblib.dump(dt, os.path.join(OUTPUT_MODELS, "decision_tree_model.pkl"))
print("Saved trained Decision Tree to outputs/models/decision_tree_model.pkl")


print(" GRAPH ANALYSIS")
coords_df = master.drop_duplicates(subset=['siteid'])[['siteid', 'decimallatitude', 'decimallongitude']].reset_index(drop=True)

#nearest neighbors (3 closest stations to every station)
# k=4 (+1 for itself)
nn = NearestNeighbors(n_neighbors=4, metric='haversine')

rad_coords = np.radians(coords_df[['decimallatitude', 'decimallongitude']])
nn.fit(rad_coords)
distances, indices = nn.kneighbors(rad_coords)

#edge list (source --> target)
edges = []
for i in range(len(coords_df)):
    source = coords_df.loc[i, 'siteid']
    #skip the self-connection
    for j in range(1, 4):
        target_idx = indices[i, j]
        target = coords_df.loc[target_idx, 'siteid']
        edges.append({'Source': source, 'Target': target, 'Distance_Rank': j})

edges_df = pd.DataFrame(edges)


edges_path = os.path.join(OUTPUT_MODELS, "spatial_edges.csv")
edges_df.to_csv(edges_path, index=False)

#SAVE
master.to_csv(os.path.join(DATA_PROCESSED, "clean_water_spatial_clustered.csv"), index=False)

print(f"Built Spatial Graph with {len(coords_df)} nodes and {len(edges_df)} connecting edges.")
print("Saved edges to outputs/models/spatial_edges.csv")
print("Saved clustered master dataset for final visualizations.")
