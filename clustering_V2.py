import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------
df = pd.read_excel("HES cu NUTS 3.xlsx", engine="openpyxl")

# ----------------------------------------------------------
# 2. BASIC CLEANUP
# ----------------------------------------------------------
df["ecContribution"] = pd.to_numeric(df["ecContribution"], errors="coerce")
df["netEcContribution"] = pd.to_numeric(df["netEcContribution"], errors="coerce")

# Use netEcContribution (if available), else ecContribution
df["funding"] = df["netEcContribution"].fillna(df["ecContribution"])

# ----------------------------------------------------------
# 3. FEATURE ENGINEERING (Features 1–10)
# ----------------------------------------------------------

# 1. num_projects (unique projects per country)
proj_unique = df[["country", "projectID"]].drop_duplicates()
num_projects = proj_unique.groupby("country").size().rename("num_projects")

# 2–3. total & average funding
funding_sum = df.groupby("country")["funding"].sum().rename("total_funding")
funding_mean = df.groupby("country")["funding"].mean().rename("avg_funding")

# 4. number of coordinators
num_coordinators = (
    df[df["role"] == "coordinator"].groupby("country").size().rename("num_coordinators")
)

# 5–7. counts per activity type / SME status
num_SME = df[df["SME"] == True].groupby("country").size().rename("num_SME")
num_HES = df[df["activityType"] == "HES"].groupby("country").size().rename("num_HES")
num_PRC = df[df["activityType"] == "PRC"].groupby("country").size().rename("num_PRC")

# 8–9. ratios (computed after merging, below)

# 10. avg partners per project in that country
partners_per_project = df.groupby(["country", "projectID"]).size()
avg_partners = partners_per_project.groupby("country").mean().rename("avg_partners")

# ----------------------------------------------------------
# 4. MERGE ALL FEATURES INTO ONE DATAFRAME
# ----------------------------------------------------------
country_features = pd.concat(
    [
        num_projects, funding_sum, funding_mean,
        num_coordinators, num_SME, num_HES, num_PRC,
        avg_partners
    ],
    axis=1
).fillna(0)

# 8. SME_ratio
country_features["SME_ratio"] = (
    country_features["num_SME"] / country_features["num_projects"].replace(0, np.nan)
)

# 9. HES_ratio
country_features["HES_ratio"] = (
    country_features["num_HES"] / country_features["num_projects"].replace(0, np.nan)
)

# Replace NaN ratios
country_features = country_features.fillna(0)

print("Feature matrix:")
print(country_features.head())

# ----------------------------------------------------------
# 5. SCALING
# ----------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(country_features)

# ----------------------------------------------------------
# 6. KMEANS CLUSTERING
# ----------------------------------------------------------
k = 5  # you can tune this later
kmeans = KMeans(n_clusters=k, random_state=0)
cluster_labels = kmeans.fit_predict(X)

country_features["cluster"] = cluster_labels

# Optional: evaluate cluster quality
sil = silhouette_score(X, cluster_labels)
print(f"Silhouette score: {sil:.3f}")

# ----------------------------------------------------------
# 7. LOAD EUROPE MAP
# ----------------------------------------------------------
world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)

world.loc[world["ADMIN"] == "France", "ISO_A2"] = "FR"
europe = world[world["CONTINENT"] == "Europe"]

# Merge clusters into the map
europe = europe.merge(
    country_features,
    left_on="ISO_A2",
    right_index=True,
    how="left"
)

# ----------------------------------------------------------
# 8. CHOROPLETH MAP OF CLUSTERS
# ----------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
europe.plot(
    column="cluster",
    cmap="tab10",
    linewidth=0.5,
    edgecolor="black",
    legend=True,
    ax=ax
)

ax.set_title("European Country Clusters (Features 1–10)", fontsize=16)
ax.axis("off")
plt.show()

clusters = country_features["cluster"].unique()
clusters = sorted(clusters)

for c in clusters:
    print(f"\n=== Cluster {c} ===")
    countries = country_features[country_features["cluster"] == c].index.tolist()
    print(", ".join(countries))