import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------------------------------
# 1. LOAD DATA (efficient for big XLSX)
# ----------------------------------------------------------
df = pd.read_excel("HES cu NUTS 3.xlsx", engine="openpyxl")

# ----------------------------------------------------------
# 2. COUNT UNIQUE PROJECTS PER COUNTRY
# ----------------------------------------------------------
# Deduplicate by projectID per country
df_unique = df[["country", "projectID"]].drop_duplicates()

# Count unique projects per country
projects_per_country = (
    df_unique.groupby("country")
    .size()
    .reset_index(name="num_projects")
)

print(projects_per_country)

# ----------------------------------------------------------
# 3. LOAD GEODATA FOR EUROPE
# ----------------------------------------------------------
# Natural Earth country boundaries (built into GeoPandas)
# Load world map from Natural Earth (GeoPandas ≥ 1.0 compatible)
world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)

# Filter Europe
europe = world[world["CONTINENT"] == "Europe"]

# Merge with your country stats
europe = europe.merge(
    projects_per_country,
    left_on="ISO_A2",
    right_on="country",
    how="left"
)

# Missing = 0
europe["num_projects"].fillna(0, inplace=True)


# ----------------------------------------------------------
# 4. CHOROPLETH MAP OF PROJECT COUNTS
# ----------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

europe.plot(
    column="num_projects",
    cmap="Blues",
    linewidth=0.8,
    edgecolor="black",
    legend=True,
    ax=ax,
)

ax.set_title("Number of Unique Projects per Country", fontsize=16)
ax.axis("off")
plt.show()

# ----------------------------------------------------------
# 5. BUILD FEATURE MATRIX FOR CLUSTERING
# ----------------------------------------------------------
# You can add more features later — for now only project count
X = europe[["num_projects"]].values

# Normalize (very important for ML)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------
# 6. KMEANS CLUSTERING (choose number of clusters)
# ----------------------------------------------------------
k = 4  # you can try 3, 4, 5 and compare
kmeans = KMeans(n_clusters=k, random_state=0)
europe["cluster"] = kmeans.fit_predict(X_scaled)

# ----------------------------------------------------------
# 7. MAP OF CLUSTERS
# ----------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

europe.plot(
    column="cluster",
    categorical=True,
    legend=True,
    cmap="tab10",
    linewidth=0.8,
    edgecolor="black",
    ax=ax,
)

ax.set_title(f"KMeans Clusters (k = {k})", fontsize=16)
ax.axis("off")
plt.show()
