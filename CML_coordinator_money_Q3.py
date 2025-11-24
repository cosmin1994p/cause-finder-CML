import pandas as pd
import numpy as np

# =========================
# 1. Clean + extract year
# =========================

df = pd.read_excel("HES cu NUTS 3.xlsx")

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Convertim data
df["contentupdatedate"] = pd.to_datetime(df["contentupdatedate"], errors="coerce")
df["year"] = df["contentupdatedate"].dt.year

print("Ani disponibili în dataset:", df["year"].unique())

# =========================
# 2. Construim tabel country-year
# =========================

# număr proiecte HES per țară în anul t
hes_counts = (
    df[df["activitytype"] == "HES"]
    .groupby(["country", "year"])["projectid"]
    .nunique()
    .rename("num_HES_projects_t")
)

# număr total proiecte per țară în anul t
total_counts = (
    df.groupby(["country", "year"])["projectid"]
    .nunique()
    .rename("num_total_projects_t")
)

# funding mediu per țară în anul t
funding_mean = (
    df.groupby(["country", "year"])["eccontribution"]
    .mean()
    .rename("avg_funding_t")
)

# un tabel combinat
country_year = (
    pd.concat([hes_counts, total_counts, funding_mean], axis=1)
    .reset_index()
    .fillna(0)
)

# shiftăm outcome cu un an în viitor
country_year["num_total_projects_next"] = (
    country_year
    .groupby("country")["num_total_projects_t"]
    .shift(-1)
)

# scoatem ultimele rânduri fără t+1
country_year = country_year.dropna(subset=["num_total_projects_next"])

print("#===== country_year")
print(country_year.head())

# =========================
# 3. Modelul cauzal (Q7)
# =========================

from dowhy import CausalModel

model7 = CausalModel(
    data=country_year,
    treatment="num_HES_projects_t",
    outcome="num_total_projects_next",
    common_causes=["country", "year", "avg_funding_t"]
)

identified_effect7 = model7.identify_effect()
print("\n=== Identified effect (Q7) ===")
print(identified_effect7)

# =========================
# 4. Estimare
# =========================

estimate7 = model7.estimate_effect(
    identified_effect7,
    method_name="backdoor.linear_regression"
)

print("\n=== Causal estimate (Q7) ===")
print("Effect:", estimate7.value)

# =========================
# 5. Refutation tests
# =========================

print("\n### Refuter 1 (Q7): Placebo Treatment ###")
ref_placebo7 = model7.refute_estimate(
    identified_effect7,
    estimate7,
    method_name="placebo_treatment_refuter"
)
print(ref_placebo7)

print("\n### Refuter 2 (Q7): Data Subset Refuter ###")
ref_subset7 = model7.refute_estimate(
    identified_effect7,
    estimate7,
    method_name="data_subset_refuter"
)
print(ref_subset7)




#===========# output 

#===== country_year
#   country  year  num_HES_projects_t  num_total_projects_t  avg_funding_t  num_total_projects_next
# 0      AT  2022                  97                    97  704433.517664                    116.0
# 1      AT  2023                 116                   116  801283.812397                    134.0
# 2      AT  2024                 134                   134  833809.598071                    657.0
# 4      BE  2022                 109                   109  747516.006032                    150.0
# 5      BE  2023                 150                   150  841966.499026                    182.0

# === Identified effect (Q7) ===
# Estimand type: EstimandType.NONPARAMETRIC_ATE

# ### Estimand : 1
# Estimand name: backdoor
# Estimand expression:
#           d                                                                
# ─────────────────────(E[num_total_projects_⟨next|year,country,⟩_funding_t])
# d[num_HES_projects_t]                                                      
# Estimand assumption 1, Unconfoundedness: If U→{num_HES_projects_t} and U→num_total_projects_next then P(num_total_projects_next|num_HES_projects_t,year,country,avg_funding_t,U) = P(num_total_projects_next|num_HES_projects_t,year,country,avg_funding_t)

# ### Estimand : 2
# Estimand name: iv
# No such variable(s) found!

# ### Estimand : 3
# Estimand name: frontdoor
# No such variable(s) found!

# ### Estimand : 4
# Estimand name: general_adjustment
# Estimand expression:
#           d                                                                
# ─────────────────────(E[num_total_projects_⟨next|year,country,⟩_funding_t])
# d[num_HES_projects_t]                                                      
# Estimand assumption 1, Unconfoundedness: If U→{num_HES_projects_t} and U→num_total_projects_next then P(num_total_projects_next|num_HES_projects_t,year,country,avg_funding_t,U) = P(num_total_projects_next|num_HES_projects_t,year,country,avg_funding_t)


# === Causal estimate (Q7) ===
# Effect: 9.016624270763714

# ### Refuter 1 (Q7): Placebo Treatment ###
# Refute: Use a Placebo Treatment
# Estimated effect:9.016624270763714
# New effect:-0.033917237853553814
# p value:0.84


# ### Refuter 2 (Q7): Data Subset Refuter ###
# Refute: Use a subset of data
# Estimated effect:9.016624270763714
# New effect:8.920333152293862
# p value:0.8

#========== intepretare: 
# 
# Analiza cauzală arată că implicarea universităților în proiecte europene generează efecte de tip spillover la nivel național. 
# Pentru fiecare proiect HES suplimentar într-un anumit an, numărul total de proiecte din țara respectivă crește 
# cu aproximativ 9 proiecte în anul următor. Acest efect este substanțial și sugerează că activitatea universităților
#  nu doar atrage finanțare directă, ci consolidează întreaga capacitate națională de a participa la programe europene de 
# cercetare și inovare. Testele de robustețe (placebo și subset) confirmă stabilitatea estimării, deși numărul limitat de 
# ani cu date valide indică faptul că rezultatul trebuie interpretat cu atenție. Per ansamblu, modelul susține puternic ipoteza 
# că universitățile acționează ca „motoare” ale ecosistemului R&I național. 