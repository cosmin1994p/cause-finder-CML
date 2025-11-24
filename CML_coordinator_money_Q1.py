import pandas as pd
import numpy as np

# =========================
# 1. Citire + curățare date
# =========================

df = pd.read_excel("HES cu NUTS 3.xlsx")

# Curățăm numele coloanelor (ca în exemplul tău)
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Ne asigurăm că eccontribution și totalcost sunt numerice
for col in ["eccontribution", "totalcost"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 2. Tabel la nivel de universitate (uni_table)
# =========================

unis = df.copy()

uni_table = (
    unis.groupby(["name", "country"], as_index=False)
        .agg(
            num_projects=("projectid", "nunique"),
            total_funding=("eccontribution", "sum"),
            avg_project_cost=("totalcost", "mean")
        )
)

uni_table = uni_table.fillna(0)

print("#===== uni_table (primele rânduri)")
print(uni_table.head())

# =========================
# 3. Calculăm câte proiecte ca coordinator are fiecare universitate
# =========================

coord_counts = (
    unis[unis["role"] == "coordinator"]
    .groupby(["name", "country"])["projectid"]
    .nunique()
    .rename("num_coord_projects")
)

# Atașăm la uni_table
uni_table = uni_table.merge(
    coord_counts,
    on=["name", "country"],
    how="left"
)

uni_table["num_coord_projects"] = uni_table["num_coord_projects"].fillna(0)

# share_coordinator = proporția de proiecte în care universitatea este coordonator
uni_table["share_coordinator"] = (
    uni_table["num_coord_projects"] / uni_table["num_projects"].replace(0, np.nan)
)

uni_table["share_coordinator"] = uni_table["share_coordinator"].fillna(0)

print("#===== uni_table cu share_coordinator")
print(uni_table.head())

# =========================
# 4. Pregătim datele pentru DoWhy
# =========================

data = uni_table.copy()

# ne asigurăm că tipurile sunt ok
data["country"] = data["country"].astype(str)
data["num_projects"] = pd.to_numeric(data["num_projects"])
data["total_funding"] = pd.to_numeric(data["total_funding"])
data["avg_project_cost"] = pd.to_numeric(data["avg_project_cost"])
data["share_coordinator"] = pd.to_numeric(data["share_coordinator"])

print("#===== date pentru modelul cauzal")
print(data[["name", "country", "num_projects", "total_funding", "avg_project_cost", "share_coordinator"]].head())

# =========================
# 5. Modelul cauzal DoWhy
# =========================

from dowhy import CausalModel

# DAG (în cuvinte):
# country → share_coordinator
# country → total_funding
# avg_project_cost → share_coordinator
# avg_project_cost → total_funding
# share_coordinator → total_funding  (efectul care ne interesează)

model = CausalModel(
    data=data,
    treatment="share_coordinator",
    outcome="total_funding",
    common_causes=["country", "avg_project_cost"],
)

# dacă ai pydot + graphviz instalate, asta îți desenează DAG-ul
# model.view_model(layout="dot")

identified_effect = model.identify_effect()
print("\n=== Identified effect (Q2) ===")
print(identified_effect)

# =========================
# 6. Estimare (linear regression pe backdoor)
# =========================

estimate = model.estimate_effect(
    identified_effect,
    method_name="backdoor.linear_regression"
)

print("\n=== Causal estimate (Q2) ===")
print("Effect of share_coordinator on total_funding:", estimate.value)

# =========================
# 7. Refutation tests (sanity checks)
# =========================

print("\n### Refuter 1: Placebo Treatment ###")
refutation_placebo = model.refute_estimate(
    identified_effect,
    estimate,
    method_name="placebo_treatment_refuter"
)
print(refutation_placebo)

print("\n### Refuter 2: Data Subset Refuter ###")
refutation_subset = model.refute_estimate(
    identified_effect,
    estimate,
    method_name="data_subset_refuter"
)
print(refutation_subset)

print("\n### Refuter 3: Add Unobserved Common Cause ###")
refutation_unobserved = model.refute_estimate(
    identified_effect,
    estimate,
    method_name="add_unobserved_common_cause"
)
print(refutation_unobserved)



#================ output-ul acestui cod: 
# === Identified effect (Q2) ===
# Estimand type: EstimandType.NONPARAMETRIC_ATE

# ### Estimand : 1
# Estimand name: backdoor
# Estimand expression:
#          d                                                    
# ────────────────────(E[total_⟨funding|country,⟩_project_cost])
# d[share_coordinator]                                          
# Estimand assumption 1, Unconfoundedness: If U→{share_coordinator} and U→total_funding then P(total_funding|share_coordinator,country,avg_project_cost,U) = P(total_funding|share_coordinator,country,avg_project_cost)

# ### Estimand : 2
# Estimand name: iv
# No such variable(s) found!

# ### Estimand : 3
# Estimand name: frontdoor
# No such variable(s) found!

# ### Estimand : 4
# Estimand name: general_adjustment
# Estimand expression:
#          d                                                    
# ────────────────────(E[total_⟨funding|⟩_project_cost,country])
# d[share_coordinator]                                          
# Estimand assumption 1, Unconfoundedness: If U→{share_coordinator} and U→total_funding then P(total_funding|share_coordinator,avg_project_cost,country,U) = P(total_funding|share_coordinator,avg_project_cost,country)


# === Causal estimate (Q2) ===
# Effect of share_coordinator on total_funding: 29755138.669675514

# ### Refuter 1: Placebo Treatment ###
# Refute: Use a Placebo Treatment
# Estimated effect:29755138.669675514
# New effect:-0.0009377337992191315
# p value:0.0


# ### Refuter 2: Data Subset Refuter ###
# Refute: Use a subset of data
# Estimated effect:29755138.669675514
# New effect:29698477.40351736
# p value:0.86


#INTERPRETARE
# Effect of share_coordinator on total_funding: 29,755,138 € adica: 
# Dacă o universitate își crește „share_coordinator” cu 1 unitate (de la 0 la 1), atunci finanțarea totală crește
# în medie cu aproximativ 29,7 milioane €, după ce controlăm pentru:
# țară (country)
# costul mediu al proiectelor (avg_project_cost)

#TESTAM cauzalitatea identificata astfel: 
#Placebo refuter: 
#  Estimated effect ≈ 29.7M €
# New effect with placebo ≈ 0
# p-value: 0.0 -> placebo treatment (o variabilă random) nu mai prezice nimic ceea ce înseamnă că estimarea noastră nu e un artefact și că relația cauzală detectată este robustă
# # Data subset refuter: 
# New effect: 29.6M €
# p-value: 0.86 -> Ce înseamnă: dacă luăm doar un subset aleator din date, efectul rămâne aproape identic, p-value > 0.8 → nici o diferență semnificativă între estimări
#Universitățile care sunt mai des în rol de coordonator (share_coordinator mai mare) 
# primesc semnificativ mai multă finanțare, chiar și după ce controlăm pentru țară 
# și costurile medii ale proiectelor. Efectul este mare din punct de vedere economic: 
# o creștere cu 10% a ponderii proiectelor coordonate este asociată cu o creștere a
#  finanțării de aproape 3 milioane € per universitate. Rezultatul trece cu succes 
# atât testul placebo, cât și testul pe subseturi de date, ceea ce arată că relația 
# pare cauzală reală, nu doar corelație sau artefact statistic.