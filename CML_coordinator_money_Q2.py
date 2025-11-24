import pandas as pd
import numpy as np

df = pd.read_excel("HES cu NUTS 3.xlsx")

# Curățăm numele coloanelor
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Ne asigurăm că avem eccontribution, totalcost numeric
for col in ["eccontribution", "totalcost"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# verificăm ce nume are coloana de NUTS3
print("Coloane disponibile:", df.columns.tolist())

# presupunem că este 'nutscode' (din exemplul anterior cu NUTS)
# dacă se cheamă altfel, modifică aici:
nuts_col = "nutscode"  # sau 'nuts_code', 'nuts3' etc., după caz

# =========================
# 2. Outcome: is_coordinator (la nivel organizație-proiect)
# =========================

df["is_coordinator"] = (df["role"] == "coordinator").astype(int)

# =========================
# 3. Mărimea consorțiului pentru fiecare proiect
# =========================

df["consortium_size"] = (
    df.groupby("projectid")["organisationid"]
      .transform("nunique")
)

# =========================
# 4. Construim un indicator de "region R&I leadership"
#    la nivel de NUTS3 (fără auto-reflecție)
# =========================

# total coordonatori și total observații pe regiune
region_sum_coord = df.groupby(nuts_col)["is_coordinator"].transform("sum")
region_count = df.groupby(nuts_col)["is_coordinator"].transform("count")

# share de coordonatori în regiune, excluzând observația curentă:
df["region_share_coord_excl"] = (
    (region_sum_coord - df["is_coordinator"]) /
    (region_count - 1)
)

# pentru regiunile cu un singur participant (division by zero) → NaN, le punem 0
df["region_share_coord_excl"] = df["region_share_coord_excl"].replace([np.inf, -np.inf], np.nan)
df["region_share_coord_excl"] = df["region_share_coord_excl"].fillna(0)

print("#===== most important columns")
print(df[[nuts_col, "is_coordinator", "region_share_coord_excl", "country", "activitytype", "sme", "consortium_size", "totalcost"]].head())

# =========================
# 5. Pregătim datele pentru DoWhy
# =========================

data3 = df.copy()

data3["country"] = data3["country"].astype(str)
data3["activitytype"] = data3["activitytype"].astype(str)
data3["sme"] = data3["sme"].astype(int)  # True/False → 1/0

# ne asigurăm că numeric
for col in ["is_coordinator", "region_share_coord_excl", "consortium_size", "totalcost"]:
    data3[col] = pd.to_numeric(data3[col], errors="coerce").fillna(0)

# =========================
# 6. Modelul cauzal (Întrebarea 3)
# =========================

from dowhy import CausalModel

# Tratament: region_share_coord_excl  (cât de "coordonatoare" e regiunea ta)
# Outcome: is_coordinator             (dacă TU ești coordonator în proiect)
# Confounders: țară, tip organizație, SME, mărime consorțiu, cost proiect

model3 = CausalModel(
    data=data3,
    treatment="region_share_coord_excl",
    outcome="is_coordinator",
    common_causes=["country", "activitytype", "sme", "consortium_size", "totalcost"],
)

identified_effect3 = model3.identify_effect()
print("\n=== Identified effect (Q2) ===")
print(identified_effect3)

# =========================
# 7. Estimare (linear probability model)
# =========================

estimate3 = model3.estimate_effect(
    identified_effect3,
    method_name="backdoor.linear_regression",
)

print("\n=== Causal estimate (Q2) ===")
print("Effect of region_share_coord_excl on is_coordinator:", estimate3.value)

# =========================
# 8. Refutation tests
# =========================

print("\n### Refuter 1 (Q2): Placebo Treatment ###")
refutation3_placebo = model3.refute_estimate(
    identified_effect3,
    estimate3,
    method_name="placebo_treatment_refuter",
)
print(refutation3_placebo)

print("\n### Refuter 2 (Q2): Data Subset Refuter ###")
refutation3_subset = model3.refute_estimate(
    identified_effect3,
    estimate3,
    method_name="data_subset_refuter",
)
print(refutation3_subset)


#=========== output

# Coloane disponibile: ['projectid', 'projectacronym', 'organisationid', 'vatnumber', 'name', 'shortname', 'sme', 'activitytype', 'street', 'postcode', 'city', 'country', 'nutscode', 'geolocation', 'organizationurl', 'contactform', 'contentupdatedate', 'rcn', 'order', 'role', 'eccontribution', 'neteccontribution', 'totalcost', 'endofparticipation']
# #===== most important columns
#   nutscode  is_coordinator  region_share_coord_excl country activitytype    sme  consortium_size  totalcost
# 0    DE600               1                 0.301370      DE          HES  False                1  1280175.0
# 1    ITH31               0                 0.408163      IT          HES  False                4        NaN
# 2    RO113               0                 0.160494      RO          HES  False                4   261250.0
# 3    NL414               0                 0.317391      NL          HES  False                4        NaN
# 4    EE001               0                 0.224000      EE          HES  False                4        NaN

# === Identified effect (Q2) ===
# Estimand type: EstimandType.NONPARAMETRIC_ATE

# ### Estimand : 1
# Estimand name: backdoor
# Estimand expression:
#             d                                                                                   
# ──────────────────────────(E[is_coordinator|activitytype,totalcost,sme,consortium_size,country])
# d[region_share_coord_excl]                                                                      
# Estimand assumption 1, Unconfoundedness: If U→{region_share_coord_excl} and U→is_coordinator then P(is_coordinator|region_share_coord_excl,activitytype,totalcost,sme,consortium_size,country,U) = P(is_coordinator|region_share_coord_excl,activitytype,totalcost,sme,consortium_size,country)

# ### Estimand : 2
# Estimand name: iv
# No such variable(s) found!

# ### Estimand : 3
# Estimand name: frontdoor
# No such variable(s) found!

# ### Estimand : 4
# Estimand name: general_adjustment
# Estimand expression:
#             d                                                                                   
# ──────────────────────────(E[is_coordinator|activitytype,totalcost,sme,consortium_size,country])
# d[region_share_coord_excl]                                                                      
# Estimand assumption 1, Unconfoundedness: If U→{region_share_coord_excl} and U→is_coordinator then P(is_coordinator|region_share_coord_excl,activitytype,totalcost,sme,consortium_size,country,U) = P(is_coordinator|region_share_coord_excl,activitytype,totalcost,sme,consortium_size,country)


# === Causal estimate (Q2) ===
# Effect of region_share_coord_excl on is_coordinator: 0.6335320420060004

# ### Refuter 1 (Q2): Placebo Treatment ###
# Refute: Use a Placebo Treatment
# Estimated effect:0.6335320420060004
# New effect:5.53270096403935e-10
# p value:0.0


# ### Refuter 2 (Q2): Data Subset Refuter ###
# Refute: Use a subset of data
# Estimated effect:0.6335320420060004
# New effect:0.6343493478040785
# p value:0.98


# ==== o prima interpretare: 
# Regiunile NUTS3 cu o tradiție puternică de coordonare în proiectele UE (adică un procent mare de organizații locale 
# care coordonează proiecte) cresc semnificativ probabilitatea ca o organizație individuală din aceeași regiune 
# să fie coordonator într-un proiect. Efectul este substanțial: o creștere de 10 puncte procentuale în 
# leadership-ul regional determină o creștere medie de aproximativ 6 puncte procentuale a probabilității 
# ca un partener să fie coordonator. Testele placebo și de robustete arată că estimarea este stabilă și 
# nu este un artefact statistic, indicând un efect cauzal puternic al ecosistemului regional asupra 
# rolurilor de coordonare în proiectele UE.