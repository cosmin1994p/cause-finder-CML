import pandas as pd

df = pd.read_excel("Filtered universities Cordis.xlsx")

# Clean column names (simplify life)
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
print("DDDDDDDFFFFF")
print(df)

print(df.head())
print(df.info())

#=== funding_by_university 

funding_by_uni = (
    df[df['university'].str.lower()=='yes']
    .groupby('name')['eccontribution']
    .sum()
    .sort_values(ascending=False)
)

print("#=== funding_by_university")
print(funding_by_uni.head(10))

#===== projects by university 
projects_by_uni = (
    df[df['university'].str.lower()=='yes']
    .groupby('name')['projectid']
    .nunique()
    .sort_values(ascending=False)
)

print("#===== projects by university ")
print(projects_by_uni.head(10))

for col in ['eccontribution', 'totalcost']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#===== university grouped
unis = df[df['university'].str.lower()=='yes'].copy()

uni_table = (
    unis.groupby(['name', 'country'], as_index=False)
        .agg(
            num_projects = ('projectid', 'nunique'),
            total_funding = ('eccontribution', 'sum'),
            avg_project_cost = ('totalcost', 'mean')
        )
)

# optional: remove infinities or NaN
uni_table = uni_table.fillna(0)
print("#===== uni table")
print(uni_table)



# Does the number of projects cause universities to receive more total funding?
# X = num_projects
# Y = total_funding
# Z (confounders) = country

#DAG - (directed acyclic graph)
# country → num_projects → total_funding
# country → total_funding


data = uni_table.copy()

# clean country as string
data['country'] = data['country'].astype(str)

# ensure numeric
data['num_projects'] = pd.to_numeric(data['num_projects'])
data['total_funding'] = pd.to_numeric(data['total_funding'])

print(data.head())

# causal model: 
# country → num_projects → total_funding
# country → total_funding

from dowhy import CausalModel

model = CausalModel(
    data=data,
    treatment="num_projects",
    outcome="total_funding",
    common_causes=["country"]
)
model.view_model()

identified_effect = model.identify_effect()
print("identified effect")
print(identified_effect)


#estimate with linear regression 
estimate = model.estimate_effect(
    identified_effect,
    method_name="backdoor.linear_regression"
)

print("Causal Estimate:", estimate.value)

##refutation tests - sanity tests (estimarea cauzala este sau nu robusta)
















# Does country influence the relationship between projects and funding?
# (heterogeneity)

# Does experience (project count) causally increase average project cost?







