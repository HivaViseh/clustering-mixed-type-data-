import pandas as pd
import scipy.stats as stats

df = pd.read_csv("data_for_risk_calculation_2.csv")
print(df.columns)
df["Cluster"] = df["Cluster"].apply(lambda x: 0 if x == 2 else 1)



contingency_table = pd.crosstab(
    index=[df['GA_permitted_status'], df['BCADescription'], df['Year_Category'], df['Source of CO']],
    columns=df['Cluster']
)
print(contingency_table)
contingency_table.to_csv("contingency_table.csv", index=False)
# Chi-square test
chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
print(f'Chi-square: {chi2}, p-value: {p}')

# Interpret the p-value
if p < 0.05:
    print("There is a significant association between these factors and fatalities.")
else:
    print("There is no significant association between these factors and fatalities.")



Incident_Info = df.groupby(['BCADescription', 'Year_Category', 'Source of CO', 'GA_permitted_status']).size().reset_index(name='Incident_Info')

# Calculate the number of buildings with fatalities in each group
Fatality_Incidence = df[df['Cluster'] == 1].groupby(['BCADescription', 'Year_Category', 'Source of CO', 'GA_permitted_status']).size().reset_index(name='Fatality_Incidence')

# Merge the two DataFrames
risk_df = pd.merge(Incident_Info, Fatality_Incidence, on=['BCADescription', 'Year_Category', 'Source of CO', 'GA_permitted_status'], how='left')
risk_df['Fatality_Incidence'] = risk_df['Fatality_Incidence'].fillna(0)

# Calculate the incidence rate of fatalities for each group
risk_df['Fatality Risk'] = risk_df['Fatality_Incidence'] / risk_df['Incident_Info']

overall_incidence_rate = df['Cluster'].mean()

# Calculate the risk ratio for each group
risk_df['FatalityRisk_Ratio'] = risk_df['Fatality Risk'] / overall_incidence_rate


print(risk_df)
risk_df.to_csv("IncidentRate_2.csv", index=False)
