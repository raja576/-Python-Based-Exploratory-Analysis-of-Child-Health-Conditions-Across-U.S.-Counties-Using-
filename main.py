import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\maraj\\Downloads\\Health_conditions_among_children_under_age_18__by_selected_characteristics__United_States.csv")
df.replace(["*", ".", "NA", "N/A", "..."], pd.NA, inplace=True)

df['ESTIMATE'] = pd.to_numeric(df['ESTIMATE'], errors='coerce')
df['SE'] = pd.to_numeric(df['SE'], errors='coerce')

df['ESTIMATE'] = df['ESTIMATE'].fillna(df.groupby('PANEL')['ESTIMATE'].transform('mean'))
df['SE'] = df['SE'].fillna(df.groupby('PANEL')['SE'].transform('mean'))
df['FLAG'] = df['FLAG'].fillna('Unknown')

df['Prevalence_Category'] = df['ESTIMATE'].apply(lambda x: 'High' if x > df['ESTIMATE'].mean() else 'Low')

avg_estimates = df.groupby('PANEL')['ESTIMATE'].mean().sort_values(ascending=False)
top10_conditions = avg_estimates.head(10)
top5_conditions = top10_conditions.head(5).index.tolist()


plt.figure(figsize=(10, 6))
sns.barplot(x=top10_conditions.values, y=top10_conditions.index, palette='viridis')
plt.xlabel("Average Estimated Prevalence (%)")
plt.title("Top 10 Health Conditions Among Children")
plt.tight_layout()
plt.show()

trend_df = df[df['PANEL'].isin(top5_conditions)].groupby(['YEAR', 'PANEL'])['ESTIMATE'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=trend_df, x='YEAR', y='ESTIMATE', hue='PANEL', marker="o")
plt.title("Trends in Top 5 Conditions Over Time")
plt.xticks(rotation=45)
plt.ylabel("Estimated Percentage (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[df['PANEL'].isin(top10_conditions.index)], x='PANEL', y='ESTIMATE', palette='Set3')
plt.xticks(rotation=45)
plt.title("Distribution of Estimates (Top 10 Conditions)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(data=df[df['PANEL'].isin(top5_conditions)], x='AGE', y='ESTIMATE', hue='PANEL')
plt.title("Estimates by Age Group (Top 5 Conditions)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.kdeplot(df['ESTIMATE'], fill=True, color='skyblue')
plt.title("Density of Health Condition Estimates")
plt.xlabel("Estimate (%)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
df['Prevalence_Category'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
plt.title("Prevalence Category Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='YEAR', palette='coolwarm')
plt.title("Data Records Per Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
age_summary = df[df['PANEL'].isin(top5_conditions)].groupby(['AGE', 'PANEL'])['ESTIMATE'].mean().reset_index()
sns.barplot(data=age_summary, x='AGE', y='ESTIMATE', hue='PANEL')
plt.title("Average Estimate by Age Group (Top 5 Conditions)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

numeric_cols = ['ESTIMATE', 'SE', 'YEAR']
pairplot_data_clean = df[df['PANEL'].isin(top5_conditions)][numeric_cols + ['PANEL']].dropna()

pairplot_data_clean['YEAR'] = pd.to_numeric(pairplot_data_clean['YEAR'], errors='coerce')

sns.set(style="whitegrid")

pair = sns.pairplot(
    pairplot_data_clean,
    hue='PANEL',
    palette='tab10',
    corner=True,
    height=2.5,
    plot_kws={'alpha': 0.7, 's': 40}
)

pair.fig.suptitle("Cleaned Pairwise Relationships (Top 5 Health Conditions)", y=1.02, fontsize=14)

pair._legend.set_title("Condition")
pair._legend.set_bbox_to_anchor((1.05, 0.5))
plt.tight_layout(rect=[0, 0, 0.9, 1])  

df.to_csv("C:\\Users\\maraj\\Downloads\\Modified_Health_Conditions_Data.csv", index=False)
