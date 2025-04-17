import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from runs.csv
df = pd.read_csv("analysis/runs.csv")

print("Available columns:", df.columns.tolist())

df = df[df['Initial_Judge_Overall_Score'].notnull() & df['Followup_Judge_Overall_Score'].notnull()]

df['Initial_Judge_Overall_Score'] = pd.to_numeric(df['Initial_Judge_Overall_Score'], errors='coerce')
df['Followup_Judge_Overall_Score'] = pd.to_numeric(df['Followup_Judge_Overall_Score'], errors='coerce')
df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')

df = df.dropna(subset=['Initial_Judge_Overall_Score', 'Followup_Judge_Overall_Score', 'temperature'])

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df,
                x='Initial_Judge_Overall_Score',
                y='Followup_Judge_Overall_Score',
                hue='temperature',
                palette='viridis', s=100)
plt.plot([0, df[['Initial_Judge_Overall_Score','Followup_Judge_Overall_Score']].max().max()],
         [0, df[['Initial_Judge_Overall_Score','Followup_Judge_Overall_Score']].max().max()],
         'k--', linewidth=1)
plt.title("Initial vs Follow-up Judge Overall Scores by Temperature")
plt.xlabel("Initial Judge Overall Score")
plt.ylabel("Follow-up Judge Overall Score")
plt.legend(title='Temperature')
plt.tight_layout()
plt.savefig("judge_score_comparison.png")
print("Saved scatter plot as judge_score_comparison.png")

plt.figure(figsize=(10, 6))
df_melted = df.melt(id_vars='temperature',
                    value_vars=['Initial_Judge_Overall_Score', 'Followup_Judge_Overall_Score'],
                    var_name='Judge', value_name='Score')
sns.boxplot(data=df_melted, x='temperature', y='Score', hue='Judge', palette='coolwarm')
plt.title("Judge Overall Score Distribution by Temperature")
plt.xlabel("Temperature")
plt.ylabel("Overall Score")
plt.tight_layout()
plt.savefig("judge_score_distribution.png")
print("Saved boxplot as judge_score_distribution.png")

for temp in sorted(df['temperature'].unique()):
    temp_df = df[df['temperature'] == temp]
    corr = temp_df['Initial_Judge_Overall_Score'].corr(temp_df['Followup_Judge_Overall_Score'])
    print(f"Correlation between initial and follow-up scores at temperature {temp}: {corr:.2f}")

mean_scores = df.groupby('temperature')[['Initial_Judge_Overall_Score', 'Followup_Judge_Overall_Score']].mean()
print("\nMean Scores by Temperature:\n", mean_scores.round(2))
