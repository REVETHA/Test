import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load and clean column names
df = pd.read_csv(r'dataset\State-FemaleLFPR.csv')
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

# Run KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Female_LFPR_%']])

# Get min/max LFPR range for each cluster
cluster_ranges = df.groupby('Cluster')['Female_LFPR_%'].agg(['min', 'max'])

# Sort clusters by LFPR average (to assign LOW/MEDIUM/HIGH meaningfully)
cluster_order = df.groupby('Cluster')['Female_LFPR_%'].mean().sort_values().index.tolist()

# Create legend labels
cluster_labels = {}
labels = ['LOW', 'MEDIUM', 'HIGH']
for idx, cluster_id in enumerate(cluster_order):
    row = cluster_ranges.loc[cluster_id]
    label = f"{labels[idx]} ({row['min']}% - {row['max']}%)"
    cluster_labels[cluster_id] = label

# Add readable labels
df['Cluster_Label'] = df['Cluster'].map(cluster_labels)

#  Plot
plt.figure(figsize=(12, 6))
sns.barplot(
    x='State', y='Female_LFPR_%', hue='Cluster_Label',  # 👈 FIXED HERE
    data=df, palette='Set1', dodge=False
)
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('Female Labour Force Participation Rate (%)')
plt.title('K-means Clustering of Indian States by Female LFPR')
plt.legend(title='Female Workforce Participation')
plt.tight_layout()
plt.show()

#  Print grouped states
print("\nGrouped States based on Female Workforce:")
for label in df['Cluster_Label'].unique():
    states = df[df['Cluster_Label'] == label]['State'].tolist()
    print(f"\n{label}:\n" + ", ".join(states))
