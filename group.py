import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("data/State-FemaleLFPR.csv")
df.columns = df.columns.str.strip()

# Custom clustering based on thresholds
def label_cluster(value):
    if value > 35:
        return "High"
    elif value > 20:
        return "Medium"
    else:
        return "Low"

df['Cluster'] = df['Female LFPR (%)'].apply(label_cluster)

# Define labels with range info
legend_labels = {
    'Low': 'Low LFPR (<= 20%)',
    'Medium': 'Medium LFPR (> 20%)',
    'High': 'High LFPR (> 35%)'
}

# Color map for clusters
color_map = {'Low': 'red', 'Medium': 'orange', 'High': 'green'}

# Plot
plt.figure(figsize=(12, 6))
for cluster in ['Low', 'Medium', 'High']:
    cluster_data = df[df['Cluster'] == cluster]
    plt.bar(cluster_data['State'], cluster_data['Female LFPR (%)'],
            color=color_map[cluster], label=legend_labels[cluster])

plt.xticks(rotation=90)
plt.ylabel("Female LFPR (%)")
plt.title("Female Labour Force Participation Rate in INDIA (statewise)(comparison-based)")
plt.legend()
plt.tight_layout()
plt.show()

# Grouped States by Cluster
print("\nFemale Labour Force Participation Rate:")
for cluster in ['Low', 'Medium', 'High']:
    states = df[df['Cluster'] == cluster]['State'].tolist()
    print(f"\n-----------{cluster} Labor rate States-------------")
    print(", ".join(states))
    #print("-" * 60)
