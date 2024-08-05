import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OrdinalEncoder
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.cm as cm
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE



col = ['GA_permitted_status', 'BCADescription','Year_Category','Source of CO', 'Fatalities']


df = pd.read_csv("rawdata.csv", usecols=col)
original_df = df.copy()



numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


nominal_cols = df.select_dtypes(include='object').columns
nominal_indices = [df.columns.get_loc(col) for col in nominal_cols]

one_hot_encoded_df = pd.get_dummies(df, columns=nominal_cols)


clusters_range = range(2, 10)
gamma_range = np.linspace(0.1, 1, 10) # gamma values between 0.1 and 1

# Placeholder variables
best_score = float('inf')
best_clusters = None
best_gamma = None

for n_clusters in clusters_range:
    for gamma in gamma_range:
        kproto = KPrototypes(n_clusters=n_clusters, gamma=gamma, init='Cao', random_state=42)
        clusters = kproto.fit_predict(one_hot_encoded_df, categorical=nominal_indices)
        score = davies_bouldin_score(one_hot_encoded_df, clusters)

        if len(set(clusters)) > 1:  # Check if there is more than one cluster
            score = davies_bouldin_score(one_hot_encoded_df, clusters)
            # Check if this configuration beats the best score
            if score < best_score:
                best_score = score
                best_clusters = n_clusters
                best_gamma = gamma

print(f"Best Davies-Bouldin Index: {best_score}")
print(f"Optimal number of clusters: {best_clusters}")
print(f"Optimal gamma value: {best_gamma}")

kproto = KPrototypes(n_clusters=best_clusters, gamma=best_gamma, init='Cao', random_state=42)
clusters = kproto.fit_predict(one_hot_encoded_df, categorical=nominal_indices)


clusters = clusters +1
original_df['Cluster'] = clusters
original_df.to_csv("data_for_risk_calculation_2.csv", index=False)
column_trans = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_cols.tolist()),
        ('cat', OrdinalEncoder(), nominal_cols.tolist())
    ],
    remainder='drop'
)


clf = RandomForestClassifier(random_state=42, class_weight='balanced')
pipeline = Pipeline([('prep', column_trans), ('clf', clf)])


X = df
print(X)
y = clusters

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

feature_importances = pipeline.named_steps['clf'].feature_importances_

# Create a DataFrame to store feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})


feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='darkorchid', height=0.5)
plt.xlabel('Importance', fontsize=14, fontweight='bold')
plt.ylabel('Feature', fontsize=14, fontweight='bold')
plt.title('Feature Importance Analysis', fontsize=14, fontweight='bold')
plt.grid(True)
plt.show()


def barcharts(original_df, firstvar):
    original_df["Cluster"] = clusters
    original_df = original_df[[firstvar, "Cluster"]]
  

    grouped = original_df.groupby([firstvar, "Cluster"]).size().reset_index(name='count')

    total_per_cluster = original_df.groupby("Cluster").size().reset_index(name='total')
    merged = pd.merge(grouped, total_per_cluster, on="Cluster")
 
    merged['percentage'] = merged['count'] / merged['total'] * 100
    pivot = merged.pivot(index="Cluster", columns=firstvar, values='percentage')

    num_colors = pivot.shape[1]
    colors = cm.get_cmap('tab20', num_colors)  # Use 'tab20' for a larger set of distinct colors

    ax = pivot.plot(kind='bar', figsize=(10, 6), color=[colors(i) for i in range(num_colors)])
    plt.xlabel("Cluster", fontsize=20)
    plt.ylabel('(%)', fontsize=20)
    counts = total_per_cluster.set_index("Cluster")['total']
    new_xticks = [f"{cluster}\n(n={counts[cluster]})" for cluster in pivot.index]
    ax.set_xticklabels(new_xticks, fontsize=15, rotation='horizontal')
    plt.yticks(fontsize=15)
    plt.title(f'Percentage of Incident by {firstvar} in Each Cluster', fontsize=20)
    plt.legend(title=firstvar, fontsize=15)
    plt.grid()
    plt.show()



for i in range(len(col)):
  barcharts(original_df, col[i])

