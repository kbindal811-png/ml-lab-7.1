import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


data = pd.read_csv("customer_segmentation.csv")

print("First 5 Records:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Records:")
print(data.duplicated().sum())


data = data.drop_duplicates()
data = data.dropna()


features = [
    "Age",
    "AnnualIncome",
    "SpendingScore",
    "PurchaseFrequency"
]

X = data[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertia = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.grid()
plt.show()


silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different Numbers of Clusters")

plt.grid()
plt.show()


optimal_k = 3

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

data["Cluster"] = kmeans.fit_predict(X_scaled)


final_silhouette_score = silhouette_score(
    X_scaled,
    data["Cluster"]
)

print("\nOptimal Number of Clusters:", optimal_k)

print(
    "Silhouette Score:",
    round(final_silhouette_score, 3)
)


print("\nCluster Distribution:")

print(
    data["Cluster"]
    .value_counts()
    .sort_index()
)


cluster_summary = data.groupby("Cluster")[features].mean()

print("\nCluster Summary:")

print(cluster_summary)


pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


pca_data = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_data["Cluster"] = data["Cluster"].values


plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=pca_data,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set1",
    s=100
)

plt.title("Customer Segmentation using K-Means")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid()
plt.show()


plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=data,
    x="AnnualIncome",
    y="SpendingScore",
    hue="Cluster",
    palette="Set1",
    s=100
)

plt.title("Customer Clusters by Income and Spending Score")

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.grid()
plt.show()


data.to_csv(
    "customer_segmentation_result.csv",
    index=False
)

print(
    "\nClustered dataset saved as "
    "customer_segmentation_result.csv"
)