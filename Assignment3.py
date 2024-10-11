from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import warnings
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
import seaborn as sns

warnings.filterwarnings("ignore")

# 1. Retrieve and load the Olivetti faces dataset
olivetti_faces = fetch_olivetti_faces()

X = olivetti_faces.data  #  flattened 1D array format
y = olivetti_faces.target  # The target labels (person identifier)

# 2. Split the training set, a validation set, and a test set using stratified sampling to ensure that there are the same number of images per person in each set

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=51)
for train_idx, temp_idx in sss.split(X, y):
    X_train, X_temp = X[train_idx], X[temp_idx]
    y_train, y_temp = y[train_idx], y[temp_idx]


sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=51)
for val_idx, test_idx in sss_val_test.split(X_temp, y_temp):
    X_val, X_test = X_temp[val_idx], X_temp[test_idx]
    y_val, y_test = y_temp[val_idx], y_temp[test_idx]
    
    # 3. Using k-fold cross validation, train a classifier to predict which person is represented in each picture, and evaluate it on the validation set
# Initialize the classifier
clf = SVC(kernel='linear', random_state=51)

# Perform 5-fold cross-validation
scores = cross_val_score(clf, X_train, y_train, cv=5)

# Fit on full training set and evaluate on validation set
clf.fit(X_train, y_train)
val_accuracy = clf.score(X_val, y_val)

print(f'Cross-validation accuracy scores: {scores}')
print(f'Average cross-validation accuracy: {scores.mean()}')
print(f'Validation accuracy: {val_accuracy}')

## 4. Using either Agglomerative Hierarchical Clustering (AHC) or Divisive Hierarchical Clustering (DHC) and using the centroid-based clustering rule, 
# reduce the dimensionality of the set by using the following similarity measures:
n_clusters = 40
# a) Euclidean Distance
ac_euc = AgglomerativeClustering(n_clusters, linkage='ward', metric='euclidean')
ac_clusters_euc = ac_euc.fit_predict(X_train)
print(f"AHC with Euclidean distance produced {len(set(ac_clusters_euc))} clusters")

# b) Minkowski Distance
# Compute the pairwise Minkowski distance matrix
distance_matrix = squareform(pdist(X_train, metric='minkowski', p=3))

cluster_minkowski = AgglomerativeClustering(n_clusters, linkage='average', metric='precomputed')
ac_clusters_min = cluster_minkowski.fit_predict(distance_matrix)
print(f"AHC with Minkowski distance produced {len(set(ac_clusters_min))} clusters")

# c) Cosine Similarity
ac_cos = AgglomerativeClustering(n_clusters, linkage='average', metric='cosine')
ac_clusters_cos = ac_cos.fit_predict(X_train)
print(f"AHC with Cosine similarity produced {len(set(ac_clusters_cos))} clusters")

def plot_clusters(X, cluster_labels, title):
    tsne = TSNE(n_components=2, random_state=51)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=cluster_labels, palette='Set1', legend='full', alpha=0.6)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Cluster')
    plt.grid()
    plt.show()

# a) Plot for Euclidean Distance
plot_clusters(X_train, ac_clusters_euc, "Clusters using AHC with Euclidean Distance")

# b) Plot for Minkowski Distance
plot_clusters(X_train, ac_clusters_min, "Clusters using AHC with Minkowski Distance")

# c) Plot for Cosine Similarity
plot_clusters(X_train, ac_clusters_cos, "Clusters using AHC with Cosine Similarity")

# 5. Use the silhouette score approach to choose the number of clusters for 4(a), 4(b), and 4(c)

range_n_clusters = range(5, 90, 5)

# Initialize lists to store silhouette scores for each metric
silhouette_scores_euclidean = []
silhouette_scores_minkowski = []
silhouette_scores_cosine = []

# Loop over the range of cluster numbers
for n_clusters in range_n_clusters:
    # Euclidean distance clustering
    clustering_euclidean = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    cluster_labels_euclidean = clustering_euclidean.fit_predict(X_train)
    silhouette_avg_euclidean = silhouette_score(X_train, cluster_labels_euclidean)
    silhouette_scores_euclidean.append(silhouette_avg_euclidean)

    # Minkowski distance clustering (precomputed)
    clustering_minkowski = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    cluster_labels_minkowski = clustering_minkowski.fit_predict(distance_matrix)
    silhouette_avg_minkowski = silhouette_score(X_train, cluster_labels_minkowski)
    silhouette_scores_minkowski.append(silhouette_avg_minkowski)

    # Cosine similarity clustering
    clustering_cosine = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    cluster_labels_cosine = clustering_cosine.fit_predict(X_train)
    silhouette_avg_cosine = silhouette_score(X_train, cluster_labels_cosine)
    silhouette_scores_cosine.append(silhouette_avg_cosine)

# Identify the best number of clusters for each distance metric based on silhouette score
best_n_euc = range_n_clusters[np.argmax(silhouette_scores_euclidean)]
print("\nBest number of clusters based on silhouette score (Euclidean):", best_n_euc)

best_n_min = range_n_clusters[np.argmax(silhouette_scores_minkowski)]
print("\nBest number of clusters based on silhouette score (Minkowski):", best_n_min)

best_n_cos = range_n_clusters[np.argmax(silhouette_scores_cosine)]
print("\nBest number of clusters based on silhouette score (Cosine):", best_n_cos)

# Plot silhouette scores for each metric
plt.figure(figsize=(12, 10))
plt.plot(range_n_clusters, silhouette_scores_euclidean, marker='o', label='Euclidean Distance')
plt.plot(range_n_clusters, silhouette_scores_minkowski, marker='o', label='Minkowski Distance')
plt.plot(range_n_clusters, silhouette_scores_cosine, marker='o', label='Cosine Similarity')

plt.title('Silhouette Scores for Different Distance Metrics')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.xticks(list(range_n_clusters))  # Display all cluster numbers on the x-axis
plt.grid(True)
plt.legend()
plt.show()

# a) Euclidean Distance
ac_euc = AgglomerativeClustering(80, linkage='ward', metric='euclidean')
ac_clusters_euc = ac_euc.fit_predict(X_train)
print(f"AHC with Euclidean distance produced {len(set(ac_clusters_euc))} clusters")

# b) Minkowski Distance
# Compute the pairwise Minkowski distance matrix
distance_matrix = squareform(pdist(X_train, metric='minkowski', p=3))

cluster_minkowski = AgglomerativeClustering(85, linkage='average', metric='precomputed')
ac_clusters_min = cluster_minkowski.fit_predict(distance_matrix)
print(f"AHC with Minkowski distance produced {len(set(ac_clusters_min))} clusters")

# c) Cosine Similarity
ac_cos = AgglomerativeClustering(80, linkage='average', metric='cosine')
ac_clusters_cos = ac_cos.fit_predict(X_train)
print(f"AHC with Cosine similarity produced {len(set(ac_clusters_cos))} clusters")

print("Silhouette Scores for Olivetti Faces Dataset:\n")
print("Euclidean Clustering:         ", silhouette_score(X_train, ac_clusters_euc))
print("Minkowski Clustering:         ", silhouette_score(distance_matrix, ac_clusters_min))
print("Cosine Similarity Clustering: ", silhouette_score(X_train, ac_clusters_cos))

def plot_clusters(X, cluster_labels, title):
    tsne = TSNE(n_components=2, random_state=51)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=cluster_labels, palette='Set1', legend='full', alpha=0.6)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Cluster')
    plt.grid()
    plt.show()

# a) Plot for Euclidean Distance
plot_clusters(X_train, ac_clusters_euc, "Clusters using AHC with Euclidean Distance")

# b) Plot for Minkowski Distance
plot_clusters(X_train, ac_clusters_min, "Clusters using AHC with Minkowski Distance")

# c) Plot for Cosine Similarity
plot_clusters(X_train, ac_clusters_cos, "Clusters using AHC with Cosine Similarity")

# 6. Use the set from (4(a), 4(b), or 4(c)) to train a classifier as in (3) using k-fold cross validation
# Initialize the classifier with the best parameters
clf = SVC(C=0.1, gamma='scale', kernel='linear', random_state=51)

# Fit the classifier on the training data
clf.fit(X_train, cluster_labels_euclidean)

# Perform cross-validation
scores = cross_val_score(clf, X_train, cluster_labels_euclidean, cv=5)
y_pred = clf.predict(X_val)
val_accuracy = clf.score(X_val, y_val)

print(f'Cross-validation accuracy scores: {scores}')
print(f'Average cross-validation accuracy: {scores.mean()}')
print(f'Validation accuracy: {val_accuracy}')

param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(SVC(kernel='linear', random_state=51), param_grid, cv=5)
grid_search.fit(X_train, cluster_labels_euclidean)
print("Best parameters found:", grid_search.best_params_)

# Function to plot the dendrogram
def plot_dendrogram(X, title):
    # Compute the linkage matrix
    Z = linkage(X, method='ward')
    
    plt.figure(figsize=(12, 10))
    dendrogram(Z)
    plt.title(title)
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()

# Plot dendrogram for the first 50 samples of the dataset
plot_dendrogram(X_train[:50], "Dendrogram for First 50 Samples with Euclidean Distance")