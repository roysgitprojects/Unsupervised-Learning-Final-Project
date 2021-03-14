from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import skfuzzy
import numpy as np


def perform_fuzzy_cmeans(points, n_clusters):
    """
    Perform FCM clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :returns: clustering labels
    """
    cntr, u, _, _, _, _, _ = skfuzzy.cluster.cmeans(points.T, c=n_clusters, m=2, error=0.005, maxiter=1000)
    predictions = np.argmax(u, axis=0)
    return predictions


def perform_hierarchical_clustering(points, n_clusters, linkage='ward'):
    """
    Perform Hierarchical clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :param linkage: the sub method
    :returns: clustering labels
    """
    # linkages = ['ward', 'average', 'complete', 'single']
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    predictions = hc.fit_predict(points)
    return predictions


def perform_kmeans(points, n_clusters):
    """
    Perform K means clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :returns: clustering labels
    """
    predictions = KMeans(n_clusters=n_clusters).fit_predict(points)
    return predictions


def perform_gmm(points, n_clusters):
    """
    Perform GMM clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :returns: clustering labels
    """
    gmm = GaussianMixture(n_components=n_clusters)
    gmm = gmm.fit(points)
    predictions = gmm.predict(points)
    return predictions


def perform_spectral_clustering(points, n_clusters):
    """
    Perform Spectral clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :returns: clustering labels
    """
    spectral = SpectralClustering(n_clusters=n_clusters)
    predictions = spectral.fit_predict(points)
    return predictions


def cluster(points, n_clusters, method, linkage='ward'):
    """
    Perform clustering and return predictions.
    :param points: points to cluster
    :param n_clusters: number of clusters
    :param method: clustering method
    :param linkage: if hierarchical clustering than linkage stands for the sub method
    :return: clustering predictions
    """
    if method == 'K means':
        return perform_kmeans(points, n_clusters)
    elif method == 'GMM':
        return perform_gmm(points, n_clusters)
    elif method == 'Fuzzy C Means':
        return perform_fuzzy_cmeans(points, n_clusters)
    elif method == 'Hierarchical':
        return perform_hierarchical_clustering(points, n_clusters, linkage=linkage)
    elif method == 'Spectral':
        return perform_spectral_clustering(points, n_clusters)


def plot_clustering(points, predictions, method):
    """
    Visualize the clustering results
    :param points: points to plot
    :param predictions: points labels according to cluster algorithms
    :param method: clustering method
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('%s clustering on the dataset (reduced data)' % method)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    ax.scatter(points[:, 0], points[:, 1], c=predictions, cmap='tab10', alpha=0.8, s=8)
    plt.show()
