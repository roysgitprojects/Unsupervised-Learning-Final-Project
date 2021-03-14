from sklearn.preprocessing import normalize, MinMaxScaler

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

import autoencoder
import data_set_preparations
import matplotlib.pyplot as plt

import clustering
import fit_to_external_classification
import predict_nuber_of_clusters

import seaborn as sns


def main():
    """
    Main function. Clusters the data and compare between clustering methods.
    Please note that in order to avoid a lot of figures on the screen,
    the next figure won't appear until the current figure is closed.
    :return: None
    """
    data_set_number = 1
    # read and prepare the data
    data = data_set_preparations.prepare_data_set(data_set_number)

    # plot a boxplot to visualize anomalies
    plot_boxplot(1)
    plot_boxplot(2)

    # anomaly detection + tSNE
    [points, anomalies, reg_points, anomalous_points] = autoencoder.main()

    reg_points = autoencoder.get_reg_points(data_set_number)
    print('len of reg ', len(reg_points))
    anomalous_points = autoencoder.get_anomalous_points(data_set_number)
    print('len of anomalous ', len(anomalous_points))
    points = autoencoder.get_all_points(data_set_number)
    is_anomaly = autoencoder.get_is_anomaly_array(data_set_number)

    # plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Anomalies detected by Autoencoder')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    ax.scatter(reg_points[:, 0], reg_points[:, 1], c='c', alpha=0.8, s=8)
    ax.scatter(anomalous_points[:, 0], anomalous_points[:, 1], c='r', alpha=0.8, s=8)
    plt.show()

    fit_to_external_classification.plot_external_tag_distribution(data_set_number=data_set_number, points=points)

    # number of real labels
    print('data 1 real labels', len(np.unique(fit_to_external_classification.get_real_labels(1))))
    print('data 2 real labels', len(np.unique(fit_to_external_classification.get_real_labels(2))))
    predict_nuber_of_clusters.perform_elbow_method(reg_points, 'K means')
    predict_nuber_of_clusters.perform_elbow_method(reg_points, 'Hierarchical')
    predict_nuber_of_clusters.perform_silhouette_method(reg_points, 'GMM')
    predict_nuber_of_clusters.perform_silhouette_method(reg_points, 'Fuzzy C Means')
    predict_nuber_of_clusters.perform_silhouette_method(reg_points, 'Spectral')

    # cluster
    clustering.plot_clustering(reg_points, clustering.cluster(reg_points, 4, 'K means'), 'K means')
    clustering.plot_clustering(reg_points, clustering.cluster(reg_points, 2, 'GMM'), 'GMM')
    clustering.plot_clustering(reg_points, clustering.cluster(reg_points, 2, 'Fuzzy C Means'), 'Fuzzy C Means')
    linkages = ['ward', 'average', 'complete', 'single']
    for linkage in linkages:
        clustering.plot_clustering(reg_points, clustering.cluster(reg_points, 3, 'Hierarchical', linkage=linkage),
                                   'Hierarchical ' + linkage)
    clustering.plot_clustering(reg_points, clustering.cluster(reg_points, 2, 'Spectral'), 'Spectral')

    # statistical tests
    # create a dictionary of method and nmi scores list
    algorithms_and_n_clusters = [['K means', 4], ['GMM', 2], ['Fuzzy C Means', 2], ['Spectral', 2]]
    algorithm_nmi_dictionary = {}
    for algorithm, n_clusters in algorithms_and_n_clusters:
        algorithm_nmi_dictionary[algorithm] = fit_to_external_classification.nmi_score(
            fit_to_external_classification.get_real_labels_without_anomalies(data_set_number, is_anomaly), reg_points,
            n_clusters=n_clusters, method=algorithm)
    linkages = ['ward', 'average', 'complete', 'single']
    for linkage in linkages:
        algorithm_nmi_dictionary['Hierarchical' + linkage] = fit_to_external_classification.nmi_score(
            fit_to_external_classification.get_real_labels_without_anomalies(data_set_number, is_anomaly), reg_points,
            n_clusters=3, method='Hierarchical', linkage=linkage)
    print('u test')
    for key1 in algorithm_nmi_dictionary:
        for key2 in algorithm_nmi_dictionary:
            if key1 != key2:
                print(key1, 'is significantly better than ', key2, 'with p-value =$',
                      fit_to_external_classification.u_test(algorithm_nmi_dictionary[key1],
                                                            algorithm_nmi_dictionary[key2]), ' <<0.05$')
    print('Average NMI Scores:')
    for key in algorithm_nmi_dictionary:
        print('for', key, 'the average NMI Score is ', sum(algorithm_nmi_dictionary[key]) / len(
            algorithm_nmi_dictionary[key]))

    methods = ['K means', 'GMM', 'Fuzzy C Means', 'Spectral']
    linkages = ['ward', 'average', 'complete', 'single']
    for method in methods:
        predict_nuber_of_clusters.compare_silhouette_scores(points, method)
    for linkage in linkages:
        predict_nuber_of_clusters.compare_silhouette_scores(points, 'Hierarchical', linkage)

    predict_nuber_of_clusters.compare_silhouette_scores_between_different_methods(points)


def perform_pca(data):
    """
    Performs PCA algorithm to 2 dimensions.
    :param data: data to perform the algorithm on
    :return: points after dimension reduction
    """
    pca = PCA(n_components=2)
    points = pca.fit_transform(data)
    return points


def plot_boxplot(data_set_number):
    """
    Plot a boxplot to visualize anomalies
    :param data_set_number: number of data set - 1 or 2
    :return: None
    """
    data = data_set_preparations.prepare_data_set(data_set_number)
    if data_set_number == 1:
        data = data.drop(data.columns[range(-1, -3, -1)], axis=1)
    plt.figure(figsize=(10, 10))
    plt.title("Box Plot")
    sns.boxplot(data=data)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
