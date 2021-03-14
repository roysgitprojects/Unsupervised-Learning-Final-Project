import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import mannwhitneyu
from clustering import cluster
import matplotlib.pyplot as plt


def get_real_labels_without_anomalies(data_set_number, is_anomaly_array):
    """
    Return real the data's real labels without the anomalous samlpes
    :param data_set_number: the number of the data set
    :param is_anomaly_array: array with 1 at location of anomaly, and 0 if regular sample.
    :return: real labels without anomalies
    """
    labels_without_anomalies = []
    labels = get_real_labels(data_set_number)
    for i in range(len(labels)):
        if is_anomaly_array[i] == 0:
            labels_without_anomalies.append(labels[i])
    return labels_without_anomalies


def get_real_labels(data_set_number):
    """
    Returns the data's real labels
    :param data_set_number: the number of the data set
    :return: list of the real label's
    """
    if data_set_number == 1:
        data = pd.read_csv("dataset/allUsers.lcl.csv", skiprows=lambda x: x % 10 != 0)
        labels = np.array(data['Class'])
        print('labels', labels)
        print('num of labels', len(labels))
        return labels
    elif data_set_number == 2:
        data = pd.read_csv("dataset/HTRU_2.csv",
                           names=['Mean of the integrated profile', 'Standard deviation of the integrated profile',
                                  'Excess kurtosis of the integrated profile', 'Skewness of the integrated profile',
                                  'Mean of the DM-SNR curve', 'Standard deviation of the DM-SNR curve',
                                  'Excess kurtosis of the DM-SNR curve', 'Skewness of the DM-SNR curve',
                                  'Class'], skiprows=lambda x: x % 3 != 0)
        labels = np.array(data['Class'])
        return labels
    else:
        raise Exception('No such dataset')


def plot_external_tag_distribution(data_set_number, points):
    """
    Plot the distribution of the external tags.
    :param data_set_number: the number of the datd set
    :param points: the reduced data
    :return: None
    """
    real_labels = get_real_labels(data_set_number)
    print('real unique labels', np.unique(real_labels))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Distribution of external tags for data set %d (reduced data)' % data_set_number)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    ax.scatter(points[:, 0], points[:, 1], c=real_labels, cmap='Paired', alpha=0.8, s=8)
    plt.show()


def nmi_score(labels_true, points, n_clusters, method, linkage='ward'):
    """
    Returns a list with 20 nmi scores.
    :param labels_true: the real labels
    :param points: the points to cluster
    :param n_clusters: the number of clusters
    :param method: clustering method
    :param linkage: if the method is Hierarchical than linkage represents the sub method
    :returns: a list with 20 nmi scores
    """
    score = []
    for i in range(0, 20):
        labels_pred = cluster(points, n_clusters, method, linkage)
        score.append(normalized_mutual_info_score(labels_true, labels_pred))
    return score


def u_test(scores_method_1, scores_method2):
    """
    Returns P value. if p<<0.05 the first scores better than the second
    :param scores_method_1: first method's scores
    :param scores_method2: second method's scores
    :returns: p value
    """
    mann_whitneyu = mannwhitneyu(scores_method_1, scores_method2, alternative='greater')
    # if p value<0.05 than we can say nmi1>nmi2. Therefore, clustering method 1 is better than 2.
    return mann_whitneyu.pvalue


if __name__ == '__main__':
    print(u_test([2 + 1 / i for i in range(1, 20)], [1 + 1 / i for i in range(1, 20)]))
