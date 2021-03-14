import torch.nn as nn

import torch
import time
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from datetime import timedelta

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import normalize

import data_set_preparations

number_of_data_set = 1
data_dim = 37 if number_of_data_set == 1 else 8


class AE(nn.Module):
    """
    Autoencoder class
    """
    def __init__(self):
        super(AE, self).__init__()
        if number_of_data_set == 1:
            self.enc = nn.Sequential(

                nn.Linear(data_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 16),
                nn.Tanh(),
                nn.Linear(16, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
                nn.Tanh()
            )
            self.dec = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 8),
                nn.Tanh(),
                nn.Linear(8, 16),
                nn.Tanh(),
                nn.Linear(16, 32),
                nn.Tanh(),
                nn.Linear(32, data_dim),
                nn.Tanh()
            )
        else:
            self.enc = nn.Sequential(
                nn.Linear(data_dim, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
                nn.Tanh()
            )
            self.dec = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, data_dim),
                nn.Tanh()
            )

    def forward(self, x):
        """
        Encode and decode
        :param x: the data
        :return: encoded and decoded data
        """
        encode = self.enc(x)
        decode = self.dec(encode)
        return encode, decode


# Core training parameters.
batch_size = 1  # 32
lr = 1e-2  # learning rate
w_d = 1e-5  # weight decay
momentum = 0.9
epochs = 15


class Loader(torch.utils.data.Dataset):
    """
    Load data
    """
    def __init__(self):
        super(Loader, self).__init__()
        self.dataset = ''

    def __len__(self):
        """
        length of dataset
        :return: length of data set
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        # row = row.drop(labels={'Class'})
        data = torch.from_numpy(np.array(row) / 255).float()
        return data


class Train_Loader(Loader):
    """
    Load training set
    """
    def __init__(self):
        super(Train_Loader, self).__init__()
        self.dataset = data_set_preparations.prepare_data_set(number_of_data_set).iloc[::5, :]  # train with 20%


class Test_Loader(Loader):
    """
    Load testing set
    """
    def __init__(self):
        super(Test_Loader, self).__init__()
        self.dataset = data_set_preparations.prepare_data_set(number_of_data_set)


def main():
    """
    Main. perform autoencoder - 20% training set. Reduces data with tSNE.
    :return: points, anomalies, reg_points and anomalous_points
    """
    train_set = Train_Loader()  # dont train with all???
    train_ = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
        pin_memory=True,
        drop_last=True
    )

    metrics = defaultdict(list)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AE()
    model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=w_d)

    # train the model
    model.train()
    start = time.time()
    for epoch in range(epochs):
        ep_start = time.time()
        running_loss = 0.0
        for bx, (data) in enumerate(train_):
            _, sample = model(data.to(device))
            loss = criterion(data.to(device), sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_set)
        metrics['train_loss'].append(epoch_loss)
        ep_end = time.time()
        print('-----------------------------------------------')
        print('[EPOCH] {}/{}\n[LOSS] {}'.format(epoch + 1, epochs, epoch_loss))
        print('Epoch Complete in {}'.format(timedelta(seconds=ep_end - ep_start)))
    end = time.time()
    print('-----------------------------------------------')
    print('[System Complete: {}]'.format(timedelta(seconds=end - start)))

    # plot whether the model converges
    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_title('Loss')
    ax.plot(metrics['train_loss'])
    plt.show()

    # prediction
    model.eval()
    loss_dist = []
    test_set = Test_Loader()  # dont train with all???
    test_ = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,  # batch_size,
        shuffle=True,
        num_workers=20,
        pin_memory=True,
        drop_last=True
    )
    encoded_data = []
    for bx, data in enumerate(test_):
        encoded_sample, sample = model(data.to(device))
        # print(encoded_sample)
        encoded_data.append(encoded_sample.detach().numpy())
        loss = criterion(data.to(device), sample)
        loss_dist.append(loss.item())
    loss_sc = []
    for i in loss_dist:
        loss_sc.append((i, i))
    plt.scatter(*zip(*loss_sc))
    upper_threshold = np.percentile(loss_dist, 98)
    lower_threshold = 0.0
    plt.axvline(upper_threshold, 0.0, 1)
    plt.show()
    df = data_set_preparations.prepare_data_set(number_of_data_set)
    ddf = pd.DataFrame(columns=df.columns)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    total_anom = 0
    anomalies = np.array([0 for _ in range(len(loss_dist))])
    for i in range(len(loss_dist)):
        if loss_dist[i] >= upper_threshold:  # if anomaly
            anomalies[i] = 1
    print(anomalies)
    print('number of anomalies', anomalies.sum(), 'out of ', len(anomalies), 'points')
    '''
    df['is anomaly'] = anomalies
    df.to_csv("first data set prepared with anomalies.csv")
    '''
    '''
    # points = model.enc(df.to_numpy())  # dimension reduction
    for step, x in enumerate(test_set):
        encoded = model.enc(x.float())
        if step > 0:
            final_encode = torch.cat((final_encode, encoded))
        else:
            final_encode = encoded
    points = final_encode
    points = points.tolist()
    points = np.asarray(points)
    encoded_data = pd.DataFrame(points)
    print(points)
    print(len(points))
    '''
    # pca = PCA(n_components=2)
    # points = pca.fit_transform(df)
    # points = data_set_preparations.scale_the_data(points)
    '''
    points = np.vstack(encoded_data) #ae
    '''
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    points = TSNE(n_components=2).fit_transform(df)
    scaler = MinMaxScaler()
    points = scaler.fit_transform(points)
    '''
    scaler = StandardScaler()
    points = scaler.fit_transform(points)
    scaler = MinMaxScaler()
    points = scaler.fit_transform(points)
    print('encoded data', points)
    '''
    anomalous_points = []
    reg_points = []
    for i in range(0, len(points)):
        if anomalies[i] == 1:
            anomalous_points.append(points[i])
        else:
            reg_points.append(points[i])
    anomalous_points = np.asarray(anomalous_points)
    reg_points = np.asarray(reg_points)
    print('anomalous: ', anomalous_points)
    print('reg: ', reg_points)
    pd.DataFrame(anomalies).to_csv(
        "dataset" + str(number_of_data_set) + "/data " + str(number_of_data_set) + " is anomaly.csv", index=False,
        header=False)
    pd.DataFrame(points).to_csv(
        "dataset" + str(number_of_data_set) + "/data " + str(number_of_data_set) + " all points.csv", index=False,
        header=False)
    pd.DataFrame(reg_points).to_csv(
        "dataset" + str(number_of_data_set) + "/data " + str(number_of_data_set) + " reg points.csv", index=False,
        header=False)
    pd.DataFrame(anomalous_points).to_csv(
        "dataset" + str(number_of_data_set) + "/data " + str(number_of_data_set) + " anomalous points.csv", index=False,
        header=False)
    return [points, anomalies, reg_points, anomalous_points]


def get_reg_points(data_set_number):
    """
    Read regular points from .csv file.
    :param data_set_number: number of the data set
    :return: regular points
    """
    return np.genfromtxt("dataset" + str(data_set_number) + "/data " + str(data_set_number) + " reg points.csv",
                         delimiter=',')


def get_anomalous_points(data_set_number):
    """
    Read anomalous points from .csv file
    :param data_set_number: the data set number
    :return: the anomalous points
    """
    return np.genfromtxt("dataset" + str(data_set_number) + "/data " + str(data_set_number) + " anomalous points.csv",
                         delimiter=',')


def get_all_points(data_set_number):
    """
    Read all points from .csv file
    :param data_set_number: the data set number
    :return: all of the points
    """
    return np.genfromtxt("dataset" + str(number_of_data_set) + "/data " + str(number_of_data_set) + " all points.csv",
                         delimiter=',')


def get_is_anomaly_array(data_set_number):
    """
    Read is_anomaly array from .csv file.
    :param data_set_number: the number of the data set
    :return: is_anomaly_array - 1 if anomaly, 0 if regular point
    """
    return np.genfromtxt("dataset" + str(number_of_data_set) + "/data " + str(number_of_data_set) + " is anomaly.csv",
                         delimiter=',')


if __name__ == '__main__':
    main()
