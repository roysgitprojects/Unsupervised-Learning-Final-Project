import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import normalize


def prepare_data_set(number_of_data_set):
    """
    Prepare the data set for the clustering.
    :param number_of_data_set: number of data set to be prepared
    :return: prepared data
    """
    if number_of_data_set == 1:
        return prepare_data_set1()
    elif number_of_data_set == 2:
        return prepare_data_set2()
    else:
        raise Exception("No such data set")


def prepare_data_set1():
    """
    Prepare the first data set to clustering.
    :return: prepared data
    """
    data = pd.read_csv("dataset/allUsers.lcl.csv", skiprows=lambda x: x % 10 != 0)
    # replace missing values with None
    data = data.replace({'?': None})
    # refer to all of the columns as numbers
    for column in data.columns:
        if data.dtypes[column] == 'object':
            data[column] = data[column].astype('float64')
    print("Missing values")
    print(data.isna().sum())
    print("Impute missing values with the median value and check again for missing values:")
    # impute with median
    for column in data.columns:
        data.loc[data[column].isnull(), column] = data[column].median()
    print(data.isna().sum())
    print("There are no missing values now")
    data = data.drop(columns=['Class'])
    print(data.columns)
    print(data.info)
    print(data.dtypes)
    # scale the data
    data = scale_the_data(data)
    # normalize the data
    data = normalize(data)
    data = pd.DataFrame(data=data[0:, 0:],  # values
                        # index=data[1:, 0],  # 1st column as index
                        columns=data[0, 0:])  # 1st row as the column names
    print("data after scale info", data.info)
    return data


def prepare_data_set2():
    """
    Prepare the second data set to clustering.
    :return: prepared data
    """
    data = pd.read_csv("dataset/HTRU_2.csv",
                       names=['Mean of the integrated profile', 'Standard deviation of the integrated profile',
                              'Excess kurtosis of the integrated profile', 'Skewness of the integrated profile',
                              'Mean of the DM-SNR curve', 'Standard deviation of the DM-SNR curve',
                              'Excess kurtosis of the DM-SNR curve', 'Skewness of the DM-SNR curve',
                              'Class'], skiprows=lambda x: x % 3 != 0)
    data = data.drop(columns=['Class'])
    print(data.columns)
    print(data.info)
    print(data.dtypes)
    # scale the data
    data = scale_the_data(data)
    # normalize the data
    data = normalize(data)
    data = pd.DataFrame(data=data[0:, 0:],  # values
                        # index=data[1:, 0],  # 1st column as index
                        columns=data[0, 0:])  # 1st row as the column names
    print("data after scale info", data.info)
    print(data.dtypes)
    return data


def scale_the_data(data):
    """
    Scales the data
    :param data: data to scale
    :return: scaled data
    """
    scaler = RobustScaler()  # MinMaxScaler()
    return scaler.fit_transform(data)


if __name__ == '__main__':
    prepare_data_set2()
