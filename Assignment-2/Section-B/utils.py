import numpy as np
import pandas as pd


class CircleDataset:
    def __init__(self, points) -> None:
        self.n = points
        self.r = 1
        self.circles = np.array([[0, 0, 0], [0, 3, 1]])

    def get(self, add_noise=False):
        self.data = np.empty(shape=(10000, 6), dtype=np.float64)
        self.buildDataset()

        if add_noise:
            self.data[:, 0] = self.data[:, 0] + np.random.randn(self.n)
            self.data[:, 1] = self.data[:, 1] + np.random.randn(self.n)

        return pd.DataFrame(data=self.data, columns=['x', 'y', 'center_x', 'center_y', 'radius', 'label'])

    def buildDataset(self):
        for i in range(10000):
            # Randomly pick any circle
            h, k, label = self.circles[np.random.choice([0, 1])]

            # Randomly pick a value of y
            y = np.random.uniform(low=-1, high=1)
            if k == 3:
                y = np.random.uniform(low=2, high=4)

            # Find corresponding value of x
            x = np.sqrt(self.r**2 - (y-k)**2) + h
            x = np.random.choice([1, -1]) * x

            # Add to Data-set
            self.data[i] = np.array([x, y, h, k, self.r, label])


def split_circle_data_into_train_test(df: pd.DataFrame, partition_size, with_bias=True):
    # x = df[df.columns[:-1]]
    x = df[['x', 'y']].copy(deep=True)
    y = df['label']

    if with_bias:
        x['bias_column'] = np.ones(x.shape[0])
    else:
        x['bias_column'] = np.zeros(x.shape[0])

    test_size = round((partition_size[1]/100) * x.shape[0])
    x_train, y_train = x.iloc[test_size:], y.iloc[test_size:]
    x_test, y_test = x.iloc[:test_size], y.iloc[:test_size]
    return x_train, y_train, x_test, y_test


def split_bit_dataset(df: pd.DataFrame):
    x = df[df.columns[:-1]]
    y = df[df.columns[-1]]
    x['bias_column'] = np.ones(x.shape[0])
    return x, y


class Perceptron:
    def signum(self, value):
        return 1 if value >= 0 else -1

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        y_train = y_train.replace(to_replace=0, value=-1)
        self.weights = np.ones(x_train.shape[1])
        self.weights[-1] = 0.0

        epoch = 0
        while epoch < 10:
            epoch += 1
        # while True:
            prev_weights = self.weights.copy()

            for i in range(x_train.shape[0]):
                error = y_train.iloc[i] - \
                    self.signum(np.dot(x_train.iloc[i], self.weights))

                if error != 0:
                    self.weights = self.weights + error * x_train.iloc[i]

            if all(np.abs(self.weights - prev_weights) < 1e-5):
                break

        self.weights = self.weights.to_numpy()

    def predict(self, x_test: pd.DataFrame):
        if x_test.empty:
            return
        y_pred = np.dot(x_test, self.weights)
        y_pred = np.vectorize(self.signum)(y_pred)
        y_pred = pd.Series(y_pred).replace(to_replace=-1, value=0)
        return y_pred

    def accuracy(self, x_test: pd.DataFrame, y_test: pd.Series):
        if x_test.empty:
            return
        y_pred = self.predict(x_test)
        test_accuracy = (y_test == y_pred).sum() / x_test.shape[0]
        return test_accuracy
