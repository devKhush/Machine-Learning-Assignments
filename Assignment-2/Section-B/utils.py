import numpy as np
import pandas as pd


# Initialize a Circle Datset
class CircleDataset:
    def __init__(self, points) -> None:
        self.n = points             # n=10000
        self.r = 1                  # radius
        self.circles = np.array([[0, 0, 0], [0, 3, 1]])     # two circles

    # get(add_noise function)
    def get(self, add_noise=False):
        # Creating Data
        self.data = np.empty(shape=(10000, 6), dtype=np.float64)
        self.buildDataset()

        # Try greater value of 'std' in noise, and observe that Perceptron can't be applied in Noisy
        # Dataset, if they are not linearly separable
        # Add noise in Dataset
        if add_noise:
            self.data[:, 0] = self.data[:, 0] + \
                np.random.normal(loc=0, scale=0.1, size=self.n)
            self.data[:, 1] = self.data[:, 1] + \
                np.random.normal(loc=0, scale=0.1, size=self.n)

        # Return a Dataframe
        return pd.DataFrame(data=self.data, columns=['x', 'y', 'center_x', 'center_y', 'radius', 'label'])

    # Helper function for building the Dataset
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


# Split the Dataset into Training & Testing Data
def split_circle_data_into_train_test(df: pd.DataFrame, partition_size, with_bias=True):
    # Try PTA in greater dimesnion (with more features, by including centers for circle and radius)
    # and we see it will give a decision boundary in greater dimension,
    # which is not possible in 2-D. Also the accuracy is 100% with noisy data, in greater dimension.
    # x = df[df.columns[:-1]]

    x = df[['x', 'y']].copy(deep=True)
    y = df['label']

    if with_bias:
        x['bias_column'] = np.ones(x.shape[0])
    else:
        x['bias_column'] = np.zeros(x.shape[0])

    test_size = round((partition_size[1]/100) * x.shape[0])
    x_train, y_train = x.iloc[test_size:], y.iloc[test_size:]  # Training Data
    x_test, y_test = x.iloc[:test_size], y.iloc[:test_size]  # Testing Data
    return x_train, y_train, x_test, y_test


# Split Bit Dataset for AND, OR, XOR
def split_bit_dataset(df: pd.DataFrame, with_bias=True):
    x = df[df.columns[:-1]]
    y = df[df.columns[-1]]
    if with_bias:
        x['bias_column'] = np.ones(x.shape[0])
    else:
        x['bias_column'] = np.zeros(x.shape[0])
    return x, y


# Perceptron Training Algorithm
class Perceptron:
    def __init__(self) -> None:
        self.epochs = 50        # epochs

    # Signum activation function
    def signum(self, value):
        return 1 if value >= 0 else -1

    # Training Phase
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        # Map output to "1 and -1" only
        y_train = y_train.replace(to_replace=0, value=-1)

        # Initialize Model weights and Bias as 0
        self.weights = np.ones(x_train.shape[1])
        self.weights[-1] = 0.0

        # Run till epochs
        for epoch in range(self.epochs):
            # while True:
            prev_weights = self.weights.copy()

            # Loop through all training examples
            for i in range(x_train.shape[0]):
                error = y_train.iloc[i] - \
                    self.signum(np.dot(x_train.iloc[i], self.weights))

                # Update weights if error is non-zero
                if error != 0:
                    self.weights = self.weights + error * x_train.iloc[i]

            # Check convergence
            if all(np.abs(self.weights - prev_weights) < 1e-5):
                break
        self.weights = self.weights.to_numpy()

    # Testing Phase
    def predict(self, x_test: pd.DataFrame):
        if x_test.empty:
            return
        y_pred = np.dot(x_test, self.weights)
        y_pred = np.vectorize(self.signum)(y_pred)
        y_pred = pd.Series(y_pred).replace(to_replace=-1, value=0)
        return y_pred

    # Calculate accuracy on Testing Data
    def accuracy(self, x_test: pd.DataFrame, y_test: pd.Series):
        if x_test.empty:
            return
        y_pred = self.predict(x_test)
        test_accuracy = (y_test == y_pred).sum() / x_test.shape[0]
        return test_accuracy
