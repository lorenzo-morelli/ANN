import os
import tensorflow as tf
import numpy as np


class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(
            os.path.join(path, 'SubmissionModel'))

    def add_categories(self, df, categories, cat=None):
        label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
        encoded = np.vectorize(label_mapping.get)(categories)
        result = []

        for i in range(df.shape[0]):
            timeseries = df[i]
            if cat is None:
                label = encoded[i]
                category = np.zeros((timeseries.shape[0], 6))
                category[:, int(label)] = 1
            else:
                category = cat[i]
            timeseries = np.expand_dims(timeseries, axis=-1)
            input_sequence = np.concatenate([timeseries, category], axis=-1)
            result.append(input_sequence)
        result = np.array(result)
        return result

    def predict(self, X, categories):
        X = self.add_categories(X, categories)
        window_size = 100
        X = X[:, -window_size:]

        autoregressive_telescope = 9
        reg_predictions = np.array([])
        X_temp = X
        for _ in range(0, 18, autoregressive_telescope):
            pred_temp = self.model.predict(X)
            if (len(reg_predictions) == 0):
                out = pred_temp
            else:
                out = np.concatenate((reg_predictions, pred_temp), axis=-1)

            pred_temp = self.add_categories(pred_temp, categories, cat=X_temp[:, :autoregressive_telescope, 1:])
            X_temp = np.concatenate(
                (X_temp[:, autoregressive_telescope:, :], pred_temp), axis=1)

        return out
