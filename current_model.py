import os
import tensorflow as tf
import numpy as np

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X, categories):

        label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
        encoded = np.vectorize(label_mapping.get)(categories)
    
        finalized_train = []

        for i in range(X.shape[0]):
            timeseries = X[i]
            label = encoded[i]
            category_info = np.zeros((timeseries.shape[0], 6))
            category_info[:, int(label)] = 1
            relevant_sequence = np.expand_dims(timeseries, axis=-1)
            input_sequence = np.concatenate([relevant_sequence, category_info], axis=-1)
            finalized_train.append(input_sequence)

        X = np.array(finalized_train)


        window_size = 150
        X_final = X[:, -window_size:]

        out = self.model.predict(X_final)
        out = out[:, :, 0] # to remove (now model output shape is only [1])
        #out = np.squeeze(out, axis=2)  # to remove (now model output shape is only [1])
        return out