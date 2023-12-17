import os
import tensorflow as tf
import numpy as np

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X, categories):
        window_size = 150
        # X_final = []
        X_final = X[:, -window_size:]

        # for serie in X:
        #     X_final.append(serie[-window_size:])

        # X_final = np.array(X_final)
        out = self.model.predict(X_final) 
        return out