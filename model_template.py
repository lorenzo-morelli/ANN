import os
import tensorflow as tf
import numpy as np


class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(
            os.path.join(path, 'SubmissionModel'))

    def build_sequences(df, window, stride, telescope):
        assert window % stride == 0
        X = []
        y = []
        temp_df = df
        padding_check = temp_df.shape[1] % window

        if (padding_check != 0):
            padding_len = window - temp_df.shape[1] % window
            padding = np.zeros((temp_df.shape[0], padding_len), dtype='float32')
            temp_df = np.concatenate((padding, temp_df), axis = 1)
            del padding
            assert temp_df.shape[1] % window == 0
        
        for j in np.arange(0, temp_df.shape[0]):
            sequences_X = []
            sequences_y = []
            for i in np.arange(0, temp_df.shape[1] - window - telescope, stride): 
                sequences_X.append(temp_df[j, i:i + window])
                sequences_y.append(temp_df[j, i + window:i + window + telescope])
            X.append(sequences_X)
            y.append(sequences_y)
            
        del temp_df
        X = np.array(X)
        y = np.array(y)
        return X, y

    def predict(self, X, categories):
        # X.shape = (BS, 200)
        # out.shape = (BS, 9) or (BS, 18)

        X, y = self.build_sequences(X)
        out = self.model.predict(X)
        out = out[:, -1, :]

        return out
