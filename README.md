# Challenges for the Artificial Neural Networks and Deep Learning course at Politecnico di Milano

## Challenge 1: [Image classification](https://github.com/lorenzo-morelli/ANN/tree/main/Image%20classification)

The goal of the first challenge revolved around a binary classification problem
focused on predicting the health status of plants depicted in labeled images. The task entailed
assigning each image to one of two classes: 'Healthy' or 'Unhealthy', using a Neural Network. The team goal was to
develop and test models tailored for this binary classification task. Our model of choice was ConvNeXt, imported from the Tensorflow package, specifically the Large version.

### Model specifications
● ConvNeXt Neural Network

● Preprocessing: data augmentation, oversampling and the built-in preprocessing layer
in the NN

● 32 batch size, early stopping with 30 patience, dropout layer with a rate of 0.35

● 92.11% accuracy on local test data, 91% on evaluation data

## Challenge 2: [Time series forecasting](https://github.com/lorenzo-morelli/ANN/tree/main/Time%20series%20forecasting)

The goal of the second challenge was to design and implement a forecasting model to learn
how to exploit past observations in the input sequences and correctly predict the future by
predicting several uncorrelated time series. The prerequisite was that the model exhibited
generalization capabilities in the forecasting domain, allowing it to transcend the constraints
of specific time domains. Our model was mainly composed of LSTM networks, imported from the Tensorflow package.
