#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: FL Client Program for ISCX-VPN 2016 weighted Averaging

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
import flwr as fl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#load dataset

x_train = np.load("x_train-client5.npy")
y_train = np.load("y_train-client5.npy")

x_test = np.load("x_test-client5.npy")
y_test = np.load("y_test-client5.npy")

#Remove IP Address
x_train = np.delete(x_train, [12,13,14,15,16,17,18,19], 1)
x_test = np.delete(x_test, [12,13,14,15,16,17,18,19], 1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#MLP Model
model = Sequential()
model.add(InputLayer(input_shape = (x_train.shape[1],))) # input layer
model.add(Dense(6, activation='relu')) # hidden layer 1
model.add(Dense(6, activation='relu')) # hidden layer 2
model.add(Dense(10, activation='softmax')) # output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

class ntcClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.cid = client_id  # Custom client ID
    
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(x_train, y_train, epochs=36, batch_size=64, shuffle = True)
        return model.get_weights(), len(x_train), {'train_loss':history.history['loss'][0], "cid": self.cid}
    
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=ntcClient(client_id="client_5"))