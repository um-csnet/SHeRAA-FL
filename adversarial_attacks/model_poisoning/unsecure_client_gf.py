#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Gradient Factor attack FL Client Program for ISCX-VPN 2016 (Unsecure client experiment)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
import flwr as fl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#load dataset

x_train = np.load("x_train-client2.npy")
y_train = np.load("y_train-client2.npy")

x_test = np.load("x_test-client2.npy")
y_test = np.load("y_test-client2.npy")

#Remove IP Address
x_train = np.delete(x_train, [12,13,14,15,16,17,18,19], 1)
x_test = np.delete(x_test, [12,13,14,15,16,17,18,19], 1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

loss_fn = tf.keras.losses.categorical_crossentropy

# Define the negative factor
negative_factor = -0.5

# Define the optimizer
optimizer = Adam(learning_rate=0.001)

#MLP Model
model = Sequential()
model.add(InputLayer(input_shape = (x_train.shape[1],))) # input layer
model.add(Dense(6, activation='relu')) # hidden layer 1
model.add(Dense(6, activation='relu')) # hidden layer 2
model.add(Dense(10, activation='softmax')) # output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define a function to compute the gradients
@tf.function
def compute_grads(x, y):
    with tf.GradientTape() as tape:
        loss = loss_fn(y, model(x))
    return tape.gradient(loss, model.trainable_weights)

# Define a function to compute the combined gradients
@tf.function
def compute_combined_grads(x, y):
    grads = compute_grads(x, y)
    combined_grads = [negative_factor * g for g in grads]
    return combined_grads

class ntcClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.cid = client_id  # Custom client ID
    
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(x_train, y_train, epochs=36, batch_size=64, shuffle = True)
        
        #Gradient Factor Attack
        combined_grads = compute_combined_grads(x_train, y_train)
        #optimizer.apply_gradients(zip(combined_grads, model.trainable_weights))

        weights = model.get_weights()
        for i, grad in enumerate(combined_grads):
            weights[i] += grad
        model.set_weights(weights)
        
        return model.get_weights(), len(x_train), {'train_loss':history.history['loss'][0], "cid": self.cid}
    
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=ntcClient(client_id="client_2"))