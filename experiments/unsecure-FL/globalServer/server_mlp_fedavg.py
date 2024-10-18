#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: FL Server Program for ISCX-VPN 2016 MLP FedAvg

import flwr as fl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
import time as timex
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MAX_ROUNDS = 3
model_name = "clf4_global_model_fedavg_mlp_6client_36epochs_3round_cpu.h5"

model = Sequential()
model.add(InputLayer(input_shape = (732,))) # input layer
model.add(Dense(6, activation='relu')) # hidden layer 1
model.add(Dense(6, activation='relu')) # hidden layer 2
model.add(Dense(10, activation='softmax')) # output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

class SaveKerasModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        agg_weights = super().aggregate_fit(server_round, results, failures)

        if (server_round == MAX_ROUNDS):
            
            model.set_weights(fl.common.parameters_to_ndarrays(agg_weights[0]))
            model.save(model_name)

        return agg_weights

strategy = SaveKerasModelStrategy(min_available_clients=6, min_fit_clients=6, min_evaluate_clients=6)

#Begin counting Time
startTime = timex.time()

fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS))

#End couting time
executionTime = (timex.time() - startTime)
executionTime = executionTime / 60
print('Execution time in minutes: ' + str(executionTime))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from tensorflow import keras

model = keras.models.load_model(model_name)

x_test = np.load("x_test-MLP-Multiclass-ISCX-740features.npy")
y_test = np.load("y_test-MLP-Multiclass-ISCX-740features.npy")
x_test = np.delete(x_test, [12,13,14,15,16,17,18,19], 1)

y_pred_class = np.argmax(model.predict(x_test),axis=1)
y_test_class = np.argmax(y_test, axis=1)
print(confusion_matrix(y_test_class, y_pred_class))
print(classification_report(y_test_class, y_pred_class, digits=4))
