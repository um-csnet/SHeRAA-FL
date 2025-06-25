#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: FL Training Program for Local Aggregator and Edge Client

import json
import numpy as np
import sys
import os
import pickle
import shutil
import pandas as pd
import subprocess
import re
import multiprocessing
import time
from datetime import datetime
import flwr as fl
import tensorflow as tf
from pathlib import Path
import statistics
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from functools import reduce
from logging import WARNING
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from flwr.server.strategy.aggregate import aggregate, aggregate_median, aggregate_trimmed_avg, aggregate_krum
from flwr.common import (
EvaluateIns,
EvaluateRes,
FitIns,
FitRes,
MetricsAggregationFn,
NDArrays,
Parameters,
Scalar,
ndarrays_to_parameters,
parameters_to_ndarrays,
NDArray,
GetPropertiesIns,
)
import time as timex
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

class SharedValue:
    trainConfig = {}
    tpmAggCache = {}
    countMal = 0
    choosen_algo = "fed_weighted_Avg"
    modelPerf = []
    gan_status = 0

class ntcClient(fl.client.NumPyClient):
    def __init__(self, client_id, verification_token, fl_config, x, y, experiment_name, delStat):
        self.cid = client_id  # client ID
        self.verification_token = verification_token
        self.fl_config = fl_config
        self.x_train = np.load(x)
        self.y_train = np.load(y)
        self.experiment_name = experiment_name
        self.delStat = delStat
        print(self.experiment_name)
        print(x)
        print(y)
        
        ##Remove Source and Destination IP From Dataset
        #print(self.x_train.shape)
        #print(self.y_train.shape)
        #self.x_train = np.delete(self.x_train, [12,13,14,15,16,17,18,19], 1)
        #print(self.x_train.shape)
        #print(self.y_train.shape)
        
        if self.fl_config['fl_training_model'] == "ntc_mlp":
            self.x_train = np.delete(self.x_train, [12,13,14,15,16,17,18,19], 1)
            self.model = Sequential()
            #self.model.add(InputLayer(input_shape = (740,))) # input layer
            self.model.add(InputLayer(input_shape = (self.x_train.shape[1],))) # input layer
            self.model.add(Dense(32, activation='relu')) # hidden layer 1
            self.model.add(Dense(64, activation='relu')) # hidden layer 2
            self.model.add(Dense(128, activation='relu')) # hidden layer 3
            self.model.add(Dense(10, activation='softmax')) # output layer
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif self.fl_config['fl_training_model'] == "ntc_cnn":
            #Add CNN model here
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif self.fl_config['fl_training_model'] == "ntc_cnn_cifar10":
            #Add CNN model here
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif self.fl_config['fl_training_model'] == "ntc_1dcnn":
            #Add CNN model here
            self.model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.x_train.shape[1], 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(self.y_train.shape[1], activation='softmax')
            ])
            self.model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        if self.delStat == "False":
            history = self.model.fit(self.x_train, self.y_train, epochs=config['training_model_epochs'], batch_size=config['training_model_batch_size'], shuffle = True, callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint(self.experiment_name + '.keras')])
        else:
            history = self.model.fit(self.x_train, self.y_train, epochs=config['training_model_epochs'], batch_size=config['training_model_batch_size'], shuffle = True, callbacks=[WandbMetricsLogger(log_freq=5)])
        return self.model.get_weights(), len(self.x_train), {'train_loss':history.history['loss'][0], "cid": self.cid, "token": self.verification_token }
        
    def get_properties(self, config) -> Dict[str, Scalar]:
        properties = {"client_id": self.cid, "verification_token": self.verification_token}
        return properties

def fl_client(config, fl_config, retrieveTPMHash, status, x, y, client_id, delStat, delStatCount):

    config_pathx = 'config.json'
    with open(config_pathx, 'r') as json_file:
        configx = json.load(json_file)
    experiment_name = configx["experiment_name"]
    
    if delStat == "True":
        experiment_name = experiment_name + "_delegate" + str(delStatCount)
    
    #tracking hyperparameters
    wandb.init(
        project="SHeRAA-Light",
        name=experiment_name,
        # track hyperparameters and run metadata with wandb.config
        config={
            "layer_1": 32,
            "activation_1": "relu",
            "layer_2": 64,
            "activation_2": "relu",
            "layer_3": 128,
            "activation_3": "relu",
            "layer_4": 10,
            "activation_4": "softmax",
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "metric": "accuracy",
            "epoch": 36,
            "batch_size": 64
        }
    )

    print("This Edge node is selected as client for domain " + fl_config['client_domain'])
    if config['enable_gpu_training'] == "False":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if status == "local_aggregator":
        time.sleep(3)
        token = retrieveTPMHash[fl_config['client_id']]['aggt']
    elif status == "local_aggregator_delegate":
        time.sleep(3)
        token = retrieveTPMHash[client_id]['dt']
    elif status == "client":
        token = retrieveTPMHash['vt']
    elif status == "client_delegate":
        time.sleep(3)
        token = retrieveTPMHash["delegated"]["dt"]
    else:
        print("Unknown FL client Status..exiting")
        exit()
        
    #Begin counting Time
    startTime = timex.time()
        
    if config['enable_flower_ssl'] == "True":
        address = fl_config['local_aggregator_host'] + ":" + str(fl_config['local_aggregator_port'])
        fl.client.start_numpy_client(server_address=address, client=ntcClient(client_id=client_id, verification_token=token, fl_config=fl_config, x=x, y=y, experiment_name=experiment_name, delStat=delStat), root_certificates=Path(fl_config['local_aggregator_cert_path']).read_bytes())
    else:
        fl.client.start_numpy_client(server_address="127.0.0.1:" + str(fl_config['local_aggregator_port']) , client=ntcClient(client_id=client_id, verification_token=token, fl_config=fl_config, x=x, y=y, experiment_name=experiment_name, delStat=delStat))
    
    #End couting time
    executionTime = (timex.time() - startTime)
    executionTime = executionTime / 60
    print('Execution time in minutes: ' + str(executionTime))

    wandb.finish()

    # Save the execution time to a file
    with open('results_' + experiment_name + '.txt', 'w') as f:
        f.write('Execution time in minutes: ' + str(executionTime) + '\n')

class SaveKerasModelStrategy(fl.server.strategy.FedAvg):
    def aggregatex(self, results: List[Tuple[NDArrays, int]], client_weights: List[float]) -> NDArrays:
        # Calculate the weighted number of examples used during training
        weighted_num_examples_total = sum(num_examples * weight for (_, num_examples), weight in zip(results, client_weights))
        # Create a list of weights, each multiplied by the related number of examples and client weight
        weighted_weights = [
            [layer * num_examples * weight for layer in weights] for (weights, num_examples), weight in zip(results, client_weights)
        ]
        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / weighted_num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime
        
    def aggregatexneo(self, results: List[Tuple[NDArrays, int]], exclude_clients: List[int]) -> NDArrays:
        # Filter out the parameters of clients to be excluded
        filtered_results = [
            (weights, num_examples) for i, (weights, num_examples) in enumerate(results) if i not in exclude_clients
        ]

        # Calculate the total number of examples used during training
        total_num_examples = sum(num_examples for _, num_examples in filtered_results)

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for (weights, num_examples) in filtered_results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / total_num_examples
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime
        
    def aggregate_medianneo(self, results: List[Tuple[NDArrays, int]], client_weights: List[float]) -> NDArrays:
        """Compute median with adjusted weights."""
        # Create a list of adjusted weights
        adjusted_weights = [
            [np.array(layer) * weight for layer in weights] for (weights, _), weight in zip(results, client_weights)
        ]

        # Compute median weight of each layer
        median_w: NDArrays = [
            np.median(np.asarray(layer), axis=0) for layer in zip(*adjusted_weights)
        ]
        return median_w
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Client selection and Configure the next round of training."""
        print("Waiting for " + str(SharedValue.trainConfig['client_count']) + " Clients")
        client_manager.wait_for(num_clients  =  SharedValue.trainConfig['client_count'])
        client_properties = {}
        standard_config = {
        "training_model_batch_size": SharedValue.trainConfig['training_model_batch_size'],
        "training_model_epochs": SharedValue.trainConfig['training_model_epochs'],
        "training_learning_rate": SharedValue.trainConfig['training_learning_rate'],
        "fl_training_model_type": SharedValue.trainConfig['fl_training_model_type']
        }
        fit_configurations = []
        for cid, client in client_manager.all().items():
            ins = GetPropertiesIns({})
            client_properties[cid] = client.get_properties(ins, timeout = 30)
            prop = client_properties[cid].properties
            if prop['client_id'] == SharedValue.trainConfig['own_id']:
                token = SharedValue.tpmAggCache[prop['client_id']]['aggt']
            else:
                try:
                    token = SharedValue.tpmAggCache[prop['client_id']]['vt']
                except:
                    token = SharedValue.tpmAggCache[prop['client_id']]['dt']
            if prop['verification_token'] == token:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                print(f"Unauthorized connection from {prop['client_id']} ")
        return fit_configurations
        
    def get_client_weight(self, client_id, token, server_round):
        """Determine the weight for a client based on client ID."""
        # Logic to return the weight based on the client ID
        if SharedValue.gan_status == 1:
            perf_threshold = 0.8
        else:
            perf_threshold = 0.6
        print("Performance Threshold: " + str(perf_threshold))
        ts = []
        for client in SharedValue.tpmAggCache:
            ts.append(SharedValue.tpmAggCache[client]['ts'])
        median_ts = statistics.median(ts)
        countMoreMedian = 0
        for client in SharedValue.tpmAggCache:
            if SharedValue.tpmAggCache[client]['ts'] >= median_ts:
                countMoreMedian += 1
        lessCheck = 0
        for perf in SharedValue.modelPerf:
            if perf < perf_threshold:
                lessCheck = 1
        client_weights = []
        if lessCheck == 0:
            if countMoreMedian == len(SharedValue.tpmAggCache):
                cw = 1 / len(SharedValue.tpmAggCache)
                for cid in client_id:
                    client_weights.append(cw)
            else:
                countLessMedian = len(SharedValue.tpmAggCache) - countMoreMedian
                SharedValue.countMal = countLessMedian
                trustScoreMore = 0.8 / countMoreMedian
                trustScoreLess = 0.2 / countLessMedian
                for cid in client_id:
                    if SharedValue.tpmAggCache[cid]['ts'] >= median_ts:
                        client_weights.append(trustScoreMore)
                    elif SharedValue.tpmAggCache[cid]['ts'] < median_ts:
                        client_weights.append(trustScoreLess)
        elif lessCheck == 1:
            morelist = 0
            lesslist = 0
            for perf in SharedValue.modelPerf:
                if perf >= perf_threshold:
                    morelist += 1
                elif perf < perf_threshold:
                    lesslist += 1
            SharedValue.countMal = lesslist
            trustScoreMore = 0.9 / morelist
            trustScoreLess = 0.1 / lesslist
            for perf in SharedValue.modelPerf:
                if perf >= perf_threshold:
                    client_weights.append(trustScoreMore)
                elif perf < perf_threshold:
                    client_weights.append(trustScoreLess)
        return client_weights
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        custom_id = []
        token = []
        clientParam = []
        for client, fit_res in results:
            custom_id.append(fit_res.metrics.get("cid"))
            token.append(fit_res.metrics.get("token"))
            clientParam.append(fit_res.parameters)
        
        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        if SharedValue.trainConfig['fl_training_model_type'] == "ntc_mlp":
            model = Sequential()
            model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            model.add(Dense(32, activation='relu')) # hidden layer 1
            model.add(Dense(64, activation='relu')) # hidden layer 2
            model.add(Dense(128, activation='relu')) # hidden layer 3
            model.add(Dense(10, activation='softmax')) # output layer
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_avg_test_model = Sequential()
            fed_avg_test_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            fed_avg_test_model.add(Dense(32, activation='relu')) # hidden layer 1
            fed_avg_test_model.add(Dense(64, activation='relu')) # hidden layer 2
            fed_avg_test_model.add(Dense(128, activation='relu')) # hidden layer 3
            fed_avg_test_model.add(Dense(10, activation='softmax')) # output layer
            fed_avg_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_weighted_avg_test_model = Sequential()
            fed_weighted_avg_test_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            fed_weighted_avg_test_model.add(Dense(32, activation='relu')) # hidden layer 1
            fed_weighted_avg_test_model.add(Dense(64, activation='relu')) # hidden layer 2
            fed_weighted_avg_test_model.add(Dense(128, activation='relu')) # hidden layer 3
            fed_weighted_avg_test_model.add(Dense(10, activation='softmax')) # output layer
            fed_weighted_avg_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_median_test_model = Sequential()
            fed_median_test_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            fed_median_test_model.add(Dense(32, activation='relu')) # hidden layer 1
            fed_median_test_model.add(Dense(64, activation='relu')) # hidden layer 2
            fed_median_test_model.add(Dense(128, activation='relu')) # hidden layer 3
            fed_median_test_model.add(Dense(10, activation='softmax')) # output layer
            fed_median_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim10_test_model = Sequential()
            fed_trim10_test_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            fed_trim10_test_model.add(Dense(32, activation='relu')) # hidden layer 1
            fed_trim10_test_model.add(Dense(64, activation='relu')) # hidden layer 2
            fed_trim10_test_model.add(Dense(128, activation='relu')) # hidden layer 3
            fed_trim10_test_model.add(Dense(10, activation='softmax')) # output layer
            fed_trim10_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim20_test_model = Sequential()
            fed_trim20_test_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            fed_trim20_test_model.add(Dense(32, activation='relu')) # hidden layer 1
            fed_trim20_test_model.add(Dense(64, activation='relu')) # hidden layer 2
            fed_trim20_test_model.add(Dense(128, activation='relu')) # hidden layer 3
            fed_trim20_test_model.add(Dense(10, activation='softmax')) # output layer
            fed_trim20_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim30_test_model = Sequential()
            fed_trim30_test_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            fed_trim30_test_model.add(Dense(32, activation='relu')) # hidden layer 1
            fed_trim30_test_model.add(Dense(64, activation='relu')) # hidden layer 2
            fed_trim30_test_model.add(Dense(128, activation='relu')) # hidden layer 3
            fed_trim30_test_model.add(Dense(10, activation='softmax')) # output layer
            fed_trim30_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_krum_test_model = Sequential()
            fed_krum_test_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            fed_krum_test_model.add(Dense(32, activation='relu')) # hidden layer 1
            fed_krum_test_model.add(Dense(64, activation='relu')) # hidden layer 2
            fed_krum_test_model.add(Dense(128, activation='relu')) # hidden layer 3
            fed_krum_test_model.add(Dense(10, activation='softmax')) # output layer
            fed_krum_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_multi_krum_test_model = Sequential()
            fed_multi_krum_test_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            fed_multi_krum_test_model.add(Dense(32, activation='relu')) # hidden layer 1
            fed_multi_krum_test_model.add(Dense(64, activation='relu')) # hidden layer 2
            fed_multi_krum_test_model.add(Dense(128, activation='relu')) # hidden layer 3
            fed_multi_krum_test_model.add(Dense(10, activation='softmax')) # output layer
            fed_multi_krum_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_median_adjusted_test_model = Sequential()
            fed_median_adjusted_test_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            fed_median_adjusted_test_model.add(Dense(32, activation='relu')) # hidden layer 1
            fed_median_adjusted_test_model.add(Dense(64, activation='relu')) # hidden layer 2
            fed_median_adjusted_test_model.add(Dense(128, activation='relu')) # hidden layer 3
            fed_median_adjusted_test_model.add(Dense(10, activation='softmax')) # output layer
            fed_median_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim10_adjusted_test_model = Sequential()
            fed_trim10_adjusted_test_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            fed_trim10_adjusted_test_model.add(Dense(32, activation='relu')) # hidden layer 1
            fed_trim10_adjusted_test_model.add(Dense(64, activation='relu')) # hidden layer 2
            fed_trim10_adjusted_test_model.add(Dense(128, activation='relu')) # hidden layer 3
            fed_trim10_adjusted_test_model.add(Dense(10, activation='softmax')) # output layer
            fed_trim10_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim20_adjusted_test_model = Sequential()
            fed_trim20_adjusted_test_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            fed_trim20_adjusted_test_model.add(Dense(32, activation='relu')) # hidden layer 1
            fed_trim20_adjusted_test_model.add(Dense(64, activation='relu')) # hidden layer 2
            fed_trim20_adjusted_test_model.add(Dense(128, activation='relu')) # hidden layer 3
            fed_trim20_adjusted_test_model.add(Dense(10, activation='softmax')) # output layer
            fed_trim20_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim30_adjusted_test_model = Sequential()
            fed_trim30_adjusted_test_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            fed_trim30_adjusted_test_model.add(Dense(32, activation='relu')) # hidden layer 1
            fed_trim30_adjusted_test_model.add(Dense(64, activation='relu')) # hidden layer 2
            fed_trim30_adjusted_test_model.add(Dense(128, activation='relu')) # hidden layer 3
            fed_trim30_adjusted_test_model.add(Dense(10, activation='softmax')) # output layer
            fed_trim30_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_krum_adjusted_test_model = Sequential()
            fed_krum_adjusted_test_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            fed_krum_adjusted_test_model.add(Dense(32, activation='relu')) # hidden layer 1
            fed_krum_adjusted_test_model.add(Dense(64, activation='relu')) # hidden layer 2
            fed_krum_adjusted_test_model.add(Dense(128, activation='relu')) # hidden layer 3
            fed_krum_adjusted_test_model.add(Dense(10, activation='softmax')) # output layer
            fed_krum_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_multi_krum_adjusted_test_model = Sequential()
            fed_multi_krum_adjusted_test_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            fed_multi_krum_adjusted_test_model.add(Dense(32, activation='relu')) # hidden layer 1
            fed_multi_krum_adjusted_test_model.add(Dense(64, activation='relu')) # hidden layer 2
            fed_multi_krum_adjusted_test_model.add(Dense(128, activation='relu')) # hidden layer 3
            fed_multi_krum_adjusted_test_model.add(Dense(10, activation='softmax')) # output layer
            fed_multi_krum_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            last_model = Sequential()
            last_model.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
            last_model.add(Dense(32, activation='relu')) # hidden layer 1
            last_model.add(Dense(64, activation='relu')) # hidden layer 2
            last_model.add(Dense(128, activation='relu')) # hidden layer 3
            last_model.add(Dense(10, activation='softmax')) # output layer
            last_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
        elif SharedValue.trainConfig['fl_training_model_type'] == "ntc_cnn":
            #Add CNN model here
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_avg_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_avg_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_weighted_avg_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_weighted_avg_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_median_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_median_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim10_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_trim10_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim20_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_trim20_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim30_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_trim30_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_krum_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_krum_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_multi_krum_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_multi_krum_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_median_adjusted_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_median_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim10_adjusted_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_trim10_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim20_adjusted_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_trim20_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim30_adjusted_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_trim30_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_krum_adjusted_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_krum_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_multi_krum_adjusted_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_multi_krum_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            last_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            last_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif SharedValue.trainConfig['fl_training_model_type'] == "ntc_cnn_cifar10":
            #Add CNN model for CIFAR10 here
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_avg_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_avg_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_weighted_avg_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_weighted_avg_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_median_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_median_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim10_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_trim10_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim20_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_trim20_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim30_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_trim30_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_krum_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_krum_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_multi_krum_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_multi_krum_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_median_adjusted_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_median_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim10_adjusted_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_trim10_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim20_adjusted_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_trim20_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_trim30_adjusted_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_trim30_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_krum_adjusted_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_krum_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            fed_multi_krum_adjusted_test_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            fed_multi_krum_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            last_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
            last_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif SharedValue.trainConfig['fl_training_model_type'] == "ntc_1dcnn":
            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            fed_avg_test_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            fed_avg_test_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            fed_weighted_avg_test_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            fed_weighted_avg_test_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            fed_median_test_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            fed_median_test_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            fed_trim10_test_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            fed_trim10_test_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            fed_trim20_test_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            fed_trim20_test_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            fed_trim30_test_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            fed_trim30_test_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            fed_krum_test_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            fed_krum_test_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            fed_multi_krum_test_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            fed_multi_krum_test_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            fed_median_adjusted_test_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            fed_median_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            fed_trim10_adjusted_test_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            fed_trim10_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            fed_trim20_adjusted_test_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            fed_trim20_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            fed_trim30_adjusted_test_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            fed_trim30_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            fed_krum_adjusted_test_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            fed_krum_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            fed_multi_krum_adjusted_test_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            fed_multi_krum_adjusted_test_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
            last_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            last_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            
        x_test = np.load(SharedValue.trainConfig['x_test'])
        y_test = np.load(SharedValue.trainConfig['y_test'])
        if SharedValue.trainConfig['fl_training_model_type'] == "ntc_mlp":
            x_test = np.delete(x_test, [12,13,14,15,16,17,18,19], 1)
        evalDataList = []
        benchGan = []
        ownCheck = 0
        #if server_round == 1 :
        if True :
            indexCheck = 0
            for weight in clientParam:
                if SharedValue.trainConfig['fl_training_model_type'] == "ntc_mlp":
                    modeleval = Sequential()
                    modeleval.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
                    modeleval.add(Dense(32, activation='relu')) # hidden layer 1
                    modeleval.add(Dense(64, activation='relu')) # hidden layer 2
                    modeleval.add(Dense(128, activation='relu')) # hidden layer 3
                    modeleval.add(Dense(10, activation='softmax')) # output layer
                    modeleval.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                    print('MLP')
                elif SharedValue.trainConfig['fl_training_model_type'] == "ntc_cnn":
                    #Add CNN model here
                    modeleval = Sequential([
                        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                        BatchNormalization(),
                        Conv2D(32, (3, 3), activation='relu', padding='same'),
                        BatchNormalization(),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.25),
                        
                        Conv2D(64, (3, 3), activation='relu', padding='same'),
                        BatchNormalization(),
                        Conv2D(64, (3, 3), activation='relu', padding='same'),
                        BatchNormalization(),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.25),

                        Flatten(),
                        Dense(128, activation='relu'),
                        BatchNormalization(),
                        Dropout(0.4),
                        Dense(10, activation='softmax')
                    ])
                    modeleval.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                    print('CNN')
                elif SharedValue.trainConfig['fl_training_model_type'] == "ntc_cnn_cifar10":
                    #Add CNN model here
                    modeleval = Sequential([
                        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                        BatchNormalization(),
                        Conv2D(32, (3, 3), activation='relu', padding='same'),
                        BatchNormalization(),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.25),
                        
                        Conv2D(64, (3, 3), activation='relu', padding='same'),
                        BatchNormalization(),
                        Conv2D(64, (3, 3), activation='relu', padding='same'),
                        BatchNormalization(),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.25),

                        Flatten(),
                        Dense(128, activation='relu'),
                        BatchNormalization(),
                        Dropout(0.4),
                        Dense(10, activation='softmax')
                    ])
                    modeleval.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                    print('CNN')
                elif SharedValue.trainConfig['fl_training_model_type'] == "ntc_1dcnn":
                    #Add CNN model here
                    modeleval = Sequential([
                        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                        MaxPooling1D(pool_size=2),
                        Conv1D(filters=128, kernel_size=3, activation='relu'),
                        MaxPooling1D(pool_size=2),
                        Flatten(),
                        Dense(128, activation='relu'),
                        Dropout(0.5),
                        Dense(10, activation='softmax')
                    ])
                    modeleval.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
                    print('1DCNN')
                modeleval.set_weights(fl.common.parameters_to_ndarrays(weight))
                y_pred_class = np.argmax(modeleval.predict(x_test),axis=1)
                y_test_class = np.argmax(y_test, axis=1)
                eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
                evalDataList.append(eva_data)
                #print(classification_report(y_test_class, y_pred_class, digits=4))
                eval_result = eva_data['accuracy']
                SharedValue.modelPerf.append(eval_result)
                if custom_id[indexCheck] == SharedValue.trainConfig['own_id']:
                    #print(custom_id[indexCheck])
                    #print(eva_data)
                    ownCheck = indexCheck
                    for cls in eva_data:
                        if cls != 'accuracy' and cls != 'macro avg' and cls != 'weighted avg':
                            benchGan.append(eva_data[cls]['f1-score'])
                indexCheck += 1
            ##Checking classes F1-score for possible GAN Attack
            possible_gan_count = 0
            cc = 0
            if server_round == 1 :
                for evalGan in evalDataList:
                    if cc != ownCheck:
                        ccc = 0
                        for cls in evalGan:
                            if cls != 'accuracy' and cls != 'macro avg' and cls != 'weighted avg':
                                #print(evalGan[cls])
                                #print(benchGan[ccc])
                                if benchGan[ccc] >= 0.1:
                                    if evalGan[cls]['f1-score'] < 0.1:
                                        possible_gan_count += 1
                                ccc += 1
                    cc += 1
                if possible_gan_count >= SharedValue.trainConfig['possible_gan_threshold']:
                    SharedValue.gan_status = 1
                    print(f"Detected attempted GAN-based attack from {possible_gan_count} malicious clients")
            print(SharedValue.modelPerf)
            
        client_weights = self.get_client_weight(custom_id, token, server_round)
        ##Printing Client Weight
        cc = 0
        for wei in client_weights:
            print(custom_id[cc] + ": " + str(wei))
            cc +=1

        adjusted_weights = self.aggregatex(weights_results, client_weights)
            
        if SharedValue.trainConfig['training_alpha_value'] == 0:
            if SharedValue.countMal == 2:
                alpha = 1.5
            elif SharedValue.countMal >= 3:
                alpha = 2.0
            else:
                alpha = 1.0
        else:
            alpha = SharedValue.trainConfig['training_alpha_value']
        print('Alpha value: ' + str(alpha))
        
        if server_round == 2 :
            
            test_fed_avg = ndarrays_to_parameters(aggregate(weights_results))
            test_weighted_avg = ndarrays_to_parameters(self.aggregatex(weights_results, client_weights))
            test_median = ndarrays_to_parameters(aggregate_median(weights_results))
            test_trim10 = ndarrays_to_parameters(aggregate_trimmed_avg(weights_results, 0.1))
            test_trim20 = ndarrays_to_parameters(aggregate_trimmed_avg(weights_results, 0.2))
            test_trim30 = ndarrays_to_parameters(aggregate_trimmed_avg(weights_results, 0.3))
            test_krum = ndarrays_to_parameters(aggregate_krum(weights_results, SharedValue.countMal, 0))
            test_multi_krum = ndarrays_to_parameters(aggregate_krum(weights_results, SharedValue.countMal, SharedValue.countMal))
            
            weights_median_adjusted = aggregate_median(weights_results)
            final_weights_median_adjusted: NDArrays = [
            (median_layer + alpha * mean_layer) / (1 + alpha)
            for median_layer, mean_layer in zip(weights_median_adjusted, adjusted_weights)
            ]
            test_median_adjusted = ndarrays_to_parameters(final_weights_median_adjusted)
            
            weights_trim10_adjusted = aggregate_trimmed_avg(weights_results, 0.1)
            final_weights_trim10_adjusted: NDArrays = [
            (trim10_layer + alpha * mean_layer) / (1 + alpha)
            for trim10_layer, mean_layer in zip(weights_trim10_adjusted, adjusted_weights)
            ]
            test_trim10_adjusted = ndarrays_to_parameters(final_weights_trim10_adjusted)
            
            weights_trim20_adjusted = aggregate_trimmed_avg(weights_results, 0.2)
            final_weights_trim20_adjusted: NDArrays = [
            (trim20_layer + alpha * mean_layer) / (1 + alpha)
            for trim20_layer, mean_layer in zip(weights_trim20_adjusted, adjusted_weights)
            ]
            test_trim20_adjusted = ndarrays_to_parameters(final_weights_trim20_adjusted)
            
            weights_trim30_adjusted = aggregate_trimmed_avg(weights_results, 0.3)
            final_weights_trim30_adjusted: NDArrays = [
            (trim30_layer + alpha * mean_layer) / (1 + alpha)
            for trim30_layer, mean_layer in zip(weights_trim30_adjusted, adjusted_weights)
            ]
            test_trim30_adjusted = ndarrays_to_parameters(final_weights_trim30_adjusted)
            
            weights_krum_adjusted = aggregate_krum(weights_results, SharedValue.countMal, 0)
            final_weights_krum_adjusted: NDArrays = [
            (krum_layer + alpha * mean_layer) / (1 + alpha)
            for krum_layer, mean_layer in zip(weights_krum_adjusted, adjusted_weights)
            ]
            test_krum_adjusted = ndarrays_to_parameters(final_weights_krum_adjusted)
            
            weights_multi_krum_adjusted = aggregate_krum(weights_results, SharedValue.countMal, SharedValue.countMal)
            final_weights_multi_krum_adjusted: NDArrays = [
            (multi_krum_layer + alpha * mean_layer) / (1 + alpha)
            for multi_krum_layer, mean_layer in zip(weights_multi_krum_adjusted, adjusted_weights)
            ]
            test_multi_krum_adjusted = ndarrays_to_parameters(final_weights_multi_krum_adjusted)
            
            fed_avg_test_model.set_weights(fl.common.parameters_to_ndarrays(test_fed_avg))
            fed_weighted_avg_test_model.set_weights(fl.common.parameters_to_ndarrays(test_weighted_avg))
            fed_median_test_model.set_weights(fl.common.parameters_to_ndarrays(test_median))
            fed_trim10_test_model.set_weights(fl.common.parameters_to_ndarrays(test_trim10))
            fed_trim20_test_model.set_weights(fl.common.parameters_to_ndarrays(test_trim20))
            fed_trim30_test_model.set_weights(fl.common.parameters_to_ndarrays(test_trim30))
            fed_krum_test_model.set_weights(fl.common.parameters_to_ndarrays(test_krum))
            fed_multi_krum_test_model.set_weights(fl.common.parameters_to_ndarrays(test_multi_krum))
            
            fed_median_adjusted_test_model.set_weights(fl.common.parameters_to_ndarrays(test_median_adjusted))
            fed_trim10_adjusted_test_model.set_weights(fl.common.parameters_to_ndarrays(test_trim10_adjusted))
            fed_trim20_adjusted_test_model.set_weights(fl.common.parameters_to_ndarrays(test_trim20_adjusted))
            fed_trim30_adjusted_test_model.set_weights(fl.common.parameters_to_ndarrays(test_trim30_adjusted))
            fed_krum_adjusted_test_model.set_weights(fl.common.parameters_to_ndarrays(test_krum_adjusted))
            fed_multi_krum_adjusted_test_model.set_weights(fl.common.parameters_to_ndarrays(test_multi_krum_adjusted))
            
            y_pred_class = np.argmax(fed_avg_test_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            fed_avg_result = eva_data['accuracy']
            
            y_pred_class = np.argmax(fed_weighted_avg_test_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            fed_weighted_avg_result = eva_data['accuracy']
            
            y_pred_class = np.argmax(fed_median_test_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            fed_median_result = eva_data['accuracy']
            
            y_pred_class = np.argmax(fed_trim10_test_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            fed_trim10_result = eva_data['accuracy']
            
            y_pred_class = np.argmax(fed_trim20_test_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            fed_trim20_result = eva_data['accuracy']
            
            y_pred_class = np.argmax(fed_trim30_test_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            fed_trim30_result = eva_data['accuracy']
            
            y_pred_class = np.argmax(fed_krum_test_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            fed_krum_result = eva_data['accuracy']
            
            y_pred_class = np.argmax(fed_multi_krum_test_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            fed_multi_krum_result = eva_data['accuracy']
            
            print("fed_Avg: " + str(fed_avg_result))
            print("fed_weighted_Avg: " + str(fed_weighted_avg_result))
            print("fed_median: " + str(fed_median_result))
            print("fed_trim10: " + str(fed_trim10_result))
            print("fed_trim20: " + str(fed_trim20_result))
            print("fed_trim30: " + str(fed_trim30_result))
            print("fed_krum: " + str(fed_krum_result))
            print("fed_multi_krum: " + str(fed_multi_krum_result))
            
            y_pred_class = np.argmax(fed_median_adjusted_test_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            fed_median_adjusted_result = eva_data['accuracy']
            
            y_pred_class = np.argmax(fed_trim10_adjusted_test_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            fed_trim10_adjusted_result = eva_data['accuracy']
            
            y_pred_class = np.argmax(fed_trim20_adjusted_test_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            fed_trim20_adjusted_result = eva_data['accuracy']
            
            y_pred_class = np.argmax(fed_trim30_adjusted_test_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            fed_trim30_adjusted_result = eva_data['accuracy']
            
            y_pred_class = np.argmax(fed_krum_adjusted_test_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            fed_krum_adjusted_result = eva_data['accuracy']
            
            y_pred_class = np.argmax(fed_multi_krum_adjusted_test_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            fed_multi_krum_adjusted_result = eva_data['accuracy']
            
            print("fed_median_adjusted: " + str(fed_median_adjusted_result))
            print("fed_trim10_adjusted: " + str(fed_trim10_adjusted_result))
            print("fed_trim20_adjusted: " + str(fed_trim20_adjusted_result))
            print("fed_trim30_adjusted: " + str(fed_trim30_adjusted_result))
            print("fed_krum_adjusted: " + str(fed_krum_adjusted_result))
            print("fed_multi_krum_adjusted: " + str(fed_multi_krum_adjusted_result))
            
            test_result_list = [fed_avg_result, fed_weighted_avg_result, fed_median_result, fed_trim10_result, fed_trim20_result, fed_trim30_result, fed_krum_result, fed_multi_krum_result, fed_median_adjusted_result, fed_trim10_adjusted_result, fed_trim20_adjusted_result, fed_trim30_adjusted_result, fed_krum_adjusted_result, fed_multi_krum_adjusted_result]
            result_max = max(test_result_list)
            
            if fed_avg_result == result_max:
                SharedValue.choosen_algo = "fed_Avg"
            elif fed_weighted_avg_result == result_max:
                SharedValue.choosen_algo = "fed_weighted_Avg"
            elif fed_median_result == result_max:
                SharedValue.choosen_algo = "fed_median"
            elif fed_trim10_result == result_max:
                SharedValue.choosen_algo = "fed_trim10"
            elif fed_trim20_result == result_max:
                SharedValue.choosen_algo = "fed_trim20"
            elif fed_trim30_result == result_max:
                SharedValue.choosen_algo = "fed_trim30"
            elif fed_krum_result == result_max:
                SharedValue.choosen_algo = "fed_krum"
            elif fed_multi_krum_result == result_max:
                SharedValue.choosen_algo = "fed_multi_krum"
            elif fed_median_adjusted_result == result_max:
                SharedValue.choosen_algo = "fed_median_adjusted"
            elif fed_trim10_adjusted_result == result_max:
                SharedValue.choosen_algo = "fed_trim10_adjusted"
            elif fed_trim20_adjusted_result == result_max:
                SharedValue.choosen_algo = "fed_trim20_adjusted"
            elif fed_trim30_adjusted_result == result_max:
                SharedValue.choosen_algo = "fed_trim30_adjusted"
            elif fed_krum_adjusted_result == result_max:
                SharedValue.choosen_algo = "fed_krum_adjusted"
            elif fed_multi_krum_adjusted_result == result_max:
                SharedValue.choosen_algo = "fed_multi_krum_adjusted"
            print("Choosen Aggregation Algorithm: " + SharedValue.choosen_algo)
        
        if SharedValue.choosen_algo == "fed_Avg":
            parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        elif SharedValue.choosen_algo == "fed_weighted_Avg":
            parameters_aggregated = ndarrays_to_parameters(self.aggregatex(weights_results, client_weights))
        elif SharedValue.choosen_algo == "fed_median":
            parameters_aggregated = ndarrays_to_parameters(aggregate_median(weights_results))
        elif SharedValue.choosen_algo == "fed_trim10":
            parameters_aggregated = ndarrays_to_parameters(aggregate_trimmed_avg(weights_results, 0.1))
        elif SharedValue.choosen_algo == "fed_trim20":
            parameters_aggregated = ndarrays_to_parameters(aggregate_trimmed_avg(weights_results, 0.2))
        elif SharedValue.choosen_algo == "fed_trim30":
            parameters_aggregated = ndarrays_to_parameters(aggregate_trimmed_avg(weights_results, 0.3))
        elif SharedValue.choosen_algo == "fed_krum":
            parameters_aggregated = ndarrays_to_parameters(aggregate_krum(weights_results, SharedValue.countMal, 0))
        elif SharedValue.choosen_algo == "fed_multi_krum":
            parameters_aggregated = ndarrays_to_parameters(aggregate_krum(weights_results, SharedValue.countMal, SharedValue.countMal))
        elif SharedValue.choosen_algo == "fed_median_adjusted":
            weights_median_adjusted = aggregate_median(weights_results)
            final_weights_median_adjusted: NDArrays = [
            (median_layer + alpha * mean_layer) / (1 + alpha)
            for median_layer, mean_layer in zip(weights_median_adjusted, adjusted_weights)
            ]
            parameters_aggregated = ndarrays_to_parameters(final_weights_median_adjusted)
        elif SharedValue.choosen_algo == "fed_trim10_adjusted":
            weights_trim10_adjusted = aggregate_trimmed_avg(weights_results, 0.1)
            final_weights_trim10_adjusted: NDArrays = [
            (trim10_layer + alpha * mean_layer) / (1 + alpha)
            for trim10_layer, mean_layer in zip(weights_trim10_adjusted, adjusted_weights)
            ]
            parameters_aggregated = ndarrays_to_parameters(final_weights_trim10_adjusted)
        elif SharedValue.choosen_algo == "fed_trim20_adjusted":
            weights_trim20_adjusted = aggregate_trimmed_avg(weights_results, 0.2)
            final_weights_trim20_adjusted: NDArrays = [
            (trim20_layer + alpha * mean_layer) / (1 + alpha)
            for trim20_layer, mean_layer in zip(weights_trim20_adjusted, adjusted_weights)
            ]
            parameters_aggregated = ndarrays_to_parameters(final_weights_trim20_adjusted)
        elif SharedValue.choosen_algo == "fed_trim30_adjusted":
            weights_trim30_adjusted = aggregate_trimmed_avg(weights_results, 0.3)
            final_weights_trim30_adjusted: NDArrays = [
            (trim30_layer + alpha * mean_layer) / (1 + alpha)
            for trim30_layer, mean_layer in zip(weights_trim30_adjusted, adjusted_weights)
            ]
            parameters_aggregated = ndarrays_to_parameters(final_weights_trim30_adjusted)
        elif SharedValue.choosen_algo == "fed_krum_adjusted":
            weights_krum_adjusted = aggregate_krum(weights_results, SharedValue.countMal, 0)
            final_weights_krum_adjusted: NDArrays = [
            (krum_layer + alpha * mean_layer) / (1 + alpha)
            for krum_layer, mean_layer in zip(weights_krum_adjusted, adjusted_weights)
            ]
            parameters_aggregated = ndarrays_to_parameters(final_weights_krum_adjusted)
        elif SharedValue.choosen_algo == "fed_multi_krum_adjusted":
            weights_multi_krum_adjusted = aggregate_krum(weights_results, SharedValue.countMal, SharedValue.countMal)
            final_weights_multi_krum_adjusted: NDArrays = [
            (multi_krum_layer + alpha * mean_layer) / (1 + alpha)
            for multi_krum_layer, mean_layer in zip(weights_multi_krum_adjusted, adjusted_weights)
            ]
            parameters_aggregated = ndarrays_to_parameters(final_weights_multi_krum_adjusted)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided") 
            
        MAX_ROUNDS = SharedValue.trainConfig['max_rounds']
        if (server_round == MAX_ROUNDS):
            last_model.set_weights(fl.common.parameters_to_ndarrays(parameters_aggregated))
            y_pred_class = np.argmax(last_model.predict(x_test),axis=1)
            y_test_class = np.argmax(y_test, axis=1)
            eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
            last_model_result = eva_data['accuracy']
            if last_model_result < 0.7:
                print("Default Algorithm")
                print(last_model_result)
                parameters_aggregated = ndarrays_to_parameters(self.aggregatex(weights_results, client_weights))
                last_model.set_weights(fl.common.parameters_to_ndarrays(parameters_aggregated))
                y_pred_class = np.argmax(last_model.predict(x_test),axis=1)
                y_test_class = np.argmax(y_test, axis=1)
                eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
                last_model_result = eva_data['accuracy']
                print(last_model_result)
                if last_model_result < 0.7:
                    print("Last Algorithm")
                    index = 0
                    exclude = []
                    for weight in clientParam:
                        if SharedValue.trainConfig['fl_training_model_type'] == "ntc_mlp":
                            modeleval = Sequential()
                            modeleval.add(InputLayer(input_shape = (SharedValue.trainConfig['fl_training_model_input_size'],))) # input layer
                            modeleval.add(Dense(32, activation='relu')) # hidden layer 1
                            modeleval.add(Dense(64, activation='relu')) # hidden layer 2
                            modeleval.add(Dense(128, activation='relu')) # hidden layer 3
                            modeleval.add(Dense(10, activation='softmax')) # output layer
                            modeleval.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                        elif SharedValue.trainConfig['fl_training_model_type'] == "ntc_cnn":
                            #Add CNN model here
                            modeleval = Sequential([
                                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                                BatchNormalization(),
                                Conv2D(32, (3, 3), activation='relu', padding='same'),
                                BatchNormalization(),
                                MaxPooling2D(pool_size=(2, 2)),
                                Dropout(0.25),
                                
                                Conv2D(64, (3, 3), activation='relu', padding='same'),
                                BatchNormalization(),
                                Conv2D(64, (3, 3), activation='relu', padding='same'),
                                BatchNormalization(),
                                MaxPooling2D(pool_size=(2, 2)),
                                Dropout(0.25),

                                Flatten(),
                                Dense(128, activation='relu'),
                                BatchNormalization(),
                                Dropout(0.4),
                                Dense(10, activation='softmax')
                            ])
                            modeleval.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                            print('CNN')
                        elif SharedValue.trainConfig['fl_training_model_type'] == "ntc_cnn_cifar10":
                            #Add CNN model here
                            modeleval = Sequential([
                                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                                BatchNormalization(),
                                Conv2D(32, (3, 3), activation='relu', padding='same'),
                                BatchNormalization(),
                                MaxPooling2D(pool_size=(2, 2)),
                                Dropout(0.25),
                                
                                Conv2D(64, (3, 3), activation='relu', padding='same'),
                                BatchNormalization(),
                                Conv2D(64, (3, 3), activation='relu', padding='same'),
                                BatchNormalization(),
                                MaxPooling2D(pool_size=(2, 2)),
                                Dropout(0.25),

                                Flatten(),
                                Dense(128, activation='relu'),
                                BatchNormalization(),
                                Dropout(0.4),
                                Dense(10, activation='softmax')
                            ])
                            modeleval.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                            print('CNN')
                        elif SharedValue.trainConfig['fl_training_model_type'] == "ntc_1dcnn":
                            modeleval = Sequential([
                                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(115, 1)),
                                MaxPooling1D(pool_size=2),
                                Conv1D(filters=128, kernel_size=3, activation='relu'),
                                MaxPooling1D(pool_size=2),
                                Flatten(),
                                Dense(128, activation='relu'),
                                Dropout(0.5),
                                Dense(10, activation='softmax')
                            ])
                            modeleval.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
                            print('1DCNN')
                        modeleval.set_weights(fl.common.parameters_to_ndarrays(weight))
                        y_pred_class = np.argmax(modeleval.predict(x_test),axis=1)
                        y_test_class = np.argmax(y_test, axis=1)
                        eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
                        eval_result = eva_data['accuracy']
                        print(eval_result)
                        if eval_result < 0.6:
                           exclude.append(index) 
                        index += 1
                    print(exclude)
                    parameters_aggregated = ndarrays_to_parameters(self.aggregatexneo(weights_results, exclude))
            model.set_weights(fl.common.parameters_to_ndarrays(parameters_aggregated))
            SharedValue.trainConfig['fl_training_model_name'] = SharedValue.choosen_algo + "_" + SharedValue.trainConfig['fl_training_model_name']
            model.save(SharedValue.trainConfig['fl_training_model_name'])
        SharedValue.modelPerf = []   
        #return agg_weights
        return parameters_aggregated, metrics_aggregated

def fl_aggregator(config, fl_config, retrieveTPMHash, client_count):
    print("This Edge node is selected as local aggregator for domain " + fl_config['client_domain'])
    if config['enable_gpu_training'] == "False":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    strategy = SaveKerasModelStrategy(min_available_clients=client_count, min_fit_clients=client_count, min_evaluate_clients=client_count)
    try:
        with open('local_aggregator_config.json', 'r') as json_file:
            agg_config = json.load(json_file) 
    except:
        print("Please perform remote attestation or domain verification process before starting FL training..end")
        exit()
    now = datetime.now()
    dateTime = now.strftime("%d%m%Y_%H%M%S")
    model_name = "domain_" + fl_config['client_domain'] + "_model_" + dateTime + ".h5"
    
    x_test = np.load(fl_config['local_dataset_x_test_path'])
    y_test = np.load(fl_config['local_dataset_y_test_path'])
    
    ##Remove Source and Destination IP From Dataset
    #print(x_test.shape)
    #print(y_test.shape)
    if agg_config['fl_training_model'] == "ntc_mlp":
        x_test = np.delete(x_test, [12,13,14,15,16,17,18,19], 1)
    #print(x_test.shape)
    #print(y_test.shape)
    SharedValue.tpmAggCache = retrieveTPMHash
    SharedValue.trainConfig = {
    "max_rounds": agg_config['fl_training_round'],
    "training_model_batch_size": agg_config['training_model_batch_size'],
    "training_model_epochs": agg_config['training_model_epochs'],
    "training_learning_rate": agg_config['training_learning_rate'],
    "fl_training_model_name": model_name,
    "fl_training_model_type": agg_config['fl_training_model'],
    "fl_training_model_input_size": x_test.shape[1],
    "training_alpha_value": agg_config['training_alpha_value'],
    "possible_gan_threshold" : agg_config['possible_gan_threshold'],
    "own_id": fl_config['client_id'],
    "client_count": client_count,
    "x_test": fl_config['local_dataset_x_test_path'],
    "y_test": fl_config['local_dataset_y_test_path']
    }
    MAX_ROUNDS = SharedValue.trainConfig['max_rounds']
    address = "0.0.0.0:" + str(agg_config['aggregator_port'])
    if config['enable_flower_ssl'] == "True":
        fl.server.start_server(server_address=address, strategy=strategy, config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS), certificates=(Path(agg_config['aggregator_cert_path']).read_bytes(),Path(agg_config['aggregator_pem_path']).read_bytes(),Path(agg_config['aggregator_key_path']).read_bytes()))
    else:
        fl.server.start_server(server_address=address, strategy=strategy, config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS))
    #Evaluate Domain-Level Model
    model = keras.models.load_model(SharedValue.trainConfig['fl_training_model_name'])
    agg_config['last_domain_model_path'] = SharedValue.trainConfig['fl_training_model_name']
    with open("local_aggregator_config.json", 'w') as file:
        json.dump(agg_config, file, indent=4)
    y_pred_class = np.argmax(model.predict(x_test),axis=1)
    y_test_class = np.argmax(y_test, axis=1)
    print(confusion_matrix(y_test_class, y_pred_class))
    print(classification_report(y_test_class, y_pred_class, digits=4))

def run_command_tpm(command):
    """Function to run a shell command and return the output."""
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout
    
class ntcClientAgg(fl.client.NumPyClient):
    def __init__(self, agg_config, fl_config, retrieveTPMHash):
        self.agg_config = agg_config
        self.fl_config = fl_config
        self.model_name = agg_config['last_domain_model_path']
        self.verification_token = retrieveTPMHash[fl_config['client_id']]['aggt']
        #MLP & CNN Model
        self.aggregator_model = keras.models.load_model(self.model_name, compile=False)
        self.aggregator_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.x_train = np.load(fl_config["local_dataset_x_train_path"])
        self.y_train = np.load(fl_config["local_dataset_y_train_path"])
        self.x_test = np.load(fl_config["local_dataset_x_test_path"])
        self.y_test = np.load(fl_config["local_dataset_y_test_path"])
        
        if self.agg_config['fl_training_model'] == "ntc_mlp":
            ##Remove Source and Destination IP From Dataset
            #print(self.x_train.shape)
            #print(self.y_train.shape)
            self.x_train = np.delete(self.x_train, [12,13,14,15,16,17,18,19], 1)
            self.x_test = np.delete(self.x_test, [12,13,14,15,16,17,18,19], 1)
            #print(self.x_train.shape)
            #print(self.y_train.shape)
        
    def get_parameters(self, config):
        return self.aggregator_model.get_weights()

    def fit(self, parameters, config):
        self.aggregator_model.set_weights(parameters)
        if self.agg_config['boost_hrf_aggregation'] == "True":
            history = self.aggregator_model.fit(self.x_train, self.y_train, epochs=self.agg_config['boost_hrf_epochs'], batch_size=self.agg_config['training_model_batch_size'], shuffle = True)
            return self.aggregator_model.get_weights(), len(self.x_train), {'train_loss':history.history['loss'][0]}
        else:
            return self.aggregator_model.get_weights(), len(self.x_train), {}
        
    def evaluate(self, parameters, config):
        self.aggregator_model.set_weights(parameters)
        loss, accuracy = self.aggregator_model.evaluate(self.x_test, self.y_test)
        y_pred_class = np.argmax(self.aggregator_model.predict(self.x_test),axis=1)
        y_test_class = np.argmax(self.y_test, axis=1)
        print(confusion_matrix(y_test_class, y_pred_class))
        print(classification_report(y_test_class, y_pred_class, digits=4))
        now = datetime.now()
        dateTime = now.strftime("%d%m%Y_%H%M%S")
        save_name = "global_model_" + dateTime + ".h5"
        self.aggregator_model.save(save_name)
        return loss, len(self.x_test), {"accuracy": float(accuracy)}
        
    def get_properties(self, config) -> Dict[str, Scalar]:
        properties = {"client_id": self.fl_config['client_id'], "verification_token": self.verification_token}
        return properties
    
def global_agg(fl_config, config, retrieveTPMHash):
    if config['enable_gpu_training'] == "False":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        with open('local_aggregator_config.json', 'r') as json_file:
            agg_config = json.load(json_file) 
    except:
        print("Please perform remote attestation or domain verification process before starting FL training..end")
        exit()
    if config['enable_flower_ssl'] == "True":
        address = config['server_host'] + ":" + str(config['server_port'])
        fl.client.start_numpy_client(server_address=address, client=ntcClientAgg(agg_config=agg_config, fl_config=fl_config, retrieveTPMHash=retrieveTPMHash), root_certificates=Path(config['server_cert_path']).read_bytes())
    else:
        fl.client.start_numpy_client(server_address="127.0.0.1:" + str(config['server_port']) , client=ntcClientAgg(agg_config=agg_config, fl_config=fl_config, retrieveTPMHash=retrieveTPMHash))

if __name__ == "__main__":
    #Load FL Training Config
    config_path = 'config.json'
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    fl_config_path = config['fl_training_config_path']
    with open(fl_config_path, 'r') as json_file:
        fl_config = json.load(json_file)
    print("Reading hash from TPM...")
    try:
        read_hash = "tpm2_nvread " + fl_config['tpm_index'] + " -C o | xxd -p -r"
        retrieveTPM = run_command_tpm(read_hash)
        retrieveTPMHash = json.loads(retrieveTPM)
    except:
        print("Please perform remote attestation or domain verification process before starting FL training..end")
        exit()
    p = []
    pc = []
    aggClient = 0
    ClientDelegate = 0
    if fl_config['role'] == 'local_aggregator' and "ts" in retrieveTPMHash[fl_config['client_id']]:
        delStat = "False"
        delStatCount = 0
        if config["skip_domain_training"] == "False":
            try:
                with open('trusted_client.json', 'r') as json_file:
                    trusted_client = json.load(json_file)
            except:
                print("Please perform remote attestation or domain verification process before starting FL training..end")
                exit()
            client_count = 0
            for client in trusted_client:
                client_count += 1
                if "delegated" in trusted_client[client]:
                    for delegate in trusted_client[client]["delegated"]:
                        client_count += 1
            x = fl_config['local_dataset_x_train_path']
            y = fl_config['local_dataset_y_train_path']
            client_id = fl_config['client_id']
            aggClient = 1
            status = "local_aggregator"
            p.append(multiprocessing.Process(target=fl_client, args=(config, fl_config, retrieveTPMHash, status, x, y, client_id, delStat, delStatCount)))
            aggProcessCount = 0
            p[aggProcessCount].start()
            if "delegated" in trusted_client[fl_config['client_id']]:
                delStat = "True"
                for delegate in trusted_client[fl_config['client_id']]["delegated"]:
                    delStatCount += 1
                    aggProcessCount += 1
                    status = "local_aggregator_delegate"
                    secure_storage_path = fl_config['home_path'] + "secureStorage/"
                    x = secure_storage_path + trusted_client[fl_config['client_id']]["delegated"][delegate]['local_dataset_x_train_name']
                    y = secure_storage_path + trusted_client[fl_config['client_id']]["delegated"][delegate]['local_dataset_y_train_name']
                    client_id = delegate
                    p.append(multiprocessing.Process(target=fl_client, args=(config, fl_config, retrieveTPMHash, status, x, y, client_id, delStat, delStatCount)))
                    p[aggProcessCount].start()
            fl_aggregator(config, fl_config, retrieveTPMHash, client_count)
            if aggClient == 1:
                c = 0
                for process in p:
                    p[c].join()
                    c += 1
        global_agg(fl_config, config, retrieveTPMHash)
    elif fl_config['role'] == 'valid_client' and "x" not in retrieveTPMHash :
        clientDelegateProcessCount = 0
        delStat = "False"
        delStatCount = 0
        if "delegated" in fl_config:
            delStat = "True"
            ClientDelegate = 1
            for delegate in fl_config["delegated"]:
                delStatCount += 1
                status = "client_delegate"
                delegate_storage_path = fl_config['home_path'] + "delegateStorage/"
                client_id = delegate
                x = delegate_storage_path + fl_config["delegated"][delegate]['local_dataset_x_train_name']
                y = delegate_storage_path + fl_config["delegated"][delegate]['local_dataset_y_train_name']
                pc.append(multiprocessing.Process(target=fl_client, args=(config, fl_config, retrieveTPMHash, status, x, y, client_id, delStat, delStatCount)))
                pc[clientDelegateProcessCount].start()
                clientDelegateProcessCount += 1
        status = "client"
        x = fl_config['local_dataset_x_train_path']
        y = fl_config['local_dataset_y_train_path']
        client_id = fl_config['client_id']
        delStat = "False"
        fl_client(config, fl_config, retrieveTPMHash, status, x, y, client_id, delStat, delStatCount)
        if ClientDelegate == 1:
            c = 0
            for process in pc:
                pc[c].join()
                c += 1
    else: 
        print("Please perform remote attestation or domain verification process before starting FL training..end")