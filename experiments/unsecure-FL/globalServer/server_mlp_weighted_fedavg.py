#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: FL Server Program for ISCX-VPN 2016 MLP Weighted FedAvg

import flwr as fl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
import time as timex
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import numpy as np
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from functools import reduce
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
)

MAX_ROUNDS = 3
model_name = "clf1_global_model_weighted_fedavg_mlp_6client_36epochs_3round_cpu.h5"

model = Sequential()
model.add(InputLayer(input_shape = (732,))) # input layer
model.add(Dense(6, activation='relu')) # hidden layer 1
model.add(Dense(6, activation='relu')) # hidden layer 2
model.add(Dense(10, activation='softmax')) # output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
        
    def get_client_weight(self, client_id: str) -> float:
        """Determine the weight for a client based on its custom ID."""
        # Implement your logic here to return the weight based on the client ID
        if client_id == "client_3":
            return 0.2
        #elif client_id == "client_2":
        #    return 0.0666666666666667
        #elif client_id == "client_6":
        #    return 0.1
        #elif client_id == "client_5":
        #   return 0.05
        else:
            return 0.16#
    
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
            
        client_weights = []
        for client, fit_res in results:
            # Retrieve custom client ID from the metrics or other means
            custom_id = fit_res.metrics.get("cid")
            # Determine the weight for this client based on the custom ID
            client_weight = self.get_client_weight(custom_id)
            #print(custom_id)
            client_weights.append(client_weight)
            
        #print(client_weights)

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(self.aggregatex(weights_results, client_weights))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")            

        if (server_round == MAX_ROUNDS):
            
            model.set_weights(fl.common.parameters_to_ndarrays(parameters_aggregated))
            model.save(model_name)

        #return agg_weights
        return parameters_aggregated, metrics_aggregated
        

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
