#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Secure FL-Based NTC - Global Server Remote Attestator and Aggregator Program

import pickle
import json
from twisted.internet import reactor, protocol, ssl, defer
from twisted.protocols.basic import LineReceiver
from twisted.internet.protocol import Protocol, Factory
import sys
import os
from twisted.python import log
import subprocess
import re
import flwr as fl
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
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
from multiprocessing import Process
import psutil
import time as timex

config_pathx = 'config.json'
with open(config_pathx, 'r') as json_file:
    configx = json.load(json_file)
experiment_name = configx["experiment_name"]

def log_cpu_memory_usage():
    with open('resource_' + experiment_name + '.txt', 'a') as f:
        try:
            while True:
                # Get CPU and memory usage
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                memory_usage = memory_info.percent

                # Log to file
                f.write(f"CPU Usage: {cpu_usage}%\n")
                f.write(f"Memory Usage: {memory_usage}%\n")
                f.write("-" * 30 + "\n")

                # Optional: print to console
                print(f"CPU Usage: {cpu_usage}% | Memory Usage: {memory_usage}%")

                # Wait for 5 seconds before checking again
                timex.sleep(5)
        except KeyboardInterrupt:
            print("Logging stopped.")
            f.close()
    f.close()

class SharedValue:
    valid_clients = {}
    countTerminate = 0
    model_name = ""
    fl_config = {}
    tpmCache = {}
    aggCount = 0

class CustomProtocol(Protocol):
    def __init__(self, factory, config, predef_client):
        self.factory = factory
        self.data = b''
        self.client_id = None
        self.client_domain = None
        self.client_ip = None
        self.port = None
        self.attestation_client_program_hash = None
        self.config = config
        self.predef_client = predef_client
        hash_command = f"openssl dgst -sha1 -engine dynamic {config['sample_attestation_program_path']}"
        self.sample_attestation_program_hash = self.run_command_hash(hash_command)
        
    def dataReceived(self, data):
        self.data += data
        peer = self.transport.getPeer()
        if self.data.endswith(b'END'):
            self.data = self.data[:-3]  # Remove the 'END' marker
            data_dict = pickle.loads(self.data)
            self.client_id = data_dict['client_id']
            hash_command = f"openssl dgst -sha1 -engine dynamic {self.predef_client[self.client_id]['client_cert_path']}"
            client_cert_hash_pre = self.run_command_hash(hash_command)
            self.client_domain = data_dict['client_domain']
            self.attestation_client_program_hash = data_dict['ap']['attestation_program_sha256']
            self.client_cert_hash = data_dict['ap']['client_cert_sha256']
            self.factory.client_protocols[self.client_id] = self
            client_protocol = self.factory.client_protocols[self.client_id]
            if peer.host == self.predef_client[self.client_id]["client_ip"] and peer.port == self.predef_client[self.client_id]["client_source_port"] and self.client_domain == self.predef_client[self.client_id]["client_domain"] and self.attestation_client_program_hash == self.sample_attestation_program_hash and self.client_cert_hash == client_cert_hash_pre:
                print("Authorized connection, from " + self.client_id )
                self.factory.received_data.append(data_dict)
                self.save_test_model(data_dict['ap'])
                data_dict["ap"]["test_model_file"] = ""
                SharedValue.valid_clients[self.client_domain][self.client_id] = {"ap": data_dict["ap"]}
                if self.predef_client[self.client_id]["test_model_accuracy"] == "":
                    model_accuracy = self.evaluate_test_model(data_dict['ap']['test_model_name'], self.config)
                    SharedValue.valid_clients[self.client_domain][self.client_id]['test_model_accuracy'] = model_accuracy
                else:
                    SharedValue.valid_clients[self.client_domain][self.client_id]['test_model_accuracy'] = self.predef_client[self.client_id]["test_model_accuracy"] #Only for development, must be empty from predefined client list
                process_count, port_count = self.evaluate_verification_list(data_dict["ap"]["verification_list"], data_dict['ap']['backdoorStatus'])
                SharedValue.valid_clients[self.client_domain][self.client_id]['process_count'] = process_count
                SharedValue.valid_clients[self.client_domain][self.client_id]['port_count'] = port_count
                SharedValue.valid_clients[self.client_domain][self.client_id]['trust_score'] = 0
                SharedValue.valid_clients[self.client_domain][self.client_id]['status'] = "Undefined"
                SharedValue.valid_clients[self.client_domain][self.client_id]['client_ip'] = peer.host
                SharedValue.valid_clients[self.client_domain][self.client_id]['client_port'] = peer.port
                SharedValue.valid_clients[self.client_domain][self.client_id]['client_host'] = data_dict['ap']['client_host']
                SharedValue.valid_clients[self.client_domain][self.client_id]['backdoorStatus'] = data_dict['ap']['backdoorStatus']
            else :
                print("Unauthorized connection, removing edge client...")
                client_protocol.transport.loseConnection()
            self.data = b''
        if len(self.factory.received_data) == len(self.predef_client):
            self.process_data(self.config)
            
    def save_test_model(self, ap):
        file_data = ap["test_model_file"]
        file_path = ap["test_model_name"]
        with open(file_path, 'wb') as file:
            file.write(file_data)
        log.msg(f"Saved test model file as {file_path}")
        
    def evaluate_test_model(self, model_name, config):
        if config['enable_gpu_training'] == "False":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        #Deep Learning logic
        print(f"Evaluating {model_name}..")
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        import numpy as np
        from tensorflow import keras
        model = keras.models.load_model(model_name)
        x_test = np.load(config["server_x_test_path"])
        y_test = np.load(config["server_y_test_path"])
        if config['fl_training_model'] == "ntc_mlp":
            #Remove IP Address
            x_test = np.delete(x_test, [12,13,14,15,16,17,18,19], 1)
        y_pred_class = np.argmax(model.predict(x_test),axis=1)
        y_test_class = np.argmax(y_test, axis=1)
        eva_data = classification_report(y_test_class, y_pred_class, digits=4, output_dict=True)
        print(eva_data['accuracy'])
        if config['show_classification_report'] == "True":
            #print(confusion_matrix(y_test_class, y_pred_class))
            print(classification_report(y_test_class, y_pred_class, digits=4))
        return eva_data['accuracy']
        
    def evaluate_verification_list(self, verification_list, backdoorStatus):
        #Can put further scrutinize on malicious process and port
        print(f"Evaluating {verification_list['client_id']} verification list")
        process_count = len(verification_list["process_list"])
        port_count = len(verification_list["port_list"])
        if backdoorStatus == "True":
            print("Potential backdoor detected")
        elif backdoorStatus == "False":
            print("No potential backdoor detected")
        return process_count, port_count
        
    def process_data(self, config):
        hash_command = f"openssl dgst -sha1 -engine dynamic {config['sample_FL_training_program_path']}"
        sample_FL_program_hash = self.run_command_hash(hash_command)
        hash_command = f"openssl dgst -sha1 -engine dynamic {config['sample_domain_verification_program_path']}"
        sample_domain_verification_program_hash = self.run_command_hash(hash_command)
        #print(sample_FL_program_hash)
        untrusted_client = {}
        for domain in SharedValue.valid_clients:
            print("Performing attestation scoring for domain " + domain)
            proc_list = []
            port_list = []
            model_list = []
            untrusted_client_list = []
            for client in SharedValue.valid_clients[domain]:
                proc_list.append(SharedValue.valid_clients[domain][client]['process_count'])
                port_list.append(SharedValue.valid_clients[domain][client]['port_count'])
                model_list.append(SharedValue.valid_clients[domain][client]['test_model_accuracy'])
            proc_min = min(proc_list)
            port_min = min(port_list)
            model_max = max(model_list)
            for client in SharedValue.valid_clients[domain]:
                #print(SharedValue.valid_clients[domain][client]['ap']['FL_program_sha256'])
                #print(SharedValue.valid_clients[domain][client]['process_count'])
                #print(SharedValue.valid_clients[domain][client]['port_count'])
                #print(SharedValue.valid_clients[domain][client]['test_model_accuracy'])
                #Examine FL Client Training Program Hash
                if SharedValue.valid_clients[domain][client]['ap']['FL_program_sha256'] == sample_FL_program_hash:
                    SharedValue.valid_clients[domain][client]['trust_score'] = SharedValue.valid_clients[domain][client]['trust_score'] + 15
                else:
                    untrusted_client_list.append(client)
                #Examine The Client Verification List
                if SharedValue.valid_clients[domain][client]['process_count'] == proc_min:
                    SharedValue.valid_clients[domain][client]['trust_score'] = SharedValue.valid_clients[domain][client]['trust_score'] + 5
                if SharedValue.valid_clients[domain][client]['port_count'] == port_min:
                    SharedValue.valid_clients[domain][client]['trust_score'] = SharedValue.valid_clients[domain][client]['trust_score'] + 5
                #Examine Backdoor Potential
                if SharedValue.valid_clients[domain][client]['backdoorStatus'] == "False":
                    SharedValue.valid_clients[domain][client]['trust_score'] = SharedValue.valid_clients[domain][client]['trust_score'] + 5
                #Examine The Client Test Model
                if SharedValue.valid_clients[domain][client]['test_model_accuracy'] == model_max:
                    SharedValue.valid_clients[domain][client]['trust_score'] = SharedValue.valid_clients[domain][client]['trust_score'] + 10
                elif SharedValue.valid_clients[domain][client]['test_model_accuracy'] >= 0.6:
                    SharedValue.valid_clients[domain][client]['trust_score'] = SharedValue.valid_clients[domain][client]['trust_score'] + 5
                #print(SharedValue.valid_clients[domain][client]['trust_score'])
            untrusted_client[domain] = untrusted_client_list
        #Client Selection Process
        #print(untrusted_client)
        for domain in SharedValue.valid_clients:
            print("Performing local aggregator selection for domain " + domain)
            score_list = []
            for client in SharedValue.valid_clients[domain]:
                score_list.append(SharedValue.valid_clients[domain][client]['trust_score'])
            trust_max = max(score_list)
            selected = 0
            choosen_aggregator = ""
            for client in SharedValue.valid_clients[domain]:
                if SharedValue.valid_clients[domain][client]['trust_score'] == trust_max and selected == 0:
                    selected = 1
                    choosen_aggregator = client
                    print(f"{client} selected as local aggregator for domain {domain} (Trust Score: {SharedValue.valid_clients[domain][client]['trust_score']})")
                    SharedValue.valid_clients[domain][client]['status'] = "local_aggregator"
                    hash_command = f"openssl rand -engine dynamic -hex 12"
                    aggregator_token = self.run_command_token(hash_command)
                    SharedValue.valid_clients[domain][client]['aggregator_token'] = aggregator_token
                else:
                    if client in untrusted_client[domain]:
                        print(f"{client} become untrusted edge client for domain {domain} (Trust Score: {SharedValue.valid_clients[domain][client]['trust_score']})")
                        SharedValue.valid_clients[domain][client]['status'] = "verified_edge_client_untrusted"
                    else:
                        print(f"{client} become edge client for domain {domain} (Trust Score: {SharedValue.valid_clients[domain][client]['trust_score']})")
                        SharedValue.valid_clients[domain][client]['status'] = "verified_edge_client"
                    hash_command = f"openssl rand -engine dynamic -hex 10"
                    verification_token = self.run_command_token(hash_command)
                    SharedValue.valid_clients[domain][client]['node_verification_token'] = verification_token
        for domain in SharedValue.valid_clients:
            selected = 0
            choosen_aggregator = ""
            for client in SharedValue.valid_clients[domain]:
                if SharedValue.valid_clients[domain][client]['status'] == "local_aggregator" and selected == 0:
                    selected = 1
                    choosen_aggregator = client
                    aggregator_token = SharedValue.valid_clients[domain][client]['aggregator_token']
                    self.send_response_aggregator(domain, client, aggregator_token, sample_FL_program_hash, sample_domain_verification_program_hash, self.config, untrusted_client)
                elif SharedValue.valid_clients[domain][client]['status'] == "verified_edge_client_untrusted":
                    verification_token = SharedValue.valid_clients[domain][client]['node_verification_token']
                    self.send_response_edge_client(domain, client, verification_token, choosen_aggregator, self.config, "True")
                else:
                    verification_token = SharedValue.valid_clients[domain][client]['node_verification_token']
                    self.send_response_edge_client(domain, client, verification_token, choosen_aggregator, self.config, "False")
        with open(config["valid_client_list_path"], 'w') as file:
            json.dump(SharedValue.valid_clients, file, indent=4)
        self.storing_hashes(config)
        print("Remote attestation process complete, Bye")
        
    def storing_hashes(self, config):
        print(f"Storing Hash to TPM to {config['tpm_index']} index")
        
        #CheckTPM whether has been initialized
        tpmIndex = config["tpm_index"]
        checktpm = "tpm2_nvreadpublic " + tpmIndex
        print("Checking TPM NV Index...")
        check = self.run_command_tpm(checktpm)
        if "friendly" in check:
            #Undefined NV
            print("Resetting the TPM NV Index..." + tpmIndex)
            undefinedtpm = "tpm2_nvundefine " + tpmIndex + " -C o"
            check = self.run_command_tpm(undefinedtpm)
        
        store_hashes = {}
        for domain in SharedValue.valid_clients:
            for client in SharedValue.valid_clients[domain]:
                store_hashes[client] = {}
                if SharedValue.valid_clients[domain][client]['status'] == "local_aggregator":
                    store_hashes[client]["x"] = SharedValue.valid_clients[domain][client]["ap"]["local_dataset_x_train_sha256"]
                    store_hashes[client]["y"] = SharedValue.valid_clients[domain][client]["ap"]["local_dataset_y_train_sha256"]
                    store_hashes[client]["aggt"] = SharedValue.valid_clients[domain][client]["aggregator_token"]
                else:
                    store_hashes[client]["x"] = SharedValue.valid_clients[domain][client]["ap"]["local_dataset_x_train_sha256"]
                    store_hashes[client]["y"] = SharedValue.valid_clients[domain][client]["ap"]["local_dataset_y_train_sha256"]
                    store_hashes[client]["vt"] = SharedValue.valid_clients[domain][client]["node_verification_token"]
        store_hashes_str = json.dumps(store_hashes)
        #Store Hashes in TPM Module
        # Define NV Index for storing SHA hash
        tpmsize = "2028"
        define_nv = "tpm2_nvdefine " + tpmIndex + " -C o -s " + tpmsize + " -a 'ownerread|ownerwrite'"
        print("Defining TPM NV Index...")
        print(self.run_command_tpm(define_nv))
        # write to TPM
        write_hash = "echo -n '" + store_hashes_str + "' | xxd -p | tr -d '\n' | tpm2_nvwrite " + tpmIndex + " -C o -i -"
        print("Writing hash to TPM...")
        print(self.run_command_tpm(write_hash))
        
    def send_response_aggregator(self, domain, client, aggregator_token, sample_FL_program_hash, sample_domain_verification_program_hash, config, untrusted_client):
        print(f"Sending aggregator token to {client}")
        client_protocol = self.factory.client_protocols[client]
        clientDomain = SharedValue.valid_clients[domain]
        response_dict = {
            'type': "aggregatorResponse",
            'aggregator_token': aggregator_token,
            'sample_FL_program_hash': sample_FL_program_hash,
            'sample_domain_verification_program_hash': sample_domain_verification_program_hash,
            'training_model_batch_size': config['training_model_batch_size'],
            'training_model_epochs': config['training_model_epochs'],
            'training_learning_rate': config['training_learning_rate'],
            'fl_training_round': config['fl_training_round'],
            'fl_training_model': config['fl_training_model'],
            'boost_hrf_aggregation': config['boost_hrf_aggregation'],
            'boost_hrf_epochs': config['boost_hrf_epochs'],
            'training_alpha_value': config['training_alpha_value'],
            'backdoor_pattern_threshold': config['backdoor_pattern_threshold'],
            'possible_gan_threshold': config['possible_gan_threshold'],
            'client_list': clientDomain,
            'untrusted_client_list': untrusted_client[domain],
            'upload_request': "False"
        }
        serialized_data = pickle.dumps(response_dict) + b'END'
        client_protocol.transport.write(serialized_data)
        
    def send_response_edge_client(self, domain, client, verification_token, choosen_aggregator, config, uploadRequest):
        if uploadRequest == "True":
            print(f"Sending node verification token & Dataset Upload Request to {client}. (Aggregator: {choosen_aggregator})")
        else:
            print(f"Sending node verification token to {client}. (Aggregator: {choosen_aggregator})")
        client_protocol = self.factory.client_protocols[client]
        with open(self.predef_client[choosen_aggregator]['client_cert_path'], 'rb') as file:
            local_aggregator_cert_file = file.read()
        response_dict = {
            'type': "clientResponse",
            'verification_token': verification_token,
            'local_aggregator_ip': SharedValue.valid_clients[domain][choosen_aggregator]['client_ip'],
            'local_aggregator_port': SharedValue.valid_clients[domain][choosen_aggregator]['client_port'],
            'local_aggregator_host': SharedValue.valid_clients[domain][choosen_aggregator]['client_host'],
            'local_aggregator_cert_path': self.predef_client[choosen_aggregator]['client_cert_path'],
            'local_aggregator_cert_file': local_aggregator_cert_file,
            'fl_training_model': config['fl_training_model'],
            'upload_request': uploadRequest
        }
        serialized_data = pickle.dumps(response_dict) + b'END'
        client_protocol.transport.write(serialized_data)

    def connectionMade(self):
        peer = self.transport.getPeer()
        print(f"Connection made from Edge Client {peer.host}:{peer.port}")

    def connectionLost(self, reason):
        if self.client_id in self.factory.client_protocols:
            SharedValue.countTerminate = SharedValue.countTerminate + 1
            print(f"{self.client_id} is terminating attestation process")
            del self.factory.client_protocols[self.client_id]
            if SharedValue.countTerminate == len(self.predef_client):
                print("All attestation clients have terminate connection..")
                reactor.stop()
    
    def run_command_hash(self, command):
        """Function to run a TPM shell command, extract hash, and return the output."""
        try:
            result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
            # Use regex to extract the hexadecimal hash value
            hash_value = re.search(r"=\s*([a-f0-9]+)", result.stdout)
            if hash_value:
                return hash_value.group(1)
            else:
                return "No hash found"
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.stderr.strip()}")
            return None
            
    def run_command_token(self, command):
        try:
            result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
            lines = result.stdout.splitlines()
            return lines[0]
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.stderr.strip()}")
            return None
            
    def run_command_tpm(self, command):
        """Function to run a shell command and return the output."""
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.stdout

class CustomFactory(Factory):
    def __init__(self, config, predef_client):
        self.received_data = []
        self.client_protocols = {}
        self.config = config
        self.predef_client = predef_client

    def buildProtocol(self, addr):
        return CustomProtocol(self, self.config, self.predef_client)

class SaveKerasModelStrategy(fl.server.strategy.FedAvg):
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Client selection and Configure the next round of training."""
        print("Waiting for " + str(SharedValue.aggCount) + " Local Aggregators")
        client_manager.wait_for(num_clients  =  SharedValue.aggCount)
        client_properties = {}
        standard_config = {}
        fit_configurations = []
        for cid, client in client_manager.all().items():
            ins = GetPropertiesIns({})
            client_properties[cid] = client.get_properties(ins, timeout = 30)
            prop = client_properties[cid].properties
            if prop['verification_token'] == SharedValue.tpmCache[prop['client_id']]['aggt']:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                print(f"Unauthorized connection from {prop['client_id']} ")
        return fit_configurations
        
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        client_properties = {}
        standard_config = {}
        eval_configurations = []
        for cid, client in client_manager.all().items():
            ins = GetPropertiesIns({})
            client_properties[cid] = client.get_properties(ins, timeout = 30)
            prop = client_properties[cid].properties
            if prop['verification_token'] == SharedValue.tpmCache[prop['client_id']]['aggt']:
                eval_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                print(f"Unauthorized connection from {prop['client_id']} ")
        return eval_configurations
        
    def aggregate_fit(self, server_round, results, failures):
        agg_weights = super().aggregate_fit(server_round, results, failures)
        MAX_ROUNDS = 1
        model_name = SharedValue.model_name
        if SharedValue.fl_config['fl_training_model'] == "ntc_mlp":
            model = Sequential()
            #model.add(InputLayer(input_shape = (740,))) # input layer
            model.add(InputLayer(input_shape = (732,))) # input layer
            model.add(Dense(32, activation='relu')) # hidden layer 1
            model.add(Dense(64, activation='relu')) # hidden layer 2
            model.add(Dense(128, activation='relu')) # hidden layer 3
            model.add(Dense(10, activation='softmax')) # output layer
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif SharedValue.fl_config['fl_training_model'] == "ntc_cnn":
            #Add CNN model
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
        elif SharedValue.fl_config['fl_training_model'] == "ntc_cnn_cifar10":
            #Add CNN model for cifar10
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
        elif SharedValue.fl_config['fl_training_model'] == "ntc_1dcnn":
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
        if (server_round == MAX_ROUNDS):
            model.set_weights(fl.common.parameters_to_ndarrays(agg_weights[0]))
            model.save(model_name)
        return agg_weights

def run_command_tpm(command):
    """Function to run a shell command and return the output."""
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def globalAggregator(config):
    print("Running Global Aggregator Server for FL Network...")
    if config['enable_gpu_training'] == "False":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        read_hash = "tpm2_nvread " + config['tpm_index'] + " -C o | xxd -p -r"
        retrieveTPM = run_command_tpm(read_hash)
        retrieveTPMHash = json.loads(retrieveTPM)
    except:
        print("Please perform remote attestation before starting FL training..end")
        exit()
    SharedValue.tpmCache = retrieveTPMHash
    SharedValue.fl_config = config
    now = datetime.now()
    dateTime = now.strftime("%d%m%Y_%H%M%S")
    model_name = "global_model_" + config['scenario'] + "_" + dateTime + ".h5"
    SharedValue.model_name = model_name
    try:
        with open(config['valid_client_list_path'], 'r') as json_file:
            client_list = json.load(json_file)
    except:
        print("Please perform remote attestation process..")
    MAX_ROUNDS = 1
    address = "0.0.0.0:" + str(config['server_port'])
    aggCount = 0
    for domain in client_list:
        aggCount += 1
    SharedValue.aggCount = aggCount
    strategy = SaveKerasModelStrategy(min_available_clients=aggCount, min_fit_clients=aggCount, min_evaluate_clients=aggCount)
    
    #Begin counting Time
    startTime = timex.time()
    #monitor_process = Process(target=log_cpu_memory_usage)
    #monitor_process.start()
    
    if config['enable_flower_ssl'] == "True":
        fl.server.start_server(server_address=address, strategy=strategy, config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS), certificates=(Path(config['server_cert_path']).read_bytes(),Path(config['server_pem_path']).read_bytes(),Path(config['server_key_path']).read_bytes()))
    else:
        fl.server.start_server(server_address=address, strategy=strategy, config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS))
    
    #End couting time
    executionTime = (timex.time() - startTime)
    executionTime = executionTime / 60
    print('Execution time in minutes: ' + str(executionTime))

    #monitor_process.terminate()
    
    #Confusion Matrix
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import numpy as np
    from tensorflow import keras
    
    # Load the classes Map from the file
    #with open('classes.pkl', 'rb') as file:
    #    loaded_classes = pickle.load(file)
    
    model = keras.models.load_model(model_name)

    x_test = np.load(config['server_x_test_path'])
    y_test = np.load(config['server_y_test_path'])
    
    if config['fl_training_model'] == "ntc_mlp":
        x_test = np.delete(x_test, [12,13,14,15,16,17,18,19], 1)

    y_pred_class = np.argmax(model.predict(x_test),axis=1)
    y_test_class = np.argmax(y_test, axis=1)
    print(confusion_matrix(y_test_class, y_pred_class))
    #print(classification_report(y_test_class, y_pred_class, target_names=loaded_classes, digits=4))
    #report = classification_report(y_test_class, y_pred_class, target_names=loaded_classes, digits=4)
    print(classification_report(y_test_class, y_pred_class, digits=4))
    report = classification_report(y_test_class, y_pred_class, digits=4)
    
    # Save the execution time to a file
    with open('results_' + experiment_name + '.txt', 'w') as f:
        f.write('Execution time in minutes: ' + str(executionTime) + '\n')
        f.write("\n\n")
        f.write("Classification Report:\n")
        f.write(report)

def main():
    #Load Global Attestator Config
    log.startLogging(sys.stdout)
    config_path = 'config.json'
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    if config["skip_attestation"] == "False":
        #Load Predefined Client List
        client_list_path = config["predefined_client_list_path"]
        with open(client_list_path, 'r') as json_file:
            predef_client = json.load(json_file)
        
        port = config["server_port"]
        ip_address = config["server_ip"]
        
        tmpdomain = []
        for client in predef_client:
            if predef_client[client]["client_domain"] not in tmpdomain:
                tmpdomain.append(predef_client[client]["client_domain"])
                SharedValue.valid_clients[predef_client[client]["client_domain"]] = {}
        
        factory = CustomFactory(config, predef_client)
        contextFactory = ssl.DefaultOpenSSLContextFactory(config["server_key_path"], config["server_cert_path"])
        reactor.listenSSL(port, factory, contextFactory, interface=ip_address)
        print(f"Server is listening on IP {ip_address} and port {port} with SSL...")
        reactor.run()
    globalAggregator(config)

if __name__ == "__main__":
    main()