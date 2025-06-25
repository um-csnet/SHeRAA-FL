#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Secure FL-Based NTC - Edge Client Remote Attestation Program

import json
import pickle
from twisted.internet import reactor, ssl
from twisted.internet.protocol import Protocol, ClientFactory
from twisted.python import log
import sys
import os
import numpy as np
import pandas as pd
import subprocess
import re
from OpenSSL import SSL

log.startLogging(sys.stdout)

class CustomClient(Protocol):
    def __init__(self, client_id, config, ap):
        self.client_id = client_id
        self.config = config
        self.ap = ap
        self.received_data = b''

    def connectionMade(self):
        print('Sending Attestation Parameters to Global Attestator Server..')
        data_dict = {
            'client_id': self.client_id,
            'client_domain': self.config['client_domain'],
            'ap': self.ap
        }
        serialized_data = pickle.dumps(data_dict) + b'END'
        self.transport.write(serialized_data)

    def dataReceived(self, data):
        self.received_data += data
        print('Received Response From Global Attestator Server..')
        if self.received_data.endswith(b'END'):
            self.received_data = self.received_data[:-3]  # Remove the 'END' marker
            response_dict = pickle.loads(self.received_data)
            if response_dict['type'] == "aggregatorResponse":
                print('This Edge Client is selected as Local Aggregator for Domain ' + self.config['client_domain'])
                print('Storing Valid Client and FL Training Information')
                aggregator_info = {
                'role': 'local_aggregator',
                'aggregator_token': response_dict['aggregator_token'],
                'sample_FL_program_hash': response_dict['sample_FL_program_hash'],
                'sample_domain_verification_program_hash': response_dict['sample_domain_verification_program_hash'],
                'aggregator_ip': self.config['client_ip'],
                'aggregator_port': self.config['client_source_port'],
                'aggregator_key_path': self.config['client_key_path'],
                'aggregator_cert_path': self.config['client_cert_path'],
                'aggregator_pem_path': self.config['client_pem_path'],
                'training_model_batch_size': response_dict['training_model_batch_size'],
                'training_model_epochs': response_dict['training_model_epochs'],
                'training_learning_rate': response_dict['training_learning_rate'],
                'fl_training_round': response_dict['fl_training_round'],
                'fl_training_model': response_dict['fl_training_model'],
                'verify_domain_verification_program': "False",
                'verify_both_dataset': "False",
                'boost_hrf_aggregation': response_dict['boost_hrf_aggregation'],
                'boost_hrf_epochs': response_dict['boost_hrf_epochs'],
                'training_alpha_value': response_dict['training_alpha_value'],
                'backdoor_pattern_threshold': response_dict['backdoor_pattern_threshold'],
                'possible_gan_threshold': response_dict['possible_gan_threshold'],
                'client_list': response_dict['client_list'],
                'untrusted_client_list': response_dict['untrusted_client_list']
                }
                training_info = {}
                training_info['role'] = "local_aggregator"
                training_info['verification_token'] = response_dict['aggregator_token']
                training_info['local_aggregator_ip'] = self.config['client_ip']
                training_info['local_aggregator_port'] = self.config['client_source_port']
                training_info['local_aggregator_cert_path'] = self.config['client_cert_path']
                training_info['local_aggregator_host'] = self.config['client_host']
                training_info['client_id'] = self.config['client_id']
                training_info['client_domain'] = self.config['client_domain']
                training_info['local_dataset_x_train_path'] = self.config['local_dataset_x_train_path']
                training_info['local_dataset_y_train_path'] = self.config['local_dataset_y_train_path']
                training_info['local_dataset_x_test_path'] = self.config['local_dataset_x_test_path']
                training_info['local_dataset_y_test_path'] = self.config['local_dataset_y_test_path']
                training_info['FL_program_path'] = self.config['FL_program_path']
                training_info['domain_verification_program_path'] = 'domain_verification_' + self.config['client_id'] +'.py'
                training_info['tpm_index'] = self.config['tpm_index']
                training_info['recalculate_hash'] = "False"
                training_info['home_path'] = self.config['home_path']
                training_info['fl_training_model'] = response_dict['fl_training_model']
                training_info['upload_request'] = response_dict['upload_request']
                with open("local_aggregator_config.json", 'w') as file:
                    json.dump(aggregator_info, file, indent=4)
                with open(self.config["fl_training_config_path"], 'w') as file:
                    json.dump(training_info, file, indent=4)
                self.storing_hashes_aggregator(self.config, response_dict['client_list'], response_dict['aggregator_token'], response_dict['sample_FL_program_hash'], response_dict['sample_domain_verification_program_hash'])
            elif response_dict['type'] == "clientResponse":
                training_info = {}
                if response_dict['upload_request'] == "False":
                    print('This Edge Client has been verified to participate in FL Training for Domain ' + self.config['client_domain'])
                    training_info['role'] = "valid_client"
                elif response_dict['upload_request'] == "True":
                    print('This Edge Client has been verified and requires to upload datasets to local aggregator ' + self.config['client_domain'])
                    training_info['role'] = "valid_client_untrusted"
                print('Storing Attestation Information')
                with open(response_dict['local_aggregator_cert_path'], 'wb') as file:
                    file.write(response_dict['local_aggregator_cert_file'])
                print(f"Saving Local Aggregator Public Certificate {response_dict['local_aggregator_cert_path']}")
                training_info['verification_token'] = response_dict['verification_token']
                training_info['local_aggregator_ip'] = response_dict['local_aggregator_ip']
                training_info['local_aggregator_port'] = response_dict['local_aggregator_port']
                training_info['local_aggregator_cert_path'] = response_dict['local_aggregator_cert_path']
                training_info['local_aggregator_host'] = response_dict['local_aggregator_host']
                training_info['client_id'] = self.config['client_id']
                training_info['client_domain'] = self.config['client_domain']
                training_info['local_dataset_x_train_path'] = self.config['local_dataset_x_train_path']
                training_info['local_dataset_y_train_path'] = self.config['local_dataset_y_train_path']
                training_info['local_dataset_x_test_path'] = self.config['local_dataset_x_test_path']
                training_info['local_dataset_y_test_path'] = self.config['local_dataset_y_test_path']
                training_info['FL_program_path'] = self.config['FL_program_path']
                training_info['domain_verification_program_path'] = 'domain_verification_' + self.config['client_id'] +'.py'
                training_info['tpm_index'] = self.config['tpm_index']
                training_info['recalculate_hash'] = "False"
                training_info['home_path'] = self.config['home_path']
                training_info['fl_training_model'] = response_dict['fl_training_model']
                training_info['upload_request'] = response_dict['upload_request']
                with open(self.config["fl_training_config_path"], 'w') as file:
                    json.dump(training_info, file, indent=4)
                self.storing_hashes_client(self.config, self.ap, response_dict['verification_token'])
            else:
                print('Unknown Response from Global Aggregator')
        self.transport.loseConnection()

    def storing_hashes_aggregator(self, config, client_list, aggregator_token, sample_FL_program_hash, sample_domain_verification_program_hash):
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
        for client in client_list:
            store_hashes[client] = {}
            if client_list[client]['status'] == "local_aggregator":
                store_hashes[client]['flp'] = sample_FL_program_hash
                store_hashes[client]['dv'] = sample_domain_verification_program_hash
                store_hashes[client]['x'] = client_list[client]['ap']['local_dataset_x_train_sha256']
                store_hashes[client]['y'] = client_list[client]['ap']['local_dataset_y_train_sha256']
                store_hashes[client]['aggt'] = client_list[client]['aggregator_token']
            else:
                store_hashes[client]['x'] = client_list[client]['ap']['local_dataset_x_train_sha256']
                store_hashes[client]['y'] = client_list[client]['ap']['local_dataset_y_train_sha256']
                store_hashes[client]['vt'] = client_list[client]['node_verification_token']
        store_hashes_str = json.dumps(store_hashes)
        #Store Hashes in TPM Module
        # Define NV Index for storing SHA hash
        tpmsize = "1448"
        define_nv = "tpm2_nvdefine " + tpmIndex + " -C o -s " + tpmsize + " -a 'ownerread|ownerwrite'"
        print("Defining TPM NV Index...")
        print(self.run_command_tpm(define_nv))
        # write to TPM
        write_hash = "echo -n '" + store_hashes_str + "' | xxd -p | tr -d '\n' | tpm2_nvwrite " + tpmIndex + " -C o -i -"
        print("Writing hash to TPM...")
        print(self.run_command_tpm(write_hash))
        
    def storing_hashes_client(self, config, ap, verification_token):
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
        store_hashes = {
        "x": ap["local_dataset_x_train_sha256"],
        "y": ap["local_dataset_y_train_sha256"],
        "vt": verification_token
        }
        store_hashes_str = json.dumps(store_hashes)
        #Store Hashes in TPM Module
        # Define NV Index for storing SHA hash
        tpmsize = "256"
        define_nv = "tpm2_nvdefine " + tpmIndex + " -C o -s " + tpmsize + " -a 'ownerread|ownerwrite'"
        print("Defining TPM NV Index...")
        print(self.run_command_tpm(define_nv))
        # write to TPM
        write_hash = "echo -n '" + store_hashes_str + "' | xxd -p | tr -d '\n' | tpm2_nvwrite " + tpmIndex + " -C o -i -"
        print("Writing hash to TPM...")
        print(self.run_command_tpm(write_hash))
    
    def run_command_tpm(self, command):
        """Function to run a shell command and return the output."""
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.stdout

class CustomClientFactory(ClientFactory):
    def __init__(self, config, ap):
        self.client_id = config["client_id"]
        self.config = config
        self.ap = ap

    def buildProtocol(self, addr):
        return CustomClient(self.client_id, self.config, self.ap)

    def clientConnectionFailed(self, connector, reason):
        print("Connection to global server failed")
        reactor.stop()

    def clientConnectionLost(self, connector, reason):
        print("Connection to global server lost")
        reactor.stop()

def trainTestModel(config):
    model_path = config["test_model_path"]
    if os.path.exists(model_path) and config["retrain_test_model"] == "False":
        print("The test model already trained " + model_path)
    else:
        print("Training test model")
        #Put Any Deep Learning Logic Here
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import InputLayer, Dense
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
        from tensorflow.keras.optimizers import Adam
        if config['enable_gpu_training'] == "False":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        x_train = np.load(config["local_dataset_x_train_path"])
        y_train = np.load(config["local_dataset_y_train_path"])
        x_test = np.load(config["local_dataset_x_test_path"])
        y_test = np.load(config["local_dataset_y_test_path"])
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        if config['fl_training_model'] == "ntc_mlp":
            ##Remove Source and Destination IP From Dataset
            x_train = np.delete(x_train, [12,13,14,15,16,17,18,19], 1)
            x_test = np.delete(x_test, [12,13,14,15,16,17,18,19], 1)
            #MLP Model
            model = Sequential()
            model.add(InputLayer(input_shape = (x_train.shape[1],))) # input layer
            model.add(Dense(32, activation='relu')) # hidden layer 1
            model.add(Dense(64, activation='relu')) # hidden layer 2
            model.add(Dense(128, activation='relu')) # hidden layer 3
            model.add(Dense(10, activation='softmax')) # output layer
            model.summary()
            # Compile model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif config['fl_training_model'] == "ntc_cnn":
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
        elif config['fl_training_model'] == "ntc_cnn_cifar10":
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
        elif config['fl_training_model'] == "ntc_1dcnn":
            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(y_train.shape[1], activation='softmax')
            ])
            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        # fit the keras model on the dataset
        model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = config["training_model_batch_size"], epochs = config["training_model_epochs"], verbose = True, shuffle = True)
        #Store Model
        model.save(config["test_model_path"])
        
def run_command_hash(command):
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
        
def run_ps_command():
    # Execute the 'ps -aux' command and capture its output
    result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE, text=True)
    # Split the output into lines
    lines = result.stdout.splitlines()
    # List to hold all processes
    processes = []
    # Skip the first line as it contains the headers
    for line in lines[1:]:
        parts = line.split(maxsplit=10)  # Split on whitespace, max 10 splits
        if len(parts) == 11:  # Ensure the line splits into the correct number of parts
            process = {
                'USER': parts[0],
                'PID': parts[1],
                'COMMAND': parts[10]
            }
            processes.append(process)
    return processes
  
def run_netstat_command():
    # Building the command
    command = "netstat --tcp --udp --programs --listening --numeric"
    # Execute the command and capture its output
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    # Check for errors
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}")
        return []
    # Split the output into lines
    lines = result.stdout.splitlines()
    # List to hold all port information
    ports = []
    # Skip the first two lines as they contain headers
    for line in lines[2:]:
        parts = line.split(maxsplit=10)
        if len(parts) == 7:
            port_info = {
                'PROTO': parts[0],
                'LOCAL_ADDRESS': parts[3],
                'FOREIGN_ADDRESS': parts[4],
                'STATE': parts[5],
                'PID/PROGRAM': parts[6]
            }
            ports.append(port_info)   
    return ports

# Find and count recurring backdoor patterns
def find_recurring_patterns_with_count(array):
    seen_patterns = {}
    recurring_patterns = {}
    for i, row in enumerate(array):
        row_tuple = tuple(row)  # Convert row to a hashable tuple
        if row_tuple in seen_patterns:
            if row_tuple in recurring_patterns:
                recurring_patterns[row_tuple]['count'] += 1
                recurring_patterns[row_tuple]['indices'].append(i)
            else:
                recurring_patterns[row_tuple] = {
                    'count': 2,  # Initial count is 2 (first occurrence and this one)
                    'indices': [seen_patterns[row_tuple], i]
                }
        else:
            seen_patterns[row_tuple] = i
    return recurring_patterns
        
def generateAttestationParameters(config):
    if os.path.exists(config["verification_list_path"]):
        print("The verification list has been generated, loading existing list " + config["verification_list_path"])
        with open(config["verification_list_path"], 'r') as json_file:
            vl = json.load(json_file)
    else:
        print("Generating Client Verification List")
        process_list = run_ps_command()
        port_list = run_netstat_command()
            
        vl = {
        "data_type": "Client_Verification_List",
        "client_id": config["client_id"],
        "client_domain": config["client_domain"],
        "process_list": process_list,
        "port_list": port_list
        }
        with open(config["verification_list_path"], 'w') as file:
            json.dump(vl, file, indent=4)
            
    if os.path.exists(config["attestation_parameters_path"]) and config["regenerate_attestation_parameters"] == "False":
        print("The attestation parameters has been generated, loading existing list " + config["attestation_parameters_path"])
        with open(config["attestation_parameters_path"], 'r') as json_file:
            ap = json.load(json_file)
    else:
        print("Generating Client Attestation Parameters")
        
        ##Check Backdoor Pattern
        checkPathx = config['local_dataset_x_train_path']
        checkPathy = config['local_dataset_y_train_path']
        x_train_check = np.load(checkPathx)
        y_train_check = np.load(checkPathy)
        array = x_train_check
        if config['fl_training_model'] == "ntc_cnn" or config['fl_training_model'] == "ntc_cnn_cifar10" or config['fl_training_model'] == "ntc_1dcnn":
            # Flatten each image (from 28x28x1 to 784 elements)
            array_flattened = array.reshape(array.shape[0], -1)  # Flattening the images to 1D
            array = array_flattened
        recurring_patterns = find_recurring_patterns_with_count(array)
        backdoor_status = 0
        if recurring_patterns:
            print("Checking for potential backdoor pattern...")
            for pattern, info in recurring_patterns.items():
                if info['count'] >= 1000:
                    print("Possible backdoor pattern detected")
                    print("Attempting to remove backdoor pattern..")
                    print(f"Pattern: {pattern[:5]}... (truncated for display), Count: {info['count']}")
                    backdoor_status = 1
                    x_train_check = np.delete(x_train_check, info['indices'], 0)
                    y_train_check = np.delete(y_train_check, info['indices'], 0)
                    np.save(checkPathx, x_train_check) # Replacing dataset
                    np.save(checkPathy, y_train_check) # Replacing dataset
        else:
            print("No recurring patterns found")
        x_train_verify = np.load(checkPathx)
        y_train_verify = np.load(checkPathy)
        print(x_train_verify.shape)
        print(x_train_verify.shape)
        
        if backdoor_status == 0:
            bs = "False"
            print("No recurring patterns found which exceed the threshold value..")
        elif backdoor_status == 1:
            bs = "True"
    
        with open(config["test_model_path"], 'rb') as file:
            test_model = file.read() 
        hash_command = f"openssl dgst -sha1 -engine dynamic {config['FL_program_path']}"
        FL_program_sha256 = run_command_hash(hash_command)
        
        hash_command = f"openssl dgst -sha1 -engine dynamic {config['verification_list_path']}"
        verification_list_sha256 = run_command_hash(hash_command)
        
        hash_command = f"openssl dgst -sha1 -engine dynamic {config['attestation_program_path']}"
        attestation_program_sha256 = run_command_hash(hash_command)
        
        hash_command = f"openssl dgst -sha1 -engine dynamic {config['local_dataset_x_train_path']}"
        local_dataset_x_train_sha256 = run_command_hash(hash_command)
        
        hash_command = f"openssl dgst -sha1 -engine dynamic {config['local_dataset_y_train_path']}"
        local_dataset_y_train_sha256 = run_command_hash(hash_command)
        
        hash_command = f"openssl dgst -sha1 -engine dynamic {config['local_dataset_x_test_path']}"
        local_dataset_x_test_sha256 = run_command_hash(hash_command)
        
        hash_command = f"openssl dgst -sha1 -engine dynamic {config['local_dataset_y_test_path']}"
        local_dataset_y_test_sha256 = run_command_hash(hash_command)
        
        hash_command = f"openssl dgst -sha1 -engine dynamic {config['client_cert_path']}"
        client_cert_sha256 = run_command_hash(hash_command)
        ap = {
        "data_type": "Client_Attestation_Parameters",
        "client_id": config["client_id"],
        "client_domain": config["client_domain"],
        "client_host": config["client_host"],
        "test_model_file": '',
        "test_model_name": config["test_model_path"],
        "FL_program_sha256": FL_program_sha256,
        "attestation_program_sha256": attestation_program_sha256,
        "local_dataset_x_train_sha256": local_dataset_x_train_sha256,
        "local_dataset_y_train_sha256": local_dataset_y_train_sha256,
        "local_dataset_x_test_sha256": local_dataset_x_test_sha256,
        "local_dataset_y_test_sha256": local_dataset_y_test_sha256,
        "client_cert_sha256": client_cert_sha256,
        "verification_list": vl,
        "verification_list_sha256": verification_list_sha256,
        "backdoorStatus": bs
        }
        with open(config["attestation_parameters_path"], 'w') as file:
            json.dump(ap, file, indent=4)
        
    with open(config["test_model_path"], 'rb') as file:
            test_model = file.read()
    ap["test_model_file"] = test_model  

    return ap

class ClientContextFactory(ssl.ClientContextFactory):
    def getContext(self):
        context = SSL.Context(SSL.TLSv1_2_METHOD)
        context.set_options(SSL.OP_NO_SSLv2)
        context.set_options(SSL.OP_NO_SSLv3)
        context.set_verify(SSL.VERIFY_NONE, lambda conn, cert, errno, depth, result: True)
        return context

def main():
    #Load Attestation Config
    config_path = 'config.json'
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    host = config["server_host"]
    port = config["server_port"]
    client_source_ip = config["client_ip"]
    client_source_port = config["client_source_port"]
    client_id = config["client_id"]
    file_path = config["test_model_path"]
    trainTestModel(config)
    ap = generateAttestationParameters(config)
    factory = CustomClientFactory(config, ap)
    if config['verify_ssl_cert'] == "True":
        contextFactory = ssl.optionsForClientTLS(hostname=host, trustRoot=ssl.Certificate.loadPEM(open(config["server_cert_path"]).read()))
        reactor.connectSSL(host, port, factory, contextFactory, bindAddress=(client_source_ip, client_source_port))
    else:
        reactor.connectSSL(host, port, factory, ClientContextFactory(), bindAddress=(client_source_ip, client_source_port))
    reactor.run()

if __name__ == "__main__":
    main()