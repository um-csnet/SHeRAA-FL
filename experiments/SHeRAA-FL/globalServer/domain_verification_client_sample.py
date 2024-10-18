#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Domain-Level Verification Program for Local Aggregator and Edge Client

import json
import pickle
from twisted.internet import reactor, ssl, defer
from twisted.internet.protocol import Protocol, ClientFactory, Factory
from twisted.protocols.basic import LineReceiver
from twisted.python import log
from twisted.protocols.basic import FileSender
from twisted.python import filepath
import sys
import os
import shutil
import numpy as np
import pandas as pd
import subprocess
import re
import multiprocessing
import time
from OpenSSL import SSL

log.startLogging(sys.stdout)

class SharedValue:
    countTerminate = 0
    countConnected = 0
    untrusted_client = {}
    trusted_client = {}
    client_list = {}

def fl_aggregator_worker(fl_config, config):
    print("Running Local Aggregator FL Client Training Program")
    time.sleep(3)
    host = fl_config["local_aggregator_host"]
    port = fl_config["local_aggregator_port"]
    factory = CustomClientFactory(fl_config, config)
    if config['verify_ssl_cert'] == "True":
        contextFactory = ssl.optionsForClientTLS(hostname=host, trustRoot=ssl.Certificate.loadPEM(open(fl_config["local_aggregator_cert_path"]).read()))
        reactor.connectSSL(host, port, factory, contextFactory)
    else:
        reactor.connectSSL(host, port, factory, ClientContextFactory())
    reactor.run()

class CustomClient(Protocol):
    def __init__(self, client_id, fl_config, retrieveTPMHash, config):
        self.client_id = client_id
        self.fl_config = fl_config
        self.config = config
        self.received_data = b''
        self.retrieveTPMHash = retrieveTPMHash
        self.state = 'DICT'
        self.file = None
        self.bytes_received = 0
        self.dataset_size = 0
        self.dictstore = {}
        
    def connectionMade(self):
        print('Checking TPM..')
        if self.fl_config['recalculate_hash'] == "False":
            attestation_parameters_path = 'attestation_parameters.json'
            with open(attestation_parameters_path, 'r') as json_file:
                at = json.load(json_file)
            local_dataset_x_train_sha256 = at['local_dataset_x_train_sha256']
            local_dataset_y_train_sha256 = at['local_dataset_y_train_sha256']
        else:
            hash_command = f"openssl dgst -sha1 -engine dynamic {self.fl_config['local_dataset_x_train_path']}"
            local_dataset_x_train_sha256 = self.run_command_hash(hash_command)
            hash_command = f"openssl dgst -sha1 -engine dynamic {self.fl_config['local_dataset_y_train_path']}"
            local_dataset_y_train_sha256 = self.run_command_hash(hash_command)
        if self.fl_config['role'] == "local_aggregator":
            self.local_dataset_x_train_sha256_tpm = self.retrieveTPMHash[self.client_id]['x']
            self.local_dataset_y_train_sha256_tpm = self.retrieveTPMHash[self.client_id]['y']
            self.verification_token = self.retrieveTPMHash[self.client_id]['aggt']
        else:
            self.local_dataset_x_train_sha256_tpm = self.retrieveTPMHash['x']
            self.local_dataset_y_train_sha256_tpm = self.retrieveTPMHash['y']
            self.verification_token = self.retrieveTPMHash['vt']
        if self.verification_token != self.fl_config['verification_token'] or self.local_dataset_x_train_sha256_tpm != local_dataset_x_train_sha256 or self.local_dataset_y_train_sha256_tpm != local_dataset_y_train_sha256:
            print(self.verification_token)
            print(self.fl_config['verification_token'])
            print(self.local_dataset_x_train_sha256_tpm)
            print(local_dataset_x_train_sha256)
            print(self.local_dataset_y_train_sha256_tpm)
            print(local_dataset_y_train_sha256)
            print('The training is compromised, terminating training process..')
            self.transport.loseConnection()
        print('Uploading Local Dataset and FL Training Program to Local Aggregator..')
        with open(self.fl_config['FL_program_path'], 'rb') as file:
            FL_program_file = file.read()
        with open(self.fl_config['local_dataset_y_train_path'], 'rb') as file:
            local_dataset_y_train_file = file.read()
        file_size = os.path.getsize(self.fl_config['local_dataset_x_train_path'])
        hash_command = f"openssl dgst -sha1 -engine dynamic {self.fl_config['domain_verification_program_path']}"
        domain_verification_program_hash = self.run_command_hash(hash_command)
        data_dict = {
            'client_id': self.client_id,
            'client_domain': self.fl_config['client_domain'],
            'verification_token': self.verification_token,
            'domain_verification_program_hash': domain_verification_program_hash,
            'FL_program_name': self.fl_config['FL_program_path'],
            'FL_program_file': FL_program_file,
            'local_dataset_x_train_name': self.fl_config['local_dataset_x_train_path'],
            'local_dataset_x_train_size': file_size,
            'local_dataset_y_train_name': self.fl_config['local_dataset_y_train_path'],
            'local_dataset_y_train_file': local_dataset_y_train_file
        }
        serialized_data = pickle.dumps(data_dict) + b'END'
        self.transport.write(serialized_data)
        file_to_send = filepath.FilePath(self.fl_config['local_dataset_x_train_path'])
        sender = FileSender()
        d = sender.beginFileTransfer(file_to_send.open('rb'), self.transport, lambda x: x)
        d.addCallback(self.waitResponse)
        d.addErrback(self.error)
        
    def dataReceived(self, data):
        if self.state == 'DICT':
            self.received_data += data
            if self.received_data.endswith(b'END'):
                print('Received Response From Local Aggregator..')
                self.received_data = self.received_data[:-3]  # Remove the 'END' marker
                response_dict = pickle.loads(self.received_data)
                if response_dict['type'] == "untrustedResponse":
                    print(response_dict['message'])
                    self.fl_config['role'] = "untrusted_client"
                    with open(self.config["fl_training_config_path"], 'w') as file:
                        json.dump(self.fl_config, file, indent=4)
                    print('Ending domain-level verification process, Bye')
                    tpmIndex = self.fl_config["tpm_index"]	    
                    #Undefined NV
                    print("Clearing the TPM Storage..." + tpmIndex)
                    undefinedtpm = "tpm2_nvundefine " + tpmIndex + " -C o"
                    check = self.run_command_tpm(undefinedtpm)
                    self.transport.loseConnection()
                elif response_dict['type'] == "trustedResponse":
                    print(response_dict['message'])
                    if response_dict['delegated_status'] == "True":
                        if not os.path.exists('delegateStorage'):
                            os.makedirs('delegateStorage')
                        else:
                            try:
                                shutil.rmtree('delegateStorage')
                            except OSError as e:
                                print(f"Error: {e.strerror} ")
                            os.makedirs('delegateStorage')
                        self.save_file(response_dict['local_dataset_y_train_name'], response_dict['local_dataset_y_train_file'])
                        self.state = 'LARGE'
                        self.data = b''
                        self.dataset_size = response_dict['local_dataset_x_train_size']
                        storage_path = self.fl_config['home_path'] + "delegateStorage/"
                        dataset_path = storage_path + response_dict['local_dataset_x_train_name']
                        self.file = open(dataset_path, 'wb')
                        print('Storing Delegated Dataset File..')
                        self.fl_config['delegated'] = {}
                        self.fl_config['delegated'][response_dict['client_id']] = {}
                        self.fl_config['delegated'][response_dict['client_id']]['verification_token'] = response_dict['verification_token']
                        self.fl_config['delegated'][response_dict['client_id']]['verification_token'] = response_dict['verification_token']
                        self.fl_config['delegated'][response_dict['client_id']]['local_dataset_x_train_name'] = response_dict['local_dataset_x_train_name']
                        self.fl_config['delegated'][response_dict['client_id']]['local_dataset_y_train_name'] = response_dict['local_dataset_y_train_name']
                        with open(self.config["fl_training_config_path"], 'w') as file:
                            json.dump(self.fl_config, file, indent=4)
                        self.dictstore = response_dict
                    else:
                        self.fl_config['role'] = "valid_client"
                        if 'delegated' in self.fl_config:
                            del self.fl_config['delegated']
                        with open(self.config["fl_training_config_path"], 'w') as file:
                            json.dump(self.fl_config, file, indent=4)
                        self.storing_hashes_verify(response_dict)
                        self.transport.loseConnection()
                else:
                    print('Unknown response from local aggregator..')
                    self.transport.loseConnection()
        elif self.state == 'LARGE':        
            self.file.write(data)
            self.bytes_received += len(data)
            if self.bytes_received >= self.dataset_size:
                self.file.close()
                print("Finish Storing Delegated Dataset")
                self.storing_hashes_delegated(self.dictstore)
                self.transport.loseConnection()
    def storing_hashes_verify(self, response_dict):
        #CheckTPM whether has been initialized
        tpmIndex = self.fl_config["tpm_index"]
        checktpm = "tpm2_nvreadpublic " + tpmIndex
        print("Checking TPM NV Index...")
        check = self.run_command_tpm(checktpm)
        if "friendly" in check:
            #Undefined NV
            print("Resetting the TPM NV Index..." + tpmIndex)
            undefinedtpm = "tpm2_nvundefine " + tpmIndex + " -C o"
            check = self.run_command_tpm(undefinedtpm)
        store_hashes = {
        "vt": self.fl_config['verification_token'],
        }
        store_hashes_str = json.dumps(store_hashes)
        print(store_hashes_str)
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
               
    def storing_hashes_delegated(self, response_dict):
        #CheckTPM whether has been initialized
        tpmIndex = self.fl_config["tpm_index"]
        checktpm = "tpm2_nvreadpublic " + tpmIndex
        print("Checking TPM NV Index...")
        check = self.run_command_tpm(checktpm)
        if "friendly" in check:
            #Undefined NV
            print("Resetting the TPM NV Index..." + tpmIndex)
            undefinedtpm = "tpm2_nvundefine " + tpmIndex + " -C o"
            check = self.run_command_tpm(undefinedtpm)
        delegated = {
        "cid": response_dict['client_id'],
        "dt": response_dict['verification_token']
        }
        store_hashes = {
        "vt": self.fl_config['verification_token'],
        "delegated": delegated
        }
        store_hashes_str = json.dumps(store_hashes)
        print(store_hashes_str)
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
    
    def save_file(self, filename, file):
        storage_path = self.fl_config['home_path'] + "delegateStorage/"
        file_data = file
        file_path = storage_path + filename
        with open(file_path, 'wb') as file:
            file.write(file_data)
        log.msg(f"Saving..{file_path}")
        
    def waitResponse(self, _):
        print('Finish Uploading FL Training Program and Dataset to Secure Storage')
        if self.fl_config['role'] == "local_aggregator":
            self.transport.loseConnection()
        
    def error(self, reason):
        print(f"Error: {reason}")
        self.transport.loseConnection()
        
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
    
    def run_command_tpm(self, command):
        """Function to run a shell command and return the output."""
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.stdout

class CustomClientFactory(ClientFactory):
    def __init__(self, fl_config, config):
        self.client_id = fl_config["client_id"]
        self.fl_config = fl_config
        self.config = config
        # Read back and display the hash stored in TPM
        print("Reading hash from TPM...")
        try:
            read_hash = "tpm2_nvread " + self.fl_config['tpm_index'] + " -C o | xxd -p -r"
            retrieveTPM = self.run_command_tpm(read_hash)
            self.retrieveTPMHash = json.loads(retrieveTPM)
        except:
            if config['restore_tpm_storage'] == "True":
                with open('backup_tpm.json', 'r') as json_file:
                    self.retrieveTPMHash = json.load(json_file)
            else:
                print("Please perform remote attestation with global attestator server..bye")
                exit()
        if config['restore_tpm_storage'] == "True":
            with open('backup_tpm.json', 'r') as json_file:
                self.retrieveTPMHash = json.load(json_file)
        if self.fl_config['role'] == "local_aggregator":
            if 'x' not in self.retrieveTPMHash[self.fl_config['client_id']]:
                print("Please perform remote attestation with global attestator server..bye")
                exit()
        else:
            if 'x' not in self.retrieveTPMHash:
                print("Please perform remote attestation with global attestator server..bye")
                exit()
        if config['backup_tpm_storage'] == "True":        
            with open('backup_tpm.json', 'w') as file:
                json.dump(self.retrieveTPMHash, file, indent=4)

    def buildProtocol(self, addr):
        return CustomClient(self.client_id, self.fl_config, self.retrieveTPMHash, self.config)

    def clientConnectionFailed(self, connector, reason):
        print("Connection to Local Aggregator failed")
        reactor.stop()

    def clientConnectionLost(self, connector, reason):
        print("Connection to Local Aggregator lost")
        reactor.stop()
        
    def run_command_tpm(self, command):
        """Function to run a shell command and return the output."""
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.stdout
        
class ClientContextFactory(ssl.ClientContextFactory):
    def getContext(self):
        context = SSL.Context(SSL.TLSv1_2_METHOD)
        context.set_options(SSL.OP_NO_SSLv2)
        context.set_options(SSL.OP_NO_SSLv3)
        context.set_verify(SSL.VERIFY_NONE, lambda conn, cert, errno, depth, result: True)
        return context

def fl_client(fl_config, config):
    print("This Edge node is selected as client for domain " + fl_config['client_domain'] )
    host = fl_config["local_aggregator_host"]
    port = fl_config["local_aggregator_port"]
    factory = CustomClientFactory(fl_config, config)
    
    if config['verify_ssl_cert'] == "True":
        contextFactory = ssl.optionsForClientTLS(hostname=host, trustRoot=ssl.Certificate.loadPEM(open(fl_config["local_aggregator_cert_path"]).read()))
        reactor.connectSSL(host, port, factory, contextFactory)
    else:
        reactor.connectSSL(host, port, factory, ClientContextFactory())
    reactor.run()

class CustomAggregatorProtocol(Protocol):
    def __init__(self, factory, fl_config, aggregator_config, retrieveTPMHash, config):
        self.factory = factory
        self.data = b''
        self.client_id = None
        self.fl_config = fl_config
        self.config = config
        self.aggregator_config = aggregator_config
        self.retrieveTPMHash = retrieveTPMHash
        self.state = 'DICT'
        self.file = None
        self.bytes_received = 0
        self.client_list_record = aggregator_config['client_list']
        self.dataset_size = 0
        self.storage_path = self.fl_config['home_path'] + "secureStorage/"
        
    def dataReceived(self, data):
        if self.state == 'DICT':
            self.data += data
            peer = self.transport.getPeer()
            if self.data.endswith(b'END'):
                self.state = 'LARGE'
                self.data = self.data[:-3]  # Remove the 'END' marker
                data_dict = pickle.loads(self.data)
                self.client_id = data_dict['client_id']
                self.factory.client_protocols[self.client_id] = self
                client_protocol = self.factory.client_protocols[self.client_id]
                try:
                    client_token_tpm = self.retrieveTPMHash[self.client_id]['vt']
                except:
                    client_token_tpm = self.retrieveTPMHash[self.client_id]['aggt']
                valid_domain_verification_hash = self.retrieveTPMHash[self.fl_config['client_id']]['dv']
                if data_dict['verification_token'] != client_token_tpm: 
                    print("Unauthorized connection, Verification Token Mismatch...removing edge client...")
                    client_protocol.transport.loseConnection()
                elif data_dict['domain_verification_program_hash'] != valid_domain_verification_hash and self.aggregator_config["verify_domain_verification_program"] == 'True':
                    print("Unauthorized connection, Using illegal Domain verification program...removing edge client...")
                    client_protocol.transport.loseConnection()
                elif data_dict['FL_program_name'] == "" or data_dict['local_dataset_y_train_name'] == "" or data_dict['local_dataset_x_train_name'] == "":
                    print("Client Does not Upload Dataset or FL Program...removing edge client...")
                    client_protocol.transport.loseConnection()
                else:
                    self.save_file(data_dict['FL_program_name'], data_dict['FL_program_file'], client_protocol)
                    self.save_file(data_dict['local_dataset_y_train_name'], data_dict['local_dataset_y_train_file'], client_protocol)
                    self.data = b''
                    print('Storing Dataset File..')
                    storage_path = self.fl_config['home_path'] + "secureStorage/"
                    dataset_path = storage_path + data_dict['local_dataset_x_train_name']
                    self.dataset_size = data_dict['local_dataset_x_train_size']
                    self.file = open(dataset_path, 'wb')
                    SharedValue.client_list[self.client_id] = {}
                    SharedValue.client_list[self.client_id]['verification_token'] = data_dict['verification_token']
                    SharedValue.client_list[self.client_id]['FL_program_name'] = data_dict['FL_program_name']
                    SharedValue.client_list[self.client_id]['local_dataset_x_train_name'] = data_dict['local_dataset_x_train_name']
                    SharedValue.client_list[self.client_id]['local_dataset_y_train_name'] = data_dict['local_dataset_y_train_name']
                    SharedValue.countConnected = SharedValue.countConnected + 1
        elif self.state == 'LARGE':
            self.file.write(data)
            self.bytes_received += len(data)
            if self.bytes_received >= self.dataset_size:
                self.file.close()
                if self.bytes_received <= 100:
                    print('Dataset received is too small, client is not uploading proper file, ending edge client')
                    client_protocol.transport.loseConnection()
                    SharedValue.countConnected = SharedValue.countConnected - 1
                print("Finish Storing Dataset")
                if SharedValue.countConnected == len(self.client_list_record):
                    self.verify_client(self.storage_path)
                    
    def verify_client(self, storage_path):
        print('All local edge client connected..')
        predef_FL_program_hash = self.retrieveTPMHash[self.fl_config['client_id']]['flp']
        for client in SharedValue.client_list:
            print("Verifying " + client)
            checkPath = storage_path + SharedValue.client_list[client]['FL_program_name']
            hash_command = f"openssl dgst -sha1 -engine dynamic {checkPath}"
            client_fl_program_hash = self.run_command_hash(hash_command)
            if self.aggregator_config["verify_both_dataset"] == 'True':
                checkPath = storage_path + SharedValue.client_list[client]['local_dataset_x_train_name']
                hash_command = f"openssl dgst -sha1 -engine dynamic {checkPath}"
                client_x_hash = self.run_command_hash(hash_command)
                client_x_hash_record = self.retrieveTPMHash[client]['x']
                checkPath = storage_path + SharedValue.client_list[client]['local_dataset_y_train_name']
                hash_command = f"openssl dgst -sha1 -engine dynamic {checkPath}"
                client_y_hash = self.run_command_hash(hash_command)
                client_y_hash_record = self.retrieveTPMHash[client]['y']
            else:
                client_x_hash = self.retrieveTPMHash[client]['x']
                client_x_hash_record = self.retrieveTPMHash[client]['x']
                checkPath = storage_path + SharedValue.client_list[client]['local_dataset_y_train_name']
                hash_command = f"openssl dgst -sha1 -engine dynamic {checkPath}"
                client_y_hash = self.run_command_hash(hash_command)
                client_y_hash_record = self.retrieveTPMHash[client]['y']
            ##Check Backdoor Pattern
            checkPathx = storage_path + SharedValue.client_list[client]['local_dataset_x_train_name']
            checkPathy = storage_path + SharedValue.client_list[client]['local_dataset_y_train_name']
            x_train_check = np.load(checkPathx)
            y_train_check = np.load(checkPathy)
            array = x_train_check
            recurring_patterns = self.find_recurring_patterns_with_count(array)
            backdoor_status = 0
            if recurring_patterns:
                print("Checking for potential backdoor pattern...")
                for pattern, info in recurring_patterns.items():
                    if info['count'] >= self.aggregator_config['backdoor_pattern_threshold']:
                        print("Possible Backdoor in " + client)
                        print("Attempting to remove backdoor pattern..")
                        print(f"Pattern: {pattern[:5]}... (truncated for display), Count: {info['count']}")
                        ##Only delegate if the backdoor is in other Edge node
                        if client != self.config['client_id']:
                            backdoor_status = 1
                        x_train_check = np.delete(x_train_check, info['indices'], 0)
                        y_train_check = np.delete(y_train_check, info['indices'], 0)
                        np.save(checkPathx, x_train_check) # Replacing dataset
                        np.save(checkPathy, y_train_check) # Replacing dataset
                    #else:
                        #print("Recurring pattern not severe..")
                        #print(f"Pattern: {pattern[:5]}... (truncated for display), Count: {info['count']}")
            else:
                print("No recurring patterns found")
            x_train_verify = np.load(checkPathx)
            y_train_verify = np.load(checkPathy)
            print(x_train_verify.shape)
            print(x_train_verify.shape)
            if client_fl_program_hash != predef_FL_program_hash:
                print(f"{client} become Untrusted client due to tempering of FL training program")
                trust_score = self.aggregator_config['client_list'][client]['trust_score'] - 5
                SharedValue.untrusted_client[client] = {}
                SharedValue.untrusted_client[client]['trust_score'] = trust_score
                SharedValue.untrusted_client[client]['FL_program_name'] = SharedValue.client_list[client]['FL_program_name']
                SharedValue.untrusted_client[client]['local_dataset_x_train_name'] = SharedValue.client_list[client]['local_dataset_x_train_name']
                SharedValue.untrusted_client[client]['local_dataset_y_train_name'] = SharedValue.client_list[client]['local_dataset_y_train_name']
            elif backdoor_status == 1:
                print(f"{client} become Untrusted client due to possible backdoor")
                trust_score = self.aggregator_config['client_list'][client]['trust_score'] - 5
                SharedValue.untrusted_client[client] = {}
                SharedValue.untrusted_client[client]['trust_score'] = trust_score
                SharedValue.untrusted_client[client]['FL_program_name'] = SharedValue.client_list[client]['FL_program_name']
                SharedValue.untrusted_client[client]['local_dataset_x_train_name'] = SharedValue.client_list[client]['local_dataset_x_train_name']
                SharedValue.untrusted_client[client]['local_dataset_y_train_name'] = SharedValue.client_list[client]['local_dataset_y_train_name']
            elif client_x_hash != client_x_hash_record or client_y_hash != client_y_hash_record:
                print(f"{client} become Untrusted client due to tempering of Training Dataset")
                trust_score = self.aggregator_config['client_list'][client]['trust_score'] - 5
                SharedValue.untrusted_client[client] = {}
                SharedValue.untrusted_client[client]['trust_score'] = trust_score
                SharedValue.untrusted_client[client]['FL_program_name'] = SharedValue.client_list[client]['FL_program_name']
                SharedValue.untrusted_client[client]['local_dataset_x_train_name'] = SharedValue.client_list[client]['local_dataset_x_train_name']
                SharedValue.untrusted_client[client]['local_dataset_y_train_name'] = SharedValue.client_list[client]['local_dataset_y_train_name']
            else:
                print(f"{client} become Trusted client for domain {self.fl_config['client_domain']}")
                SharedValue.trusted_client[client] = {}
                SharedValue.trusted_client[client]['verification_token'] = SharedValue.client_list[client]['verification_token']
                SharedValue.trusted_client[client]['trust_score'] = self.aggregator_config['client_list'][client]['trust_score']
                SharedValue.trusted_client[client]['status'] = self.aggregator_config['client_list'][client]['status']
        if len(SharedValue.untrusted_client) == 0:
            print("No untrusted Client..")
        else:
            print(f"Delegating the the training task for {len(SharedValue.untrusted_client)} untrusted client")
            for untrusted in SharedValue.untrusted_client:
                print("Delegating " + untrusted)
                find = 0
                hash_command = f"openssl rand -engine dynamic -hex 10"
                delegation_token = self.run_command_token(hash_command)
                delegateName = "delegate_" + untrusted
                for trusted in SharedValue.trusted_client:
                    if 'delegated' in SharedValue.trusted_client[trusted]:
                        delegate_in = 1
                    else:
                        delegate_in = 0
                    if SharedValue.trusted_client[trusted]['status'] != "local_aggregator" and delegate_in == 0 and find == 0:
                        find = 1
                        SharedValue.trusted_client[trusted]['delegated'] = {}
                        SharedValue.trusted_client[trusted]['delegated'][delegateName] = SharedValue.untrusted_client[untrusted]
                        SharedValue.trusted_client[trusted]['delegated'][delegateName]['delegation_token'] = delegation_token
                if find == 0:
                    if 'delegated' in SharedValue.trusted_client[self.fl_config['client_id']]:
                        SharedValue.trusted_client[self.fl_config['client_id']]['delegated'][delegateName] = {}
                        SharedValue.trusted_client[self.fl_config['client_id']]['delegated'][delegateName] = SharedValue.untrusted_client[untrusted]
                        SharedValue.trusted_client[self.fl_config['client_id']]['delegated'][delegateName]['delegation_token'] = delegation_token
                    else:
                        SharedValue.trusted_client[self.fl_config['client_id']]['delegated'] = {}
                        SharedValue.trusted_client[self.fl_config['client_id']]['delegated'][delegateName] = {}
                        SharedValue.trusted_client[self.fl_config['client_id']]['delegated'][delegateName] = SharedValue.untrusted_client[untrusted]
                        SharedValue.trusted_client[self.fl_config['client_id']]['delegated'][delegateName]['delegation_token'] = delegation_token
        if len(SharedValue.untrusted_client) != 0:
            for untrusted in SharedValue.untrusted_client:
                self.send_response_untrusted(untrusted)
        if len(SharedValue.trusted_client) != 0:
            for trusted in SharedValue.trusted_client:
                if SharedValue.trusted_client[trusted]['status'] != "local_aggregator":
                    self.send_response_trusted(trusted)
        with open("trusted_client.json", 'w') as file:
            json.dump(SharedValue.trusted_client, file, indent=4)
        self.storing_hashes_aggregator()
    
    def storing_hashes_aggregator(self):
        #CheckTPM whether has been initialized
        tpmIndex = self.fl_config["tpm_index"]
        checktpm = "tpm2_nvreadpublic " + tpmIndex
        print("Checking TPM NV Index...")
        check = self.run_command_tpm(checktpm)
        if "friendly" in check:
            #Undefined NV
            print("Resetting the TPM NV Index..." + tpmIndex)
            undefinedtpm = "tpm2_nvundefine " + tpmIndex + " -C o"
            check = self.run_command_tpm(undefinedtpm)
        store_hashes = {}
        for client in SharedValue.trusted_client:
            store_hashes[client] = {}
            if 'delegated' in SharedValue.trusted_client[client]:
                delegate_dict = SharedValue.trusted_client[client]['delegated']
                if SharedValue.trusted_client[client]['status'] == "local_aggregator":
                    store_hashes[client]['aggt'] = SharedValue.trusted_client[client]['verification_token']
                    store_hashes[client]['ts'] = SharedValue.trusted_client[client]['trust_score']
                else:
                    store_hashes[client]['vt'] = SharedValue.trusted_client[client]['verification_token']
                    store_hashes[client]['ts'] = SharedValue.trusted_client[client]['trust_score']
                for delegate in delegate_dict:
                    store_hashes[delegate] = {}
                    store_hashes[delegate]['dt'] = delegate_dict[delegate]['delegation_token']
                    store_hashes[delegate]['ts'] = delegate_dict[delegate]['trust_score']
            else:
                if SharedValue.trusted_client[client]['status'] == "local_aggregator":
                    store_hashes[client]['aggt'] = SharedValue.trusted_client[client]['verification_token']
                    store_hashes[client]['ts'] = SharedValue.trusted_client[client]['trust_score']
                else:
                    store_hashes[client]['vt'] = SharedValue.trusted_client[client]['verification_token']
                    store_hashes[client]['ts'] = SharedValue.trusted_client[client]['trust_score']
        store_hashes_str = json.dumps(store_hashes)
        print(store_hashes_str)
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
    
    def send_response_trusted(self, client):
        client_protocol = self.factory.client_protocols[client]
        if 'delegated' in SharedValue.trusted_client[client]:
            storage_path = self.fl_config['home_path'] + "secureStorage/"
            message = f"{client} has been verified, considered as trusted edge client and delegated with training task"
            for delegate in SharedValue.trusted_client[client]['delegated']:
                delegateName = delegate
            datasety_path = storage_path + SharedValue.trusted_client[client]['delegated'][delegateName]['local_dataset_y_train_name']
            with open(datasety_path, 'rb') as file:
                local_dataset_y_train_file = file.read()
            dataset_path = storage_path + SharedValue.trusted_client[client]['delegated'][delegateName]['local_dataset_x_train_name']
            file_size = os.path.getsize(dataset_path)
            response_dict = {
            'type': "trustedResponse",
            'message': message,
            'delegated_status': "True",
            'client_id': delegateName,
            'verification_token': SharedValue.trusted_client[client]['delegated'][delegateName]['delegation_token'],
            'local_dataset_x_train_name': SharedValue.trusted_client[client]['delegated'][delegateName]['local_dataset_x_train_name'],
            'local_dataset_x_train_size': file_size,
            'local_dataset_y_train_name': SharedValue.trusted_client[client]['delegated'][delegateName]['local_dataset_y_train_name'],
            'local_dataset_y_train_file': local_dataset_y_train_file
            }
            serialized_data = pickle.dumps(response_dict) + b'END'
            client_protocol.transport.write(serialized_data)
            dataset_path = storage_path + SharedValue.trusted_client[client]['delegated'][delegateName]['local_dataset_x_train_name']
            file_to_send = filepath.FilePath(dataset_path)
            sender = FileSender()
            d = sender.beginFileTransfer(file_to_send.open('rb'), client_protocol.transport, lambda x: x)
            d.addCallback(self.waitResponse)
            d.addErrback(self.error)
        else:
            message = f"{client} has been verified and considered as trusted edge client"
            response_dict = {
                'type': "trustedResponse",
                'message': message,
                'delegated_status': "False"
            }
            serialized_data = pickle.dumps(response_dict) + b'END'
            client_protocol.transport.write(serialized_data)
            
    def waitResponse(self, _):
        print('Finish Delegating Training Task')
        
    def error(self, reason):
        print(f"Error: {reason}")

    def send_response_untrusted(self, client):
        client_protocol = self.factory.client_protocols[client]
        message = "The training process of this edge client has been delegated to other trusted edge client"
        response_dict = {
            'type': "untrustedResponse",
            'message': message
        }
        serialized_data = pickle.dumps(response_dict) + b'END'
        client_protocol.transport.write(serialized_data)
    
    def save_file(self, filename, file, client_protocol):
        storage_path = self.fl_config['home_path'] + "secureStorage/"
        file_data = file
        file_path = storage_path + filename
        with open(file_path, 'wb') as file:
            file.write(file_data)
        file_size = os.path.getsize(file_path)
        if file_size <= 100:
            print('Dataset or FL program received is too small, client is not uploading proper file, ending edge client')
            client_protocol.transport.loseConnection()
        log.msg(f"Saving..{file_path}")
        
    # Find and count recurring backdoor patterns
    def find_recurring_patterns_with_count(self, array):
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
        
    def connectionMade(self):
        peer = self.transport.getPeer()
        print(f"Connection made from Edge Client {peer.host}:{peer.port}")

    def connectionLost(self, reason):
        if self.client_id in self.factory.client_protocols:
            if self.file and not self.file.closed:
                self.file.close()
            SharedValue.countTerminate = SharedValue.countTerminate + 1
            print(f"{self.client_id} is terminating attestation process")
            del self.factory.client_protocols[self.client_id]
            if SharedValue.countTerminate == len(self.aggregator_config['client_list']):
                print("All edge clients have terminate connection..")
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

class CustomAggregatorFactory(Factory):
    def __init__(self, fl_config, aggregator_config, config):
        self.received_data = []
        self.client_protocols = {}
        self.fl_config = fl_config
        self.config = config
        self.aggregator_config = aggregator_config
        # Read back and display the hash stored in TPM
        print("Reading hash from TPM...")
        try:
            read_hash = "tpm2_nvread " + self.fl_config['tpm_index'] + " -C o | xxd -p -r"
            retrieveTPM = self.run_command_tpm(read_hash)
            self.retrieveTPMHash = json.loads(retrieveTPM)
        except:
            if config['restore_tpm_storage'] == "True":
                with open('backup_tpm.json', 'r') as json_file:
                    self.retrieveTPMHash = json.load(json_file)
            else:
                print("Please perform remote attestation with global attestator server..bye")
                exit()
        if config['restore_tpm_storage'] == "True":
            with open('backup_tpm.json', 'r') as json_file:
                self.retrieveTPMHash = json.load(json_file)
        if 'x' not in self.retrieveTPMHash[self.fl_config['client_id']]:
            print("Please perform remote attestation with global attestator server..bye")
            exit()
        if config['backup_tpm_storage'] == "True":        
            with open('backup_tpm.json', 'w') as file:
                json.dump(self.retrieveTPMHash, file, indent=4)

    def buildProtocol(self, addr):
        return CustomAggregatorProtocol(self, self.fl_config, self.aggregator_config, self.retrieveTPMHash, self.config)
        
    def run_command_tpm(self, command):
        """Function to run a shell command and return the output."""
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.stdout
   
def fl_aggregator(fl_config, config):
    print("This Edge node is selected as local aggregator for domain " + fl_config['client_domain'] )
    aggregator_config_path = 'local_aggregator_config.json'
    if os.path.isfile(aggregator_config_path):
        with open(aggregator_config_path, 'r') as json_file:
            aggregator_config = json.load(json_file)
    else:
        print("Local Aggregator config file is not found, please perform remote attestation with global attestator")
        exit()
    if not os.path.exists('secureStorage'):
        os.makedirs('secureStorage')
    else:
        try:
            shutil.rmtree('secureStorage')
        except OSError as e:
            print(f"Error: {e.strerror} ")
        os.makedirs('secureStorage')
    port = aggregator_config["aggregator_port"]
    ip_address = aggregator_config["aggregator_ip"]
    factory = CustomAggregatorFactory(fl_config, aggregator_config, config)
    contextFactory = ssl.DefaultOpenSSLContextFactory(aggregator_config["aggregator_key_path"], aggregator_config["aggregator_cert_path"])
    reactor.listenSSL(port, factory, contextFactory, interface=ip_address)
    print(f"Local Aggregator is listening on IP {ip_address} and port {port} with SSL...")
    reactor.run()
    
if __name__ == "__main__":
    #Load FL Training Config
    config_path = 'config.json'
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    fl_config_path = 'fl_training_config.json'
    aggClient = 0
    with open(fl_config_path, 'r') as json_file:
        fl_config = json.load(json_file)
    if fl_config['role'] == 'local_aggregator':
        aggClient = 1
        p = multiprocessing.Process(target=fl_aggregator_worker, args=(fl_config, config))
        p.start()
        fl_aggregator(fl_config, config)
    elif fl_config['role'] == 'valid_client':
        fl_client(fl_config, config)
    else:
        print("Please perform remote attestation process with global aggregator server..end")
    if aggClient == 1:
        p.join()