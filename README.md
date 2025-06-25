# SHeRAA-FL Framework
Paper- Mitigating Adversarial Attacks in Federated Learning Based Network Traffic Classification Applications using Secure Hierarchical Remote Attestation and Adaptive Aggregation Framework

# Deployed on:
## Hosts
* AMD Ryzen 7 7840HS 8-core CPU
* 16GB DDR5 Memory
* 16GB RAM
* Nvidia RTX 4070 GPU
* Ubuntu 20.04 LTS 

## Software Dependencies
* Python 3.8
* Scikit-learn 1.5.1
* PyShark 0.3.6
* PyShark 0.3.6
* TensorFlow & Keras 2.12.1
* CUDNN 8.9
* Twisted 18.9.0
* Flower 1.6.0 
* CUDA 12.6
* IBM TPM 2.0 simulator
* TSS 3.1.0
* ABRMD 2.3.1
* TSS-engine 1.1.0
* TPM Tools 4.3.2
* OpenSSL 1.1.2

# To deploy:
## Data Pre-Processing
1. Download relevant datasets and put in folder and run rawPacketsPreprocessing.py from data_preprocessing folder.
	ISCX-VPN 2016 from here: https://www.unb.ca/cic/datasets/vpn.html
	NBaIoT from here: https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot
	FashionMNIST and CIFAR-10 can be obtained from Tensorflow datasets
2. Put the processed raw data into a folder and combine.py script from the data_preprocessing folder to combine the datasets.
4. After that run normalize.py from the data_preprocessing folder to normalize the datasets.
3. Run the split.py script from the data_preprocessing folder to split the dataset for six FL Clients.

## TPM 2.0 Setup
1. The framework requires TPM 2.0 as dependencies.
2. You setup TPM 2.0 dependencies at your host by following this guide: https://francislampayan.medium.com/how-to-setup-tpm-simulator-in-ubuntu-20-04-25ec673b88dc

## SHeRAA-FL Framework Setup
1. The framework source code is available in the /experiments/SHeRAA-FL Folder.
2. Its contains one globalServer and six Client Folders.
3. You need to generate Private & Public Key for server and each clients, and generate self-signed SSL certificates. You can follow this guide: https://medium.com/@pasanglamatamang/creating-a-valid-self-signed-ssl-certificate-ee466d6fca31
4. Copy the public certificates of the server to each clients' folders. You also need to copy each clients' public certificates to server's folder.
5. Put training and testing datasets to server and clients' folders.
6. You need to configure IP address, domain for server and clients by modifying the config.json file
7. To start training process, first run global_server_attestator.py from the globalServer folder.
8. Then start to run remote_attestation_clientx.py script from each clients' folder to begin remote attestation process.
9. After remote attestation finished, run domain_verification_clientx.py script from each clients' folder to begin domain verification process.
10. Lastly, run fl_training_clientx.py from each clients' folder to begin model training process.
*You can also used Server and Client GUI program to run SHeRAA-FL

## Unsecure FL Setup
1. The FL setup source code is available in the /experiments/unsecure-FL Folder.
2. Its contains one globalServer and six Client Folders.
3. To start training, you need to run the server script from the globalServer folder. You can either choose to run server based on different aggregation algorithms 
	server_mlp_fedavg.py
	server_mlp_fedmedian.py
	server_mlp_fedtrim.py
	server_mlp_krum.py (For multi-KRUM increase set the num_clients_to_keep=1)
	server_mlp_weighted_fedavg.py
4. Then run the client_mlp_x.py script from each of the clients' folders

## Control Experiment
1. Start with Unsecure FL by running the FL training without any adversarial clients. Set number of client appropriately in the script. Train by using different aggregation algorithms.
2. After that run FL training using the SHeRAA-FL framework. Set server IP,port,domain appropriately and make sure server is running.
3. To evaluate, use the experiments/evaluate/evaluate.py script
4. Make sure the client[1-6] and server test datasets are in the same folder as client/server program or you can configure file path appropriately

## All-Label Flipping Attack
1. Run /adversarial_attacks/label_flipping/labelflip.py script to flip choosen clients datasets.
2. Swap the choosen malicious clients datasets with flipped datasets
3. Start FL training with Unsecure FL. Set number of client appropriately in the script. Use different aggregation algorithm and change model name appropriately according to experiment.
4. Then start FL training using the SHeRAA-FL framework. Make sure to swap the datasets with poisoned datasets.
5. To evaluate, use the experiments/evaluate/evaluate.py script
6. Make sure the server test datasets are in the same folder

## Class-Label Flipping Attack
1. Run /adversarial_attacks/label_flipping/labelflipclass.py script to flip choosen clients datasets.
2. Swap the choosen malicious clients datasets with flipped datasets
3. Start FL training with Unsecure FL. Set number of client appropriately in the script. Use different aggregation algorithm and change model name appropriately according to experiment.
4. Then start FL training using the SHeRAA-FL framework. Make sure to swap the datasets with poisoned datasets.
5. To evaluate, use the experiments/evaluate/evaluate.py script
6. Make sure the server test datasets are in the same folder

## Model Poisoning - Model Cancelling Attack
1. Starts the FL server using the Unsecure FL Setup. For this setup, swap the malicious client client program with /adversarial_attacks/model_poisoning/unsecure_client_mc.py
2. Then starts the FL server using the SHeRAA-FL framework. For this setup, swap the malicious client client program with /adversarial_attacks/model_poisoning/SHeRAA-FL_client_mc.py
3. To evaluate, use the experiments/evaluate/evaluate.py script
4. Make sure the server test datasets are in the same folder

## Model Poisoning - Gradient Factor Attack
1. Starts the FL server using the Unsecure FL Setup. For this setup, swap the malicious client client program with /adversarial_attacks/model_poisoning/unsecure_client_gf.py
2. Then starts the FL server using the SHeRAA-FL framework. For this setup, swap the malicious client client program with /adversarial_attacks/model_poisoning/SHeRAA-FL_client_gf.py
3. To evaluate, use the experiments/evaluate/evaluate.py script
4. Make sure the server test datasets are in the same folder

## Backdoor Attack
1. Run /adversarial_attacks/backdoor/backdoor_injector.py script to inject backdoor pattern into client's datasets.
2. Swap the choosen malicious clients datasets with backdoor datasets
3. Start FL training with Unsecure FL. Set number of client appropriately in the script. Use different aggregation algorithm and change model name appropriately according to experiment.
4. Then start FL training using the SHeRAA-FL framework. Make sure to swap the datasets with poisoned datasets.
5. To evaluate, use the experiments/evaluate/evaluate_backdoor.py script
6. Make sure the server test datasets are in the same folder

## GAN-Based Attack
1. Run /adversarial_attacks/GAN/gan_ntc.py script to generate synthetic traffic data for certain class. Configure the target class in the script. Set synthetic data file appropriately in the script.
2. Run /adversarial_attacks/GAN/gan_embedded.py script to inject the synthetic traffic data to target class. Set synthetic data file appropriately in the script.
3. Swap the choosen malicious clients datasets with backdoor datasets
4. Start FL training with Unsecure FL. Set number of client appropriately in the script. Use different aggregation algorithm and change model name appropriately according to experiment.
5. Then start FL training using the SHeRAA-FL framework. Make sure to swap the datasets with poisoned datasets.
6. To evaluate, use the experiments/evaluate/evaluate.py script
7. Make sure the server test datasets are in the same folder

For any inquiries you can email [azizi.mohdariffin@gmail.com]
