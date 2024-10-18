#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Packet Bytes data pre-processing for deep learning network traffic classification

import pyshark
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import warnings
import logging
warnings.filterwarnings("ignore")

#Function to padding raw packet bytes
def padding(byteValue) :
    byteValue = byteValue.ljust(2960, '0')
    return byteValue

#Function to truncate raw packet bytes
def truncate(byteValue) :
    byteValue = byteValue[0:2960]
    return byteValue
    
#Function to process raw packet bytes
def processPacket(byteValue) :
    byteLen = len(byteValue)
    if byteLen > 2960 :
        byteValue = truncate(byteValue)
    elif byteLen < 2960 :
        byteValue = padding(byteValue)
    n = 2
    byteList = []
    for i in range(0, len(byteValue), n):
        temp = byteValue[i:i+n]
        base16INT = int(temp, 16)
        byteList.append(base16INT)
    return byteList

#Specify the packet data input and output directories
input_directory = '/home/mazizi/pre_datasets'
output_directory = '/home/mazizi/proc_datasets'

fileName = "skype_chat1a.pcap"
label = 'skypechat'

log_directory = '/home/mazizi/pre_logs/'

logging.basicConfig(
filename= log_directory + filename + '.log',  # Specify the log file name
filemode='a',        # Append mode (use 'w' to overwrite)
format='%(asctime)s - %(levelname)s - %(message)s',  # Format for log entries
level=logging.DEBUG  # Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
)

input = input_directory + "/" + filename
print('Reading PCAP..')
logging.info('Reading PCAP..')
c = pyshark.FileCapture(input, use_json=True, include_raw=True)
print('Done reading ' + filename)
logging.info('Done reading ' + filename)

#create empty panda frame with 1480 column for each bytes
countDf = 0
df = pd.DataFrame()
while countDf <= 1479 :
    index = 'B' + str(countDf)
    df[index] = np.nan
    countDf += 1

output = output_directory + '/' + filename + '.csv'

count = 0
countSave = 0
countIndex = 0
countUnknown = 0
n = 2
counttls = 0
countdns = 0
counttcp = 0
countudp = 0
counthttp = 0
countmdns = 0
countdtls = 0
countstun = 0
countgquic = 0
countrtcp = 0
countbtdht = 0
countbttracker = 0
countquic = 0
counthipercont = 0
countclasstun = 0
countwg = 0
countsteamihs = 0
checkString = ""

for x in c :
    if 'TLS' in c[count] :
        counttls += 1
        try:
            byteValue = c[count].ip_raw.value + c[count].tcp_raw.value + c[count].tls_raw.value
        except:
            byteValue = c[count].ip_raw.value + c[count].tcp_raw.value + c[count].tls_raw.value[0]
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'HTTP' in c[count] :
        counthttp += 1
        try : 
            byteValue = c[count].ip_raw.value + c[count].tcp_raw.value + c[count].http_raw.value + c[count].data_raw.value
        except :
            byteValue = c[count].ip_raw.value + c[count].tcp_raw.value + c[count].http_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'MDNS' in c[count] :
        countmdns += 1
        try :
            byteValue = c[count].ipv6_raw.value + c[count].udp_raw.value + c[count].mdns_raw.value
        except :
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].mdns_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'DNS' in c[count] and 'UDP' in c[count] :
        countdns += 1
        byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].dns_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'DTLS' in c[count] :
        countdtls +=1
        byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].dtls_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'GQUIC' in c[count] :
        countgquic += 1
        byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].gquic_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'RTCP' in c[count] :
        countrtcp += 1
        try:
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].rtcp_raw.value
        except:
            try:
                byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].rtcp_raw.value[0]
            except:
                try:
                    byteValue = c[count].ip_raw.value + c[count].udp_raw.value
                except:
                    byteValue = c[count].ip_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'BT-DHT' in c[count] :
        countbtdht += 1
        try:
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].bt-dht_raw.value
        except:
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'BT-TRACKER' in c[count] :
        countbttracker += 1
        try:
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].bt-tracker_raw.value
        except:
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'QUIC' in c[count] :
        countquic += 1
        try:
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].quic_raw.value
        except:
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'WG' in c[count] :
        countwg += 1
        try:
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].wg_raw.value
        except:
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'STEAM_IHS_DISCOVERY' in c[count] :
        countsteamihs += 1
        try:
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].steam_ihs_discovery_raw.value
        except:
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'CLASSICSTUN' in c[count] :
        countclasstun += 1
        try:
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].classicstun_raw.value
        except:
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'HIPERCONTRACER' in c[count] :
        counthipercont += 1
        try:
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].hipercontracer_raw.value
        except:
            try:
                byteValue = c[count].ip_raw.value + c[count].udp_raw.value
            except:
                byteValue = c[count].ip_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'STUN' in c[count] :
        countstun += 1
        try :
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].stun_raw.value
        except :
            try :
                byteValue = c[count].ip_raw.value + c[count].tcp_raw.value + c[count].stun_raw.value
            except :
                try:
                    byteValue = c[count].ip_raw.value + c[count].stun_raw.value
                except:
                    byteValue = c[count].ip_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'TCP' in c[count] :
        if 'DATA' in c[count] :
            counthttp += 1
            try :
                byteValue = c[count].ip_raw.value + c[count].tcp_raw.value + c[count].http_raw.value + c[count].data_raw.value
                counthttp += 1
            except :
                byteValue = c[count].ip_raw.value + c[count].tcp_raw.value + c[count].data_raw.value
                counttcp += 1
        else :
            counttcp += 1
            byteValue = c[count].ip_raw.value + c[count].tcp_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    elif 'UDP' in c[count] and 'DATA' in c[count] :
        countudp += 1
        byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].data_raw.value
        df.loc[countIndex] = processPacket(byteValue)
        countIndex += 1
    else :
        checkString = str(c[count].layers)
        if 'RTCP' in checkString :
            byteValue = c[count].ip_raw.value + c[count].udp_raw.value + c[count].rtcp_raw.value
            df.loc[countIndex] = processPacket(byteValue)
            countIndex += 1
            countrtcp += 1
            print('RTCP')
            logging.debug('RTCP')
        else :
            countUnknown += 1
            print('Unrecognized protocol in row:' + str(count) + ' ' + filename + ' ' + c[count].highest_layer + ' unknownCount: ' + str(countUnknown) )
            logging.warning('Unrecognized protocol in row:' + str(count) + ' ' + filename + ' ' + c[count].highest_layer + ' unknownCount: ' + str(countUnknown))
    if countSave == 2000:
        print('Saving ' + filename)
        print('Savepoint at Packet Count: ' + str(count))
        logging.info('Saving ' + filename)
        logging.info('Savepoint at Packet Count: ' + str(count))
        df.to_csv(output, index=False)
        countSave = 0
        print('Done processing ' + filename)
        print('Total packets inspected:', count)
        print('Total packets extracted:', counttls + countdns + counttcp + countudp + counthttp + countmdns + countdtls + countstun + countgquic + countrtcp + countbtdht + countbttracker + countquic + counthipercont + countclasstun)
        print('Total unprocessed packets:', countUnknown)
        print('Total TLS:', counttls)
        print('Total DNS:', countdns)
        print('Total TCP:', counttcp)
        print('Total UDP:', countudp)
        print('Total HTTP:', counthttp)
        print('Total MDNS:', countmdns)
        print('Total DTLS:', countdtls)
        print('Total STUN:', countstun)
        print('Total GQUIC:', countgquic)
        print('Total RTCP:', countrtcp)
        print('Total BTDHT:', countbtdht)
        print('Total BTTracket:', countbttracker)
        print('Total QUIC:', countquic)
        print('Total HIPERCONTRACER:', counthipercont)
        print('Total CLASSICSTUN:', countclasstun)
        print('Total WG:', countwg)
        print('Total STEAMIHS:', countsteamihs)

        logging.info('Done processing ' + filename)
        logging.info('Total packets inspected:' + str(count))
        logging.info('Total packets extracted:' + str(counttls + countdns + counttcp + countudp + counthttp + countmdns + countdtls + countstun + countgquic + countrtcp + countbtdht + countbttracker + countquic + counthipercont + countclasstun))
        logging.info('Total unprocessed packets:' + str(countUnknown))
        logging.debug('Total TLS:' + str(counttls))
        logging.debug('Total DNS:' + str(countdns))
        logging.debug('Total TCP:' + str(counttcp))
        logging.debug('Total UDP:' + str(countudp))
        logging.debug('Total HTTP:' + str(counthttp))
        logging.debug('Total MDNS:' + str(countmdns))
        logging.debug('Total DTLS:' + str(countdtls))
        logging.debug('Total STUN:' + str(countstun))
        logging.debug('Total GQUIC:' + str(countgquic))
        logging.debug('Total RTCP:' + str(countrtcp))
        logging.debug('Total BTDHT:' + str(countbtdht))
        logging.debug('Total BTTracket:' + str(countbttracker))
        logging.debug('Total QUIC:' + str(countquic))
        logging.debug('Total HIPERCONTRACER:' + str(counthipercont))
        logging.debug('Total CLASSICSTUN:' + str(countclasstun))
        logging.debug('Total WG:' + str(countwg))
        logging.debug('Total STEAMIHS:' + str(countsteamihs))
    #if count == 2000: #Remove Later
    #    break
    count += 1
    countSave += 1
df['label'] = label
df.to_csv(output, index=False)

print('Done processing ' + filename)
print('Total packets inspected:', count)
print('Total packets extracted:', counttls + countdns + counttcp + countudp + counthttp + countmdns + countdtls + countstun + countgquic + countrtcp + countbtdht + countbttracker + countquic + counthipercont)
print('Total unprocessed packets:', countUnknown)
print('Total TLS:', counttls)
print('Total DNS:', countdns)
print('Total TCP:', counttcp)
print('Total UDP:', countudp)
print('Total HTTP:', counthttp)
print('Total MDNS:', countmdns)
print('Total DTLS:', countdtls)
print('Total STUN:', countstun)
print('Total GQUIC:', countgquic)
print('Total RTCP:', countrtcp)
print('Total BTDHT:', countbtdht)
print('Total BTTracket:', countbttracker)
print('Total QUIC:', countquic)
print('Total HIPERCONTRACER:', counthipercont)
print('Total CLASSICSTUN:', countclasstun)
print('Total WG:', countwg)
print('Total STEAMIHS:', countsteamihs)

logging.info('Done processing ' + filename)
logging.info('Total packets inspected:' + str(count))
logging.info('Total packets extracted:' + str(counttls + countdns + counttcp + countudp + counthttp + countmdns + countdtls + countstun + countgquic + countrtcp + countbtdht + countbttracker + countquic + counthipercont))
logging.info('Total unprocessed packets:' + str(countUnknown))
logging.debug('Total TLS:' + str(counttls))
logging.debug('Total DNS:' + str(countdns))
logging.debug('Total TCP:' + str(counttcp))
logging.debug('Total UDP:' + str(countudp))
logging.debug('Total HTTP:' + str(counthttp))
logging.debug('Total MDNS:' + str(countmdns))
logging.debug('Total DTLS:' + str(countdtls))
logging.debug('Total STUN:' + str(countstun))
logging.debug('Total GQUIC:' + str(countgquic))
logging.debug('Total RTCP:' + str(countrtcp))
logging.debug('Total BTDHT:' + str(countbtdht))
logging.debug('Total BTTracket:' + str(countbttracker))
logging.debug('Total QUIC:' + str(countquic))
logging.debug('Total HIPERCONTRACER:' + str(counthipercont))
logging.debug('Total CLASSICSTUN:' + str(countclasstun))
logging.debug('Total WG:' + str(countwg))
logging.debug('Total STEAMIHS:' + str(countsteamihs))

print('The End')
logging.info('The End')