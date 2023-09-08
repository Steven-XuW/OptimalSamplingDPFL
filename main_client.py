import torch.nn as nn
import torch.nn.functional as F
import importlib
import warnings
import pickle
import paramiko
import math
import json
import time
import sys
import os

from src.models.client import Client
from src.models.model import choose_model
from src.utils.worker_utils import MiniDataset
from torch.utils.data import DataLoader
from torch import tensor
from src.optimizers.gd import GD
from src.models.worker import LrdWorker



class FederatedClient(Client):
    def __init__(self, dataset, options):
        self.cid = options['cid']
        self.group = options['group']

        users, groups, train_data, test_data = dataset
        self.train_data = train_data[str(self.cid)]
        self.test_data = test_data[str(self.cid)]
        # print(f'===============test data is {len(self.train_data)}=================')
        self.train_dataloader = DataLoader(self.train_data, batch_size=options['batch_size'], shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=options['batch_size'], shuffle=False)

        self.model = choose_model(options)
        self.optimizer = GD(self.model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        self.num_epoch = options['num_epoch']
        self.worker = LrdWorker(self.model, self.optimizer, options)
        self.params_file = None

    def save_update(self, updated_params):
        # save with pkl to minimize the size of the file
        with open('params.pkl', 'wb') as f:
            pickle.dump(updated_params, f)

    def send_update(self, file_path):
        host = '192.168.2.211'
        port = 22
        username = 'spole'
        password = 'xwj20021231001'

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh.connect(hostname=host, port=port, username=username, password=password)

        local_file_path = 'params.pkl'
        remote_target_path = file_path + f'/clients_params/{str(self.cid).zfill(2)}/params.pkl'
        '''remote_target_path = 'C:/Users/spole/Desktop/params.pkl'  '''
        print(remote_target_path)
        sftp = ssh.open_sftp()
        sftp.put(local_file_path, remote_target_path)

        sftp.close()
        ssh.close()
        
def print_color_text(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")
    
def read_fedata(train_dataset, test_dataset):
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    print('>>> Read data from:' + train_dataset)

    with open(train_dataset, 'rb') as inf:
        cdata = pickle.load(inf)
    clients.extend(cdata['users'])
    for cid, v in cdata['user_data'].items():
        train_data[cid] = MiniDataset(v['x'], v['y'])

    print('>>> Read data from:' + test_dataset)
    with open(test_dataset, 'rb') as inf:
        cdata = pickle.load(inf)
    for cid, v in cdata['user_data'].items():
        test_data[cid] = MiniDataset(v['x'], v['y'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data




options = {'cid': 1,
           'group': None,
           'model': 'logistic',
           'input_shape': 784,
           'num_class': 62,
           'lr': 1,
           'wd': 0,
           'num_epoch': 5,
           'batch_size': 32,
           'gpu': False,
           }

train_path = 'merged_data_train.pkl'
test_path = 'merged_data_test.pkl'
all_data_info = read_fedata(train_path, test_path)

client = FederatedClient(all_data_info, options)

while True:
    success = False
    max_attempts = 5
    current_attempt = 0
    # load data from the server
    if os.path.exists('input.pkl'): 
        begin_time = time.time()
        while current_attempt < max_attempts:
            try:
                with open('input.pkl', 'rb') as param_file:
                    input_dict = pickle.load(param_file)
                success = True
                break
            except Exception:
                current_attempt += 1
                time.sleep(0.5)
                print_color_text(f'Attempt to reload params {current_attempt}/{max_attempts} failed. Retrying ...', 33)
        if not success:
            print(f'Loading data in round {round_i} failed, skip this round')
            continue
        round_i = input_dict['round']
        send_to_path = input_dict['path']
        latest_model = input_dict['lastest_model']

        client.set_flat_model_params(latest_model)
        params = client.local_train()

        client.save_update(params)
        client.send_update(send_to_path)

        with open('params.pkl', 'rb') as f:
          params = pickle.load(f)

        print(params)
        os.remove('input.pkl')
        end_time = time.time()
        print_color_text(f'round {round_i}: local used time is {end_time - begin_time}', 36)
    else:
        print('no input yet')
    time.sleep(1)
    




