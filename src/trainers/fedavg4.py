from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import Worker, GradientPerburtedWorker, OutputPerturbedWorker
from src.optimizers.gd import GD
from src.utils.autograd_hacks import *
from src.models.client import Client
from src.utils.privacy_accountant import noise_injection
from src.utils.get_epsilon_bound import PrivacyAccountant
from src.optimizers.paraller_cvx import GradientPerturbedOptimizor, OutputPerturbedOptimizor

import time
import numpy as np
import random
import torch
import os
import json
import math

criterion = torch.nn.CrossEntropyLoss()


class FedAvg4Trainer(BaseTrainer):

    def __init__(self, options, dataset):
        self.gamma = options['lr']
        self.balance_l = float(options['balance_l'])
        seed_value = 0
        random.seed(int(options['epsilon_seed']))
        # pattern should be in ['grad', 'output_local', 'output_server']
        self.injection_pattern = options['injection_pattern']
        self.sampling_scheme = options['sampling_scheme']

        model = choose_model(options)
        add_hooks(model)

        self.move_model_to_gpu(model, options)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        self.num_epoch = options['num_epoch']
        worker = Worker(model, self.optimizer, options)
        # set the pattern of noise injection
        if self.injection_pattern == 'grad':
            self.dpworker = GradientPerburtedWorker(model, self.optimizer, options)  # for gradient disturbution
        elif self.injection_pattern == 'output_local':
            self.dpworker = OutputPerturbedWorker(model, self.optimizer, options)  # for output disturbution

        super(FedAvg4Trainer, self).__init__(options, dataset, worker=worker)
        self.prob = self.compute_prob()

        self.d = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.s = [options['batch_size'] / len(c.train_data) for c in self.clients]
        self.clip_size = float(options['clip_size'])
        self.delta = 1e-4
        self.p = self.prob
        self.q = self.prob

        process_data = {}
        if options['read_exist_data'] == 'True':
            with open('processed_data.json', 'r') as json_file:
                recovered_data = json.load(json_file)
            self.epsilon = recovered_data['epsilon']
            self.sigma = recovered_data['sigma']
            self.q = recovered_data['q']
            process_data = recovered_data
            if self.injection_pattern != 'output_server':
                self.set_clients_sigma()
        else:
            self.epsilon = [round(random.uniform(16, 32), 4) for _ in range(self.num_clients)]
            if self.sampling_scheme == 'all':
                self.sigma = [0] * self.num_clients
            else:
                self.q = self.get_user_sampling_rate()
                PLAT = PrivacyAccountant(options, self)
                self.sigma = PLAT.get_sigma()
                if self.injection_pattern != 'output_server':
                    self.set_clients_sigma()

            process_data = {'delta': self.delta,
                            'clip_size': self.clip_size,
                            'epsilon': list(self.epsilon),
                            'q': list(self.q),
                            'sigma': list(self.sigma)}
        print(f'q is {self.q}')
        current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        t1 = options['sampling_scheme']
        t2 = options['injection_pattern']
        t3 = options['lr']
        t4 = options['epsilon_seed']
        t5 = options['num_round']
        self.save_folder = os.path.join('mydata', f'{current_time}_{t1}_{t2}_lr{round(t3, 2)}_seed{t4}_t{t5}')
        os.makedirs(self.save_folder, exist_ok=True)
        json_file_path = os.path.join(self.save_folder, 'processed_data.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(process_data, json_file, indent=4)

    def get_user_sampling_rate(self):
        if self.sampling_scheme == 'optimal':
            if self.injection_pattern == 'grad':
                solver = GradientPerturbedOptimizor(self)
                return solver.get_q()
            else:
                solver = OutputPerturbedOptimizor(self)
                return solver.get_q()

        elif self.sampling_scheme == 'uniform':
            return [1 / self.num_clients] * self.num_clients
        elif self.sampling_scheme == 'weighted':
            return self.p
        elif self.sampling_scheme == 'epsilon_weighted':
            return [ep / sum(self.epsilon) for ep in self.epsilon]
        else:
            raise ValueError("not valid scheme!")

    def set_clients_sigma(self):
        for cid, c in enumerate(self.clients):
            c.worker.set_sigma(self.sigma[cid])

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        for round_i in range(self.num_round):

            # Test latest model on train data
            # This is quite time comsuming! There is no need to do so
            # self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            # Choose K clients prop to data size
            if self.simple_average:
                selected_clients, repeated_times = self.select_clients_with_prob(seed=round_i)
            else:
                selected_clients = self.select_clients(seed=round_i)
                repeated_times = None

            # Solve minimization locally
            if self.sampling_scheme == 'all':
                solns, stats = self.local_train(round_i, self.clients)
            else:
                solns, stats = self.local_train(round_i, selected_clients)

            # Track communication cost
            # self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model = self.aggregate(solns, repeated_times=repeated_times)
            # self.optimizer.inverse_prop_decay_learning_rate(round_i)

        # Test final model on train data
        # self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        # Save tracked information
        json_file_path = os.path.join(self.save_folder, 'result_data.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(self.result_data, json_file, indent=4)
        # self.metrics.write()

    def compute_prob(self):
        probs = []
        for c in self.clients:
            probs.append(len(c.train_data))
        return np.array(probs) / sum(probs)

    def select_clients_with_prob(self, seed=1):
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        index = np.random.choice(len(self.clients), num_clients, p=self.prob)
        index = sorted(index.tolist())

        select_clients = []
        select_index = []
        repeated_times = []
        for i in index:
            if i not in select_index:
                select_clients.append(self.clients[i])
                select_index.append(i)
                repeated_times.append(1)
            else:
                repeated_times[-1] += 1
        return select_clients, repeated_times

    # if the params was not perturbed, inject noise on the server end
    def aggregate_with_orignal_params(self, solns, **kwargs):
        averaged_solution = torch.zeros_like(self.latest_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        repeated_times = kwargs['repeated_times']
        if self.sampling_scheme == 'all':
            repeated_times = [1] * self.num_clients

        offset = 0
        averaged_solution = []

        layer_norm = []
        for p in self.worker.model.parameters():
            layer_norm.append(float(torch.norm(p.grad.data, 2)))
        mean_squared_average = np.mean(np.array(layer_norm) ** 2) + 1e-5
        layer_norm = [float(self.clip_size) / mean_squared_average * norm for norm in layer_norm]

        for net_idx, p in enumerate(self.worker.model.parameters()):
            # get the size of the current layer
            param_size = p.data.numel()
            # restore clients' params on current layer
            solns_restored = [soln[2][offset:offset + param_size] for soln in solns]
            # get the adaptive max norm of the params
            max_norm = np.median([float(torch.norm(local_solution, 2))
                                  for local_solution in solns_restored])
            max_norm = min(max_norm, layer_norm[net_idx])
            print(f'layer {net_idx}: param size is {param_size}, max norm this layer is {max_norm}')
            # clip the local solutions
            local_solutions_clipped = torch.stack(
                [local_solution / max(1, float(torch.norm(local_solution, 2)) / max_norm)
                 for local_solution in solns_restored])

            sigma = [self.sigma[cid] for (cid, num_sample, local_solution) in solns]
            # inject noise to the clients' params
            solutions_disturbed = [
                noise_injection(local_solution_clipped, sigma[i], max_norm, self.batch_size, self.gpu)
                for i, local_solution_clipped in enumerate(local_solutions_clipped)]
            # aggregate the local model
            solution_averaged = torch.zeros(local_solutions_clipped[0].shape)
            if self.gpu:
                solution_averaged = solution_averaged.mps()
            for i, (cid, num_sample, local_solution) in enumerate(solns):
                solution_averaged = self.p[cid] / (self.q[cid] * self.clients_per_round) * \
                                    solutions_disturbed[i] * repeated_times[i] + solution_averaged
            averaged_solution.append(solution_averaged)
            # update the offset while entering next layer
            offset += param_size
        # flat back the averaged solution
        flat_averaged_solution = torch.cat(averaged_solution)
        # free cuda memory
        return flat_averaged_solution.detach()

    # def aggregate_with_perturbed_params(self, solns, **kwargs):
    #     repeated_times = kwargs['repeated_times']
    #     if self.sampling_scheme == 'all':
    #         repeated_times = [1] * self.num_clients
    #
    #     offset = 0
    #     averaged_solution = []
    #
    #     layer_norm = []
    #     for p in self.worker.model.parameters():
    #         layer_norm.append(float(torch.norm(p.grad.data, 2)))
    #     mean_squared_average = np.mean(np.array(layer_norm) ** 2) + 1e-5
    #     layer_norm = [float(self.clip_size) / mean_squared_average * norm for norm in layer_norm]
    #
    #     for net_idx, p in enumerate(self.worker.model.parameters()):
    #         # get the size of the current layer
    #         param_size = p.data.numel()
    #         # restore clients' params on current layer
    #         solns_restored = [soln[2][offset:offset + param_size] for soln in solns]
    #         # get the adaptive max norm of the params
    #         max_norm = layer_norm[net_idx]
    #         print(f'layer {net_idx}: param size is {param_size}, max norm this layer is {max_norm}')
    #         # clip the local solutions
    #         local_solutions_clipped = torch.stack(
    #             [local_solution / max(1, float(torch.norm(local_solution, 2)) / max_norm)
    #              for local_solution in solns_restored])
    #
    #         # inject noise to the clients' params
    #         # aggregate the local model
    #         solution_averaged = torch.zeros(local_solutions_clipped[0].shape)
    #         if self.gpu:
    #             solution_averaged = solution_averaged.cuda()
    #         for i, (cid, num_sample, local_solution) in enumerate(solns):
    #             solution_averaged = self.p[cid] / (self.q[cid] * self.clients_per_round) * \
    #                                 local_solutions_clipped[i] * repeated_times[i] + solution_averaged
    #         averaged_solution.append(solution_averaged)
    #         # update the offset while entering next layer
    #         offset += param_size
    #     # flat back the averaged solution
    #     flat_averaged_solution = torch.cat(averaged_solution)
    #     # free cuda memory
    #     return flat_averaged_solution.detach()

    def aggregate_with_perturbed_params(self, solns, **kwargs):
        averaged_solution = torch.zeros_like(self.latest_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        repeated_times = kwargs['repeated_times']
        if self.sampling_scheme == 'all':
            repeated_times = [1] * self.num_clients
        # assert len(solns) == len(repeated_times)

        offset = 0

        for i, (cid, num_sample, local_solution) in enumerate(solns):
            solution_norm = np.linalg.norm(local_solution.cpu())
            if solution_norm > self.clip_size:
                local_solution = local_solution / solution_norm * self.clip_size
            averaged_solution += repeated_times[i] * self.p[cid] / \
                                 (self.q[cid] * self.clients_per_round) * local_solution
        # free cuda memory
        return averaged_solution.detach()

    def aggregate(self, solns, **kwargs):
        if self.injection_pattern == 'output_server' and self.sampling_scheme != 'all':
            return self.aggregate_with_orignal_params(solns, **kwargs)
        else:
            return self.aggregate_with_perturbed_params(solns, **kwargs)

    def setup_clients(self, dataset):
        """Instantiates clients based on given train and test data directories

        Returns:
            all_clients: List of clients
        """
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]

        all_clients = []
        for user, group in zip(users, groups):
            if isinstance(user, str) and len(user) >= 5:
                user_id = int(user[-5:])
            else:
                user_id = int(user)
            self.all_train_data_num += len(train_data[user])
            if self.injection_pattern == 'output_server':
                c = Client(user_id, group, train_data[user], test_data[user], self.batch_size, self.worker)
            else:
                c = Client(user_id, group, train_data[user], test_data[user], self.batch_size, self.dpworker)
            all_clients.append(c)
        return all_clients
