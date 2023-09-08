import numpy as np
from src.utils.flops_counter import get_model_complexity_info
from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
from src.utils.privacy_accountant import noise_injection
from src.utils.autograd_hacks import *
import torch.nn as nn
import torch

criterion = nn.CrossEntropyLoss()
mseloss = nn.MSELoss()


class Worker(object):
    """
    Base worker for all algorithm. Only need to rewrite `self.local_train` method.

    All solution, parameter or grad are Tensor type.
    """

    def __init__(self, model, optimizer, options):
        # Basic parameters
        self.model = model
        self.optimizer = optimizer
        self.batch_size = options['batch_size']
        self.num_epoch = options['num_epoch']
        self.gpu = options['gpu'] if 'gpu' in options else False
        if options["model"] == '2nn' or options["model"] == 'logistic':
            self.flat_data = True
        else:
            self.flat_data = False

        # Setup local model and evaluate its statics
        self.flops, self.params_num, self.model_bytes = \
            get_model_complexity_info(self.model, options['input_shape'], gpu=options['gpu'])

    @property
    def model_bits(self):
        return self.model_bytes * 8

    def flatten_data(self, x):
        if self.flat_data:
            current_batch_size = x.shape[0]
            return x.reshape(current_batch_size, -1)
        else:
            return x

    def get_model_params(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def load_model_params(self, file):
        model_params_dict = get_state_dict(file)
        self.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def get_flat_grads(self, dataloader):
        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.to(torch.device('mps')), y.to(torch.device('mps'))
            pred = self.model(x)
            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def local_train(self, train_dataloader, **kwargs):
        """Train model locally and return new parameter and computation cost

        Args:
            train_dataloader: DataLoader class in Pytorch

        Returns
            1. local_solution: updated new parameter
            2. stat: Dict, contain stats
                2.1 comp: total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS)
                2.2 loss
        """
        self.model.train()
        train_loss = train_acc = train_total = 0
        for epoch in range(self.num_epoch):
            x, y = list(train_dataloader)[0]
            if self.gpu:
                x, y = x.to(torch.device('mps')), y.to(torch.device('mps'))

            self.optimizer.zero_grad()
            clear_backprops(self.model)
            pred = self.model(x)

            loss = criterion(pred, y)
            loss.backward(retain_graph=True)
            compute_grad1(self.model)

            torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)

            self.optimizer.step()

        _, predicted = torch.max(pred, 1)
        correct = predicted.eq(y).sum().item()
        target_size = y.size(0)

        train_loss += loss.item() * y.size(0)
        train_acc += correct
        train_total += target_size

        local_solution = self.get_flat_model_params()

        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict

    def local_test(self, test_dataloader):
        self.model.eval()
        self.model.to(torch.device('mps'))
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                # if self.gpu:
                x, y = x.to(torch.device('mps')), y.to(torch.device('mps'))

                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()

                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)
        # self.model.to(torch.device('cpu'))
        return test_acc, test_loss


class GradientPerburtedWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.sigma = None
        self.clip = 100
        self.num_epoch = options['num_epoch']
        super(GradientPerburtedWorker, self).__init__(model, optimizer, options)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def local_train(self, train_dataloader, **kwargs):
        """Train model locally and return new parameter and computation cost

        Ags:
            train_dataloader: DataLoader class in Pytorch

        Returns
            1. local_solution: updated new parameter
            2. stat: Dict, contain stats
                2.1 comp: total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS)
                2.2 loss
        """
        self.model.train()
        train_loss = train_acc = train_total = 0
        for epoch in range(self.num_epoch):
            train_loss = train_acc = train_total = 0

            x, y = list(train_dataloader)[0]
            if self.gpu:
                x, y = x.to(torch.device('mps')), y.to(torch.device('mps'))

            self.optimizer.zero_grad()
            clear_backprops(self.model)
            pred = self.model(x)

            loss = criterion(pred, y)
            loss.backward(retain_graph=True)
            compute_grad1(self.model)


            layer_norm = []
            for p in self.model.parameters():
                layer_norm.append(float(torch.norm(p.grad.data, 2)))
            mean_squared_average = np.mean(np.array(layer_norm) ** 2) + 1e-5
            layer_norm = [float(self.clip) / mean_squared_average * norm for norm in layer_norm]

            # torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
            for net_idx, p in enumerate(self.model.parameters()):
                # clipping single gradients
                # heuristic: otherwise, use max_norm constant
                max_norm = np.median([float(grad.data.norm(2)) for grad in p.grad1.data])
                max_norm = min(max_norm, layer_norm[net_idx])
                # print(f'{epoch+1}/{self.num_epoch}: {net_idx} {max_norm}')

                grad1 = torch.stack(
                    [grad / max(1, float(grad.data.norm(2)) / max_norm) for grad in p.grad1.data])
                p.grad.data = torch.mean(grad1, dim=0)
                # DP mechanism
                p.grad.data = noise_injection(p.grad.data, self.sigma, max_norm, self.batch_size, self.gpu)

            self.optimizer.step()

            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)

            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_total += target_size
            # print()

        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict


class OutputPerturbedWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.clip = float(options['clip_size'])
        self.sigma = None
        self.num_epoch = options['num_epoch']
        super(OutputPerturbedWorker, self).__init__(model, optimizer, options)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def local_train(self, train_dataloader, **kwargs):
        """Train model locally and return new parameter and computation cost

        Args:
            train_dataloader: DataLoader class in Pytorch

        Returns
            1. local_solution: updated new parameter
            2. stat: Dict, contain stats
                2.1 comp: total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS)
                2.2 loss
        """
        self.model.train()
        train_loss = train_acc = train_total = 0
        for epoch in range(self.num_epoch):
            train_loss = train_acc = train_total = 0

            x, y = list(train_dataloader)[0]
            if self.gpu:
                x, y = x.to(torch.device('mps')), y.to(torch.device('mps'))

            self.optimizer.zero_grad()
            clear_backprops(self.model)
            pred = self.model(x)

            loss = criterion(pred, y)
            loss.backward(retain_graph=True)
            compute_grad1(self.model)

            torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)

            self.optimizer.step()

            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)

            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_total += target_size
            # print()

        layer_norm = []
        for p in self.model.parameters():
            layer_norm.append(float(torch.norm(p.grad.data, 2)))
        mean_squared_average = np.mean(np.array(layer_norm)**2) + 1e-5
        layer_norm = [self.clip / mean_squared_average * norm for norm in layer_norm]

        for net_idx, p in enumerate(self.model.parameters()):
            # get the size of the current layer
            max_norm = layer_norm[net_idx]
            # print(f'layer {net_idx}: max norm this layer is {max_norm}')
            # clip the local solutions
            if max_norm < 1e-5:
                max_norm = 1e-5
            p.grad.data = p.grad.data / max(1., float(torch.norm(p.grad.data, 2)) / max_norm)
            # inject noise to the clients' params
            p.grad.data = noise_injection(p.grad.data, self.sigma, max_norm, self.batch_size, self.gpu)

        local_solution = self.get_flat_model_params()

        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict
