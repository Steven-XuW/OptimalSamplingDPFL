import torch.nn as nn
import torch.nn.functional as F
import importlib
import math


class Logistic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        logit = self.layer(x)
        return logit


class TwoHiddenLayerFc(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(TwoHiddenLayerFc, self).__init__()
        self.fc1 = nn.Linear(input_shape, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, out_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class DeepNN(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(DeepNN, self).__init__()

        # Input layer to first hidden layer
        self.fc1 = nn.Linear(input_shape, 256)

        # Additional hidden layers
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)

        # Output layer
        self.fc_out = nn.Linear(32, out_dim)

    def forward(self, x):
        # Activation functions are ReLU for hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # No activation function for the output layer
        x = self.fc_out(x)
        return x

class LeNet(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class TwoConvOneFc(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(TwoConvOneFc, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, out_dim)

    # def forward(self, x):
    #     out = []
    #     out.append(F.relu(self.conv1(x)))
    #     out[-1].retain_grad()
    #
    #     out.append(F.max_pool2d(out, 2))
    #     out[-1].retain_grad()
    #
    #     out.append(F.relu(self.conv2(out)))
    #     out[-1].retain_grad()
    #
    #     out.append(F.max_pool2d(out, 2))
    #     out[-1].retain_grad()
    #
    #     out.append(out[-1].view(out[-1].size(0), -1))
    #     out[-1].retain_grad()
    #
    #     out.append(out.view(out.size(0), -1))
    #     out = F.relu(self.fc1(out))
    #     out = self.fc2(out)
    #     return out
    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.max_pool2d(out1, 2)
        out3 = F.relu(self.conv2(out2))
        out4 = F.max_pool2d(out3, 2)
        out5 = out4.view(out4.size(0), -1)
        out6 = F.relu(self.fc1(out5))
        out7 = self.fc2(out6)

        return out7

class CifarCnn(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(CifarCnn, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*5*5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def choose_model(options):
    model_name = str(options['model']).lower()
    if model_name == 'logistic':
        return Logistic(options['input_shape'], options['num_class'])
    elif model_name == '2nn':
        return TwoHiddenLayerFc(options['input_shape'], options['num_class'])
    elif model_name == 'dnn':
        return DeepNN(options['input_shape'], options['num_class'])
    elif model_name == 'cnn':
        return TwoConvOneFc(options['input_shape'], options['num_class'])
    elif model_name == 'ccnn':
        return CifarCnn(options['input_shape'], options['num_class'])
    elif model_name == 'lenet':
        return LeNet(options['input_shape'], options['num_class'])
    elif model_name.startswith('vgg'):
        mod = importlib.import_module('src.models.vgg')
        vgg_model = getattr(mod, model_name)
        return vgg_model(options['num_class'])
    else:
        raise ValueError("Not support model: {}!".format(model_name))
