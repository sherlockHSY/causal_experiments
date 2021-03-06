
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from libs.distance_measure.mmd import mix_rbf_mmd2
from libs.common_utils import cos_act, normalize_mixing_matrix
from scipy.stats import semicircular
from scipy.stats import hypsecant
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import argparse
from libs.pytorch_pgm import PGM, prox_soft, prox_plus
import matplotlib.pyplot as plt
import plot_function

# standard pytorch dataset
class dataset_simul(Dataset):
    def __init__(self, data_path):
        data_arrs = np.load(data_path)
        self.mixtures = Tensor(data_arrs['arr_0'])
        self.components = data_arrs['arr_1']
        self.A = Tensor(data_arrs['arr_2'])
        self.data_size = self.mixtures.shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        mixtures_sample = self.mixtures[idx, :]
        components_sample = self.components[idx, :]
        return mixtures_sample, components_sample

    def get_real_A(self):
        return self.A

    def get_real_components(self, batch_size):
        assert batch_size <= self.data_size
        np.random.shuffle(self.components)
        return self.components[0:batch_size, :]


# transform random noise into components
class Gaussian_transformer(nn.Module):
    def __init__(self, num_components):
        super().__init__()
        self.num_components = num_components
        self.m = 1  # number of gaussian components for each channel in our non-gaussian noise generation model
        self.random_feature_mapping = nn.ModuleList()
        self.D = 8
        self.models = nn.ModuleList()
        for i in range(num_components):
            random_feature_mapping = nn.Linear(self.m, self.D)
            torch.nn.init.normal_(random_feature_mapping.weight, mean=0, std=1)
            torch.nn.init.uniform_(random_feature_mapping.bias, a=0, b=2 * math.pi)
            random_feature_mapping.weight.requires_grad = False
            random_feature_mapping.bias.requires_grad = False
            self.random_feature_mapping.append(random_feature_mapping)
        for i in range(num_components):  # different channels have different networks to guarantee independent
            model = nn.Sequential(
                nn.Linear(self.D, 2 * self.D),
                nn.ELU(),
                # nn.Linear(2*self.D, 4*self.D),
                # nn.ELU(),
                # nn.Linear(4 * self.D, 2*self.D),
                # nn.ELU(),
                nn.Linear(2 * self.D, 1)
            )
            self.models.append(model)

    def forward(self, batch_size):
        # gaussianNoise = Tensor(np.random.normal(0, 1, [batch_size, num_components, self.m])).to(device)
        gaussianNoise = Tensor(np.random.uniform(-1, 1, [batch_size, self.num_components, self.m])).to(device)
        output = Tensor(np.zeros([batch_size, self.num_components])).to(device)  # batchSize * k * channels
        cos_act_func = cos_act()
        for i in range(self.num_components):  # output shape [batchSize, k, n]
            tmp = self.random_feature_mapping[i](gaussianNoise[:, i, :])
            tmp = cos_act_func(tmp)
            output[:, i] = self.models[i](tmp).squeeze()
        return output


# the generative process mimic the mixing procedure from components to mixtures
class Generative_net(nn.Module):
    def __init__(self, num_mixtures, num_components, A):
        super().__init__()
        # for simulation exp, we initialize A with it's true value added with some large noise to avoid local optimum.
        # all methods are compared under the same initialization
        self.A = nn.Parameter(A + torch.Tensor(np.random.uniform(-0.2, 0.2, [num_mixtures, num_components])))


    def forward(self, components):
        batch_size = components.shape[0]
        result = torch.mm(components, self.A.t())
        return result


parser = argparse.ArgumentParser()
parser.add_argument('--num_mixtures', type=int, default=5)
parser.add_argument('--num_components', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=2000)
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--lr_T', type=float, default=0.01)
parser.add_argument('--lr_G', type=float, default=0.001)
parser.add_argument('--reg_lambda', type=float, default=0)
parser.add_argument('--print_int', type=int, default=500)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()
sigmaList = [0.001, 0.01]
print_int = 100
if (args.cuda):
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

dataset = dataset_simul(
    'data/OICA_data/OICA_data_{}mixtures_{}components.npz'.format(args.num_mixtures, args.num_components))
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# ???????????????
num_components = 10
num_mixtures = 5
lat = np.arange(num_mixtures,num_components)
A1 = dataset.get_real_A().numpy()
A2 = np.concatenate((A1, np.eye(num_components)[num_mixtures:,:]),axis=0)
real_B = np.eye(num_components) - np.linalg.inv(A2)

plot_function.plotmodel(real_B,lat,'o_origin_dag')

transformer = Gaussian_transformer(args.num_components).to(device)
generator = Generative_net(args.num_mixtures, args.num_components, dataset.get_real_A()).to(device)
transformer_optimizer = optim.Adam(transformer.parameters(), lr=args.lr_T)
generator_optimizer = optim.Adam(generator.parameters(), lr=args.lr_G)

for epoch in range(args.num_epochs):
    if epoch % print_int == 0:
        print("??????????????? %d ???" % epoch)
    for step, (real_mixtures, real_components) in enumerate(dataloader):
        generator.zero_grad()
        transformer.zero_grad()
        real_mixtures = real_mixtures.to(device)
        batch_size_i = real_mixtures.shape[0]
        fake_components = transformer.forward(batch_size_i)
        fake_mixtures = generator.forward(fake_components)
        MMD = torch.sqrt(F.relu(mix_rbf_mmd2(fake_mixtures, real_mixtures, sigmaList)))
        MMD.backward()
        transformer_optimizer.step()
        generator_optimizer.step()
    if epoch % args.print_int == 0 and epoch != 0:
        print('##########epoch{}##########'.format(epoch))
        MSE_func = nn.MSELoss()
        real_A = torch.abs(dataset.get_real_A())
        fake_A = torch.abs(list(generator.parameters())[0]).detach().cpu()
        real_A, fake_A = normalize_mixing_matrix(real_A, fake_A)
        # for i in range(num_components):
        #     fake_A[:, i]/=normalize_factor[i]
        print('estimated A', fake_A)
        print('real A', real_A)
        MSE = MSE_func(real_A, fake_A)
        print('MSE: {}, MMD: {}'.format(MSE, MMD))

print('##########epoch{}##########'.format(epoch))
MSE_func = nn.MSELoss()
real_A = torch.abs(dataset.get_real_A())
fake_A = torch.abs(list(generator.parameters())[0]).detach().cpu()
real_A, fake_A = normalize_mixing_matrix(real_A, fake_A)
# for i in range(num_components):
#     fake_A[:, i]/=normalize_factor[i]
print('estimated A', fake_A)
print('real A', real_A)
MSE = MSE_func(real_A, fake_A)
print('MSE: {}, MMD: {}'.format(MSE, MMD))

# ????????????

# ????????? B
fake_A_nd = fake_A.numpy()
fake_A_nd = np.append(fake_A_nd,np.eye(num_components)[num_mixtures:,:],axis=0)
fake_B = np.eye(num_components)-np.linalg.inv(fake_A_nd)
plot_function.plotmodel(fake_B, lat,'o_origin_dag')