import numpy as np
from scipy.linalg.decomp_svd import null_space
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
from torch.utils.data import Dataset, DataLoader
import argparse
from libs.pytorch_pgm import PGM, prox_soft, prox_plus
from scipy.linalg import lu
import itertools
import LFOICA_prune

# 使用gpu
device = torch.device('cuda:0')
# standard pytorch dataset
class dataset_simul(Dataset):
    def __init__(self, data_arrs):
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

def perms(x):
    """Python equivalent of MATLAB perms."""
    x = range(x)
    return np.vstack(list(itertools.permutations(x)))[::-1]

def matchbases(Aest, N):
    Nest = len(Aest)
    Amatched = []

    Amatched.append(Aest[0])
    A1 = Amatched[0]

    for j in range(1,Nest):
    
        # Go through all permutations, pick best fit
        bestval = np.inf
        allperms = perms(N) # 所有可能的排列 组合
        nperms = allperms.shape[0]
        A2 = Aest[j]
        # 比较，选出最优的排列
        Am = None
        for i in range(nperms):
            A2p = A2[:,(allperms[i,:])]
            A2pp = A2p.conj().transpose()
            s = np.sign(np.diag(A2pp @ A1))
            A2p = A2p @ np.diag(s)
            c = np.sum(np.sum(abs(A2p - A1)))
            if c<bestval:
                bestval = c
                Am = A2p
	
        Amatched.append(Am)

    return Amatched



def LFOICA_exp(data_arrs, num_mixtures, num_components, means,eng,num_epochs = 100, batch_size = 80, print_int = 10):
    
    # 参数设置
    sigmaList = [0.001, 0.01]
    lr_T = 0.01
    lr_G = 0.001
    reg_lambda = 0.
    # 加载数据
    dataset = dataset_simul(data_arrs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 构造transformer，用于把
    transformer = Gaussian_transformer(num_components).to(device)

    generator = Generative_net(num_mixtures, num_components, dataset.get_real_A()).to(device)

    transformer_optimizer = optim.Adam(transformer.parameters(), lr=lr_T)

    generator_optimizer = optim.Adam(generator.parameters(), lr=lr_G)

    for epoch in range(num_epochs):
        # if epoch % print_int == 0:
        #     print("正在迭代第 %d 次" % epoch)
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
    
    # print('共迭代 {} 次'.format(epoch+1))
    MSE_func = nn.MSELoss()
    real_A = torch.abs(dataset.get_real_A())
    fake_A = torch.abs(list(generator.parameters())[0]).detach().cpu()
    real_A, fake_A = normalize_mixing_matrix(real_A, fake_A)
    # for i in range(num_components):
    #     fake_A[:, i]/=normalize_factor[i]
    # print('estimated A', fake_A)
    # print('real A', real_A)
    MSE = MSE_func(real_A, fake_A)
    print('MSE: {}, MMD: {}'.format(MSE, MMD))

    
    # 需要剪枝
    real_A_nd = real_A.numpy()
    fake_A_nd = fake_A.numpy()
    # np.savez('test_data_9_1',fake_A=fake_A_nd,means=means)
    # 要进行剪枝，利用lvlingam的剪枝策略
    lvmodelset = LFOICA_prune.estimate2model(fake_A_nd, means, eng)
    return fake_A_nd,lvmodelset
    
    # 计算均值
    means =  dataset.get_means()

    # 利用ica_lingam的剪枝策略
    B1 = LFOICA_prune.Prune(X, real_A, fake_A, num_components)
    

    # 恢复出 B

    # fake_A_nd = np.append(fake_A_nd, np.eye(num_components)[num_mixtures:,:],axis=0)
    
    
    # bootstrap obtaining a set of estimates Ai representing our uncertainty regarding the elements of the mixing matrix
    # Nest = 20
    # Aest = []
    # mest = []
    # noiselevel = 0.1 # 估计时的噪声
    # for i in range(Nest):
    #     rp = np.random.permutation(num_components)
    #     Aest.append(fake_A_nd[:, rp])
    #     # Aest[i] = Aest[i] + noiselevel * np.random.randn(num_mixtures, num_components)
    #     mest.append(noiselevel * np.random.randn(means.shape[0],means.shape[1]))

    # # Match up all the bases
    # Amatched = matchbases(Aest, num_components)
  
    # # Calculate mean and variance of 'estimated' ICA coefficients
    # Am = np.zeros(fake_A_nd.shape)
    # mm = np.zeros(means.shape)
    # for i in range(Nest):
    #     Am = Am + Amatched[i]
    #     mm = mm + mest[i]

    # Am = Am / Nest
    # mm = mm / Nest;
    # Av = np.zeros(Am.shape)
    # for i in range(Nest):
    #   Av = Av + (Amatched[i] - Am) ** 2
    
    # Av = Av / Nest
  
    # # Infer zeros and set them to zero
    # threshold = 1
    # zeromat = np.abs(Am) < (threshold * np.sqrt(Av))
    # for i in range(zeromat.shape[0]):
    #     for j in range(zeromat.shape[1]):
    #        if zeromat[i,j] == True:
    #            Am[i,j] = 0
    # print('indetify zeros in A')
    # print(zeromat)
    # real_zeromat = (real_A_nd == 0)
    # print(real_zeromat)
    # print(zeromat)
    # for i in range(Nest):
    #     Amatched[i](zeromat) = 0;

    
    # Calculate statistics on success in identifying zeros



    
    # 寻找因果顺序
    # 1e12


    # identify zero
    
    p,l,u =  lu(fake_A_nd)
    print(u)

    fake_B = np.eye(num_components)-np.linalg.inv(fake_A_nd)

    print('fake_B',fake_B)
    return fake_B