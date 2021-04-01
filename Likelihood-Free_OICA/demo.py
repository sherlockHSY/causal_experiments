import numpy as np
import itertools

from torch.nn.functional import mse_loss
import LFOICA_prune
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

num_obs = 4
num_lat = 1
test_data = np.load('test_result_4_1.npz')
X = test_data['X']
A_real = test_data['A']
e = test_data['e']
B_real = test_data['B_real']
B_RCD =  test_data['B_RCD']
B_LFOICA_set = test_data['B_LFOICA_set']
B_LFOICA = B_LFOICA_set[0]
N = B_real.shape[0]

# confounder
TP,FP,TN,FN =0,0,0,0
B_lat = B_real[:,num_obs:]
positive = len(np.nonzero(B_lat)[0])

for j in range(num_obs,N):
    for i in range(num_obs):
        if B_real[i][j] != 0. and len(np.nonzero(np.isnan(B_RCD[i]))[0]) > 0:
            TP = TP + 1
        if B_real[i][j] == 0. and len(np.nonzero(np.isnan(B_RCD[i]))[0]) > 0:
            FP = FP + 1   

precision_rcd_l = TP/(TP+FP)
recall_rcd_l = TP/positive
F1_rcd_l = (2 * precision_rcd_l * recall_rcd_l)/(precision_rcd_l + recall_rcd_l)

print(precision_rcd_l)
print(recall_rcd_l)
print(F1_rcd_l)


# X1 = e @ A_real.T
# m1 = mean_squared_error(X, X1)
# print(m1)




