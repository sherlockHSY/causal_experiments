from scipy.optimize import linear_sum_assignment
from scipy.linalg import lu
import itertools
import numpy as np
from scipy.optimize.optimize import main
from sklearn import linear_model
from sklearn.linear_model import LassoLarsIC, LinearRegression
from dec2bin import *
import matlab.engine
from scipy.linalg import lu
import itertools
import plot_function
import math
import numpy.matlib

def search_causal_order(matrix):
    """Obtain a causal order from the given matrix strictly.

    Parameters
        ----------
    matrix : array-like, shape (n_features, n_samples)
            Target matrix.

    Return
        ------
    causal_order : array, shape [n_features, ]
        A causal order of the given matrix on success, None otherwise.
    """
    causal_order = []

    row_num = matrix.shape[0]
    original_index = np.arange(row_num)
    
    while 0 < len(matrix):
        # find a row all of which elements are zero
        row_index_list = np.where(np.sum(np.abs(matrix), axis=1) == 0)[0]
        if len(row_index_list) == 0:
            break

        target_index = row_index_list[0]
    
        # append i to the end of the list
        causal_order.append(original_index[target_index])
        original_index = np.delete(original_index, target_index, axis=0)

        # remove the i-th row and the i-th column from matrix
        mask = np.delete(np.arange(len(matrix)), target_index, axis=0)
        matrix = matrix[mask][:, mask]

    if len(causal_order) != row_num:
        causal_order = None

    return causal_order

def estimate_causal_order(matrix):
    """Obtain a lower triangular from the given matrix approximately.

    Parameters
        ----------
    matrix : array-like, shape (n_features, n_samples)
            Target matrix.
        
    Return
    ------
    causal_order : array, shape [n_features, ]
        A causal order of the given matrix on success, None otherwise.
    """
    causal_order = None

    # set the m(m + 1)/2 smallest(in absolute value) elements of the matrix to zero
    pos_list = np.argsort(np.abs(matrix), axis=None)
    pos_list = np.vstack(np.unravel_index(pos_list, matrix.shape)).T
    initial_zero_num = int(matrix.shape[0] * (matrix.shape[0] + 1) / 2)
    for i, j in pos_list[:initial_zero_num]:
        matrix[i, j] = 0

    for i, j in pos_list[initial_zero_num:]:
        # set the smallest(in absolute value) element to zero
        matrix[i, j] = 0

        causal_order = search_causal_order(matrix)
        if causal_order is not None: 
            break

    return causal_order

def predict_adaptive_lasso(X, predictors, target, gamma=1.0):
    """Predict with Adaptive Lasso.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    predictors : array-like, shape (n_predictors)
        Indices of predictor variable.
    target : int
        Index of target variable.

    Returns
    -------
    coef : array-like, shape (n_features)
        Coefficients of predictor variable.
    """
    lr = LinearRegression()
    lr.fit(X[:, predictors], X[:, target])
    weight = np.power(np.abs(lr.coef_), gamma)
    reg = LassoLarsIC(criterion='bic')
    reg.fit(X[:, predictors] * weight, X[:, target])
    return reg.coef_ * weight

def estimate_adjacency_matrix(X, causal_order):
    """Estimate adjacency matrix by causal order.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        prior_knowledge : array-like, shape (n_variables, n_variables), optional (default=None)
            Prior knowledge matrix.

        Returns
        -------
        self : object
            Returns the instance itself.
    """
    sink_vars = []
    exo_vars = []

    B = np.zeros([X.shape[1], X.shape[1]], dtype='float64')
    for i in range(1, len(causal_order)):
        target = causal_order[i]
        predictors = causal_order[:i]

        # target is not used for prediction if it is included in exogenous variables
        if target in exo_vars:
            continue

        # sink variables are not used as predictors
        predictors = [v for v in predictors if v not in sink_vars]

        B[target, predictors] = predict_adaptive_lasso(
            X, predictors, target)

    return B

def prune(X, real_A, fake_A, N):
    W = np.linalg.inv(fake_A)
 # obtain a permuted W_ica
    _, col_index = linear_sum_assignment(1 / np.abs(W))
    PW_ica = np.zeros_like(W)
    PW_ica[col_index] = W

    # obtain a vector to scale
    D = np.diag(PW_ica)[:, np.newaxis]

    # estimate an adjacency matrix
    W_estimate = PW_ica / D
    B_estimate = np.eye(len(W_estimate)) - W_estimate

    causal_order = estimate_causal_order(B_estimate)
    
    B = estimate_adjacency_matrix(X, causal_order)
    return B


def consolidatebasis(Aobs):
    # Given an ICA basis matrix A, this function simply removes
    # zero columns and also combines any columns which are in
    # the same direction, so that the result is a basis that
    # might have been found by overcomplete ICA on sample data.

    # 去除全 0 列
    index = np.nonzero(np.sum(abs(Aobs), axis=0) > 1e-12)
    Aobs = Aobs[:, np.array(index).flatten()]

    # Next combine any vectors which are in the same directions because
    # these indicate non-confounding hidden variables that would not be
    # detected in the data anyway
    while 1:
        changed = 0
        N = Aobs.shape[1]
        #
        nAobs = Aobs / (np.ones((Aobs.shape[0], 1)) @ np.sqrt(np.sum(Aobs ** 2, axis=0)).reshape((1, N)))
        for i in range(N):
            for j in range(i + 1, N):
                if changed == 0:
                    ip = abs(nAobs[:, i].conj().T @ nAobs[:, j])
                    if ip >= 1 - 1e-12:
                        ind = np.hstack((i, j))
                        nind = np.delete(np.arange(N), ind)
                        oldnorms = np.sqrt(np.sum(Aobs[:, ind] ** 2, axis=0))
                        newnorm = np.sqrt(np.sum(oldnorms ** 2))
                        cvec = (Aobs[:, i] / np.linalg.norm(Aobs[:, i])) * newnorm
                        Aobs = np.hstack((Aobs[:, nind], cvec.reshape((cvec.shape[0],1))))
                        changed = 1

        if changed == 0:
            break
    # set any zeros with rounding errors to exact zeros
    Aobs[np.nonzero(abs(Aobs) < 1e-9)] = 0

    return Aobs 

def iperm(p):
    q = np.empty(len(p), dtype=np.int)
    for i in range(len(p)):
        t = np.nonzero(p == i)
        q[i] = np.nonzero(p == i)[0]
        
    return q

def basis2model(Aobs, means, eng):
    
    # basis2model - give possible canonical lvmodels for given overcomplete Aobs
    # # return lvmodelset 即lvmodel的集合
    if Aobs.shape[1] < Aobs.shape[0]:
        return []

    # Consolidate the observed basis
    # Aobs = consolidatebasis(Aobs)
    # 参数
    N = Aobs.shape[1]
    No = Aobs.shape[0]
    Nh = N - No
    # means = np.zeros((No,1))

    lvmodelset = []

    # 如果没有 confounding latent variables
    if Nh == 0:
        Aobs[abs(Aobs) < 1e-9] = 0

        # Dulmage–Mendelsohn分解，这里调用matlab来执行这个函数
        # 分解后矩阵会变成上三角矩阵，且返回使之变换成上三角的行列变换 p、q
        b = eng.dmperm(matlab.double(Aobs.tolist()), nargout=6)
        p = np.array(b[0], dtype=np.int).flatten()
        q = np.array(b[1], dtype=np.int).flatten()
        p = p - 1
        q = q - 1
        # 翻转的目的是为了获得下三角矩阵的行列变换
        p = np.fliplr(p.reshape((1, p.shape[0]))).flatten()
        q = np.fliplr(q.reshape((1, p.shape[0]))).flatten()

        Ap = Aobs[p, :]
        Ap = Ap[:, q]  # Ap就是下三角矩阵
        if np.sum(abs(np.triu(Ap, 1))) > 1e-12:
            return []
        Ap = Ap[iperm(p), :]
        Ap = Ap[:, iperm(p)]
        W = np.linalg.inv(Ap)
        e = 1 / np.diag(W)
        a = np.diag(W)
        a = a.reshape((N,1))
        b = np.tile(a, (1, N))
        W = W / b

        B = np.eye(N) - W
        B[np.nonzero(abs(B) < 1e-9)] = 0
        # ci = (np.eye(N) - B) * means
        # No = N
        # ci = ci.reshape((N, 1))
        # e = e.reshape((e.shape[0], 1))
        # lvmodel = Lvmodel(B, e, ci, No)
        lvmodelset.append(B)

        return lvmodelset

    # Get all possible N choose Nh combinations
    # N =5 Nh =1 No =4
    poss = []
    hidden = np.arange(Nh)
    j = 0
    while 1:
        poss.append(hidden.copy())
        if all(hidden == np.arange(No, N)):
            break
        updating = Nh

        while hidden[updating-1] == (N - Nh + updating - 1):
            updating = updating - 1
        hidden[updating-1] = hidden[updating-1] + 1
        if updating < Nh:
            for i in range(updating, Nh):
                hidden[i] = hidden[i - 1] + 1
        j = j + 1

    # Go through all possibilities for which are the hidden variables
    poss = np.array(poss)

    for i in range(poss.shape[0]):
        hidd = poss[i, :]

        v = np.ones((1, N))
        v[:, hidd] = 0
        obs = np.nonzero(v.flatten())
        obs = np.array(obs)

        # Generate augmented ICA basis matrix
        t1 = np.hstack((Aobs[:, hidd], Aobs[:, obs.flatten()]))
        t2 = np.hstack((np.eye(Nh), np.zeros((Nh, No))))
        Aaug = np.vstack((t1, t2))
        Aaug[np.nonzero(abs(Aaug) < 1e-12)] = 0

        # matlab engine
        # 分解后矩阵会变成上三角矩阵，且返回使之变换成上三角的行列变换 p、q
        b = eng.dmperm(matlab.double(Aaug.tolist()), nargout=6)
        p = np.array(b[0], dtype=np.int).flatten()
        q = np.array(b[1], dtype=np.int).flatten()
        p = p - 1
        q = q - 1

        # 目的是为了得到一个下三角矩阵 Aaug
        Aaug = Aaug[p, :]
        Aaug = Aaug[:, q]
        vn = np.arange(N-1,-1,-1,dtype=np.int)
        Aaug = Aaug[vn, :]
        Aaug = Aaug[:, vn]

        # p1,l1,u1 = lu(Aaug)
        # print(u1)
        # u2 = u1[vn, :]
        # u2 = u2[:, vn]
        # print(np.sum(abs(np.triu(Aaug, 1))) > 1e-12 or np.linalg.cond(Aaug) > 1e+12)
        if np.sum(abs(np.triu(Aaug, 1))) > 1e-12 or np.linalg.cond(Aaug) > 1e+12:
            continue

        # Waug 是对Aaug的每一列，除以其对角线元素
        Waug = np.linalg.inv(Aaug)
        e = 1 / np.diag(Waug)  # ?
        Waug = np.diag(1 / np.diag(Waug)) @ Waug

        # Bnew就是 I-A^-1
        Bnew = np.eye(N) - Waug
        Bnew[np.where(np.abs(Bnew) < 1e-12)] = 0

        fp = np.fliplr(p.reshape((1, p.shape[0]))).flatten()
        # iperm 就对原先所做的p行变换，求其逆变换
        Bnew = Bnew[iperm(fp), :]
        Bnew = Bnew[:, iperm(fp)]
        e = e[iperm(fp)]

        # Do pairwise check to see if result is stable/faithful
        failed = 0
        for i in range(No):
            for j in range(i+1, No):
                Ap = Aobs[np.hstack((i, j)), :]
                Ap = Ap[:, np.nonzero(np.sum(abs(Ap), axis=0) > 1e-12)[0]]
                if np.any(np.abs(Ap[0, :]) < 1e-12) and np.any(np.abs(Ap[1, :]) < 1e-12):
                    if (Bnew[i, j] != 0) or (Bnew[j, i] != 0):
                        failed = 1
                elif np.any(np.abs(Ap[0, :]) < 1e-12):
                    if Bnew[i, j] != 0:
                        failed = 1
                elif np.any(np.abs(Ap[1, :]) < 1e-12):
                    if Bnew[j, i] != 0:
                        failed = 1

        if failed==1:
            continue

        B = Bnew
        # b1 = Bnew[np.arange(No), :]
        # b1 = b1[:, np.arange(No)]
        # b2 = (np.eye(No) - b1) @ means

        # ci = np.zeros(N)
        # ci[0:No] = b2.flatten()
        # ci[No:N] = 0
        # ci = ci.reshape((N, 1))

        # e = e.reshape((e.shape[0], 1))

        # lvmodel = Lvmodel(B, e, ci, No)
        lvmodelset.append(B)

    return lvmodelset

def estimate2model(Aobs, means,eng):
    # 调用matlab引擎
    
    N = Aobs.shape[1]
    No = Aobs.shape[0]
    Nh = N - No
    # Test setting to zero components until we get an acceptable model(or model set). 
    # Basically, we go through all 'binary numbers' 0000 -> 1111 (where 1 means set to zero, 0 means leave alone) in
    # the natural order 0001, 0010, 0011, etc, where the entries are arranged in decreasing absolute value 
    # (so smallest absolute value are set to zero first). 
    # Any setting with less than No*(No-1)/2 forced zeros are disregarded as they cannot lead to acceptable models.
    dummy = np.sort(-abs(Aobs.flatten('F')))
    sortind = np.unravel_index(np.argsort(-abs(Aobs), axis=None), Aobs.shape)
    nind = len(Aobs.flatten('F'))
    minzeros = int(No * (No + 1) / 2)
    patnum = 0
    
    existzeros = len(np.nonzero(Aobs)[0])
    lvmodelset = []
    cutpercent = 0
    # 把绝对值最小的 minzeros 个元素置0
    pos_list = np.argsort(np.abs(Aobs), axis=None)
    pos_list = np.vstack(np.unravel_index(pos_list, Aobs.shape)).T
    for i, j in pos_list[:minzeros]:
        Aobs[i, j] = 0
    existzeros_1 = len(np.nonzero(Aobs)[0])
    
    for i, j in pos_list[minzeros:]:
        Aobs[i, j] = 0
        Azero = Aobs.copy()
        Azero = consolidatebasis(Azero) 
        if np.any(np.sum(abs(Azero), axis=1) < 1e-9):
            continue
        lvmodelset = basis2model(Azero, means, eng)
        
        if len(lvmodelset)>0:
            # 如果lvmodelset为空，那么说明没有隐变量，或者Azero的行数大于列数
            break
        

    # while 1:
    #     patnum = patnum+1
    #     if patnum >= 2 ** nind:
    #         break
    
    #     binpattern = dec2bin(patnum)
    #     zeros_prefix = '0'
    #     if len(binpattern) < nind:
    #         for i in range(1,nind - len(binpattern)):
    #             zeros_prefix = zeros_prefix + '0'
    #     binpattern = zeros_prefix + binpattern
        
    #     Azero = Aobs.copy()
    #     for i in reversed(range(nind)):
    #         if binpattern[i]=='1': 
    #             Azero[sortind[0][i],sortind[1][i]] = 0
    
    #     cutpercent = sum(abs(Azero.flatten('F')-Aobs.flatten('F')))/sum(abs(Aobs.flatten('F')))
    #     #[give possible canonical lvmodels for basis estimate Azero]
    #     Azero = consolidatebasis(Azero) 
    
    #     if np.any(np.sum(abs(Azero), axis=1) < 1e-9):
    #         continue
    
    #     # lvmodelset = basis2model(Azero, means, eng)
    #     if len(lvmodelset)>0:
    #         # 如果lvmodelset为空，那么说明没有隐变量，或者Azero的行数大于列数
    #         break

    return lvmodelset


if __name__ == '__main__':
    
    test_data_1 = np.load('test_data_9_1_B.npz')
    X = test_data_1['X']
    B = test_data_1['B']
    lat = test_data_1['lat']
    test_data = np.load('test_data_9_1.npz')
    fake_A =  test_data['fake_A']
    means= test_data['means']
    N = 5
    Nh = 1

    a = math.factorial(90) / (math.factorial(36) * math.factorial(54))
    b = 2 ** 90
    c = b-a
    lvmodelset,cutpercent = estimate2model(fake_A, means)
    print('--真实 B--')
    print(B)
    print('--估计 B--')
    print(lvmodelset[0])
    plot_function.plotmodel(B, lat,'LFOICA_dag_0')
    plot_function.plotmodel(lvmodelset[0], lat,'LFOICA_dag_1')
        
    plot_function.plotmodel(lvmodelset[1], lat,'LFOICA_dag_2')
    print('end')
