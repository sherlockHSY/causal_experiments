import numpy as np
from scipy.optimize.optimize import main
from sklearn import linear_model
from sklearn.linear_model import LassoLarsIC, LinearRegression
import matlab.engine

import itertools



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
        try:
             W = np.linalg.inv(Ap)
        except:
            return []

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
    minzeros = int(No * (No - 1) / 2)
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
