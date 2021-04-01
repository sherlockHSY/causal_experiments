from numpy.core.fromnumeric import nonzero, reshape, shape
from scipy.sparse.construct import random
from my_LFOICA import *
import matplotlib.pyplot as plt
import plot_function
from compute_result import calculate_lfoica,calculate_rcd,print_result
import lingam
from lingam.utils import print_causal_directions, print_dagc, make_dot
import warnings
import sympy as sym
from scipy.linalg import lu
import matlab.engine
import pandas as pd
import time

def gen_data(indegree, p, randomseed,num_measured_var,data_size=1000):
    
    np.random.seed(randomseed)
    print("当前使用随机种子：",randomseed)
    # 生成非高斯噪声的函数
    # get_noises = lambda n: np.array(np.random.normal(0, 0.5, n)**3)
    get_noises = lambda n: np.array(np.random.uniform(-0.5, 0.5,n))

    num_obs = int(num_measured_var * (1-p))  # 观察变量数量
    num_lat = int(num_measured_var - num_obs) # 隐变量数量
    # 生成噪声
    e = np.zeros((data_size, num_measured_var)) # (1000,10)
    for i in range(num_measured_var):
        c = get_noises(data_size)
        e[:,i] = c
    
    # 基于lvlingam，我们得到的模型应该是canonical的
    # 也就是隐变量放置最后列，且所有隐变量都是root结点
    m_var = np.arange(num_measured_var)
    obs = m_var[:num_obs]
    lat = m_var[num_obs:]
    
    # 生成邻接矩阵B，这里要控制平均入度，还要控制 causal strength 在(-1,-0.2]U[0.2,1)
    B_obs_tril = np.random.rand(num_obs, num_obs)
    B_temp = np.random.randn(num_obs, num_obs)
    B_obs_tril = np.sign(B_temp) * (abs(B_obs_tril))
    B_obs_tril = np.tril(B_obs_tril, -1) # 下三角，为了不能成环
    
    # 创建隐变量与观察变量之间的联系
    B_lat = np.zeros((num_obs, num_lat))
    for i in range(num_lat):
        while 1:
            b_temp = np.random.randn(num_obs, 1)
            b = np.random.rand(num_obs, 1)
            b = np.sign(b_temp) * (abs(b))
            # 限制隐变量孩子节点个数不大于3
            b = b.flatten()
            for bi in range(len(b)):
                if b[bi]<0.2:
                    b[bi]=0.
        
            if len(np.nonzero(b)[0]) <= 3 and len(np.nonzero(b)[0]) >= 1:
                break
            
        B_lat[:,i] = b
    
    # 控制因果强度在 [-1,0.2)U(0.2,1]
    for i in range(num_obs):
        for j in range(num_obs):
            if B_obs_tril[i ,j] != 0:
                if B_obs_tril[i, j] < 0.2 and B_obs_tril[i, j] > 0:
                    B_obs_tril[i, j] = B_obs_tril[i, j] + 0.2
                if B_obs_tril[i, j] > -0.2 and B_obs_tril[i, j] < 0:
                    B_obs_tril[i, j] = B_obs_tril[i, j] - 0.2

    # 合并观察变量与隐变量，此时的观察变量是下三角的
    B1 = np.concatenate((B_obs_tril, B_lat),axis=1)
    B1 = np.concatenate((B1,np.zeros((num_lat,num_measured_var))),axis=0)
                
    # 控制平均入度，即需要将多少个点变成0或赋值
    num_do = int(len(np.nonzero(B1)[0]) - indegree * num_measured_var)
    if num_do > 0:
        order = np.random.permutation(len(np.nonzero(B1)[0]))[:num_do]
        flag = 0
        for i in range(num_measured_var):
            for j in range(num_measured_var):
                if B1[i ,j] != 0 :
                    if np.isin(flag, order) == True:
                        B1[i, j] = 0
                    flag = flag + 1
    elif num_do < 0:
        order = np.random.permutation(int(num_obs *(num_obs-1) / 2))[:abs(num_do)]
        # print('需要添加一些边') 
        flag = abs(num_do)
        for i in range(num_obs): # 我们先在观察变量加
            for j in range(num_obs):
                if B1[i ,j] == 0 :
                    if i>j and flag>0:
                        B1[i ,j] = np.random.rand(1)[0] + 0.2
                        flag = flag - 1
                    
        if flag>0: # 还需要在隐变量加
            for i in range(num_lat):
                b1 = B1[:,num_obs+i]
                add_num = 3 -len( np.nonzero(b1)[0])
                for j in range(num_obs):
                    if add_num>0 and B1[j,num_obs+i] ==0 and flag>0:
                        B1[j,num_obs+i] = np.random.rand(1)[0] + 0.2
                        flag = flag - 1
                        add_num = add_num -1   
                if flag==0:
                    break;
          
                
    # 检验一下平均入度
    a = len(np.nonzero(B1[:,num_obs:])[0])
    b = len(np.nonzero(B1[:num_obs,:num_obs])[0])
    aver_degree = len(np.nonzero(B1)[0])/num_measured_var
    if aver_degree == indegree:
        print('平均入度校验正确')
    else:
        print('平均入度校验失败,当前真实平均入度为: {}'.format(aver_degree))
     
    # 获取经过调整边权的关系矩阵
    B_obs_tril = B1[:num_obs,:]
    B_obs_tril = B_obs_tril[:,:num_obs]
    
    # 对观察数据引入线性关系
    X = e.copy()
    for i in range(num_obs):
        X[:,i] = X.dot(B1[i].T) + e[:,i]
    # 得到观察数据X
    X_obs = X[:,:num_obs]

    # 对 X 和噪声e 和 B  随机变换，获得一组因果顺序
    k = np.random.permutation(num_obs)
    X_obs = X_obs[:,k]
    e_obs = e[:,:num_obs]
    e_obs = e_obs[:,k]
    e_lat = e[:,num_obs:]
    e = np.concatenate((e_obs,e_lat),axis=1)

    B_obs = B_obs_tril[k, :]
    B_obs = B_obs[:, k]
    B_lat = B_lat[k,:]
    # 真正的非下三角的B
    B = np.concatenate((B_obs,B_lat), axis=1)
    B = np.concatenate((B, np.zeros((num_lat,num_measured_var))),axis=0)

    # 因果顺序
    ki = k.copy()
    for i in range(len(k)):
        ki[k[i]] = i
    causal_order = np.concatenate((lat,ki))

    # 混合矩阵 A 注意是 overcomplete 的
    A = np.linalg.inv(np.eye(num_measured_var)-B)
    A = A[:num_obs,:]

    # 均值
    means = np.mean(X,axis=0)
    means = means.reshape((means.shape[0],1))
    
    return X_obs,e_obs,B,A,means,lat,causal_order,num_obs,num_lat

    
if __name__ == '__main__':
 
    print('--正在调用matlab引擎--')
    eng = matlab.engine.start_matlab()
    print('--调用成功--')

    indegree_arr = [2]
    p_arr = [0.2]
    randomseed_arr = [0,10,20,30,40,50,60,70,80,90]
    num_measured_var = 10


    df = pd.DataFrame(columns=['p', 'indegree', 'algorithm','Precision_causality','Recall_causaltity','F1_causality','Precision_latent','Recall_latent','F1_latent','mse','time'])
    for p in p_arr:
        for indegree in indegree_arr:
            i = 0
            P_c,R_c,F1_c,P_l,R_l,F1_l,mse,aver_time = 0.,0.,0.,0.,0.,0.,0.,0.
            for randomseed in randomseed_arr:
                i = i +1
                # 生成数据
                X,e,B,A,means,lat,causal_order,num_obs,num_lat = gen_data(indegree, p, randomseed,num_measured_var)
                # print('---真实因果顺序---')
                # print(causal_order)
                plot_function.plotmodel(B, lat, p,indegree,0,i)
                data_arrs = {
                    'arr_0': X,
                    'arr_1': e,
                    'arr_2': A
                }

                # 调用LFOICA 
                print('--开始LFOICA实验--')
                algorithm = 'LFOICA'
                start = time.time()
                # np.savez('test_data_9_1_B',B=B,X=X,lat=lat)
                fake_A,lvmodelset = LFOICA_exp(data_arrs, num_obs, num_measured_var,means,eng)
                B_est = lvmodelset[0]
                end = time.time()
                run_time = end-start
                # 计算指标
                precision_c,recall_c,F1_score_c,precision_l,recall_l,F1_score_l,my_mse = calculate_lfoica(X,e,B,B_est,num_obs,num_lat)
                plot_function.plotmodel(lvmodelset[0], lat, p,indegree,1,i)
                    
                # 调用RCD
                # print('--开始RCD实验--')
                # algorithm = 'RCD'
                # start = time.time()
                # model = lingam.RCD()
                # warnings.filterwarnings('ignore', category=UserWarning)
                # model.fit(X)
                # B_RCD = model.adjacency_matrix_
                # end = time.time()
                # run_time = end -start
                # precision_c,recall_c,F1_score_c,precision_l,recall_l,F1_score_l,my_mse = calculate_rcd(X,e,B,B_RCD,num_obs,num_lat)
                # plot_function.plotmodel(B_RCD,lat,p,indegree,2,i)
                
                # ancestors_list = model.ancestors_list_
                # # for i, ancestors in enumerate(ancestors_list):
                # #     print(f'M{i}={ancestors}')
                # dot = make_dot(model.adjacency_matrix_)
                # dot.format = 'png'
                # dot.render('RCD_dag.png')
                # plot_function.plotmodel(B_RCD,lat,'RCD_dag')
                # warnings.filterwarnings('ignore', category=UserWarning)
                
                # 调用ParceLiNGAM
                # print('--开始ParceLiNGAM实验--')
                # algorithm = 'ParceLiNGAM'
                # start = time.time()
                # model2 = lingam.BottomUpParceLiNGAM()
                # model2.fit(X)
                # end = time.time()
                # run_time = end -start
                # B_Parce = model2.adjacency_matrix_
                # precision_c,recall_c,F1_score_c,precision_l,recall_l,F1_score_l,my_mse = calculate_rcd(X,e,B,B_Parce,num_obs,num_lat)
                # plot_function.plotmodel(B_Parce,lat,p,indegree,3,i)

                aver_time = aver_time + run_time
                P_c = P_c + precision_c
                R_c = R_c + recall_c
                F1_c = F1_c + F1_score_c
                P_l = P_l + precision_l
                R_l = R_l + recall_l
                F1_l = F1_l + F1_score_l
                mse = mse + my_mse

            iter = len(randomseed_arr)
            aver_time = aver_time / iter
            P_c = P_c / iter
            R_c = R_c / iter
            F1_c = F1_c / iter
            P_l = P_l / iter
            R_l  = R_l / iter
            F1_l = F1_l / iter
            mse = mse / iter
            print_result(p,indegree,algorithm,P_c,R_c,F1_c,P_l,R_l,F1_l,mse,aver_time)
            # 存储实验结果
            df = df.append({'p':p,
                'indegree': indegree,
                'algorithm': algorithm,
                'Precision_causality': P_c,
                'Recall_causaltity': R_c,
                'F1_causality': F1_c,
                'Precision_latent': P_l,
                'Recall_latent': R_l,
                'F1_latent': F1_l,
                'mse': mse,
                'time': aver_time
                },ignore_index=True)
    
    
    df.to_csv('experiment_result.csv')
    print('---end---')
    
