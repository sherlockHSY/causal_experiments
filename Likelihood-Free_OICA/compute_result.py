import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def print_result(p,indegree,algorithm,precision_c,recall_c,F1_c,precision_l,recall_l,F1_l,mse,run_time):
    print('---p: {0} , indegree: {1}  算法: {2}的结果---'.format(p,indegree,algorithm))
    print('Precision of causality: {}'.format(precision_c))
    print('Recall of causality: {}'.format(recall_c))
    print('F1 score of causality: {}'.format(F1_c))
    print('Precision of lantent confounder: {}'.format(precision_l))
    print('Recall of lantent confounder: {}'.format(recall_l))
    print('F1 score of lantent confounder: {}'.format(F1_l))
    print('MSE: {}'.format(mse))
    print('running time: {}'.format(run_time))


# 计算RCD算法的指标
def calculate_rcd(X,e,B_real,B_est,num_obs,num_lat):
    N = num_obs + num_lat
    # causality
    TP,FP,TN,FN =0,0,0,0
    for i in range(num_obs):
        for j in range(num_obs):
            if B_real[i][j] !=0. and B_est[i][j] != 0.:
                TP = TP + 1
            elif B_real[i][j] !=0. and B_est[i][j] == 0.:   
                FN = FN + 1
            elif B_real[i][j] ==0. and B_est[i][j] != 0.:  
                FP = FP + 1            

    precision_c = TP/(TP+FP) if (TP+FP)!=0 else 0
    recall_c = TP/(TP+FN) if (TP+FN)!=0 else 0
    F1_c = (2 * precision_c * recall_c) / (precision_c + recall_c) if (precision_c + recall_c)!=0 else 0
    
    # confounder
    TP,FP,TN,FN =0,0,0,0
    B_lat = B_real[:,num_obs:]
    positive = len(np.nonzero(B_lat)[0])

    for j in range(num_obs,N):
        for i in range(num_obs):
            if B_real[i][j] != 0. and len(np.nonzero(np.isnan(B_est[i]))[0]) > 0:
                TP = TP + 1
            if B_real[i][j] == 0. and len(np.nonzero(np.isnan(B_est[i]))[0]) > 0:
                FP = FP + 1   

    precision_l = TP/(TP+FP) if (TP+FP)!=0 else 0
        
    recall_l = TP/positive if (positive)!=0 else 0
    F1_l = (2 * precision_l * recall_l)/(precision_l + recall_l) if (precision_l + recall_l)!=0 else 0
    # MSE
    mse = 0.
    for i in range(num_obs):
        for j in range(num_obs):
            if np.isnan(B_est[i,j]):
                B_est[i,j] = 0
    
    X1 = (X.dot(B_est.T)) + e
    # B_real_obs.T
    mse = mean_squared_error(X,X1)

    return precision_c,recall_c,F1_c,precision_l,recall_l,F1_l,mse

# 计算LFOICA算法的指标
def calculate_lfoica(X,e,B_real,B_est,num_obs,num_lat):
    N = num_obs + num_lat
    # causality
    TP,FP,TN,FN =0,0,0,0
    for i in range(num_obs):
        for j in range(num_obs):
            if B_real[i][j] !=0. and B_est[i][j] != 0.:
                TP = TP + 1
            elif B_real[i][j] !=0. and B_est[i][j] == 0.:   
                FN = FN + 1      
            elif B_real[i][j] ==0. and B_est[i][j] != 0.:  
                FP = FP + 1            

    precision_c = TP/(TP+FP) if (TP+FP)!=0 else 0
    recall_c = TP/(TP+FN) if (TP+FN)!=0 else 0
    F1_c = (2 * precision_c * recall_c) / (precision_c + recall_c) if (precision_c + recall_c)!=0 else 0
    
    # confounder
    TP,FP,TN,FN =0,0,0,0
    for j in range(num_obs,N):
        for i in range(N):
            if B_real[i][j] !=0. and B_est[i][j] != 0.:
                TP = TP + 1
            elif B_real[i][j] !=0. and B_est[i][j] == 0.:   
                FN = FN + 1
            elif B_real[i][j] ==0. and B_est[i][j] != 0.:  
                FP = FP + 1            

    precision_l = TP/(TP+FP) if (TP+FP)!=0 else 0
    recall_l = TP/(TP+FN) if (TP+FN)!=0 else 0
    F1_l = (2 * precision_l * recall_l)/(precision_l + recall_l) if (precision_l + recall_l)!=0 else 0
    
    # MSE
    # A = np.linalg.inv(np.eye(N)-B_est)
    # A = A[:num_obs,:]
    B_est = B_est[:num_obs,:num_obs]
    X1 = (X.dot(B_est.T)) + e
    mse = mean_squared_error(X,X1)
    return precision_c,recall_c,F1_c,precision_l,recall_l,F1_l,mse