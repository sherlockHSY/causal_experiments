from numpy.core.fromnumeric import nonzero
from my_LFOICA import *
import matplotlib.pyplot as plt
import plot_function
import lingam
from lingam.utils import print_causal_directions, print_dagc, make_dot
import warnings
import sympy as sym
from scipy.linalg import lu

if __name__ == '__main__':
    # 随机种子
    randomseed = 10
    np.random.seed(randomseed)
    print("当前使用随机种子：",randomseed)
    # 生成测试数据 两组setting：平均入度 indegree、隐变量比例 p
    # 固定参数设置
    data_size = 1000
    num_measured_var = 5
    pvalue = 0.05  # 因果强度p值
    # 生成非高斯噪声的函数
    get_noises = lambda n: np.array(np.random.normal(0, 0.5, n)**3)
    
    # average indegree 7组
    indegree_arr = [1,1.5,2,2.5,3,3.5]
    
    # ratio of lantent confounder p 5组
    p_arr = [0.1,0.2,0.3,0.4,0.5]
    
    # 假设取值
    p = p_arr[1]
    indegree = indegree_arr[1]

    num_obs = int(num_measured_var * (1-p))  # 观察变量数量
    num_lat = int(num_measured_var - num_obs) # 隐变量数量
    # 生成噪声
    e = np.zeros((data_size,num_measured_var)) # (1000,10)
    for i in range(num_measured_var):
        c = get_noises(data_size)
        e[:,i] = c
    
    # 基于lvlingam，我们得到的模型应该是canonical的
    # 也就是隐变量放置最后列，且所有隐变量都是root结点
    m_var = np.arange(num_measured_var)
    obs = m_var[:num_obs]
    lat = m_var[num_obs:]
    
    # 生成邻接矩阵B，这里要控制平均入度，还要控制 causal strength 在(-1,-0.2]U[0.2,1)
    B = np.random.rand(num_measured_var, num_measured_var)
    B_temp = np.random.randn(num_measured_var, num_measured_var)
    B = np.sign(B_temp) * (abs(B))
    B = np.tril(B, -1) # 下三角，为了不能成环
    
    # 先随机变换观察变量
    rp = np.random.permutation(num_obs)
    Bo = B[rp, :]
    Bo = Bo[:, rp]
    
    # 创建隐变量与观察变量之间的联系
    Bl = np.zeros((num_obs, num_lat))
    for i in range(num_lat):
        while 1:
            b_temp = np.random.randn(num_obs, 1)
            b = np.random.rand(num_obs, 1)
            b = np.sign(b_temp) * (abs(b))
            # 限制隐变量孩子节点个数不大于3
            b = b.flatten()
            for bi in range(len(b)):
                if b[bi]<pvalue:
                    b[bi]=0.
        
            if len(np.nonzero(b)[0]) <= 3 and len(np.nonzero(b)[0]) >= 1:
                break
            
        Bl[:,i] = b
    
    # 合并观察变量与隐变量
    B = np.concatenate((Bo,Bl),axis=1)
    B = np.concatenate((B,np.zeros((num_lat,num_measured_var))),axis=0)

    # 控制平均入度，即需要将多少个点变成0或赋值
    print(len(np.nonzero(B)[0]))
    num_do = int(len(np.nonzero(B)[0]) - indegree * num_measured_var)
    if num_do > 0:
        order = np.random.permutation(len(np.nonzero(B)[0]))[:num_do]
        print('需要去掉一些边')
    else:
        order = np.random.permutation(num_measured_var**2 - len(np.nonzero(B)[0]))[:abs(num_do)-1]
        print('需要添加一些边')
    flag = 0
    for i in range(num_measured_var):
        for j in range(num_measured_var):

            if num_do > 0: # 去边比较简单
                if B[i ,j] != 0:
                    if B[i, j] < 0.2 and B[i, j] > 0:
                        B[i, j] = B[i, j] + 0.2
                    if B[i, j] > -0.2 and B[i, j] < 0:
                        B[i, j] = B[i, j] - 0.2
                    if np.isin(flag, order) == True: 
                        B[i, j] = 0
                    flag = flag + 1
            else: # 加边比较麻烦，不能成环，实际在这里好像不会出现这种情况
                if B[i ,j] == 0:      
                  if np.isin(flag, order) == True: # 
                    B[i, j] = np.random.rand()
                flag = flag + 1  
                
    # 检验一下平均入度
    aver_degree = len(np.nonzero(B)[0])/num_measured_var
    if aver_degree == indegree:
        print('平均入度校验正确')
    
    # 观察数据 X = BX + e
    X = e
    # LU分解可以得到上三角矩阵
    p,l,u =  lu(B)
    pl = p.dot(l)
    # 将上三角变为下三角
    f1 = np.arange(num_measured_var-1,-1,-1)
    B_tril = u[f1,:]
    B_tril = B_tril[:,f1]    
    print(B)
    print(B_tril)

    X = e.copy()
    for i in range(num_measured_var):
        X[:,i] = X.dot(B_tril[i].T) + e[:,i]

    # 将X也做相同的列变换，得到的其实就是因果顺序 k
    k = m_var.reshape((num_measured_var, 1))
    k = k[f1]
    k = pl.dot(k)
    k = k.flatten().astype(int)
    X = X[:,k]
    # 只取观察变量部分
    X = X[:,:num_obs]

    # 可视化真实数据的DAG
    plot_function.plotmodel(B,lat,'origin_dag')

    # 混合矩阵 A 注意是 overcomplete 的
    A = np.linalg.inv(np.eye(num_measured_var)-B)
    A = A[:num_obs,:]


    # 调用LFOICA
    data_arrs = {
        'arr_0': X,
        'arr_1': e,
        'arr_2': A
    }
    B_LFOICA = LFOICA_exp(data_arrs, num_obs, num_measured_var)
    
    # # 可视化LFOICA学到的DAG
    # plot_function.plotmodel(B_LFOICA,lat,'LFOICA_dag')
    
    # 调用RCD
    # model = lingam.RCD()
    # model.fit(X)
    
    # ancestors_list = model.ancestors_list_
    # for i, ancestors in enumerate(ancestors_list):
    #     print(f'M{i}={ancestors}')
    # B_RCD = model.adjacency_matrix_
    # dot = make_dot(model.adjacency_matrix_)
    # dot.format = 'png'
    # dot.render('RCD_dag.png')
    # plot_function.plotmodel(B_RCD,lat,'RCD_dag')
    # warnings.filterwarnings('ignore', category=UserWarning)


    print('end')
    
