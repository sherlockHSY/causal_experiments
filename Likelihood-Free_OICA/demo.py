import numpy as np

# 参数
data_size = 3000
num_lantents = 1
num_components = 4
num_mixtures = 3  # 观察变量数量

sigmaList = [0.001, 0.01]
batch_size = 2000
num_epochs = 1000
lr_T = 0.01
lr_G = 0.001
reg_lambda = 0.
print_int = 500

# 生成高斯噪声的函数
def get_noises(n):
    x = np.random.normal(0.0, 0.5, n) ** 3
    return np.array(x)

# 生成测试数据
components = np.zeros((data_size,num_components)) # (3000,4)
for i in range(num_components):
    c = get_noises(data_size)
    components[:,i] = c
    
# 原始数据X
X = np.zeros((data_size,num_components)) # (3000,3)
X[:,0] = components[:,0]
X[:,3] = components[:,3] # x3 是隐变量
X[:,1] = components[:,1] + 2 * X[:,0] + X[:,3] 
X[:,2] = components[:,2] + 0.3 * X[:,1]

B = np.array([[0,0,0,0],[2,0,0,1],[0,0.3,0,0],[0,0,0,0]])
A = np.linalg.inv(np.eye(num_components)-B)
mixtures = X[:,:num_components-1]
A = A[:num_components-1,:]

data_arrs = {
    'arr_0':mixtures,
    'arr_1':components,
    'arr_2':A
}
data_arrs['arr_0']
data_arrs['arr_1']
data_arrs['arr_2'] 


i = 1