import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm

from cProfile import label
import re
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import math
import matplotlib.ticker
import sys
import time
import copy


# definition of global parameters
sigma_0=0.05 # variance of ground truths
d=28*28 # dimension of feature vectors
r=20 # rank of feature vectors
s=28 # samples of feature vectors per task
M=[1,4,7] # number of experts
flg=len(M) 
global_content=[[1],[4],[7]]
N=len(global_content) # number of ground truths, with K=3
K=2 
_eta=0.3 # learning rate
_lambda=0.005 # upper bound of the random noise
_alpha=0.3 # scalar of the auxiliary loss function
T=60 # number of task arrivals
# r=1-s/d

curr_path = os.path.dirname(os.path.abspath(__file__))
data_file_name = os.path.join(curr_path, "NN_CL_t2_cor.pkl")

SIZE = 28  # rescale the size of the image to SIZE x SIZE (original image is 28x28)
# global_content=[[0, 1, 2],[7, 8, 9]]


class TaskData():
    def __init__(self, _opt_record) -> None:
        self.data = torchvision.datasets.MNIST(
            'Datasets', train=True, download=True, transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((SIZE, SIZE)),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
        self.shuffle = torch.randperm(len(self.data))
        # print(self.data[self.shuffle[100]])
        self.task_num = len(_opt_record)
        self.nt = 100  # num of training samples for each training task
        self.nv = 1000  # num of test samples for each training task
        self.task_content = []
        for i in range(self.task_num):
            self.task_content.append(global_content[_opt_record[i]])
        # print(self.task_content)
        self.train_data = []
        self.test_data = []
        for i in range(self.task_num):
            self.train_data.append(self.get_task_data_block(i))
            self.test_data.append(
                self.get_task_data_block(i, is_test=True))

    def get_task_data(self, task_index, data_index, is_test=False):
        begin = task_index * (self.nt + self.nv)
        if begin>=59000:
            begin-=59000
        if is_test:
            begin += self.nt
        return self.data[self.shuffle[begin + data_index]]

    def get_task_data_block(self, task_index, is_test=False):
        _n = self.nt
        if is_test:
            _n = self.nv
        input_data = torch.zeros(_n, 1, SIZE, SIZE)
        label_data = torch.zeros(_n)
        for i in range(_n):
            temp = self.get_task_data(task_index, i, is_test)
            # print(temp)
            input_data[i, 0, :, :] = temp[0][0]
            if temp[1] in self.task_content[task_index]:
                label_data[i] = 1.0
            else:
                label_data[i] = 0.0
        # add noise
        noise_input = torch.randn(_n, 1, SIZE, SIZE)
        return input_data, label_data, noise_input


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def get_test_loss(task_index, a, expert):
    with torch.no_grad():
        outputs = net[expert](a.test_data[task_index][0].to(device))
        test_loss = loss_fn[expert](outputs, a.test_data[task_index][1].view(a.nv, 1).to(device))
        return test_loss.item()


torch.manual_seed(0)
net1 = Net()
net2 = Net()
net3 = Net()
net4 = Net()
net5 = Net()
net6 = Net()
net7 = Net()
net8 = Net()
net=[net1, net2, net3, net4, net5, net6, net7, net8]

loss_fn1 = nn.MSELoss()
loss_fn2 = nn.MSELoss()
loss_fn3 = nn.MSELoss()
loss_fn4 = nn.MSELoss()
loss_fn5 = nn.MSELoss()
loss_fn6 = nn.MSELoss()
loss_fn7 = nn.MSELoss()
loss_fn8 = nn.MSELoss()
loss_fn=[loss_fn1,loss_fn2,loss_fn3,loss_fn4,loss_fn5,loss_fn6,loss_fn7,loss_fn8]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)
for i in range(len(net)):
    net[i].to(device)
    loss_fn[i].to(device)
    init_params = copy.deepcopy(net[i].state_dict())

def training_for_each_option(_opt_record,seed,expert):
    torch.manual_seed(seed)
    net[expert].load_state_dict(init_params)
    a = TaskData(_opt_record)
    optimizer = optim.SGD(net[expert].parameters(), lr=0.1)
    # Continual learning
    training_loss_all_tasks = []
    test_loss_all_tasks = []
    task_index=a.task_num-1
    for epoch in range(600):
        optimizer.zero_grad()
        outputs = net[expert](a.train_data[task_index][0].to(device)) # data input
        loss = loss_fn[expert](outputs, a.train_data[task_index][1].view(a.nt, 1).to(device)) # label data
        loss.backward()
        optimizer.step()
    training_loss_all_tasks.append(loss.item())
    for i in range(task_index + 1):
        test_loss_all_tasks.append(get_test_loss(i, a, expert))
    return test_loss_all_tasks


def generate_ground_truth():
    Xt_average=np.zeros((N,d))
    for n in range(N):
        Xt=[]
        _data_cnt=0
        _data=TaskData([n])
        for i in range(_data.nt):
            if _data.train_data[0][1][i]==1.0:
                _data_cnt+=1
                Xt.append(_data.train_data[0][0][i].tolist()[0])
        for k in range(_data_cnt):
            for i in range(SIZE):
                for j in range(SIZE):
                    Xt_average[n][i*SIZE+j] += Xt[k][i][j]
    Xt_average = Xt_average/np.linalg.norm(Xt_average) # d times 1
    Xt_average[0] =Xt_average[0]-0.5
    Xt_average[1] =Xt_average[1]+0.5
    return Xt_average

def update_theta(task_grad,theta_init,flag):
    theta_t=np.zeros((M[flag],d))
    for i in range(0,M[flag]):
        for j in range(0,d):
            theta_t[i][j]=theta_init[i][j]-_eta*task_grad[i][j]
    return theta_t


def soft_max(h_value,flag):
    sum_exp=0
    softmax_m=np.zeros(M[flag])
    for i in range(0,M[flag]):
        sum_exp += np.exp(h_value[i])
    for i in range(0,M[flag]):
        softmax_m[i] = np.exp(h_value[i])/sum_exp
    return softmax_m

def cal_gradient(pi_t,test_loss,X_t,t,mt,flag):
    loc_grad=np.zeros((M[flag],d))
    aux_grad=np.zeros((M[flag],d))
    task_grad=np.zeros((M[flag],d))
    # print(test_loss)
    # print(pi_t)
    for i in range(0,M[flag]):
        if i==mt:
            for j in range(0,d):
                loc_grad[i][j]=pi_t[mt]*test_loss*X_t[j]
                aux_grad[i][j]=pi_t[mt]*_alpha*M[flag]/(t+1)*X_t[j]
        else:
            for j in range(0,d):
                loc_grad[i][j]=-pi_t[mt]*X_t[j]*test_loss
                aux_grad[i][j]=-pi_t[mt]*X_t[j]*_alpha*M[flag]/(t+1)
        task_grad[i]=[x + y for x, y in zip(loc_grad[i], aux_grad[i])] 
    # print(task_grad)
    return task_grad



# v_0 = np.zeros((N, d))
v_pool=generate_ground_truth() # generate ground truth pool
# print(v_pool)
data_error01=[x - y for x, y in zip(v_pool[0], v_pool[1])]
data_error02=[x - y for x, y in zip(v_pool[0], v_pool[2])]
data_error12=[x - y for x, y in zip(v_pool[1], v_pool[2])]
print(np.linalg.norm(data_error01),np.linalg.norm(data_error02),np.linalg.norm(data_error12))
# print(np.linalg.norm(data_error01))
#---------------------------------------------------------------------------------------with interrupt

#parameters intializaton

index_record=[]
for i in range(T):
    index_record.append(np.random.randint(0,N))
# index_record=[1, 1, 2, 1, 0, 1, 0, 1, 0, 0, 0, 0, 2, 2, 2, 0, 1, 2, 0, 2, 2, 1, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 0, 1, 0, 2, 1, 0, 2, 2, 1, 1, 0, 1, 0, 2, 2, 0, 2, 1, 2, 2, 1, 2, 2, 0, 0, 0, 2, 1, 2, 2, 2, 2]
# index_record=[1,0,1,0,0,1,1,0,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,0,0,1,1,0,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,0,0,1,1,0,1,0,1,0,1,1,0,1,0,1,1,0]
Ft_record=[]
Gt_record=[]
for i in range(flg):
    Ft_record.append([])
    Gt_record.append([])

# run MoE for different M's
for flag in range(flg):
    Im=np.zeros((flg,M[flag])) # convergence flag in Algorithm 1
    mt_record=np.arange(T)
    for i in range(T):
        mt_record[i]=1
    theta=np.zeros((T,M[flag],d)) # M times d


    expert_record=[]
    expert_loss=[]
    for i in range(0,M[flag]):
        expert_record.append([])
        expert_loss.append([])

    theta_gradient=np.zeros((T,M[flag],d))
    softmax_value=np.zeros((T,M[flag]))
    seed=[3,4,8]

    index_record_t=[]
    for t in tqdm(range(0,T)):
        ht=np.zeros(M[flag])
        forget_col=[]
        gene_col=[]
        Ft=0
        Gt=0
        Xt_average = v_pool[index_record[t]]
        # print(theta[t])
        # print(Xt_average)
        # print(len(Xt_average))
        ht=theta[t] @ Xt_average
        print("task index: ", index_record[t])
        # print(Xt_average)
        print(ht)
        rt = np.random.uniform(0,_lambda,M[flag]) # generate noise r_t^m
        ht_noise=[x + y for x, y in zip(ht, rt)] 
        # print(ht_noise)
        softmax_value[t]=soft_max(ht_noise,flag) 
        for i in range(M[flag]):
            if softmax_value[t][i] == np.max(softmax_value[t]):
                mt_record[t]=i # routing strategy to decide m_t
                print("Selected expert: ",i)
                expert_record[i].append(index_record[t])
                temp = training_for_each_option(expert_record[i],seed[flag]+i,i)
                expert_loss[i] = temp
        if t >= M[flag]/_eta:
            for i in range(0,M[flag]):
                if abs(ht_noise[i]-ht_noise[mt_record[t]])<sigma_0**(1.25):
                        Im[flag][i]=1
        if sum(Im[flag])!=M[flag]:
        # if t<T-1:
            theta_gradient[t] = cal_gradient(softmax_value[t],temp[len(temp)-2],Xt_average,t,mt_record[t],flag)
            # print(theta_gradient[t])
            theta_temp = update_theta(theta_gradient[t],theta[t],flag)
            print(t)
            if t <T-1:
                theta[t+1]=theta_temp
                # print(theta[t+1])
        else:
            if t <T-1:
                theta[t+1]=theta[t]
        # Calculate forgetting and generalization error
        if t == 0:
            Ft_record[flag].append(0)
            Gt_record[flag].append(temp[t])
        else:
            for i in range(M[flag]):
                if len(expert_loss[i])==1:
                    Ft+=0
                    Gt+=expert_loss[i][0]
                elif len(expert_loss[i])>=2:
                    for j in range(len(expert_loss[i])):
                        Ft+=abs(expert_loss[i][len(expert_loss[i])-1] - expert_loss[i][j])
                        Gt+=expert_loss[i][j]
            Ft_record[flag].append(Ft/t)
            Gt_record[flag].append(Gt/(t+1))

    for i in range(0,M[flag]):
        print(expert_record[i])

#---------------------------------------------------------------------------------------

plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 16})
all_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
_style = ['-', '--', '-.', ':', '-']
x_axis=np.arange(T)

f = plt.figure(1)

for i in range(flg):
    plt.plot(x_axis, Ft_record[i], _style[i], color=all_colors[i], label='$M={}$'.format(M[i]))
plt.xlabel("Rounds")
plt.ylabel("forgetting $F_t$")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(os.path.join(curr_path, 'NN_MoE_forgetting.pdf'),
            format='pdf', bbox_inches='tight')

g = plt.figure(2)
for i in range(flg):
    plt.plot(x_axis, Gt_record[i], _style[i], color=all_colors[i], label='$M={}$'.format(M[i]))
plt.xlabel("Rounds")
plt.ylabel("error $G_t$")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig(os.path.join(curr_path, 'NN_MoE_error.pdf'),
            format='pdf', bbox_inches='tight')


plt.show()

