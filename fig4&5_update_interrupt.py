import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm

curr_path = os.path.dirname(os.path.abspath(__file__))

# definition of global parameters
sigma_0=0.4 # variance of ground truths
d=10 # dimension of feature vectors
r=6 # rank of feature vectors
s=8 # samples of feature vectors per task
M=5 # number of experts
N=30 # number of ground truths
K=3
_eta=0.5 # learning rate
_lambda=sigma_0 # upper bound of the random noise
_alpha=0.5 # scalar of the auxiliary loss function
T=500 # number of task arrivals
# r=1-s/d
_exp=[0.75,1,1.25,1.5]
flg=len(_exp)

loc_grad_record=[]
aux_grad_record=[]
pi_record=[]


def generate_ground_truth(w_0,v_0):
    for i in range(0,N):
        temp_w = np.random.randn(r)
        temp_w = sigma_0*temp_w/np.linalg.norm(temp_w)
        temp_v = temp_w/np.linalg.norm(temp_w)*5
        w_0[i,:r]=temp_w
        v_0[i,:r]=temp_v
    for i in range(K,N):
        # temp_w=w_0[i-K,:r]+sigma_0/100*np.random.randn(r)
        temp_w=w_0[i-K,:r]+sigma_0/100*np.random.randn(r)
        temp_v = temp_w/np.linalg.norm(temp_w)*5
        w_0[i,:r]=temp_w
        v_0[i,:r]=temp_v
    return w_0, v_0

def update_w(w_last, X_t,y_t):
    X=X_t
    y=y_t
    w_init=w_last
    wt =  (y - w_init @ X) @ np.linalg.pinv(X) + w_init
    # print(np.linalg.norm(wt)-np.linalg.norm(w_init))
    return wt

def generate_feature_matrix(w_0,v_0,w_index):
    X_t=np.zeros((d,s))
    y_t=np.zeros(s)
    for i in range(0,s):
        # temp_X = _eta*np.random.randn(r)
        temp_X = np.random.randn(r)
        temp_X = 0.01*temp_X/np.linalg.norm(temp_X)
        X_t[:r,i]=temp_X
    v_position=np.random.randint(0,s)
    # w_index=np.random.randint(0,N)
    # print(w_index)
    for i in range(0,d):
        X_t[i,v_position]=v_0[w_index,i]
    for j in range(r,d):
        if j-r != v_position:
            X_t[j,:]=X_t[j-r,:]*np.random.randn(1)*0.001
        else:
            X_t[j,:]=X_t[j-r,:]*np.random.randn(1)*0.0001
    y_t=w_0[w_index] @ X_t
    return X_t, y_t

def update_theta(task_grad,theta_init):
    theta_t=np.zeros((M,d))
    for i in range(0,M):
        for j in range(0,d):
            theta_t[i][j]=theta_init[i][j]-_eta*task_grad[i][j]
    return theta_t


def soft_max(h_value):
    sum_exp=0
    softmax_m=np.zeros(M)
    for i in range(0,M):
        sum_exp += np.exp(h_value[i])
    for i in range(0,M):
        softmax_m[i] = np.exp(h_value[i])/sum_exp
    return softmax_m

def cal_gradient(pi_t,w_init,wt_m,X_t,t,f_t,mt):
    loc_grad=np.zeros((M,d))
    aux_grad=np.zeros((M,d))
    task_grad=np.zeros((M,d))
    model_error=[x - y for x, y in zip(wt_m, w_init)]
    error_norm=np.linalg.norm(model_error) # calculate ||w_t-w_{t-1}||_2
    sum_X=np.zeros(d)
    for i in range(0,s): # calculate \sum_{i=0}^{s} X_{t,i}
        temp_sum=[x + y for x, y in zip(sum_X, np.transpose(X_t)[i])]
        sum_X=temp_sum

    for i in range(0,M):
        if i==mt:
            for j in range(0,d):
                # loc_grad[i][j]=pi_t[mt]*(1-pi_t[mt])*error_norm*sum_X[j]
                # aux_grad[i][j]=pi_t[mt]*(1-pi_t[mt])*_alpha*M/(t+1)*sum_X[j]
                loc_grad[i][j]=pi_t[mt]*error_norm*sum_X[j]
                aux_grad[i][j]=pi_t[mt]*_alpha*M/(t+1)*sum_X[j]
        else:
            for j in range(0,d):
                # loc_grad[i][j]=-pi_t[mt]*pi_t[i]*sum_X[j]*error_norm
                # aux_grad[i][j]=-pi_t[mt]*pi_t[i]*sum_X[j]*_alpha*M/(t+1)
                loc_grad[i][j]=-pi_t[mt]*sum_X[j]*error_norm
                aux_grad[i][j]=-pi_t[mt]*sum_X[j]*_alpha*M/(t+1)
        task_grad[i]=[x + y for x, y in zip(loc_grad[i], aux_grad[i])] 

    loc_grad_record.append(np.linalg.norm(loc_grad[mt]))
    aux_grad_record.append(np.linalg.norm(aux_grad[mt]))
    pi_record.append(pi_t[mt])
    return task_grad


def cal_F_and_G(t, mt_record, wt_record,w_record):
    Ft=0
    Gt=0
    model_error_t = np.zeros(d)
    model_error_i = np.zeros(d)
    for i in range(0,t-1):
        model_error_t=[x - y for x, y in zip(wt_record[t+1][mt_record[i]], w_record[i])]
        error_t= np.linalg.norm(model_error_t)
        model_error_i=[x - y for x, y in zip(wt_record[i+1][mt_record[i]], w_record[i])]
        error_i= np.linalg.norm(model_error_i)
        Ft +=  abs(error_t - error_i) 
    Ft /= t
    for i in range(0,t):
        model_error=[x - y for x, y in zip(wt_record[t+1][mt_record[i]], w_record[i])]
        error_G= np.linalg.norm(model_error)
        Gt += abs(error_G)
    Gt /= (t+1)
    return Ft, Gt


w_0 = np.zeros((N, d))
v_0 = np.zeros((N, d))
w_pool, v_pool=generate_ground_truth(w_0,v_0) # generate ground truth pool

#---------------------------------------------------------------------------------------with interrupt

#parameters intializaton

index_record=[]
for i in range(T):
    index_record.append(np.random.randint(0,N))
Ft_record=np.zeros((flg,T))
Gt_record=np.zeros((flg,T))

expert_record=[]
expert_t_record=[]
router_record=[]
expert_task_record=[]

for i in range(flg):
    expert_record.append([])
    expert_t_record.append([])
    router_record.append([])
    expert_task_record.append([])
# run MoE for different exponents
cnt=100
Ft_record_average=np.zeros((flg,T))
Gt_record_average=np.zeros((flg,T))
for count in tqdm(range(cnt)):
    for flag in range(flg):

        for i in range(0, N):
            expert_record[flag].append([])
            router_record[flag].append([])
        # run MoE for different exponents
        for i in range(0, M):
            expert_t_record[flag].append([])
            expert_task_record[flag].append([])

        Im=np.zeros((M)) # convergence flag in Algorithm 1
        mt_record=np.arange(T)
        for i in range(T):
            mt_record[i]=-1
        wt_record=np.zeros((T+1,M,d))
        theta=np.zeros((T,M,d)) # M times d
        w_record=np.zeros((T,d))

        ft=0
        theta_gradient=np.zeros((T,M,d))
        softmax_value=np.zeros((T,M))

        for t in range(0,T):
            ht=np.zeros(M)
            w_index=index_record[t]
            Xt, yt = generate_feature_matrix(w_pool, v_pool, w_index)
            w_record[t]=w_pool[w_index]
            # index_record[flag][t]=w_index
            ht_maxtrix=theta[t] @ Xt # theta: M times d, Xt: d times s
            for i in range(0,M):
                for j in range(0,s):
                    ht[i] += ht_maxtrix[i,j] # output of the gating network
            rt = np.random.uniform(0,_lambda,M) # generate noise r_t^m
            ht_noise=[x + y for x, y in zip(ht, rt)] 
            softmax_value[t]=soft_max(ht_noise) 
            if flag>1 and t>=130+20*flag: 
                for i in range(0,M):
                    # print(expert_t_record[flag][i])
                    # print(expert_task_record[flag][i])
                    if expert_t_record[flag][i][-1]<t-np.random.randint(18,25,1) and (expert_task_record[flag][i][-1]==w_index or abs(expert_task_record[flag][i][-1]-w_index)==K):
                        mt_record[t]=i
            if mt_record[t]==-1:
                for i in range(0,M):
                    if softmax_value[t][i] == np.max(softmax_value[t]):
                        mt_record[t]=i # routing strategy to decide m_t
            for i in range(0,M):
                if i == mt_record[t]:
                    wt_record[t+1][i]=update_w(wt_record[t][i], Xt,yt) # update w_t^{m_t}
                    expert_record[flag][w_index].append(i+1)
                    router_record[flag][w_index].append(t)
                    expert_task_record[flag][i].append(w_index)
                    expert_t_record[flag][i].append(t)
                else:
                    wt_record[t+1][i]=wt_record[t][i] # keep w_t^m=w_{t-1}^m

            if t >= M/_eta:
                for i in range(0,M):
                    if i != mt_record[t]:
                        if abs(ht_noise[i]-ht_noise[mt_record[t]])<sigma_0**(_exp[flag]):
                            Im[i]=1
            if sum(Im)!=M or t<120:
                theta_gradient[t] = cal_gradient(softmax_value[t],wt_record[t][mt_record[t]],wt_record[t+1][mt_record[t]],Xt,t,ft,mt_record[t])
                theta_temp = update_theta(theta_gradient[t],theta[t])
                if t <T-1:
                    theta[t+1]=theta_temp
            else:
                ft+=1
                if t <T-1:
                    theta[t+1]=theta[t]
            # Calculate forgetting and generalization error
            if t >=1:
                Ft_record[flag][t], Gt_record[flag][t]=cal_F_and_G(t, mt_record, wt_record,w_record)
            else:
                Ft_record[flag][t] = 0
                model_error=[x - y for x, y in zip(wt_record[t+1][:,mt_record[i]], w_record[i])]
                error_G = np.linalg.norm(model_error)
                Gt_record[flag][t] = error_G
        Ft_record_average[flag] += Ft_record[flag]
        Gt_record_average[flag] += Gt_record[flag]

Ft_record_average/=cnt
Gt_record_average/=cnt
    # print(Im)
    # for i in range(0,M):
    #     print(expert_record[i])

#---------------------------------------------------------------------------------------

plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 16})
all_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
_style = ['-', '--', '-.', ':']
x_axis=np.arange(T)


# e1 = plt.figure(1)

# for i in range(N):
#     plt.scatter(router_record[0][i], expert_record[0][i], color=all_colors[i], label='$Task: {}$'.format(i+1))

# plt.xlabel("Rounds")
# plt.ylabel("Expert ID")
# plt.legend(loc='upper left')
# plt.savefig(os.path.join(curr_path, 'sigma75_expert_ID.pdf'),
#             format='pdf', bbox_inches='tight')

# e1 = plt.figure(2)

# for i in range(N):
#     plt.scatter(router_record[1][i], expert_record[1][i], color=all_colors[i], label='$Task: {}$'.format(i+1))

# plt.xlabel("Rounds")
# plt.ylabel("Expert ID")
# plt.legend(loc='upper left')
# plt.savefig(os.path.join(curr_path, 'sigma100_expert_ID.pdf'),
#             format='pdf', bbox_inches='tight')

# e1 = plt.figure(3)

# for i in range(N):
#     plt.scatter(router_record[2][i], expert_record[2][i], color=all_colors[i], label='$Task: {}$'.format(i+1))

# plt.xlabel("Rounds")
# plt.ylabel("Expert ID")
# plt.legend(loc='upper left')
# plt.savefig(os.path.join(curr_path, 'sigma125_expert_ID.pdf'),
#             format='pdf', bbox_inches='tight')

# e1 = plt.figure(4)

# for i in range(N):
#     plt.scatter(router_record[3][i], expert_record[3][i], color=all_colors[i], label='$Task: {}$'.format(i+1))

# plt.xlabel("Rounds")
# plt.ylabel("Expert ID")
# plt.legend(loc='upper left')
# plt.savefig(os.path.join(curr_path, 'sigma150_expert_ID.pdf'),
#             format='pdf', bbox_inches='tight')

f = plt.figure(1)


for i in range(flg):
    plt.plot(x_axis, Ft_record_average[i], _style[i], color=all_colors[i], label='$\sigma_0^{}$'.format({_exp[i]}))
plt.xlabel("Rounds")
plt.ylabel("forgetting $F_t$")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(os.path.join(curr_path, 'forgetting_algo1.pdf'),
            format='pdf', bbox_inches='tight')

g = plt.figure(2)
for i in range(flg):
    plt.plot(x_axis, Gt_record_average[i], _style[i], color=all_colors[i], label='$\sigma_0^{}$'.format({_exp[i]}))
plt.xlabel("Rounds")
plt.ylabel("error $G_t$")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig(os.path.join(curr_path, 'error_algo1.pdf'),
            format='pdf', bbox_inches='tight')

plt.show()

