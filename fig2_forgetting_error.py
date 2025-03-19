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
s=6 # samples of feature vectors per task
M=[1,5,10,20] # number of experts
flg=len(M) 
N=7 # number of ground truths, with K=3
K=3 
_eta=0.5 # learning rate
_lambda=0.3 # upper bound of the random noise
_alpha=0.5 # scalar of the auxiliary loss function
T=2000 # number of task arrivals
# r=1-s/d

def generate_ground_truth(w_0,v_0):
    for i in range(0,K):
        temp_w = np.random.randn(r)
        temp_w = sigma_0*temp_w/np.linalg.norm(temp_w)
        temp_v = temp_w/np.linalg.norm(temp_w)*6
        w_0[i,:r]=temp_w
        v_0[i,:r]=temp_v
    for i in range(K,N):
        temp_w=w_0[i-K,:r]+sigma_0/200*np.random.randn(r)
        temp_v = temp_w/np.linalg.norm(temp_w)*6
        w_0[i,:r]=temp_w
        v_0[i,:r]=temp_v
    return w_0, v_0

def update_w(w_last, X_t,y_t):
    X=X_t
    y=y_t
    w_init=w_last
    wt =  (y - w_init @ X) @ np.linalg.pinv(X) + w_init
    return wt

def generate_feature_matrix(w_0,v_0,w_index):
    X_t=np.zeros((d,s))
    y_t=np.zeros(s)
    for i in range(0,s):
        temp_X = np.random.randn(r)
        temp_X = 0.1*temp_X/np.linalg.norm(temp_X)
        X_t[:r,i]=temp_X
    v_position=np.random.randint(0,s)
    # w_index=np.random.randint(0,N)
    for i in range(0,d):
        X_t[i,v_position]=v_0[w_index,i]
    for j in range(r,d):
        if j-r != v_position:
            X_t[j,:]=X_t[j-r,:]*np.random.randn(1)*0.001
        else:
            X_t[j,:]=X_t[j-r,:]*np.random.randn(1)*0.00001
    y_t=w_0[w_index] @ X_t
    return X_t, y_t

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

def cal_gradient(pi_t,w_init,wt_m,X_t,t,f_t,mt,flag):
    loc_grad=np.zeros((M[flag],d))
    aux_grad=np.zeros((M[flag],d))
    task_grad=np.zeros((M[flag],d))
    model_error=[x - y for x, y in zip(wt_m, w_init)]
    error_norm=np.linalg.norm(model_error) # calculate ||w_t-w_{t-1}||_2
    sum_X=np.zeros(d)
    for i in range(0,s): # calculate \sum_{i=0}^{s} X_{t,i}
        temp_sum=[x + y for x, y in zip(sum_X, np.transpose(X_t)[i])]
        sum_X=temp_sum

    for i in range(0,M[flag]):
        if i==mt:
            for j in range(0,d):
                loc_grad[i][j]=pi_t[mt]*(1-pi_t[mt])*error_norm*sum_X[j]
                aux_grad[i][j]=pi_t[mt]*(1-pi_t[mt])*_alpha*M[flag]/(t+1)*sum_X[j]
        else:
            for j in range(0,d):
                loc_grad[i][j]=-pi_t[mt]*pi_t[i]*sum_X[j]*error_norm
                aux_grad[i][j]=-pi_t[mt]*pi_t[i]*sum_X[j]*_alpha*M[flag]/(t+1)
        task_grad[i]=[x + y for x, y in zip(loc_grad[i], aux_grad[i])] 

    return task_grad

def max_len(expert_set,flag):
    max_length=-1
    res=0
    for i in range(M[flag]):
        if len(expert_set[i]) > max_length:
            max_length = len(expert_set[i])
            res = i
    return res

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

# run MoE for different M's
cnt=1
Ft_record_average=np.zeros((flg,T))
Gt_record_average=np.zeros((flg,T))
for count in tqdm(range(cnt)):
    for flag in range(flg):
        # T1=151+M[flag]*3
        Im=np.zeros((flg,M[flag])) # convergence flag in Algorithm 1
        mt_record=np.arange(T)
        for i in range(T):
            mt_record[i]=1
        wt_record=np.zeros((T+1,M[flag],d))
        theta=np.zeros((T,M[flag],d)) # M times d
        w_record=np.zeros((T,d))


        expert_record=[]
        for i in range(0,M[flag]):
            expert_record.append([])

        ft=0
        theta_gradient=np.zeros((T,M[flag],d))
        softmax_value=np.zeros((T,M[flag]))

        for t in range(0,T):
            ht=np.zeros(M[flag])
            w_index=index_record[t]
            Xt, yt = generate_feature_matrix(w_pool, v_pool, w_index)
            w_record[t]=w_pool[w_index]
            ht_maxtrix=theta[t] @ Xt # theta: M times d, Xt: d times s
            for i in range(0,M[flag]):
                for j in range(0,s):
                    ht[i] += ht_maxtrix[i,j] # output of the gating network
            rt = np.random.uniform(0,_lambda,M[flag]) # generate noise r_t^m
            ht_noise=[x + y for x, y in zip(ht, rt)] 
            softmax_value[t]=soft_max(ht_noise,flag) 
            for i in range(0,M[flag]):
                if softmax_value[t][i] == np.max(softmax_value[t]):
                    mt_record[t]=i # routing strategy to decide m_t
                    wt_record[t+1][i]=update_w(wt_record[t][i], Xt,yt) # update w_t^{m_t}
                    expert_record[i].append(w_index)
                else:
                    wt_record[t+1][i]=wt_record[t][i] # keep w_t^m=w_{t-1}^m
            if t >= M[flag]/_eta:
                for i in range(0,M[flag]):
                    if i != mt_record[t]:
                        if abs(ht_noise[i]-ht_noise[mt_record[t]])<sigma_0**(1.25):
                            Im[flag][i]=1
            if sum(Im[flag])!=M[flag] or t<80+M[flag]*3:
                theta_gradient[t] = cal_gradient(softmax_value[t],wt_record[t][mt_record[t]],wt_record[t+1][mt_record[t]],Xt,t,ft,mt_record[t],flag)
                theta_temp = update_theta(theta_gradient[t],theta[t],flag)
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
    # for i in range(0,M[flag]):
    #     print(expert_record[i])
Ft_record_average/=cnt
Gt_record_average/=cnt

#---------------------------------------------------------------------------------------

plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 16})
all_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
_style = ['-', '--', '-.', ':', '-']
x_axis=np.arange(T)

f = plt.figure(1)


for i in range(flg):
    plt.plot(x_axis, Ft_record_average[i], _style[i], color=all_colors[i], label='$M={}$'.format(M[i]))
plt.xlabel("Rounds")
plt.ylabel("forgetting $F_t$")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(os.path.join(curr_path, 'forgetting_algo1.pdf'),
            format='pdf', bbox_inches='tight')

g = plt.figure(2)
for i in range(flg):
    plt.plot(x_axis, Gt_record_average[i], _style[i], color=all_colors[i], label='$M={}$'.format(M[i]))
plt.xlabel("Rounds")
plt.ylabel("error $G_t$")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig(os.path.join(curr_path, 'error_algo1.pdf'),
            format='pdf', bbox_inches='tight')

# f1 = plt.figure(3)


# for i in range(flg):
#     plt.plot(x_axis, Ft_record1[i], _style[i], color=all_colors[i], label='$M={}$'.format(M[i]))
# plt.xlabel("Rounds")
# plt.ylabel("forgetting $F_t$")
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.savefig(os.path.join(curr_path, 'forgetting_algo2.pdf'),
#             format='pdf', bbox_inches='tight')

# g = plt.figure(4)
# for i in range(flg):
#     plt.plot(x_axis, Gt_record1[i], _style[i], color=all_colors[i], label='$M={}$'.format(M[i]))
# plt.xlabel("Rounds")
# plt.ylabel("error $G_t$")
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# plt.savefig(os.path.join(curr_path, 'error_algo2.pdf'),
#             format='pdf', bbox_inches='tight')

plt.show()

