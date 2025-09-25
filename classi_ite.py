import copy, numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.stats

#%%
#generate data 
def data_generation(sigma_x, thresh, size):
    z=np.random.normal(0, 1, size+1)
    x=np.random.normal(0, sigma_x, size)     
    zp=np.random.normal(0, np.sqrt(1-sigma_x*sigma_x), size+1)
    y=np.zeros(size+1)
    
    for i in range(size):
        if y[i]<thresh:
            y[i+1]=z[i+1]
        else:
            y[i+1]=x[i]+zp[i+1]  
            
    y_ts=y[0:size]
    y_ts=y_ts.reshape(-1,1)
  
    y_t=y[1:size+1]
    y_t=y_t.reshape(-1,1)
    x=x.reshape(-1,1)
    
    data = np.concatenate((x,y_t,y_ts),axis=1)
    return data

#%%
#generate data bar{y} 
def data_gen_zbar(data, size_g, data_net):
    data_z = data[:,2].reshape(-1,1)  
    data_z_ten = torch.FloatTensor(data_z)
    data_zbar = data_net(data_z_ten)
    sample_dataxyzbar = np.concatenate((data[:,[0,1]],data_zbar.detach().numpy()),axis=1)
    Jacobian_weight = torch.zeros((data_net.fc1.weight.shape[0], size_g)) 
    for i in range(size_g):
        output = torch.zeros(size_g,1)
        output[i] = 1
         #   each column is dz_1/dw_1, \cdots, dz_1/dw_n
        Jacobian_weight[:,i:i+1] = torch.autograd.grad(data_zbar,data_net.fc1.weight,
                                                           grad_outputs = output, 
                                                           retain_graph = True)[0]
    
    return sample_dataxyzbar, Jacobian_weight

#%%
#reconstruct data to create training data set and evaluation data set 
def recons_data(rho_data, size, train_size):  

    total_size = size
    train_index = np.random.choice(range(total_size), size=train_size, replace=False)
    test_index =  np.delete(np.arange(total_size), train_index, 0)
    joint_train = rho_data[train_index][:]
    joint_test = rho_data[test_index][:]

    marg_data, joint_index, marginal_index= sample_batch(rho_data, size, 
                                                         batch_size=size, 
                                                         sample_mode='marginal')
    
    marg_train = marg_data[train_index][:]
    marg_test = marg_data[test_index][:]
    
    train_data = np.vstack((joint_train,marg_train))
 
    joint_label = np.ones(train_size)
    marg_label = np.zeros(train_size)  
    label = np.vstack((joint_label,marg_label)).flatten()
    return train_data, joint_test, marg_test, label, train_index, marginal_index
#%%
def sample_batch(data, input_size, batch_size, sample_mode='joint'):
    joint_index=0
    marginal_index2=0
    if input_size==2:
        if sample_mode == 'joint':
            joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            batch = np.concatenate((data[joint_index][:,0].reshape(-1,1),data[joint_index][:,-1].reshape(-1,1)),axis=1)
        else:
            marginal_index1 = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            marginal_index2 = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            batch = np.concatenate((data[marginal_index1][:,0].reshape(-1,1),data[marginal_index2][:,-1].reshape(-1,1)),axis=1)
    else:
        if sample_mode == 'joint':
            joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            batch =data[joint_index]
        else:
            marginal_index1 = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            marginal_index2 = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            batch = np.concatenate((data[marginal_index1][:,0].reshape(-1,1),data[marginal_index2][:,[1,2]]),axis=1)
    return batch, joint_index, marginal_index2
#%%
# define varitional auto encoder 
class VAE(nn.Module):
    def __init__(self,VAE_input_size=1, VAE_hidden_size=200):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(VAE_input_size, VAE_hidden_size)
        self.fc21 = nn.Linear(VAE_hidden_size, 1)
        self.fc22 = nn.Linear(VAE_hidden_size, 1)

        nn.init.normal_(self.fc1.weight,std=0.2)
        self.fc1.weight.requires_grad = True
        nn.init.normal_(self.fc21.weight,std=0.2)
        nn.init.constant_(self.fc21.bias, 0)
        nn.init.normal_(self.fc22.weight,std=0.2)
        nn.init.constant_(self.fc22.bias, 0)
            
    def encode(self, x):
        h1 = F.elu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        # std = exp(0.5*logvar) without in-place ops; sample eps with correct dtype/device
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        zbar = self.reparametrize(mu, logvar)
        return zbar 


#%%
#classifier   
class Class_Net(nn.Module):
    def __init__(self, input_size, hidden_size, std):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        nn.init.normal_(self.fc1.weight, std=std)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=std)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=std)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self,input):
        m = nn.Sigmoid()
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        logit = self.fc3(output)
        prob = m(logit)
        return logit, prob
#%% 
 #training proccess        
def train(rho_data, size, train_size, mine_net, optimizer, iteration, input_size, tau):
    criterion = nn.BCEWithLogitsLoss()
    diff_et = torch.tensor(0.0)
    grads = None
    grads_placeholder = None
    last_index1 = None
    data, test_p0, test_q0, label, train_index, marg_index = recons_data(rho_data, size, 
                                                                           train_size)
    default_batch_size = max(1, int(len(data) / 4))
    for i in range(iteration):   

        batch_size = int(len(data)/4)
        if input_size == 2:  
            test_p = torch.FloatTensor(test_p0[:,[0,2]])
            test_q = torch.FloatTensor(test_q0[:,[0,2]])
            
        else: 
            test_p = torch.FloatTensor(test_p0)
            test_q = torch.FloatTensor(test_q0)
        
        train_batch, index1, index2 = sample_batch(data, input_size, 
                                                   batch_size = batch_size, 
                                                   sample_mode = 'joint')
        label_batch = label[index1]
        train_batch = torch.autograd.Variable(torch.FloatTensor(train_batch), requires_grad=True)
        if grads_placeholder is None:
            grads_placeholder = torch.zeros_like(train_batch)
        label_batch = torch.FloatTensor(label_batch)
        last_index1 = index1
        
        logit = mine_net(train_batch)[0]
        loss = criterion(logit.reshape(-1), label_batch)
        
        # Always backprop before stepping; on last iter, capture input gradients
        optimizer.zero_grad()
        loss.backward()
        if i == iteration - 1:
            if train_batch.grad is not None:
                grads = train_batch.grad.detach().clone()
            else:
                grads = grads_placeholder.detach().clone()
        optimizer.step()
        
        if i >= iteration - 101:
            with torch.no_grad():
                prob_p = mine_net(test_p)[1]
                rn_est_p = prob_p / (1 - prob_p)
                finp_p = torch.log(torch.abs(rn_est_p))

                prob_q = mine_net(test_q)[1]
                rn_est_q = prob_q / (1 - prob_q)
                a = torch.abs(rn_est_q)
                clip = torch.max(torch.min(a, torch.exp(tau)), torch.exp(-tau))
                diff_et = diff_et + torch.max(
                    torch.mean(finp_p) - torch.log(torch.mean(clip)),
                    torch.tensor(0.0),
                )
            
    safe_grads = grads if grads is not None else grads_placeholder
    safe_index = last_index1 if last_index1 is not None else np.arange(default_batch_size)
    return (diff_et/100).detach().cpu().numpy(), safe_grads, safe_index, train_index, marg_index

#%%
def mi(rho_data, size, train_size, model, optimizer, repo, tau, input_size):    
    mi, grad, index, train_index, marg_index = train(rho_data, size, train_size,
                                                     model, optimizer, repo, 
                                                     input_size, tau=tau)
    
    return mi, grad, index, train_index, marg_index
#%%
def ma(a, window_size=20):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]    

def main():
    # parameters
    rho = 0.9
    repi = int(200)
    repo = int(200)
    rep = int(1000)

    alpha = 0.001
    tau = torch.tensor(0.9)
    quan = 7
    # os.environ["FAST_DEBUG"] = "1"
    # FAST_DEBUG mode to speed up debug runs
    if os.environ.get("FAST_DEBUG", "0") == "1":
        print("FAST_DEBUG mode enabled: using tiny iterations for a quick check.")
        repi = 10
        repo = 5
        rep = 20
        quan = 3
        alpha = 0.001

    thresh_vec = np.linspace(-3, 3, quan)
    size = 4000
    train_size = 3000
    if os.environ.get("FAST_DEBUG", "0") == "1":
        size = 400
        train_size = 300
    test_size = size - train_size

    realization = 10
    if os.environ.get("FAST_DEBUG", "0") == "1":
        realization = 1
    print("ite, realization", realization)

    total_te = np.zeros(shape=(realization, quan))
    total_ite = np.zeros(shape=(realization, quan))
    total_ste = np.zeros(shape=(realization, quan))

    for i in range(realization):
        condiMI = []
        ground_truth = []
        result_ITE = []

        for thresh in thresh_vec:
            rho_data = data_generation(rho, thresh, size)

            modelP = Class_Net(input_size=3, hidden_size=130, std=0.08)
            modelQ = Class_Net(input_size=2, hidden_size=100, std=0.02)

            optimizerP = torch.optim.Adam(modelP.parameters(), lr=1e-3)
            optimizerQ = torch.optim.Adam(modelQ.parameters(), lr=1e-3)
            # conditional mi
            mi_p = mi(rho_data, size, train_size, modelP, optimizerP, rep, tau, input_size=3)[0]
            mi_q = mi(rho_data, size, train_size, modelQ, optimizerQ, rep, tau, input_size=2)[0]

            condi_mi = mi_p - mi_q
            condiMI.append(condi_mi * 1.4427)
            # ground truth
            p = scipy.stats.norm(0, 1).cdf(thresh)
            ground_value = -(1 - p) * 0.5 * np.log(1 - rho * rho) * 1.4427
            ground_truth.append(ground_value)
            print("TE", condiMI[-1])
            print("ground_truth", ground_truth[-1])

            

        total_te[i, :] = condiMI

    plt.figure(1)
    max_te = np.amax(total_te, axis=0)
    min_te = np.min(total_te, axis=0)
    total = [sum(x) for x in zip(max_te, min_te)]
    mid_te = [x / 2 for x in total]

  
    plt.plot(thresh_vec, mid_te, color='orange', alpha=.9, label='TE')
    plt.fill_between(thresh_vec, max_te, min_te, color='orange', alpha=.9)

    plt.plot(thresh_vec, ground_truth, 'b--', label='Ground Truth of TE')

    plt.xlabel(r"threshold $\lambda$")
    plt.legend()
    plt.savefig("comte_variance.pdf")

    plt.figure(1)
    plt.plot(thresh_vec, condiMI, marker='^', color='g', label='TE')
    plt.plot(thresh_vec, ground_truth, 'b--', label='Ground Truth')

    plt.xlim(-3, 3)
    plt.xlabel(r"threshold $\lambda$")
    plt.legend()
    plt.savefig("comq.pdf")
    plt.show()


if __name__ == "__main__":
    main()
    
