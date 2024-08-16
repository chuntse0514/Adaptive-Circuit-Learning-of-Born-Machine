import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import (
    epsilon, evaluate
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data import *
import json
from scipy.sparse.csgraph import minimum_spanning_tree

def chowliu_tree(pdata):
    
    X = mutual_information(pdata)
    Tcsr = -minimum_spanning_tree(-X)
    Tcoo = Tcsr.tocoo()
    pairs = list(zip(Tcoo.row, Tcoo.col))
    print('Chow-Liu tree pairs = %s'%pairs)
    return pairs

def mutual_information(pdata):

    sl = [0, 1]  # possible states
    d = len(sl)  # number of possible states
    num_bit = int(np.round(np.log(len(pdata))/np.log(2)))
    basis = np.arange(2**num_bit, dtype='uint32')

    pxy = np.zeros([num_bit, num_bit, d, d])
    px = np.zeros([num_bit, d])
    pdata2d = np.broadcast_to(pdata[:,None], (len(pdata), num_bit))
    pdata3d = np.broadcast_to(pdata[:,None,None], (len(pdata), num_bit, num_bit))
    offsets = np.arange(num_bit-1,-1,-1)

    for s_i in sl:
        mask_i = (basis[:,None]>>offsets)&1 == s_i
        px[:,s_i] = np.ma.array(pdata2d, mask=~mask_i).sum(axis=0)
        for s_j in sl:
            mask_j = (basis[:,None]>>offsets)&1 == s_j
            pxy[:,:,s_i,s_j] = np.ma.array(pdata3d, mask=~(mask_i[:,None,:]&mask_j[:,:,None])).sum(axis=0)

    # mutual information
    pratio = pxy/np.maximum(px[:,None,:,None]*px[None,:,None,:], 1e-15)
    for i in range(num_bit):
        pratio[i, i] = 1
    I = (pxy*np.log(pratio)).sum(axis=(2,3))
    return I

class Generator1(nn.Module):

    def __init__(self, n_qubit: int, k: int, pairs=None):
        super().__init__()
        
        dev = qml.device('default.qubit.torch', wires=n_qubit)
        
        self.params = nn.ParameterList(
            # [nn.Parameter((torch.rand(n_qubit * 3) * 2 - 1) * np.pi, requires_grad=True) for _ in range(k+1)]
            [nn.Parameter(torch.normal(mean=torch.zeros(n_qubit * 3), std=torch.Tensor([np.pi/8])), requires_grad=True) for _ in range(k+1)]
        )

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit():
            for layer in range(k):
                for q in range(n_qubit):
                    qml.RX(self.params[layer][3*q], wires=q)
                    qml.RY(self.params[layer][3*q+1], wires=q)
                    qml.RX(self.params[layer][3*q+2], wires=q)
                if pairs:
                    for i, j in pairs:
                        qml.CZ(wires=[i, j])
                else:
                    for q in range(n_qubit):
                        qml.CZ(wires=[q, (q+1) % n_qubit])
            for q in range(n_qubit):
                qml.RX(self.params[-1][3*q], wires=q)
                qml.RY(self.params[-1][3*q+1], wires=q)
                qml.RX(self.params[-1][3*q+2], wires=q)
            return qml.probs()

        self.model = circuit

    def forward(self):
        return self.model()
    
class Generator2(nn.Module):
    
    def __init__(self, n_qubit: int, k: int, pairs=None):
        super().__init__()
        
        dev = qml.device('default.qubit.torch', wires=n_qubit)
        
        self.params = nn.ParameterList(
            # [nn.Parameter((torch.rand(n_qubit * 2) * 2 - 1) * np.pi, requires_grad=True) for _ in range(k)]
            [nn.Parameter(torch.normal(mean=torch.zeros(n_qubit * 2), std=torch.Tensor([np.pi/8])), requires_grad=True) for _ in range(k)]
        )
        
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit():
            for layer in range(k):
                for q in range(n_qubit):
                    qml.RY(self.params[layer][q], wires=q)
                if pairs:
                    for i, j in pairs:
                        qml.CRZ(self.params[layer][n_qubit+q], wires=[i, j])
                else:
                    for q in range(n_qubit):
                        qml.CRZ(self.params[layer][n_qubit+q], wires=[q, (q+1) % n_qubit])
                    
            return qml.probs()

        self.model = circuit

    def forward(self):
        return self.model()
    


class MMD(nn.Module):

    def __init__(self, sigmas: list, n_qubit: int, log=False):
        super().__init__()
        self.n_qubit = n_qubit
        self.log = log
        self.K = nn.Parameter(self.make_K(sigmas), requires_grad=False)

    def forward(self, x, y):
        x_y = (x - y).unsqueeze(-1)
        
        if self.log:
            return torch.log2(x_y.transpose(0, 1) @ self.K @ x_y)
        
        return x_y.transpose(0, 1) @ self.K @ x_y

    def make_K(self, sigmas: list):
        
        sigmas = torch.Tensor(sigmas)
        x_range = torch.arange(2 ** self.n_qubit)
        norm_square = torch.abs(x_range[:, None] - x_range[None, :]) ** 2
        K = (torch.exp(-norm_square.unsqueeze(-1) / (2 * sigmas ** 2)).sum(dim=-1) / len(sigmas))
        return K.double()


class QCBM:

    def __init__(self, 
                 data_class: LogNormal | Triangular | Bimodal | 
                             BarAndStripes | RealImage, 
                 n_epoch: int, 
                 reps: int, 
                 lr: float,
                 ):
        self.data_class = data_class
        self.n_qubit = data_class.n_bit
        self.n_epoch = n_epoch
        self.reps = reps
        self.lr = lr
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
        self.target_prob = torch.Tensor(data_class.get_data()).double().to(self.device)
        
        # apply MMD loss when the distribution is sparse.
        if data_class.dist_property == 'sparse':
            self.generator = Generator1(self.n_qubit, 
                                        k=reps,
                                        pairs=chowliu_tree(self.target_prob.detach().cpu().numpy())).to(self.device)
            self.mmd = MMD([0.5, 1., 2., 4.], n_qubit=self.n_qubit, log=False).to(self.device)
        
        # apply Log MMD loss when the distribution is dense.
        # this can circumvent the exponential decay of loss function.
        elif data_class.dist_property == 'dense':
            self.generator = Generator2(self.n_qubit, 
                                        k=reps,
                                        pairs=chowliu_tree(self.target_prob.detach().cpu().numpy())).to(self.device)
            self.mmd = MMD([0.5, 1., 2., 4.], n_qubit=self.n_qubit, log=True).to(self.device)

        self.optim = optim.Adam(params=self.generator.parameters(), lr=lr, amsgrad=True)
        self.loss_history = []
        self.kl_history = []
        self.js_history = []
        self.grad_norm_history = []

        self.filename = f'./images/QCBM/QCBM(data={data_class.name}, lr={lr}, reps={reps}).png'
        self.result_file = f'./results/QCBM/QCBM(data={data_class.name}, lr={lr}, reps={reps}).json'

    def plot_training_result(self, prob):
        
        fig = plt.figure(figsize=(10, 9))
        gs = gridspec.GridSpec(2, 1, figure=fig)
 
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        
        ax1_ = ax1.twinx()
        
        n_epoch = len(self.kl_history)

        ax1.semilogy(np.arange(n_epoch)+1, self.kl_history, label='KL divergence', color='red')
        ax1.semilogy(np.arange(n_epoch)+1, self.js_history, label='JS divergence', color='blue')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('KL / JS divergence')
        
        if self.data_class.dist_property == 'sparse':
            ax1_.semilogy(np.arange(n_epoch)+1, self.loss_history, label='MMD loss', color='yellowgreen')
            ax1_.set_ylabel('MMD loss')
        else:
            ax1_.plot(np.arange(n_epoch)+1, self.loss_history, label='Log MMD loss', color='yellowgreen')
            ax1_.set_ylabel('Log MMD loss')
        
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles1_, labels1_ = ax1_.get_legend_handles_labels()

        # Combine handles and labels in the order you want them in the legend
        handles = handles1 + handles1_
        labels = labels1 + labels1_

        # Create a single legend
        ax1.legend(handles, labels, loc='upper right')
        
        ax1.grid()

        if isinstance(self.data_class, RealImage):
            ax2.imshow(prob.detach().cpu().numpy().reshape(256, 256))
            
        else:
            ax2.bar(np.arange(prob.shape[0])+1, self.target_prob.detach().cpu().numpy(), alpha=0.5, color='blue', label='target')
            ax2.bar(np.arange(prob.shape[0])+1, prob.detach().cpu().numpy(), alpha=0.5, color='red', label='approx')
            ax2.legend(loc='upper right')

        plt.savefig(self.filename)

        plt.savefig(self.filename)
    
    def fit(self):
        
        threshold = 1e-5 if self.data_class.dist_property == 'sparse' else 1e-3

        for i_epoch in range(self.n_epoch):
            prob = self.generator()
            mmd_loss = self.mmd(self.target_prob, prob)

            self.optim.zero_grad()
            mmd_loss.backward()
            self.optim.step()
            self.loss_history.append(mmd_loss.item())
            kl_div, js_div = evaluate(self.target_prob, prob)
            self.kl_history.append(kl_div)
            self.js_history.append(js_div)

            grad_vec = torch.cat([params.grad for params in self.generator.params])
            grad_norm = torch.linalg.vector_norm(grad_vec)
            
            self.grad_norm_history.append(grad_norm.item())

            print(kl_div, grad_norm.item())
            
            if (i_epoch + 1) % 5 == 0:
                if self.data_class.dist_property == 'sparse':
                    print(f'epoch: {i_epoch+1} | MMD loss: {mmd_loss.item():6f} | KL divergence: {kl_div:6f} | JS divergence: {js_div:6f}')
                else:
                    print(f'epoch: {i_epoch+1} | Log MMD loss: {mmd_loss.item():6f} | KL divergence: {kl_div:6f} | JS divergence: {js_div:6f}')
                

            if grad_norm < threshold:
                break
            
        self.plot_training_result(prob)

        with open(self.result_file, 'w') as f:
            data_dict = {
                'pmf': prob.detach().cpu().tolist(),
                'kl div': self.kl_history,
                'js div': self.js_history,
                'loss history': self.loss_history,
                'grad norm': self.grad_norm_history
            }
            json.dump(data_dict, f)


if __name__ == '__main__':
    
    model = QCBM(
        data_class=DATA_HUB['real image 3'],
        n_epoch=8000,
        reps=20,
        lr=0.05,
    )

    model.fit()