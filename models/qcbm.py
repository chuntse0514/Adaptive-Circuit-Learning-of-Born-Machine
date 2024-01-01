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
import pickle
from scipy.sparse.csgraph import minimum_spanning_tree

def chowliu_tree(pdata):
    '''
    generate chow-liu tree.

    Args:
        pdata (1darray): empirical distribution in dataset

    Returns:
        list: entangle pairs.
    '''
    X = mutual_information(pdata)
    Tcsr = -minimum_spanning_tree(-X)
    Tcoo = Tcsr.tocoo()
    pairs = list(zip(Tcoo.row, Tcoo.col))
    print('Chow-Liu tree pairs = %s'%pairs)
    return pairs

def mutual_information(pdata):
    '''
    calculate mutual information I = \sum\limits_{x,y} p(x,y) log[p(x,y)/p(x)/p(y)]

    Args:
        pdata (1darray): empirical distribution in dataset

    Returns:
        2darray: mutual information table.
    '''
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

class Generator(nn.Module):

    def __init__(self, n_qubit: int, k: int, pairs=None):
        super().__init__()
        self.params = nn.ParameterList(
            [nn.Parameter((torch.rand(n_qubit * 3) * 2 - 1) * np.pi, requires_grad=True) for _ in range(k+1)]
        )
        dev = qml.device('default.qubit.torch', wires=n_qubit)
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit():
            for layer in range(k):
                for q in range(n_qubit):
                    qml.RZ(self.params[layer][3*q], wires=q)
                    qml.RX(self.params[layer][3*q+1], wires=q)
                    qml.RZ(self.params[layer][3*q+2], wires=q)
                if pairs:
                    for i, j in pairs:
                        qml.CNOT(wires=[i, j])
                else:
                    for q in range(n_qubit):
                        qml.CNOT(wires=[q, (q+1) % n_qubit])
            for q in range(n_qubit):
                qml.RZ(self.params[-1][3*q], wires=q)
                qml.RX(self.params[-1][3*q+1], wires=q)
                qml.RZ(self.params[-1][3*q+2], wires=q)
            return qml.probs()

        self.model = circuit

    def forward(self):
        return self.model()


class MMD(nn.Module):

    def __init__(self, sigmas: list, n_qubit: int):
        super().__init__()
        self.n_qubit = n_qubit
        self.K = nn.Parameter(self.make_K(sigmas), requires_grad=False)

    def forward(self, x, y):
        x_y = (x - y).unsqueeze(-1)
        return x_y.transpose(0, 1) @ self.K @ x_y

    def to_binary(self, x):
        r = torch.arange(self.n_qubit)
        to_binary_op = torch.ones_like(r) << r  # (n_qubit,)
        return ((x.unsqueeze(-1) & to_binary_op) > 0).long()

    def make_K(self, sigmas: list):
        sigmas = torch.Tensor(sigmas)
        r = self.to_binary(torch.arange(2 ** self.n_qubit)).float()  # (2 ** n_qubit, n_qubit)

        x = r.unsqueeze(1)  # (2 ** n_qubit, 1, n_qubit)
        y = r.unsqueeze(0)  # (1, 2 ** n_qubit, n_qubit)
        
        norm_square = (x** 2 + y ** 2 - 2 * x * y).sum(dim=-1)  # (2 ** n_qubit, 2 ** n_qubit)
        
        K = (-norm_square.unsqueeze(-1) / (2 * sigmas)).exp().sum(dim=-1)  # (2 ** n_qubit, 2 ** n_qubit)
        return K.double()


class QCBM:

    def __init__(self, 
                 data_class: LogNormal | Triangular | Bimodal | 
                             BarAndStripes | RealImage, 
                 n_epoch: int, 
                 reps: int, 
                 lr: float,
                 sample_size=1000000 
                 ):
        self.data_class = data_class
        self.n_qubit = data_class.n_bit
        self.n_epoch = n_epoch
        self.reps = reps
        self.lr = lr
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
        self.target_prob = torch.Tensor(data_class.get_data(num=sample_size)).double().to(self.device)
        
        self.generator = Generator(self.n_qubit, 
                                   k=reps,
                                   pairs=chowliu_tree(self.target_prob.detach().cpu().numpy())).to(self.device)
        self.optim = optim.Adam(params=self.generator.parameters(), lr=lr, amsgrad=True)
        self.mmd = MMD([0.5, 1., 2., 4.], n_qubit=self.n_qubit).to(self.device)

        self.loss_history = []
        self.kl_history = []
        self.js_history = []

        self.filename = f'./images/QCBM(data={data_class.name}, lr={lr}, reps={reps}).png'
        self.tensor_file = f'./results/QCBM(data={data_class.name}, lr={lr}, reps={reps}).pt'
        self.result_file = f'./results/QCBM(data={data_class.name}, lr={lr}, reps={reps}).pkl'

    def fit(self):

        plt.ion()
        fig = plt.figure(figsize=(15, 9))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        if isinstance(self.data_class, RealImage):
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[0, 1])
            ax4 = fig.add_subplot(gs[1, 1])
        else:
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[:, 1])

        for i_epoch in range(self.n_epoch):
            prob = self.generator()
            mmd_loss = self.mmd(self.target_prob, prob)

            self.optim.zero_grad()
            mmd_loss.backward()
            self.optim.step()
            self.loss_history.append(mmd_loss.item())
            kl_div, js_div = evaluate(self.target_prob.detach().cpu().numpy(), prob.detach().cpu().numpy())
            self.kl_history.append(kl_div)
            self.js_history.append(js_div)

            grad_vec = torch.cat([params.grad for params in self.generator.params])
            grad_norm = torch.linalg.vector_norm(grad_vec)

            if (i_epoch + 1) % 5 == 0:
                print(f'epoch: {i_epoch+1} | MMD loss: {mmd_loss.item():6f} | KL divergence: {kl_div:6f} | JS divergence: {js_div:6f}')

            if grad_norm < 1e-3:
                break

        ax1.clear()
        ax1.plot(np.arange(len(self.kl_history))+1, self.kl_history, label='KL divergence', color='red', marker='^', markerfacecolor=None)
        ax1.plot(np.arange(len(self.js_history))+1, self.js_history, label='JS divergence', color='blue', marker='x', markerfacecolor=None)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('KL / JS divergence')
        ax1.grid()
        ax1.legend()

        ax2.clear()
        ax2.plot(np.arange(len(self.loss_history))+1, self.loss_history, color='royalblue', marker='P')
        ax2.set_xlabel('iteration')
        ax2.set_ylabel('MMD loss')
        ax2.grid()

        ax3.clear()
        ax3.bar(np.arange(prob.shape[0])+1, self.target_prob.detach().cpu().numpy(), alpha=0.5, color='blue', label='target')
        ax3.bar(np.arange(prob.shape[0])+1, prob.detach().cpu().numpy(), alpha=0.5, color='red', label='approx')
        ax3.legend()

        if isinstance(self.data_class, RealImage):
            ax4.clear()
            ax4.imshow(prob.detach().cpu().numpy().reshape(256, 256))

        plt.pause(0.01)
        plt.savefig(self.filename)
        
        plt.ioff()
        plt.show()

        torch.save(prob.detach().cpu(), self.tensor_file)
        with open(self.result_file, 'wb') as f:
            pickle.dump((self.loss_history, self.js_history, self.kl_history), f)


if __name__ == '__main__':
    
    model = QCBM(
        data_class=DATA_HUB['triangular 10'],
        n_epoch=3160,
        reps=16,
        lr=1e-2,
        sample_size=100000000
    )

    model.fit()