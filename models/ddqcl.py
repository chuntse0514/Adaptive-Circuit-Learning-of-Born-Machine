import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from pennylane import numpy as np
from utils import epsilon, evaluate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data import *
    
class DDQCL:

    def __init__(self,
                 data_class: LogNormal | Triangular | Bimodal | 
                             BarAndStripes | RealImage,
                 n_epoch: int, 
                 reps: int, 
                 lr: float,
                 loss_fn: str,
                 sample_size=1000000
                 ):
        
        self.data_class = data_class
        self.n_qubit = data_class.n_bit
        self.reps = reps
        self.n_epoch = n_epoch
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.target_prob = torch.Tensor(data_class.get_data(num=sample_size)).double().to(self.device)

        self.params = nn.ParameterDict({
            'su2': nn.Parameter((torch.rand(self.n_qubit, 2) * 2 - 1) * np.pi, requires_grad=True),
            'ry': nn.Parameter((torch.rand(self.n_qubit, reps) * 2 - 1) * np.pi, requires_grad=True)
        }).to(self.device)

        self.loss_history = []
        self.kl_history = []
        self.js_history = []

        if isinstance(data_class, RealImage):
            self.normalize_const = data_class.get_normalize_const() / 255

        Criterion = {
            'KL divergence': lambda p, q: -torch.inner(p[p>0], torch.log2(q[p>0] / p[p>0])) - torch.sum(p==0),
            'Renyi-0.5 divergence': lambda p, q: -2 * torch.log2(torch.sum(torch.sqrt(p[p>0] * q[p>0]))),
            'Renyi-2 divergence': lambda p, q: torch.log2(torch.sum(p[p>0] ** 2 / (q[p>0]))),
            'Renyi-inf divergence': lambda p, q: torch.log2(torch.max(p[p>0] / (q[p>0]))),
            'Quantum relative entropy': lambda p, q: 1 - torch.sum(torch.sqrt(p[p>0] * q[p>0])) ** 2,
            'MSE': lambda p, q: torch.sum(((p - q) * self.normalize_const) ** 2) / (2 ** self.n_qubit)
        }
        
        self.loss_fn = loss_fn
        self.criterion = Criterion[loss_fn]

        self.filename = f'./images/DDQCL(data={data_class.name}, lr={lr}, loss={loss_fn}, reps={reps}).png'

    def get_circuit(self):
        
        for q in range(self.n_qubit):
            qml.RY(self.params['su2'][q, 0], wires=q)
            qml.RZ(self.params['su2'][q, 1], wires=q)

        for rep in range(self.reps):
            for q in range(self.n_qubit):
                qml.CZ(wires=[q, (q+1) % self.n_qubit])
            
            for q in range(self.n_qubit):
                qml.RY(self.params['ry'][q, rep], wires=q)

        return qml.probs(wires=list(range(self.n_qubit)))

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

        dev = qml.device('default.qubit.torch', wires=self.n_qubit)
        circuit = self.get_circuit
        model = qml.QNode(circuit, dev, interface='torch', diff_method='backprop')
        opt = optim.Adam(self.params.values(), lr=self.lr)

        for i_epoch in range(self.n_epoch):

            prob = model()
            loss = self.criterion(self.target_prob, prob)
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.loss_history.append(loss.item())

            prob = torch.squeeze(prob)

            kl_div, js_div = evaluate(self.target_prob.detach().cpu().numpy(), prob.detach().cpu().numpy())
            self.kl_history.append(kl_div)
            self.js_history.append(js_div)

            grad_vec = torch.cat((self.params['su2'].grad.view(-1) , self.params['ry'].grad.view(-1)))
            grad_norm = torch.linalg.vector_norm(grad_vec)

            print(loss.item(), grad_norm.item())

            if (i_epoch + 1) % 5 == 0:
                print(f'epoch: {i_epoch+1}  |  loss: {self.loss_history[-1]: 6f}  |  KL divergence: {kl_div:6f}  |  JS divergence: {js_div:6f}')


            if (i_epoch + 1) % (self.n_epoch // 5) == 0: 
                ax1.clear()
                ax1.plot(np.arange(len(self.kl_history))+1, self.kl_history, label='KL divergence', color='red', marker='^', markerfacecolor=None)
                ax1.plot(np.arange(len(self.js_history))+1, self.js_history, label='JS divergence', color='blue', marker='x', markerfacecolor=None)
                ax1.set_xlabel('epoch')
                ax1.set_ylabel('KL / JS divergence')
                ax1.grid()
                ax1.legend()

                ax2.clear()
                ax2.plot(np.arange(len(self.loss_history))+1, self.loss_history, color='green', marker='P')
                ax2.set_xlabel('iteration')
                ax2.set_ylabel(self.loss_fn)
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

        print(qml.draw(model)())
            
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    
    from data import DATA_HUB

    model = DDQCL(
        data_class=DATA_HUB['real image 1'],
        n_epoch=2000,
        reps=20,
        lr=1e-3,
        loss_fn='MSE',
        sample_size=100000
    )
    
    model.fit()