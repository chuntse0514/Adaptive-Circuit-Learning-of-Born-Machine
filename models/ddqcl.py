import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from pennylane import numpy as np
from utils import epsilon, evaluate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data import *
import json
    
class DDQCL:

    def __init__(self,
                 data_class: LogNormal | Triangular | Bimodal | 
                             BarAndStripes | RealImage,
                 n_epoch: int, 
                 reps: int, 
                 lr: float,
                 ):
        
        self.data_class = data_class
        self.n_qubit = data_class.n_bit
        self.reps = reps
        self.n_epoch = n_epoch
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.target_prob = torch.Tensor(data_class.get_data()).double().to(self.device)

        # self.params = nn.ParameterList(
        #     [nn.Parameter((torch.rand(self.n_qubit * 3) * 2 -1) * np.pi, requires_grad=True) for _ in range(reps+1)]
        # ).to(self.device)
        
        self.params = nn.ParameterList(
            [nn.Parameter((torch.rand(self.n_qubit * 2) * 2 - 1) * np.pi, requires_grad=True) for _ in range(reps)]
        ).to(self.device)

        self.loss_history = []
        self.kl_history = []
        self.js_history = []

        if isinstance(data_class, RealImage):
            self.normalize_const = data_class.get_normalize_const() / 255
        
        self.criterion = lambda p, q: torch.inner(p[p>0], torch.log(p[p>0] / q[p>0]))

        self.filename = f'./images/DDQCL(data={data_class.name}, lr={lr}, reps={reps}).png'
        self.result_file = f'./results/DDQCL(data={data_class.name}, lr={lr}, reps={reps}).json'
    
    # def circuit(self):
    #     for layer in range(self.reps):
    #         for q in range(self.n_qubit):
    #             qml.RX(self.params[layer][3*q], wires=q)
    #             qml.RY(self.params[layer][3*q+1], wires=q)
    #             qml.RX(self.params[layer][3*q+2], wires=q)
    #         for q in range(self.n_qubit):
    #             qml.CZ(wires=[q, (q+1) % self.n_qubit])
    #     for q in range(self.n_qubit):
    #         qml.RX(self.params[-1][3*q], wires=q)
    #         qml.RY(self.params[-1][3*q+1], wires=q)
    #         qml.RX(self.params[-1][3*q+2], wires=q)
    #     return qml.probs()
    
    def circuit(self):
        for layer in range(self.reps):
            for q in range(self.n_qubit):
                qml.RY(self.params[layer][q], wires=q)

            for q in range(self.n_qubit):
                qml.CRZ(self.params[layer][self.n_qubit+q], wires=[q, (q+1) % self.n_qubit])
                
        return qml.probs()

    def fit(self):

        dev = qml.device('default.qubit.torch', wires=self.n_qubit)
        circuit = self.circuit
        model = qml.QNode(circuit, dev, interface='torch', diff_method='backprop')
        opt = optim.Adam(self.params, lr=self.lr)
        # scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor=0.1, total_iters=5000)

        for i_epoch in range(self.n_epoch):

            prob = model()
            loss = self.criterion(self.target_prob, prob)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # scheduler.step()
            self.loss_history.append(loss.item())

            prob = torch.squeeze(prob)

            kl_div, js_div = evaluate(self.target_prob.detach().cpu().numpy(), prob.detach().cpu().numpy())
            self.kl_history.append(kl_div)
            self.js_history.append(js_div)

            grad_vec = torch.cat([params.grad for params in self.params])
            grad_norm = torch.linalg.vector_norm(grad_vec)

            print(loss.item(), grad_norm.item())

            if (i_epoch + 1) % 5 == 0:
                print(f'epoch: {i_epoch+1}  |  loss: {self.loss_history[-1]: 6f}  |  KL divergence: {kl_div:6f}  |  JS divergence: {js_div:6f}')

            if grad_norm < 1e-3:
                break

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

        ax1.plot(np.arange(len(self.kl_history))+1, self.kl_history, label='KL divergence', color='red', markerfacecolor=None)
        ax1.plot(np.arange(len(self.js_history))+1, self.js_history, label='JS divergence', color='blue', markerfacecolor=None)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('KL / JS divergence')
        ax1.grid()
        ax1.legend()

        ax2.plot(np.arange(len(self.loss_history))+1, self.loss_history, color='green')
        ax2.set_xlabel('iteration')
        ax2.set_ylabel('KL divergence')
        ax2.grid()

        ax3.bar(np.arange(prob.shape[0])+1, self.target_prob.detach().cpu().numpy(), alpha=0.5, color='blue', label='target')
        ax3.bar(np.arange(prob.shape[0])+1, prob.detach().cpu().numpy(), alpha=0.5, color='red', label='approx')
        ax3.legend()

        if isinstance(self.data_class, RealImage):
            ax4.clear()
            ax4.imshow(prob.detach().cpu().numpy().reshape(256, 256))

        plt.savefig(self.filename)

        with open(self.result_file, 'w') as f:
            data_dict = {
                'pmf': prob.detach().cpu().tolist(),
                'kl div': self.kl_history,
                'js div': self.js_history,
                'loss history': self.loss_history
            }
            json.dump(data_dict, f)


if __name__ == '__main__':

    model = DDQCL(
        data_class=DATA_HUB['real image 3'],
        n_epoch=8000,
        reps=25,
        lr=0.01,
    )
    
    model.fit()