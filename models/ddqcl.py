import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from pennylane import numpy as np
from utils import epsilon, evaluate
from circuits.templates import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Generator(nn.Module):

    def __init__(self, n_qubit: int, circuit_depth: int):

        super().__init__()
        self.n_qubit = n_qubit
        self.circuit_depth = circuit_depth
        self.params = nn.Parameter((torch.rand(circuit_depth, n_qubit) * 2 - 1) * np.pi, requires_grad=True)
        self.su2_params = nn.Parameter((torch.rand(n_qubit, 3) * 2 - 1) * np.pi, requires_grad=True)
        self.circuit = RYCircuitModel(n_qubit, circuit_depth)
        
        # self.params = nn.Parameter((torch.rand(2, n_qubit, circuit_depth) * 2 - 1) * np.pi, requires_grad=True)
        # self.su2_params = nn.Parameter((torch.rand(n_qubit, 3) * 2 - 1) * np.pi, requires_grad=True)
        # self.circuit = MølmerSørensenXXCircuitModel(n_qubit, circuit_depth)

        # self.params = nn.Parameter((torch.zeros(countTotalParameter(n_qubit), 4) * 2 - 1) * np.pi, requires_grad=True)
        # self.su2_params = nn.Parameter((torch.zeros(n_qubit, 3) * 2 - 1) * np.pi, requires_grad=True)
        # self.circuit = TwoQubitEntangleCircuitModel(n_qubit)

        # self.ent_params = nn.Parameter((torch.zeros(n_qubit-1, circuit_depth, 2)), requires_grad=True)
        # self.rot_params = nn.Parameter((torch.zeros(n_qubit, circuit_depth)), requires_grad=True)
        # self.su2_params = nn.Parameter((torch.rand(n_qubit, 3) * 2 - 1) * np.pi, requires_grad=True)
        # self.circuit = HardwareEfficientModel(n_qubit, circuit_depth)

    def forward(self):
        return self.circuit(self.params, self.su2_params)
    
class DDQCL:

    def __init__(self, n_qubit: int, n_epoch: int, circuit_depth: int, lr: float):
        self.n_qubit = n_qubit
        self.circuit_depth = circuit_depth
        self.n_epoch = n_epoch
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(n_qubit, circuit_depth).to(self.device)
        self.opt = optim.Adam(
            list(self.generator.parameters()),
            lr=lr
        )

        self.loss_history = []
        self.kl_history = []
        self.js_history = []

    def fit(self, target_prob: torch.Tensor):

        fig = plt.figure(figsize=(15, 9))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[:, 1])

        for i_epoch in range(self.n_epoch):

            prob = self.generator.forward()
            loss = -torch.inner(target_prob, torch.log2(prob + epsilon))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.loss_history.append(loss.item())

            prob = torch.squeeze(prob)

            kl_div, js_div = evaluate(target_prob.detach().cpu().numpy(), prob.detach().cpu().numpy())
            self.kl_history.append(kl_div)
            self.js_history.append(js_div)

            if (i_epoch + 1) % 5 == 0:
                print(f'epoch: {i_epoch+1}  |  cross entropy: {self.loss_history[-1]: 6f}  |  KL div: {kl_div:6f}  |  JS div: {js_div:6f}')

        ax1.plot(np.arange(self.n_epoch)+1, self.kl_history, label='KL divergence', color='red', marker='^', markerfacecolor=None)
        ax1.plot(np.arange(self.n_epoch)+1, self.js_history, label='JS divergence', color='blue', marker='x', markerfacecolor=None)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('KL / JS divergence')
        ax1.grid()
        ax1.legend()

        ax2.plot(np.arange(len(self.loss_history))+1, self.loss_history, color='green', marker='P')
        ax2.set_xlabel('iteration')
        ax2.set_ylabel('cross entropy')
        ax2.grid()

        ax3.bar(np.arange(prob.shape[0])+1, target_prob.detach().cpu().numpy(), alpha=0.5, color='blue', label='target', width=100)
        ax3.bar(np.arange(prob.shape[0])+1, prob.detach().cpu().numpy(), alpha=0.5, color='red', label='approx', width=100)
        ax3.legend()

        plt.savefig('ddqcl_bas4x4.png')

        return self.kl_history, self.js_history

if __name__ == '__main__':
    from data import DATA_HUB

    n_qubit = 16
    n_epoch = 2000
    circuit_depth = 20
    lr = 1e-1

    data = torch.Tensor(DATA_HUB['bas 4x4']().get_data(10000000)).double().to(torch.device("cuda:0"))
    model = DDQCL(n_qubit, n_epoch, circuit_depth, lr)
    model.fit(data)