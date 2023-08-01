import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from pennylane import numpy as np
from utils import epsilon, evaluate
from circuits.templates import *

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
                print(f'epoch: {i_epoch+1} nll: {self.loss_history[-1]} KL_divergence: {kl_div} JS_divergence: {js_div}')

        # print(qml.draw(self.generator.circuit)(self.generator.params, self.generator.su2_params))

        return self.kl_history, self.js_history