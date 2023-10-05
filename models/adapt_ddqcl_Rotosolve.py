import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pennylane import numpy as np
from utils import (
    epsilon, evaluate, mutual_information, entanglement_of_formation
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import partial
from pprint import pprint
from data import *
from optimize.Rotosolve import Rotosolve_Torch
 
def operator_pool(n_qubit):

    pool = []
    gate_description = []

    for i in range(n_qubit):
        for j in range(n_qubit):
            if i != j:
                pool.append(partial(PauliStringRotation, pauliString=('XY', [i, j])))
                pool.append(partial(PauliStringRotation, pauliString=('YZ', [i, j])))
                gate_description.append(f'e^[X{i} Y{j}]')
                gate_description.append(f'e^[Y{i} Z{j}]')
        
    for i in range(n_qubit):
        pool.append(partial(PauliStringRotation, pauliString=('Y', [i])))
        gate_description.append(f'e^[Y{i}]')

    return pool, gate_description

def operator_pool_V(n_qubit):
    
    pool = []
    gate_description = []

    for i in reversed(range(n_qubit)):
        pool.append(partial(PauliStringRotation, pauliString=('Z' * i + 'Y', list(range(i)) + [i])))
        if i > 1:
            gate_description.append(f'e^[Z0:{i-1} Y{i}]')
        elif i == 1:
            gate_description.append(f'e^[Z0 Y1]')
        else:
            gate_description.append(f'e^[Y0]')
    for i in reversed(range(n_qubit-2)):
        pool.append(partial(PauliStringRotation, pauliString=('Z' * i + 'Y', list(range(i)) + [i+2])))
        if i > 1:
            gate_description.append(f'e^[Z0:{i-1}] Y{i+2}')
        elif i == 1:
            gate_description.append(f'e^[Z0 Y2]')
        else:
            gate_description.append(f'e^[Y1]')

    return pool, gate_description

def PauliStringRotation(theta, pauliString: tuple[str, list[int]]):
    
    # Basis rotation
    for pauli, qindex in zip(*pauliString):
        if pauli == 'X':
            qml.RY(-np.pi / 2, wires=qindex)
        elif pauli == 'Y':
            qml.RX(np.pi / 2, wires=qindex)
    
    # CNOT layer
    for q, q_next in zip(pauliString[1][:-1], pauliString[1][1:]):
        qml.CNOT(wires=[q, q_next])
    
    # Z rotation
    qml.RZ(theta, pauliString[1][-1])

    # CNOT layer
    for q, q_next in zip(reversed(pauliString[1][:-1]), reversed(pauliString[1][1:])):
        qml.CNOT(wires=[q, q_next])

    # Basis rotation
    for pauli, qindex in zip(*pauliString):
        if pauli == 'X':
            qml.RY(np.pi / 2, wires=qindex)
        elif pauli == 'Y':
            qml.RX(-np.pi / 2, wires=qindex)

class adapt_DDQCL:

    def __init__(self, 
                 data_class: LogNormal | Triangular | Bimodal | 
                             BarAndStripes | RealImage,
                 n_epoch: int,
                 lr: float,
                 sample_size=1000000
                 ):
        
        self.data_class = data_class
        self.n_qubit = data_class.n_bit
        self.n_epoch = n_epoch
        self.lr = lr
        self.threshold1 = 1e-5
        self.threshold2 = 1e-3
        self.Ng = 100
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.target_prob = torch.Tensor(data_class.get_data(num=sample_size)).double().to(self.device)
        self.pool, self.gate_description = operator_pool(self.n_qubit)

        self.params = nn.ParameterDict({
            'ry': nn.Parameter(torch.full((self.n_qubit,), np.pi/2), requires_grad=True),
            'freeze': nn.Parameter(torch.Tensor([]), requires_grad=False),
            'append': nn.Parameter(torch.Tensor([]), requires_grad=True)
        }).to(self.device)

        self.operatorID = {
            'append': [],
            'freeze': []
        }
        self.loss_history = []
        self.kl_history = []
        self.js_history = []

        if isinstance(data_class, RealImage):
            self.normalize_const = data_class.get_normalize_const() / 255

        self.criterion = lambda p, q: -torch.cosine_similarity(p, q, dim=0)

        self.filename = f'./images/ADAPT-DDQCL(data={data_class.name}, lr={lr}, t1={self.threshold1}, t2={self.threshold2}).png'

    def circuit(self, ry, freeze, append, mode=None, eval_gate=None):

        if mode == 'train':

            for q in range(self.n_qubit):
                qml.RY(ry[q], wires=q)

            for i, id in enumerate(self.operatorID['freeze']):
                gate = self.pool[id]
                gate(freeze[i].detach())

            for i, id in enumerate(self.operatorID['append']):
                gate = self.pool[id]
                gate(append[i])
            
            return qml.probs(wires=list(range(self.n_qubit)))

        elif mode == 'eval':

            for q in range(self.n_qubit):
                qml.RY(ry[q], wires=q)

            for i, id in enumerate(self.operatorID['freeze']):
                gate = self.pool[id]
                gate(freeze[i].detach())

            eval_gate(append)
            
            return qml.probs(wires=list(range(self.n_qubit)))

    def select_operator(self):
        
        dev = qml.device('default.qubit.torch', wires=self.n_qubit)
        
        circuit = self.circuit
        model = qml.QNode(circuit, dev, interface='torch', diff_method='backprop')
        min_vals = []
        self.params['append'] = nn.Parameter(torch.zeros(1), requires_grad=True).to(self.device)
        self.params['ry'].requires_grad = False
        self.params['freeze'].requires_grad = False

        for eval_gate in self.pool:
            opt = Rotosolve_Torch(
                params=[self.params['ry'], self.params['freeze'], self.params['append']],
                qnode=model,
                criterion=self.criterion,
                target_prob=self.target_prob,
                full_output=True,
                mode='eval',
                eval_gate=eval_gate
            )
            min_vals.append(opt.step(mode='eval')[0])

        self.params['ry'].requires_grad = True
        self.params['freeze'].requires_grad = True
        
        selected_indices = np.argsort(np.array(min_vals))[:self.Ng].tolist()
        selected_gates = [self.gate_description[i] for i in selected_indices]
        min_losses = [min_vals[i] for i in selected_indices]

        return min_losses, selected_indices, selected_gates
    
    def entanglement_measure(self):

        state = torch.sqrt(self.target_prob)
        ent_list = []
        
        for i in range(self.n_qubit):
            for j in range(i+1, self.n_qubit):
                MI = mutual_information(state, subsystems=(i, j))
                EOF = entanglement_of_formation(state, subsystems=(i, j))
                ent_list.append([(i, j), MI, EOF])

        ent_list = sorted(ent_list, key=lambda x: x[1], reverse=True)

        for ent_description in ent_list:
            (i, j) = ent_description[0]
            MI = ent_description[1]
            EOF = ent_description[2]
            print(f'subsystem: ({i}, {j}) | mutual information: {MI: 3f} | entanglement of formation: {EOF:3f}')
    
    def fit(self):
        
        dev = qml.device('default.qubit.torch', wires=self.n_qubit)
        
        # plt.ion()
        # fig = plt.figure(figsize=(15, 9))
        # gs = gridspec.GridSpec(2, 2, figure=fig)

        # if isinstance(self.data_class, RealImage):
        #     ax1 = fig.add_subplot(gs[0, 0])
        #     ax2 = fig.add_subplot(gs[1, 0])
        #     ax3 = fig.add_subplot(gs[0, 1])
        #     ax4 = fig.add_subplot(gs[1, 1])
        # else:
        #     ax1 = fig.add_subplot(gs[0, 0])
        #     ax2 = fig.add_subplot(gs[1, 0])
        #     ax3 = fig.add_subplot(gs[:, 1])

        self.entanglement_measure()

        for i_epoch in range(self.n_epoch):

            min_losses, selected_indices, selected_gates = self.select_operator()

            print(f'==== Found minimum losses')
            pprint(min_losses)
            print('of gates')
            pprint(', '.join(selected_gates))
            print('====')

            self.operatorID['append'] = selected_indices
            self.params['append'] = nn.Parameter(torch.zeros(self.Ng), requires_grad=True).to(self.device)
            circuit = self.circuit
            model = qml.QNode(circuit, dev, interface='torch', diff_method='backprop')

            opt = Rotosolve_Torch(
                params=[self.params['ry'], self.params['freeze'], self.params['append']],
                qnode=model,
                criterion=self.criterion,
                target_prob=self.target_prob,
                full_output=True,
                mode='train'
            )

            print(opt.step())
            prob =  model(self.params['ry'], self.params['freeze'], self.params['append'], mode='train').squeeze()
            loss = self.criterion(self.target_prob, prob)
            self.loss_history.append(loss.item())
            self.operatorID['freeze'] += self.operatorID['append']
            self.params['freeze'] = torch.cat((self.params['freeze'], self.params['append'].clone()))

            kl_div, js_div = evaluate(self.target_prob.detach().cpu().numpy(), prob.detach().cpu().numpy())
            self.kl_history.append(kl_div)
            self.js_history.append(js_div)
            print(f'epoch: {i_epoch+1}  |   loss: {self.loss_history[-1]:.6f}  |   KL divergence: {kl_div:.6f}  |  JS divergence: {js_div:.6f}')

            # ax1.clear()
            # ax1.plot(np.arange(len(self.kl_history))+1, self.kl_history, label='KL divergence', color='red', marker='^', markerfacecolor=None)
            # ax1.plot(np.arange(len(self.js_history))+1, self.js_history, label='JS divergence', color='blue', marker='X', markerfacecolor=None)
            # ax1.set_xlabel('epoch')
            # ax1.set_ylabel('KL / JS divergence')
            # ax1.grid()
            # ax1.legend()

            # ax2.clear()
            # ax2.plot(np.arange(len(self.loss_history))+1, self.loss_history, color='green', marker='P')
            # ax2.set_xlabel('iteration')
            # ax2.set_ylabel(self.loss_fn)
            # ax2.grid()

            # ax3.clear()
            # ax3.bar(np.arange(prob.shape[0])+1, self.target_prob.detach().cpu().numpy(), alpha=0.5, color='blue', label='target')
            # ax3.bar(np.arange(prob.shape[0])+1, prob.detach().cpu().numpy(), alpha=0.5, color='red', label='approx')
            # ax3.legend()

            # if isinstance(self.data_class, RealImage):
            #     ax4.clear()
            #     ax4.imshow(prob.detach().cpu().numpy().reshape(256, 256))
                

            # plt.pause(0.01)
            # plt.savefig(self.filename)
            
        print(qml.draw(model)(self.params['ry'], self.params['freeze'], self.params['append'], mode='train'))
        
        plt.ioff()
        plt.show()

if __name__ == '__main__':

    from data import DATA_HUB

    model = adapt_DDQCL(
        data_class=DATA_HUB['real image 1'],
        n_epoch=40,
        lr=1,
        sample_size=100000
    )
    
    model.fit()