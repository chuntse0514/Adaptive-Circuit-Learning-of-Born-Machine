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
import pickle

def operator_pool(n_qubit):

    pool = []
    gate_description = []

    for i in range(n_qubit):
        for j in range(n_qubit):
            if i != j:
                pool.append(partial(PauliStringRotation, pauliString=('XY', [i, j])))
                pool.append(partial(PauliStringRotation, pauliString=('YZ', [i, j])))
                pool.append(partial(qml.CRY, wires=[i, j]))
                gate_description.append(f'e^[X{i} Y{j}]')
                gate_description.append(f'e^[Y{i} Z{j}]')
                gate_description.append(f'CRY[{i}, {j}]')
    for i in range(n_qubit):
        pool.append(partial(qml.RY, wires=i))
        gate_description.append(f'RY[{i}]')
        
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
                 loss_fn: str,
                 sample_size=1000000
                 ):
        
        self.data_class = data_class
        self.n_qubit = data_class.n_bit
        self.n_epoch = n_epoch
        self.lr = lr
        self.threshold1 = 1e-3
        self.threshold2 = 5e-3
        self.ratio = 0.1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.target_prob = torch.Tensor(data_class.get_data(num=sample_size)).double().to(self.device)
        self.pool, self.gate_description = operator_pool(self.n_qubit)

        self.params = nn.ParameterDict({
            'ry': nn.Parameter(torch.full((self.n_qubit,), np.pi/2), requires_grad=True),
            'eval-grad': nn.Parameter(torch.Tensor([]), requires_grad=True),
            'append': nn.Parameter(torch.Tensor([]), requires_grad=True)
        }).to(self.device)

        self.operatorID = []
        self.loss_history = []
        self.kl_history = []
        self.js_history = []

        if isinstance(data_class, RealImage):
            self.normalize_const = data_class.get_normalize_const() / 255

        Criterion = {
            'KL divergence': lambda p, q: -torch.inner(p[p>0], torch.log(q[p>0] / p[p>0])),
            'Renyi-0.5 divergence': lambda p, q: -2 * torch.log(torch.sum(torch.sqrt(p[p>0] * q[p>0]))),
            'Renyi-2 divergence': lambda p, q: torch.log(torch.sum(p[p>0] ** 2 / (q[p>0]))),
            'Quantum relative entropy': lambda p, q: 1 - torch.sum(torch.sqrt(p[p>0] * q[p>0])) ** 2,
            'MSE': lambda p, q: torch.sum(((p - q) * self.normalize_const) ** 2) / (2 ** self.n_qubit),
            'negative cosine similarity': lambda p, q: -torch.cosine_similarity(p, q, dim=0)
        }
        
        self.loss_fn = loss_fn
        self.criterion = Criterion[loss_fn]

        self.filename = f'./images/ADAPT-DDQCL-4(data={data_class.name}, lr={lr}, loss={loss_fn}, ratio={self.ratio}, t1={self.threshold1}, t2={self.threshold2}).png'
        self.tensor_file = f'./results/ADAPT-DDQCL-4(data={data_class.name}, lr={lr}, loss={loss_fn}, ratio={self.ratio}, t1={self.threshold1}, t2={self.threshold2}).pt'
        self.result_file = f'./results/ADAPT-DDQCL-4(data={data_class.name}, lr={lr}, loss={loss_fn}, ratio={self.ratio}, t1={self.threshold1}, t2={self.threshold2}).pkl'

    def circuit(self, ry, append, eval_params=None, mode=None):

        if mode == 'train':

            for q in range(self.n_qubit):
                qml.RY(ry[q], wires=q)

            for i, id in enumerate(self.operatorID):
                gate = self.pool[id]
                gate(append[i])
            
            return qml.probs(wires=list(range(self.n_qubit)))

        elif mode == 'eval-grad':

            for q in range(self.n_qubit):
                qml.RY(ry[q].detach(), wires=q)

            for i, id in enumerate(self.operatorID):
                gate = self.pool[id]
                gate(append[i].detach())

            for i, gate in enumerate(self.pool):
                gate(eval_params[i])
            
            return qml.probs(wires=list(range(self.n_qubit)))

    def select_operator(self):
        
        dev = qml.device('default.qubit.torch', wires=self.n_qubit)
        
        circuit = self.circuit
        self.params['eval-grad'] = nn.Parameter(torch.zeros(len(self.pool)), requires_grad=True)
        model = qml.QNode(circuit, dev, interface='torch', diff_method='backprop')
        prob = model(self.params['ry'], self.params['append'], self.params['eval-grad'], mode='eval-grad')
        loss = self.criterion(self.target_prob, prob)
        loss.backward()
        grads = self.params['eval-grad'].grad.detach().cpu().numpy()
        grads = np.abs(grads)
        
        selected_index = np.argmax(grads)
        selected_gate = self.gate_description[selected_index]
        max_grad = grads[selected_index]

        return max_grad, selected_index, selected_gate
    
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
        circuit = self.circuit
        model = qml.QNode(circuit, dev, interface='torch', diff_method='backprop')
        opt = optim.Adam(self.params.values(), lr=self.lr)

        self.entanglement_measure()

        for i_epoch in range(self.n_epoch):

            max_grad, max_grad_index, max_grad_gate = self.select_operator()

            pprint(f'==== Found maximium gradient {max_grad} of gate ' + max_grad_gate + ' ====')
            if max_grad < self.threshold1:
                print('Convergence criterion has reached, break the loop!')
                break

            self.operatorID += [max_grad_index]
            self.params['append'] = torch.cat((self.params['append'], nn.Parameter(torch.zeros(1), requires_grad=True).to(self.device)))
            opt.param_groups[0]['params'][0] = self.params['append']

            while True:
                opt.zero_grad()
                prob =  model(self.params['ry'], self.params['append'], mode='train')
                loss = self.criterion(self.target_prob, prob)
                loss.backward()
                opt.step()
                self.loss_history.append(loss.item())

                grad_vec = torch.cat((self.params['append'].grad, self.params['ry'].grad))
                grad_norm = torch.linalg.vector_norm(grad_vec)
                print(loss.item(), grad_norm.item())
                if grad_norm < self.threshold2:
                    break

            kl_div, js_div = evaluate(self.target_prob.detach().cpu().numpy(), prob.detach().cpu().numpy())
            self.kl_history.append(kl_div)
            self.js_history.append(js_div)
            print(f'epoch: {i_epoch+1}  |   loss: {self.loss_history[-1]:.6f}  |   KL divergence: {kl_div:.6f}  |  JS divergence: {js_div:.6f}')
            
        print(qml.draw(model)(self.params['ry'], self.params['append'], mode='train'))
    

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


        ax1.plot(np.arange(len(self.kl_history))+1, self.kl_history, label='KL divergence', color='red', marker='^', markerfacecolor=None)
        ax1.plot(np.arange(len(self.js_history))+1, self.js_history, label='JS divergence', color='blue', marker='X', markerfacecolor=None)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('KL / JS divergence')
        ax1.grid()
        ax1.legend()

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
            ax4.imshow(prob.detach().cpu().numpy().reshape(256, 256), cmap='gray')

        plt.pause(0.01)
        plt.savefig(self.filename)

        torch.save(prob.detach().cpu(), self.tensor_file)
        with open(self.result_file, 'wb') as f:
            pickle.dump((self.loss_history, self.js_history, self.kl_history), f)


if __name__ == '__main__':

    from data import DATA_HUB

    model = adapt_DDQCL(
        data_class=DATA_HUB['real image 1'],
        n_epoch=150,
        lr=1e-3,
        loss_fn='KL divergence',
        sample_size=100000
    )
    
    model.fit()
