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
    
class Generator1(nn.Module):

    def __init__(self, n_qubit: int, k: int):
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
    
    def __init__(self, n_qubit: int, k: int):
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

                for q in range(n_qubit):
                    qml.CRZ(self.params[layer][n_qubit+q], wires=[q, (q+1) % n_qubit])
                    
            return qml.probs()

        self.model = circuit

    def forward(self):
        return self.model()
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
        
        if data_class.dist_property == 'sparse':
            self.generator = Generator1(self.n_qubit, k=reps).to(self.device)
        
        else:
            self.generator = Generator2(self.n_qubit, k=reps).to(self.device)

        self.loss_history = []
        self.kl_history = []
        self.js_history = []
        self.grad_norm_history = []
        
        self.criterion = lambda p, q: torch.inner(p[p>0], torch.log(p[p>0] / q[p>0]))
        self.optim = optim.Adam(params=self.generator.parameters(), lr=lr, amsgrad=True)
        
        self.filename = f'./images/DDQCL/DDQCL(data={data_class.name}, lr={lr}, reps={reps}).png'
        self.result_file = f'./results/DDQCL/DDQCL(data={data_class.name}, lr={lr}, reps={reps}).json'
    
    def plot_training_result(self, prob):
        
        fig = plt.figure(figsize=(10, 9))
        gs = gridspec.GridSpec(2, 1, figure=fig)
 
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        
        n_epoch = len(self.kl_history)

        ax1.semilogy(np.arange(n_epoch)+1, self.kl_history, label='KL divergence', color='red')
        ax1.semilogy(np.arange(n_epoch)+1, self.js_history, label='JS divergence', color='blue')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('KL / JS divergence')
        ax1.grid()
        ax1.legend(loc='upper right')

        if isinstance(self.data_class, RealImage):
            ax2.imshow(prob.detach().cpu().numpy().reshape(256, 256))
            
        else:
            ax2.bar(np.arange(prob.shape[0])+1, self.target_prob.detach().cpu().numpy(), alpha=0.5, color='blue', label='target')
            ax2.bar(np.arange(prob.shape[0])+1, prob.detach().cpu().numpy(), alpha=0.5, color='red', label='approx')
            ax2.legend(loc='upper right')

        plt.savefig(self.filename)

    def fit(self):

        for i_epoch in range(self.n_epoch):

            prob = self.generator()
            loss = self.criterion(self.target_prob, prob)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.loss_history.append(loss.item())

            prob = torch.squeeze(prob)

            kl_div, js_div = evaluate(self.target_prob, prob)
            self.kl_history.append(kl_div)
            self.js_history.append(js_div)

            grad_vec = torch.cat([params.grad for params in self.generator.params])
            grad_norm = torch.linalg.vector_norm(grad_vec).item()
            
            self.grad_norm_history.append(grad_norm)

            print(loss.item(), grad_norm)

            if (i_epoch + 1) % 5 == 0:
                print(f'epoch: {i_epoch+1}  |  loss: {self.loss_history[-1]: 6f}  |  KL divergence: {kl_div:6f}  |  JS divergence: {js_div:6f}')

            if grad_norm < 1e-3:
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

    model = DDQCL(
        data_class=DATA_HUB['real image 1_1'],
        n_epoch=8000,
        reps=20,
        lr=0.01,
    )
    
    model.fit()