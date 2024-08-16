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

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, batch: torch.Tensor):
        return self.layers(batch)


class QGAN:

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
        
        if data_class.dist_property == 'sparse':
            self.generator = Generator1(self.n_qubit, k=reps).to(self.device)
        
        else:
            self.generator = Generator2(self.n_qubit, k=reps).to(self.device)
            
        self.discriminator = Discriminator().to(self.device)
        self.g_optim = optim.Adam(params=self.generator.parameters(), lr=lr, amsgrad=True)
        self.d_optim = optim.Adam(params=self.discriminator.parameters(), lr=lr, amsgrad=True)

        self.loss_history = {
            'g loss': [],
            'd loss': []
        }
        self.kl_history = []
        self.js_history = []

        self.filename = f'./images/QGAN/QGAN(data={data_class.name}, lr={lr}, reps={reps}).png'
        self.result_file = f'./results/QGAN/QGAN(data={data_class.name}, lr={lr}, reps={reps}).json'
        
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
        
        ax1_.plot(np.arange(n_epoch)+1, self.loss_history['g loss'], label='G loss', color='turquoise')
        ax1_.plot(np.arange(n_epoch)+1, self.loss_history['d loss'], label='D loss', color='yellowgreen')
        ax1_.set_ylabel('G / D loss')
        
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
            ax2.legend()

        plt.savefig(self.filename)


    def fit(self):

        g_losses, d_losses = [], []
        best_epoch = 0
        best_kl = np.inf
        
        for i_epoch in range(self.n_epoch):
            
            d_loss = self.train_discriminator()
            d_losses.append(d_loss)
            g_loss = self.train_generator()
            g_losses.append(g_loss)
            
            prob = self.generator()
            kl_div, js_div = evaluate(self.target_prob, prob)
            self.kl_history.append(kl_div)
            self.js_history.append(js_div)
            
            if kl_div < best_kl:
                best_epoch = i_epoch
                best_kl = kl_div
                best_pmf = prob
            
            g_grad_vec = torch.cat([params.grad for params in self.generator.params])
            g_grad_norm = torch.linalg.vector_norm(g_grad_vec).item()
            d_grad_vec = torch.cat([params.grad.flatten() for params in self.discriminator.parameters()])
            d_grad_norm = torch.linalg.vector_norm(d_grad_vec).item()
            
            print(kl_div, "  ", g_grad_norm, "  ", d_grad_norm)    
            
            if np.sqrt(g_grad_norm ** 2 + d_grad_norm ** 2) < 0.001:
                break

            if (i_epoch + 1) % 5 == 0:
                print(f'epoch: {i_epoch+1} | G_loss: {g_losses[-1]:6f} | D_loss: {d_losses[-1]:6f} | KL divergence: {kl_div:6f} | JS divergence: {js_div:6f}')

        self.loss_history['g loss'] = g_losses[:best_epoch+1]
        self.loss_history['d loss'] = d_losses[:best_epoch+1]
        self.kl_history = self.kl_history[:best_epoch+1]
        self.js_history = self.js_history[:best_epoch+1]
        
        self.plot_training_result(best_pmf)
        
        with open(self.result_file, 'w') as f:
            data_dict = {
                'pmf': best_pmf.detach().cpu().tolist(),
                'kl div': self.kl_history,
                'js div': self.js_history,
                'loss history': self.loss_history
            }
            json.dump(data_dict, f)


    def train_generator(self):

        gen_prob = self.generator()
        sample = torch.arange(2**self.n_qubit).view(-1, 1).float().to(self.device)
        disc_prob = self.discriminator(sample).double().squeeze()
        
        self.g_optim.zero_grad()
        g_loss = -torch.inner(gen_prob[gen_prob>0], torch.log(disc_prob[gen_prob>0]+epsilon))
        g_loss.backward()
        self.g_optim.step()
        return g_loss.item()

    def train_discriminator(self):

        gen_prob = self.generator()
        sample = torch.arange(2**self.n_qubit).view(-1, 1).float().to(self.device)
        disc_prob = self.discriminator(sample).double().squeeze()

        self.d_optim.zero_grad()
        real_loss = -torch.inner(self.target_prob[gen_prob>0], torch.log(disc_prob[gen_prob>0]+epsilon))
        fake_loss = -torch.inner(gen_prob, torch.log(1 - disc_prob+epsilon))
        d_loss =  (real_loss + fake_loss) / 2
        d_loss.backward()
        self.d_optim.step()
        return d_loss.item()
    
if __name__ == '__main__':

    model = QGAN(
        data_class=DATA_HUB['bimodal 10'],
        n_epoch=1000,
        reps=5,
        lr=0.002,
    )
    
    model.fit()