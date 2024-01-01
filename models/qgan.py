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


class Generator(nn.Module):

    def __init__(self, n_qubit: int, k: int):
        super().__init__()
        self.params = nn.ParameterList(
            [nn.Parameter((torch.rand(n_qubit * 3) * 2 - 1) * np.pi, requires_grad=True) for _ in range(k+1)]
        )
        dev = qml.device('default.qubit.torch', wires=n_qubit)
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
                 sample_size=1000000 
                 ):
        self.data_class = data_class
        self.n_qubit = data_class.n_bit
        self.n_epoch = n_epoch
        self.reps = reps
        self.lr = lr
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
        self.target_prob = torch.Tensor(data_class.get_data(num=sample_size)).double().to(self.device)
        
        self.generator = Generator(self.n_qubit, k=reps).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.g_optim = optim.Adam(params=self.generator.parameters(), lr=lr, amsgrad=True)
        self.d_optim = optim.Adam(params=self.discriminator.parameters(), lr=lr, amsgrad=True)

        self.loss_history = {
            'g loss': [],
            'd loss': []
        }
        self.kl_history = []
        self.js_history = []

        self.filename = f'./images/QGAN(data={data_class.name}, lr={lr}, reps={reps}).png'
        self.tensor_file = f'./results/QGAN(data={data_class.name}, lr={lr}, reps={reps}).pt'
        self.result_file = f'./results/QGAN(data={data_class.name}, lr={lr}, reps={reps}).pkl'


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

        g_losses, d_losses = [], []
        for i_epoch in range(self.n_epoch):
            
            g_loss = self.train_generator()
            g_losses.append(g_loss)
            d_loss = self.train_discriminator()
            d_losses.append(d_loss)
            
            prob = self.generator()
            kl_div, js_div = evaluate(self.target_prob.detach().cpu().numpy(), prob.detach().cpu().numpy())
            self.kl_history.append(kl_div)
            self.js_history.append(js_div)

            if (i_epoch + 1) % 5 == 0:
                print(f'epoch: {i_epoch+1} | G_loss: {g_losses[-1]:6f} | D_loss: {d_losses[-1]:6f} | KL divergence: {kl_div:6f} | JS divergence: {js_div:6f}')

        ax1.clear()
        ax1.plot(np.arange(len(self.kl_history))+1, self.kl_history, label='KL divergence', color='red', marker='^', markerfacecolor=None)
        ax1.plot(np.arange(len(self.js_history))+1, self.js_history, label='JS divergence', color='blue', marker='x', markerfacecolor=None)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('KL / JS divergence')
        ax1.grid()
        ax1.legend()

        ax2.clear()
        ax2.plot(np.arange(len(g_losses))+1, g_losses, color='royalblue', marker='P', label='G loss')
        ax2.plot(np.arange(len(d_losses))+1, d_losses, color='magenta', marker='P', label='D loss')
        ax2.set_xlabel('iteration')
        ax2.set_ylabel('loss')
        ax2.legend()
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

        self.loss_history['g loss'] = g_losses
        self.loss_history['d loss'] = d_losses
        torch.save(prob.detach().cpu(), self.tensor_file)
        with open(self.result_file, 'wb') as f:
            pickle.dump((self.loss_history, self.js_history, self.kl_history), f)


    def train_generator(self):

        fake_prob = self.generator()
        fake_data = torch.arange(2**self.n_qubit).view(-1, 1).float().to(self.device)
        fake_score = self.discriminator(fake_data).double().squeeze()
        g_loss = -torch.inner(fake_prob, torch.log(fake_score+epsilon))
        self.g_optim.zero_grad()
        g_loss.backward()
        self.g_optim.step()
        return g_loss.item()

    def train_discriminator(self):

        fake_prob = self.generator()
        fake_data = torch.arange(2**self.n_qubit).view(-1, 1).float().to(self.device)
        fake_score = self.discriminator(fake_data).double().squeeze()

        real_loss = -torch.inner(self.target_prob, torch.log(fake_score+epsilon))
        fake_loss = -torch.inner(fake_prob, torch.log(1 - fake_score+epsilon))
        d_loss =  (real_loss + fake_loss) / 2
        self.d_optim.zero_grad()
        d_loss.backward()
        self.d_optim.step()
        return d_loss.item()
    
if __name__ == '__main__':

    model = QGAN(
        data_class=DATA_HUB['bas 4x4'],
        n_epoch=2000,
        reps=20,
        lr=1e-3,
        sample_size=100000000
    )
    
    model.fit()