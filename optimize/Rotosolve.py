import torch
from torch.optim import Optimizer
import numpy as np
import pennylane as qml
from inspect import signature

def numeric_solver(func, n_steps, n_points, device):

    def torch_brute(func, interval: tuple):
        X = torch.linspace(interval[0], interval[1], steps=n_points).to(device)
        Y = torch.Tensor([func(x) for x in X])
        index = torch.argmin(Y)
        return X[index], Y[index]
    
    width = 2 * np.pi
    center = 0
    for _ in range(n_steps):
        interval = (center - width / 2, center + width / 2)
        x_min, y_min = torch_brute(func, interval)
        center = x_min
        width /= n_points

    return x_min.item(), y_min.item()

        
def consine_similarity_reconstruction(params, index, ix, qnode, target_prob, device, **kwargs):

    shift_vec = torch.zeros_like(params[index])
    shift_vec[ix] = 1
    probs = []

    shifts = torch.linspace(-np.pi / 2, np.pi / 2, 5).to(device)
    freqencies = torch.Tensor([1, 2]).to(device)

    for x in shifts:
        prob = qnode(*params[:index], params[index] + x * shift_vec, *params[index+1:], **kwargs)
        probs.append(prob)

    prob_squares = [torch.inner(prob, prob) for prob in probs]
    inner_prods = [torch.inner(target_prob, probs[i]) for i in [0, 2, 4]]

    # reconstruct numerator
    f_minus, f0, f_plus = inner_prods

    a3 = (f_plus + f_minus) / 2
    a2 = torch.arctan2(f_minus - a3, f0 - a3)
    a1 = torch.sqrt((f0 - a3) ** 2 + 0.25 * (f_plus - f_minus) ** 2)
    
    numerator = lambda theta: a1 * torch.cos(theta + a2) + a3

    # reconstruct denominator
    C0 = torch.ones(5, 1).to(device)
    C1 = torch.cos(torch.outer(shifts, freqencies)).to(device)
    C2 = torch.sin(torch.outer(shifts, freqencies)).to(device)
    C = torch.cat([C0, C1, C2], dim=1)
    
    W = torch.linalg.inv(C) @ torch.Tensor(prob_squares).to(device)
    a0 = W[0]
    a = W[1:3]
    b = W[3:5]

    denominator = lambda theta: a0 + torch.dot(a, torch.cos(freqencies * theta)) + torch.dot(b, torch.sin(freqencies * theta))

    return lambda theta: -numerator(theta) / (torch.sqrt(denominator(theta)) * torch.sqrt(torch.inner(target_prob, target_prob)))

class Rotosolve_Torch(Optimizer):

    def __init__(self,
                 params,
                 qnode,
                 criterion,
                 target_prob,
                 full_output=False,
                 **kwargs
                 ):
        defaults = dict(device=params[0].device,
                        qnode=qnode,
                        criterion=criterion,
                        target_prob=target_prob,
                        full_output=full_output,
                        kwargs=kwargs
                        )
        super().__init__(params, defaults)

    def step(self, mode='train'):
        
        for group in self.param_groups:
            
            params = group['params']
            device = self.defaults['device']
            qnode = self.defaults['qnode']
            criterion = self.defaults['criterion']
            target_prob = self.defaults['target_prob']
            full_output = self.defaults['full_output']
            kwargs = self.defaults['kwargs']

            # Get the argument name of the qfunc of the input qnode
            qfunc = qnode.func
            params_name = list(signature(qfunc).parameters.keys())
            requires_grad = {
                param_name: param.requires_grad for param_name, param in zip(params_name, params)
            }

            if full_output:
                loss_history = []

            for index, (param, param_name) in enumerate(zip(params, params_name)):

                if not requires_grad[param_name]:
                    continue

                for ix, x in enumerate(param):

                    reconstructed_func = consine_similarity_reconstruction(params,
                                                                           index,
                                                                           ix,
                                                                           qnode,
                                                                           target_prob,
                                                                           device,
                                                                           **kwargs)
                    
                    x_min, y_min = numeric_solver(reconstructed_func,
                                                  n_steps=2, 
                                                  n_points=50,
                                                  device=device)

                    if mode == 'train':
                        params[index].data[ix].add_(x_min)
                    if full_output:
                        loss_history.append(y_min)
                    
                    # import matplotlib.pyplot as plt

                    # fig = plt.figure()
                    # ax = fig.add_subplot(1, 1, 1)

                    # reconst = []
                    # real = []
                    # original_params_x = params[index].data[ix].item()

                    # for theta in np.linspace(-np.pi, np.pi, 50):
                    #     reconst.append(reconstructed_func(theta).item())
                    #     params[index].data[ix] = theta
                    #     prob = qnode(*params, **kwargs).squeeze()
                    #     real.append(criterion(target_prob, prob).item())

                    # params[index].data[ix] = original_params_x

                    # ax.clear()
                    # ax.plot(np.linspace(-np.pi, np.pi, 50), reconst, ls=':', label='reconst')
                    # # ax.plot(np.linspace(-np.pi, np.pi, 50), real, label='real')
                    # ax.plot([x_min], [y_min], marker='X', label='optimal')
                    # ax.legend()
                    # plt.pause(0.01)
            
            if full_output:
                return loss_history