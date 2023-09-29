import torch
from torch.optim import Optimizer
import numpy as np
import pennylane as qml
from inspect import signature

def construct_univariate_func(params, index, ix, qnode, criterion, target_prob, **kwargs):
    
    variable_vec = torch.zeros_like(params[index])
    variable_vec[ix] = 1

    def shift_func(x):
        prob = qnode(*params[:index], params[index] + x * variable_vec, *params[index+1:], **kwargs)
        return criterion(target_prob, prob)
    
    return shift_func

def analytic_solver(func, f0=None):
        
    if f0 is None:
        f0 = func(0)

    f_plus = func(np.pi / 2)
    f_minus = func(-np.pi / 2)

    a3 = (f_plus + f_minus) / 2
    a2 = torch.arctan2(2 * f0 - f_plus - f_minus, f_plus - f_minus)
    a1 = torch.sqrt((f0 - a3) ** 2 + 0.25 * (f_plus - f_minus) ** 2)
    
    x_min = -np.pi / 2 - a2
    y_min = func(x_min)
    
    if x_min < -np.pi:
        x_min += 2 * np.pi

    return x_min, y_min

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

def full_reconstruction_eq(func, R, device, f0=None):

    if not f0:
        f0 = func(0)

    X_mu = torch.linspace(-R, R, steps=2*R+1).to(device) * (2*np.pi) / (2*R + 1)
    E_mu = torch.Tensor([func(x_mu) for x_mu in X_mu[:R]] + [f0] + [func(x_mu) for x_mu in X_mu[R+1:]]).to(device)
    
    def reconstruction_func(x):

        # Definition of torch.sinc(x) is sin(\pi * x) / (\pi * x)
        # However, our definition is sinc(x) = sin(x) / x
        
        kernel = torch.sinc((2*R + 1) / (2*np.pi) * (x - X_mu)) / torch.sinc(1 / (2 * np.pi) * (x - X_mu))
        return torch.inner(E_mu, kernel)
    
    return reconstruction_func
        


class Rotosolve_Torch(Optimizer):

    def __init__(self,
                 params,
                 qnode,
                 criterion,
                 target_prob,
                 num_freqs,
                 full_output=False,
                 **kwargs
                 ):
        defaults = dict(device=params[0].device,
                        qnode=qnode,
                        criterion=criterion,
                        target_prob=target_prob,
                        num_freqs=num_freqs,
                        full_output=full_output,
                        kwargs=kwargs
                        )
        super().__init__(params, defaults)

    def step(self):
        
        for group in self.param_groups:
            
            params = group['params']

            # Get the argument name of the qfunc of the input qnode
            qnode = self.defaults['qnode']
            criterion = self.defaults['criterion']
            target_prob = self.defaults['target_prob']
            num_freqs = self.defaults['num_freqs']
            full_output = self.defaults['full_output']
            kwargs = self.defaults['kwargs']

            qfunc = qnode.func
            params_name = list(signature(qfunc).parameters.keys())
            requires_grad = {
                param_name: param.requires_grad for param_name, param in zip(params_name, params)
            }

            if full_output:
                loss_history = []

            init_prob = qnode(*params, **kwargs)
            init_func_val = criterion(target_prob, init_prob)

            for index, (param, param_name) in enumerate(zip(params, params_name)):

                if not requires_grad[param_name]:
                    continue

                for ix, x in enumerate(param):

                    import matplotlib.pyplot as plt
                    value_list = []
                    origin_x = params[index].data[ix].item()
                    for param_x in torch.linspace(-np.pi, np.pi, 100):
                        params[index].data[ix] = param_x
                        prob = qnode(*params, **kwargs)
                        value = criterion(target_prob, prob).item()
                        value_list.append(value)

                    plt.plot(np.linspace(-np.pi, np.pi, 100), value_list)
                    plt.plot([origin_x], [init_func_val.item()], marker='X')
                    plt.savefig('test.png')

                    params[index].data[ix] = origin_x
                    total_freq = num_freqs[param_name][ix]
                    univariate_func = construct_univariate_func(params, index, ix, qnode, criterion, target_prob, **kwargs)

                    if total_freq == 1:
                        x_min, y_min = analytic_solver(univariate_func, f0=init_func_val)
                        params[index].data[ix].add_(x_min)
                        init_func_val = y_min
                        prob = qnode(*params, **kwargs)
                        func_val = criterion(target_prob, prob).item()
                        print(y_min.item(), func_val)
                        loss_history.append(y_min.item())

                    else:
                        reconstructed_func = full_reconstruction_eq(univariate_func, 
                                                                    total_freq, 
                                                                    device=self.device,
                                                                    f0=init_func_val)
                        
                        x_min, y_min = numeric_solver(reconstructed_func,
                                                    n_steps=2, 
                                                    n_points=50,
                                                    device=self.device)
                        self.params[index].data[ix].add_(x_min)
                        init_func_val = y_min
                        loss_history.append(y_min)
            
            if full_output:
                return loss_history
        