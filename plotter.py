import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from argparse import ArgumentParser


def plotter(data):
    with open(f'./results/QGAN-{data}.csv', 'r') as qgan, \
         open(f'./results/QCBM-{data}.csv', 'r') as qcbm, \
         open(f'./results/DDQCL-{data}.csv', 'r') as ddqcl, \
         open(f'./results/QAE-{data}.csv', 'r') as qae:
            
            qgan_reader = list(csv.reader(qgan, quoting=csv.QUOTE_NONNUMERIC))
            qcbm_reader = list(csv.reader(qcbm, quoting=csv.QUOTE_NONNUMERIC))
            ddqcl_reader = list(csv.reader(ddqcl, quoting=csv.QUOTE_NONNUMERIC))
            qae_reader = list(csv.reader(qae, quoting=csv.QUOTE_NONNUMERIC))

            qgan_kl = np.array(qgan_reader[:10])
            qgan_js = np.array(qgan_reader[10:])
            qcbm_kl = np.array(qcbm_reader[:10])
            qcbm_js = np.array(qcbm_reader[10:])
            ddqcl_kl = np.array(ddqcl_reader[:10])
            ddqcl_js = np.array(ddqcl_reader[10:])
            qae_kl = np.array(qae_reader[:10])
            qae_js = np.array(qae_reader[10:])

            qgan_kl_mean, qgan_kl_std = np.mean(qgan_kl, axis=0), np.std(qgan_kl, axis=0)
            qcbm_kl_mean, qcbm_kl_std = np.mean(qcbm_kl, axis=0), np.std(qcbm_kl, axis=0)
            ddqcl_kl_mean, ddqcl_kl_std = np.mean(ddqcl_kl, axis=0), np.std(ddqcl_kl, axis=0)
            qae_kl_mean, qae_kl_std = np.mean(qae_kl, axis=0), np.std(qae_kl, axis=0)

            epochs = len(qgan_kl_mean)

            qgan_js_mean, qgan_js_std = np.mean(qgan_js, axis=0), np.std(qgan_js, axis=0)
            qcbm_js_mean, qcbm_js_std = np.mean(qcbm_js, axis=0), np.std(qcbm_js, axis=0)
            ddqcl_js_mean, ddqcl_js_std = np.mean(ddqcl_js, axis=0), np.std(ddqcl_js, axis=0)
            qae_js_mean, qae_js_std = np.mean(qae_js, axis=0), np.std(qae_js, axis=0)

            figure(figsize=(8, 6))
            plt.plot([i+1 for i in range(epochs)], qgan_kl_mean, '^', color='darkviolet', label='QGAN', markerfacecolor='none')
            plt.fill_between([i+1 for i in range(epochs)], qgan_kl_mean-qgan_kl_std, qgan_kl_mean+qgan_kl_std, alpha=0.25, color='darkviolet')
            plt.plot([i+1 for i in range(epochs)], qcbm_kl_mean, 's', color='blue', label='QCBM', markerfacecolor='none')
            plt.fill_between([i+1 for i in range(epochs)], qcbm_kl_mean-qcbm_kl_std, qcbm_kl_mean+qcbm_kl_std, alpha=0.25, color='blue')
            plt.plot([i+1 for i in range(epochs)], ddqcl_kl_mean, '+', color='olivedrab', label='DDQCL')
            plt.fill_between([i+1 for i in range(epochs)], ddqcl_kl_mean-ddqcl_kl_std, ddqcl_kl_mean+ddqcl_kl_std, alpha=0.25, color='olivedrab')
            plt.plot([i+1 for i in range(epochs)], qae_kl_mean, 'x', color='red', label='QAE')
            plt.fill_between([i+1 for i in range(epochs)], qae_kl_mean-qae_kl_std, qae_kl_mean+qae_kl_std, alpha=0.25, color='red')
            plt.ylim(bottom=0.0)
            plt.title(f'KL divergence of dataset {data}')
            plt.xlabel('epoch')
            plt.ylabel('KL divergence')
            plt.legend()
            plt.savefig(f'./images/{data}-KL.png')

            figure(figsize=(8, 6))
            plt.plot([i+1 for i in range(epochs)], qgan_js_mean, '^', color='darkviolet', label='QGAN', markerfacecolor='none')
            plt.fill_between([i+1 for i in range(epochs)], qgan_js_mean-qgan_js_std, qgan_js_mean+qgan_js_std, alpha=0.25, color='darkviolet')
            plt.plot([i+1 for i in range(epochs)], qcbm_js_mean, 's', color='blue', label='QCBM', markerfacecolor='none')
            plt.fill_between([i+1 for i in range(epochs)], qcbm_js_mean-qcbm_js_std, qcbm_js_mean+qcbm_js_std, alpha=0.25, color='blue')
            plt.plot([i+1 for i in range(epochs)], ddqcl_js_mean, '+', color='olivedrab', label='DDQCL')
            plt.fill_between([i+1 for i in range(epochs)], ddqcl_js_mean-ddqcl_js_std, ddqcl_js_mean+ddqcl_js_std, alpha=0.25, color='olivedrab')
            plt.plot([i+1 for i in range(epochs)], qae_js_mean, 'x', color='red', label='QAE')
            plt.fill_between([i+1 for i in range(epochs)], qae_js_mean-qae_js_std, qae_js_mean+qae_js_std, alpha=0.25, color='red')
            plt.ylim(bottom=0.0)
            plt.title(f'JS divergence of dataset {data}')
            plt.xlabel('epoch')
            plt.ylabel('JS divergence')
            plt.legend()
            plt.savefig(f'./images/{data}-JS.png')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', type=str)
    args = parser.parse_args()
    plotter(args.d)
                    