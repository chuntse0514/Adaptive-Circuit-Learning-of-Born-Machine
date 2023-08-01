import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import csv
from data import DATA_HUB
from models import MODEL_HUB
from config import CONFIG
import torch

def trainer():

    print('------------start training the dataset {}------------'.format(CONFIG['data']))

    data = DATA_HUB[CONFIG['data']]()
    data_points = data.get_data(num=CONFIG['N'])
    if CONFIG['show_data_hist']:
        plt.hist(data_points, bins= 2 ** data.n_bit, range=(0, 2 ** data.n_bit))
        plt.show()


    iterator = range(CONFIG['repeat'])
    if CONFIG['repeat'] > 1:
        iterator = tqdm(iterator)

    kl_results = []
    js_results = []

    for _ in iterator:
        seed = np.random.randint(0, 10000)

        model = MODEL_HUB[CONFIG['model']](
            n_qubit=data.n_bit,
            n_epoch=CONFIG['n_epoch'],
            circuit_depth=CONFIG['circuit_depth'],
            lr=CONFIG['lr']
        )

        kl_result, js_result = model.fit(torch.Tensor(data_points).double().to(model.device))

        kl_results.append(kl_result)
        js_results.append(js_result)

    kl_results = np.array(kl_results)
    js_results = np.array(js_results)

    prob = model.generator.forward().detach().cpu().numpy()
    plt.hist(prob, bins=2 **data.n_bit)
    plt.savefig('dist.png')

    print('results of the data set {}:'.format(CONFIG['data']))
    print('KL_divergence: {} ± {}'.format(np.mean(kl_results[:, -1]), np.std(kl_results[:, -1])))
    print('JS_divergnece: {} ± {}'.format(np.mean(js_results[:, -1]), np.std(js_results[:, -1])))

    data_plotter(kl_results, js_results)
    save_result(kl_results, js_results)

def data_plotter(kl_results, js_results):

    kl_mean = np.mean(kl_results, axis=0)
    kl_std = np.std(kl_results, axis=0)
    js_mean = np.mean(js_results, axis=0)
    js_std = np.std(js_results, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    ax1.plot([i+1 for i in range(len(kl_mean))], kl_mean, linewidth=0.8, color='g', label='KL_divergence')
    ax1.plot([i+1 for i in range(len(kl_mean))], kl_mean-kl_std, linewidth=0.8, color='g', alpha=0.25)
    ax1.plot([i+1 for i in range(len(kl_mean))], kl_mean+kl_std, linewidth=0.8, color='g', alpha=0.25)
    ax1.fill_between([i+1 for i in range(len(kl_mean))], kl_mean-kl_std, kl_mean+kl_std, color='g', alpha=0.1)
    ax1.set_ylim(bottom=0.0)
    ax1.set_xlabel('episode')
    ax1.set_ylabel('KL divergence')
    ax1.legend(loc="best")
    ax1.grid()
    ax1.set_title('KL divergence of {} data'.format(CONFIG['data']))

    ax2.plot([i+1 for i in range(len(js_mean))], js_mean, linewidth=0.8, color='r', label='JS_divergence')
    ax2.plot([i+1 for i in range(len(js_mean))], js_mean-js_std, linewidth=0.8, color='r', alpha=0.25)
    ax2.plot([i+1 for i in range(len(js_mean))], js_mean+js_std, linewidth=0.8, color='r', alpha=0.25)
    ax2.fill_between([i+1 for i in range(len(js_mean))], js_mean-js_std, js_mean+js_std, color='r', alpha=0.1)
    ax2.set_ylim(bottom=0.0)
    ax2.set_xlabel('episode')
    ax2.set_ylabel('JS divergence')
    ax2.legend(loc="best")
    ax2.grid()
    ax2.set_title('JS divergence of {} data'.format(CONFIG['data']))

    # plt.savefig('./images/{}-{}.png'.format(CONFIG['model'], CONFIG['data']))
    plt.savefig('out.png')

def save_result(kl_results, js_results):
    
    print('writing the results to the file {}-{}.csv...'.format(CONFIG['model'], CONFIG['data']))
    with open('./results/{}-{}.csv'.format(CONFIG['model'], CONFIG['data']), 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for line in kl_results:
            csv_writer.writerow(line)
        for line in js_results:
            csv_writer.writerow(line)
    print('successfully write the results to the csv file!')

if __name__ == '__main__':
    trainer()