import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import random


IGNOREDDIRECTORY = '5b49576b'
DIRECTORY = 'math_analysis_4/images'

# Define seahorse colors
seahorse_colors = {
    'blue': '#006994',
    'green': '#008047',
    'red': '#D9001B',
    'orange': '#FF7B00',
    'gray': '#5C5C5C',
    'pink': '#D60060',
    'yellow': '#FFC900'
}

def save_plot(name):
    dir = os.path.join(DIRECTORY, IGNOREDDIRECTORY)
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, name)
    plt.savefig(path, dpi=300, transparent=True)

def get_domain():
    d = np.array([1/i for i in range(1, 100000)]) # 'discrete'
    si1 = np.arange(1, 2, 0.0001) # 'semi interval 1'
    si2 = np.arange(3, 2, -0.0001) # 'semi interval 2'
    p = np.array([3]) # 'point'
    i = np.arange(3, 4, 0.0001) # 'interval'
    fd = np.array([random.uniform(0, 1) * np.power(k, 1.5) * np.power(-1, n) for n, k in enumerate(d)])
    fsi1 = np.array([-1 / 0.1845 * np.power(np.e, -(1/(1/1.3 * (k - 1))**2)) for k in si1])
    fsi2 = np.array([1 / 0.1845 * np.power(np.e, -(1/(1/1.3 * (3-k))**2)) for k in si2])
    fp = 1/2 * p
    fi = np.array([(k-3)**2 * np.sin(1/(4-k)) for k in i])
    return {
        'dis': [d, fd],
        'si1': [si1, fsi1],
        'si2': [si2, fsi2],
        'poi': [p, fp],
        'int': [i, fi]
    }

def function_prototype():
    domain = get_domain()
    fig, ax = plt.subplots()
    # ax.plot(*domain['dis'], *domain['si1'], *domain['si2'], *domain['poi'], *domain['int'])
    ax.plot(*domain['dis'], 'o', color=seahorse_colors['red'], markersize=1, alpha=0.7)
    ax.plot(*domain['si1'], color=seahorse_colors['red'], markersize=0.35, alpha=0.7, label =r'$f(x)$')
    ax.plot(*domain['si2'], color=seahorse_colors['red'], markersize=0.35, alpha=0.7)
    ax.plot(*domain['poi'], 'o', color=seahorse_colors['red'], markersize=3, alpha=0.7)
    ax.plot(*domain['int'], color=seahorse_colors['red'], markersize=0.35, alpha=0.7)
    ax.legend(fontsize=12)
    save_plot('function_prototype')


function_prototype()
plt.show()
