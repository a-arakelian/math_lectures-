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


def remove_from_plot(*args):
    for a in args:
        a.remove()

def configure_axis(ax, x_range, y_range, xlabel='$x$', ylabel='$f(x)$'):
    # Move the left and bottom spines to x = 0 and y = 0, respectively.
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.text(-2, 0.95, ylabel, transform=ax.get_xaxis_transform(), fontsize=16)
    ax.text(0.95, -0.25, xlabel, transform=ax.get_yaxis_transform(), fontsize=16)
    ax.set_yticks([])
    ax.set_xticks([])


def set_epsilon_range(ax, epsilon, sequence_limit):
    up_limit = sequence_limit + epsilon
    down_limit = sequence_limit - epsilon
    ul = ax.axhline(up_limit, color='black', linewidth=0.5, label='$y = 1$')
    ll = ax.axhline(sequence_limit, color='green', linewidth=0.25)
    dl = ax.axhline(down_limit, color='black', linewidth=0.5)
    sp = ax.axhspan(up_limit, down_limit, facecolor='gray', alpha=0.3, label='Էպսիլոն միջակայք')
    return ul, dl, sp, ll

def set_delta_range(ax, delta, x_point):
    up_limit = x_point + delta
    down_limit = x_point - delta
    ul = ax.axvline(up_limit, color='black', linewidth=0.5)
    dl = ax.axvline(down_limit, color='black', linewidth=0.5)
    sp = ax.axvspan(up_limit, down_limit, facecolor='gold', alpha=0.3, label='դելտա միջակայք')
    return ul, dl, sp

def set_image(ax, delta):
    up_limit = 1
    down_limit = np.sin(delta)/delta
    sp = ax.axhspan(up_limit, down_limit, facecolor='green', alpha=0.3, label='դելտա միջակայքի պատկեր')
    return sp


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

def sinx_x(epsilon_list = [0.7, 0.35, 0.2]):
    fig, ax = plt.subplots(figsize=(16, 4))
    configure_axis(ax, (-4*np.pi, 4*np.pi), (-0.5, 1.75))
    x = np.linspace(-4*np.pi, 4*np.pi, 10000)
    y = np.sin(x) / x
    ax.plot(x, y, label=r'$f(x) = \frac{\sin(x)}{x}$')
    ax.legend(fontsize=14)
    save_plot("sinx_x_0")
    #\epsilon
    for n, epsilon in enumerate(epsilon_list, 1):
        eps_ul, eps_dl, eps_sp, ll = set_epsilon_range(ax, epsilon, 1)
        ax.legend(fontsize=14)
        save_plot(f"sinx_x_e_{n}")
        delta = np.min((epsilon, np.pi/2))
        del_ul, del_dl, del_sp = set_delta_range(ax, delta, 0)
        ax.legend(fontsize=14)
        save_plot(f"sinx_x_d_{n}")
        ima_sp = set_image(ax, delta)
        leg = ax.legend(fontsize=14)
        save_plot(f"sinx_x_i_{n}")
        remove_from_plot(eps_ul, eps_dl, eps_sp, del_ul, del_dl, del_sp, ima_sp, leg, ll)

def discontinuous_function(delta_list = [0.7, 0.35, 0.2]):
    fig, ax = plt.subplots(figsize=(8, 6))
    configure_axis(ax, (-8, 8), (-6, 6))
    x = np.linspace(-8, 8, 1000000)
    y = np.sign(x) + x
    ax.plot(x, y, label=r'$f(x) = \mathrm{sign}(x) + x$')
    ax.legend(fontsize=14)
    save_plot("discontinuous_function_0")
    for n, delta in enumerate(delta_list, 1):
        eps_ul, eps_dl, eps_sp, ll = set_epsilon_range(ax, 0.5, 0)
        if n == 1:
            ax.legend(fontsize=14)
            save_plot("discontinuous_function_1")
        del_ul, del_dl, del_sp = set_delta_range(ax, delta, 0)
        ax.legend(fontsize=14)
        save_plot(f"discontinuous_function_d_{n}")
        ima_sp_1 = ax.axhspan(-1, -1-delta, facecolor='red', alpha=0.3, label='դելտա միջակայքի պատկեր')
        ima_sp_2 = ax.axhspan(1, 1+delta, facecolor='red', alpha=0.3)
        leg = ax.legend(fontsize=14)
        save_plot(f"discontinuous_function_i_{n}")
        remove_from_plot(eps_ul, eps_dl, eps_sp, del_ul, del_dl, del_sp, ima_sp_1, ima_sp_2, leg, ll)


        

function_prototype()
sinx_x()
discontinuous_function()
plt.show()
