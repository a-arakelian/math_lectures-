import os
import matplotlib.pyplot as plt
import numpy as np


def equivalent_sequences(a, b, x_range=range(1, 71), mode=0):
    # if a is not callable or b is not callable:
    #     raise TypeError('a and b should be callable')
    N = [int(n) for n in x_range]
    N_even = [int(n) for n in x_range if int(n)%2 == 0]
    N_odd = [int(n) for n in x_range if int(n)%2 == 1]
    a_n = [a(n) for n in N]
    a_n_even = [a(n) for n in N_even]
    a_n_odd = [a(n) for n in N_odd]
    b_n = [b(n) for n in N]
    b_n_even = [b(n) for n in N_even]
    b_n_odd = [b(n) for n in N_odd]
    c_n = []
    for n in N:
        c = a(n) if n%2 == 0 else b(n)
        c_n.append(c)

    fig, ax = plt.subplots()
    ax.set_xlabel('N', fontsize=16)
    ax.set_ylabel('R', fontsize=16)
    if mode == 0:
        plt.plot(N, a_n, 'o', color='#008047', alpha=0.6, label=r'$(a_n)$', markersize=3)
        plt.plot(N, b_n, 'o', color='#FFC900', alpha=0.6, label=r'$(b_n)$', markersize=3)
    elif mode == 1:
        plt.plot(N_even, a_n_even, 'o', color='#008047', alpha=0.65, label=r'$(a_n)$', markersize=3)
        plt.plot(N_odd, a_n_odd, 'o', color='#008047', alpha=0.3, markersize=2.5)
        plt.plot(N_even, b_n_even, 'o', color='#FFC900', alpha=0.3, markersize=2.5)
        plt.plot(N_odd, b_n_odd, 'o', color='#FFC900', alpha=0.65, label=r'$(b_n)$', markersize=3)
    elif mode == 2:
        plt.plot(N, c_n, 'o', color='#006994', alpha=0.6, label=r'$(c_n)$', markersize=3.5)
        plt.plot(N_odd, a_n_odd, 'o', color='#008047', alpha=0.2, markersize=2)
        plt.plot(N_even, b_n_even, 'o', color='#FFC900', alpha=0.2, markersize=2)
    plt.legend()
    plt.title('Իրական թվի նախատիպը', fontsize=16)

    directory = '/home/srohund/Documents/1991/math_analysis_2/images'
    file_name = f'equivalent_sequences_{mode}.png'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, file_name), dpi=300, transparent=True)


def a(n):
    return np.power(2, 1/2) + 1/n


def b(n):
    return np.power(2, 1/2) - 1/n


# equivalent_sequences(a, b, mode=0)
# equivalent_sequences(a, b, mode=1)
# equivalent_sequences(a, b, mode=2)

def plot_steps(mode):
    fig, ax = plt.subplots()
    ax.set_xlabel('R', fontsize=16)
    ax.set_ylabel('R', fontsize=16)
    N_x = [i for i in range(-7, 8)]
    N_y = [i for i in range(-7, 8)]
    plt.step(N_x, N_y, where='post', color='tab:blue', label=r'$[]$')
    plt.plot(N_x, N_y, 'o', color='tab:blue', alpha=0.6)
    if mode is not None:
        mode = round(mode, 2)
        if mode < 0:
            sign = -1
            shift = 2.15
            y = int(mode - 1)
        else:
            sign = 1
            shift = 3.15
            y = int(mode)
        plt.axhline(y=y, color='orange', alpha=0.3)
        plt.axvline(x=mode, color='orange', alpha=0.3)
        ax.annotate(r'$x='+f'{round(mode, 2)}'r'$', xy=(mode - sign * 0.15, -mode + sign * 0.15), xytext=(mode - sign * shift, -mode + sign * 1.15),
                arrowprops=dict(facecolor='black', shrink=0.05))
        ax.annotate(r'$y='+f'{y}'r'$', xy=(-y, y - sign * 0.15), xytext=(-y - sign * 1.15, y - sign * 2.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
        plt.plot([mode], [y], 'o', color='orange', alpha=0.6)
    plt.legend()
    plt.title('Թվի ամբողջ մասը', fontsize=16)
    directory = '/home/srohund/Documents/1991/math_analysis_2/images'
    file_name = f'steps_{mode}.png'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, file_name), dpi=300, transparent=True)

plot_steps(None)


def plot_number_line(start, end, points=[]):
    fig, ax = plt.subplots()
    plt.figure(figsize=(8, 1))
    plt.plot([start, end], [0, 0], color='black', linewidth=2)
    plt.scatter(points, [0] * len(points), color='red', marker='o', label='Points')
    plt.xlabel('Number Line')
    plt.yticks([])
    plt.legend()
    directory = '/home/srohund/Documents/1991/math_analysis_2/images'
    file_name = f'number_line.png'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, file_name), dpi=300, transparent=True)

# Example usage
start_value = -5
end_value = 5
data_points = [-3, 0, 2]

plot_number_line(start_value, end_value, points=data_points)
