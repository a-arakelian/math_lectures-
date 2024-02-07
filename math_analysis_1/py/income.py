import os
import matplotlib.pyplot as plt
import numpy as np


def function(n, p=5, start=50000):
    return start * (1 + p / 100) ** int(n)


def continuous_function(x, p=5, start=50000):
    return start * np.power((1 + p / 100), x)


def make_plot(
        x_range=(0, 10),
        percent=15,
        start=50000,
        frequency=[1],
        add_continuous=False
):
    fig, ax = plt.subplots()
    int_x = list(range(int(x_range[0]), int(x_range[1]) + 1))
    int_y = [function(n, percent, start) for n in int_x]
    plt.plot(int_x, int_y, 'o', color='grey', alpha=0.3)
    label_names = {
        1: 'տարեկան տոկոսադրույք',
        2: 'կես տարեկան տոկոսադրույք',
        12: 'ամսական, տոկոսադրույք',
        24: 'կես ամսական, տոկոսադրույք',
        365: 'օրական տոկոսադրույք',
        730: 'կես օրական տոկոսադրույք',
        8766: 'ժամային տոկոսադրույք',
        31560000: 'վարկյանային տոկոսադրույք'
    }
    frequency = sorted(frequency)
    steps_list = [
        np.arange(x_range[0], x_range[1] + 1/s, 1/s) for s in frequency
    ]
    percent_list = [100*(np.power(1+percent/100, 1/s)-1) for s in frequency]
    y_list = [
        np.array([function(f*n, p, start) for n in x])
        for f, x, p in zip(frequency, steps_list, percent_list)
    ]
    for x, y, f, p in zip(steps_list, y_list, frequency, percent_list):
        p = round(p, 4)
        label = f'{p}% ' + label_names.get(f, f'1/{f}տարեկան տոկոսադրույք')
        plt.step(x, y, where='post', label=label, alpha=0.5)
    if add_continuous:
        x_values = np.linspace(x_range[0], x_range[1], 100)
        y_values = continuous_function(x_values, percent, start)
        plt.plot(
            x_values, y_values,
            label='անընդհատ աճ',
            linestyle='--', color='grey', alpha=0.3
        )
    plt.legend()
    plt.title('Գումարի աճը', fontsize=16)
    ax.set_xlabel('տարի', fontsize=16)
    ax.set_ylabel('դրամ', fontsize=16)
    plt.tight_layout()
    name = '_'.join(map(str, frequency))
    name += '_' + 'x'.join(map(lambda x: str(int(10*x)), x_range))
    if add_continuous:
        name += '_c'
    name += '.png'
    directory = 'home/srohund/Documents/1991/math_analysis_1/images' # '/home/srohund/Documents/1991/images/'
    file_name = __file__.split('/')[-1].split('.')[0]
    file_name += '_' + name
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, name), dpi=300)


make_plot(
        x_range=(0, 10),
        percent=15,
        start=50000,
        frequency=[1, 2, 12, 365, 8766],
        add_continuous=True
    )
