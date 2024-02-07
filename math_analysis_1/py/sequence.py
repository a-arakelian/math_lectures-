import os
import matplotlib.pyplot as plt
import numpy as np

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


def find_last_none(lst):
    # Find the index of the last None in reverse order
    last_none_index = next((i for i, val in enumerate(reversed(lst)) if val is None), None)
    
    if last_none_index is not None:
        # If a None is found, slice the list accordingly
        result = lst[-last_none_index:]
    else:
        # If no None is found, return the original list
        result = lst
    return result


def my_function(n):
    return 50000 * np.power((1 + 15 / 100), 1/n) ** int(n * np.pi)


def n2one(n):
    return np.arctan(n)


def make_plot(
        x_range=(0, 10),
        function=my_function,
        add_line=False
):
    fig, ax = plt.subplots()
    start = 1 if x_range[0] < 1 else int(x_range[0])
    end = 6 if x_range[1] < 5 else int(x_range[1]) + 1
    int_x = [n for n in range(start, end, 1)]
    int_y = [function(n) for n in range(start, end, 1)]
    plt.plot(int_x, int_y, 'o', color='grey', alpha=0.75, markersize=3)
    # for x, y in zip(int_x, int_y):
    if add_line:
        x = int_x[add_line-1]
        y = int_y[add_line-1]
        plt.vlines(x, 76000, y, alpha=0.5, color='orange', linewidth=1)
        plt.hlines(y, 0, x, alpha=0.5, color='green', linewidth=1)
        plt.plot([x], [y], 'o', color='grey', alpha=0.75, markersize=5)
    # plt.legend()
    plt.title('Հաջորդականություն', fontsize=16)
    ax.set_xlabel('N', fontsize=16)
    ax.set_ylabel('R', fontsize=16)
    plt.tight_layout()
    name = 'd'
    if add_line:
        name = f'line{add_line}_' + name
    name += '.png'
    directory = 'home/srohund/Documents/1991/math_analysis_1/images'
    file_name = __file__.split('/')[-1].split('.')[0]
    file_name += '_' + name
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, file_name), dpi=300, transparent=True)


def make_poly_plot(
        x_range=(0, 10),
        function_list=[my_function],
        label_list=['a'],
        color_pallet=[seahorse_colors['green']],
        alpha_list=[0.75],
        delta=False,
        naming='',
        title='Հաջորդականություններ',
        style=None,
        v_line=False,
        inf=False
):
    fig, ax = plt.subplots()
    start = 1 if x_range[0] < 1 else int(x_range[0])
    end = 6 if x_range[1] < 5 else int(x_range[1]) + 1
    int_x = [n for n in range(start, end, 1)]
    list_int_y = []
    if style is None:
        style = len(function_list) * [False]
    for function, color, alpha, label, st in zip(function_list, color_pallet, alpha_list, label_list, style):
        list_int_y.append(find_last_none([function(n) for n in range(start, end, 1)]))
        if alpha <= 0.05:
            label = None
        if not st:
            y = list_int_y[-1]
            x = int_x[-len(y):]
            plt.plot(
                x, y, 'o', color=color, alpha=alpha, markersize=3, label=label,
            )
        else:
            red_y = [(n, y) for n, y in enumerate(list_int_y[-1]) if y < delta and y > -delta]
            green_y = [(n, y) for n, y in enumerate(list_int_y[-1]) if y > delta or y < -delta]
            plt.plot(
                [t[0] for t in green_y],
                [t[1] for t in green_y],
                'o',
                color=color,
                alpha=alpha,
                markersize=3,
                label=label
            )
            plt.plot(
                [t[0] for t in red_y],
                [t[1] for t in red_y],
                'o',
                color='red',
                alpha=alpha,
                markersize=3,
                # label=label
            )
    plt.title(title, fontsize=16)
    ax.set_xlabel('N', fontsize=16)
    ax.set_ylabel('R', fontsize=16)
    if delta:
        ax.axhspan(delta, -delta, facecolor='gray', alpha=0.3, label='դելտա միջակայք')
        ax.set_yticks([delta, -delta])
        ax.set_yticklabels([r'$\delta$', r'$-\delta$'])
    ax.axhline(0, color='black', linestyle='-', label=r'$y=0$', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    name = 'poly_plot_' + naming
    name2 = name ##.copy()
    name += '.png'
    directory = 'home/srohund/Documents/1991/math_analysis_1/images'
    file_name = __file__.split('/')[-1].split('.')[0]
    file_name2 = __file__.split('/')[-1].split('.')[0]
    file_name += '_' + name
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, file_name), dpi=300, transparent=True)
    if v_line:
        min_len = min([len(y) for y in list_int_y])
        print(min_len)
        vertical_line_x = int_x[-min_len]
        plt.axvline(x=vertical_line_x,)      # linestyle='--', color='red', label='Vertical Line'

        # Shade the left side of the line
        #plt.fill_betweenx(list_int_y[1], int_x, where=(np.array(int_x) <= vertical_line_x), color='lightgray', alpha=0.5)
        name = '.png'
        name2 += 'v_l.png'
        file_name2 += '_' + name2
        plt.savefig(os.path.join(directory, file_name2), dpi=300, transparent=True)



def a(n):
    b = (np.power(2*n, 1/n)+2)*(np.sin(np.log(n+n**2))/n*2+1)+0.5
    if abs(b) > 10:
        b = np.log(b)
    return -b/3


def b(n):
    n += 5
    a = np.power(-1, n) * (np.power(15 * n, 1 / (5*n)) - 1) + 1
    if a > 1:
        a = np.power(a, 5)
    else:
        a -= 10/n
    if abs(a) > 10:
        a = np.log(abs(a))
    return a


def apb(n):
    return a(n)+b(n)


def amb(n):
    return a(n)*b(n)


def afb(n):
    if abs(b(n)) < 11/20:
        return None
    return a(n)/b(n)


make_poly_plot(
    (0, 75),
    [a, b, apb, amb, afb],
    [r'$(a_n)$', r'$(b_n)$', r'$(a_n+b_n)$', r'$(a_nb_n)$', r'$\left(\frac{a_n}{b_n}\right)$'],
    [
        seahorse_colors['green'],
        seahorse_colors['orange'],
        seahorse_colors['blue'],
        seahorse_colors['pink'],
        seahorse_colors['gray']
    ],
    [0.75, 0.75, 0, 0, 0],
    False,
    '',
    'Հաջորդականություններ'
)


make_poly_plot(
    (0, 75),
    [a, b, apb, amb, afb],
    [r'$(a_n)$', r'$(b_n)$', r'$(a_n+b_n)$', r'$(a_nb_n)$', r'$\left(\frac{a_n}{b_n}\right)$'],
    [
        seahorse_colors['green'],
        seahorse_colors['orange'],
        seahorse_colors['blue'],
        seahorse_colors['pink'],
        seahorse_colors['gray']
    ],
    [0.35, 0.35, 0.75, 0, 0],
    False,
    'sum',
    'Հաջորդականությունների գումարը'
)

make_poly_plot(
    (0, 75),
    [a, b, apb, amb, afb],
    [r'$(a_n)$', r'$(b_n)$', r'$(a_n+b_n)$', r'$(a_nb_n)$', r'$\left(\frac{a_n}{b_n}\right)$'],
    [
        seahorse_colors['green'],
        seahorse_colors['orange'],
        seahorse_colors['blue'],
        seahorse_colors['pink'],
        seahorse_colors['gray']
    ],
    [0.35, 0.35, 0, 0.75, 0],
    False,
    'mul',
    'Հաջորդականությունների արտադրյալ'
)


make_poly_plot(
    (0, 75),
    [a, b, apb, amb, afb],
    [r'$(a_n)$', r'$(b_n)$', r'$(a_n+b_n)$', r'$(a_nb_n)$', r'$\left(\frac{a_n}{b_n}\right)$'],
    [
        seahorse_colors['green'],
        seahorse_colors['orange'],
        seahorse_colors['blue'],
        seahorse_colors['pink'],
        seahorse_colors['gray']
    ],
    [0.35, 0.75, 0, 0, 0],
    11/20,
    'frac',
    'Հաջորդականությունների քանորդ',
    [False, True, False, False, False],
    False
)

make_poly_plot(
    (0, 75),
    [a, b, apb, amb, afb],
    [r'$(a_n)$', r'$(b_n)$', r'$(a_n+b_n)$', r'$(a_nb_n)$', r'$\left(\frac{a_n}{b_n}\right)$'],
    [
        seahorse_colors['green'],
        seahorse_colors['orange'],
        seahorse_colors['blue'],
        seahorse_colors['pink'],
        seahorse_colors['gray']
    ],
    [0.35, 0.35, 0, 0, 0.75],
    11/20,
    'frac_2',
    'Հաջորդականությունների քանորդ',
    [False, True, False, False, False],
    True
)

make_poly_plot(
    (0, 75),
    [a, b, apb, amb, afb],
    [r'$(a_n)$', r'$(b_n)$', r'$(a_n+b_n)$', r'$(a_nb_n)$', r'$\left(\frac{a_n}{b_n}\right)$'],
    [
        seahorse_colors['green'],
        seahorse_colors['orange'],
        seahorse_colors['blue'],
        seahorse_colors['pink'],
        seahorse_colors['gray']
    ],
    [0.35, 0.35, 0, 0.75, 0],
    False,
    'mul',
    'Հաջորդականությունների արտադրյալ'
)


def my_seq(n):
    c = 0
    x = np.sin(n) * ((n**3/(2 - n**2) + n))
    if n == 10:
        n = 5 
        c = 0.5
    if n <= 10 and n > 2:
        a = np.sin(1) * ((1**3/(2 - 1**2) + 1))
        b = 0.2
        x = (-1)**n * a * (1 - n/10) + b * (n/10)
        if n == 8:
            x += 0.15
    return x - c


def boundness_fundamental_sequence(
        x_range=(0, 10),
        function=my_function,
        N_epsilon = 10,
        epsilon = 1
):
    fig, ax = plt.subplots()
    start = 1 if x_range[0] < 1 else int(x_range[0])
    end = 6 if x_range[1] < 5 else int(x_range[1]) + 1
    int_x = [n for n in range(start, end, 1)]
    N_e_x_b = [n for n in int_x if n <= N_epsilon]
    N_e_x_a = [n for n in int_x if n > N_epsilon]
    int_y = [function(n) for n in int_x]
    int_y_b = [function(n) for n in N_e_x_b]
    int_y_a = [function(n) for n in N_e_x_a]
    plt.plot(int_x, int_y, 'o', color='grey', alpha=0.3, markersize=3)
    plt.plot(N_e_x_a, int_y_a, 'o', color='green', alpha=0.3, markersize=3)
    plt.plot(N_e_x_b, int_y_b, 'o', color='red', alpha=0.3, markersize=3)
    ax.annotate(r'$a_{N_\varepsilon + 1}$', xy=(N_epsilon + 1 + 0.25, function(N_epsilon + 1) + 0.05), xytext=(N_epsilon + 7.25, function(N_epsilon + 1) + 0.75),
            arrowprops=dict(facecolor='black', shrink=0.05))
    ax.set_xticks([N_epsilon])
    ax.set_xticklabels([r'$N_\varepsilon$'])

    vertical_line_x = N_epsilon
    horizontal_line = function(N_epsilon + 1)
    plt.axhline(y=horizontal_line, color='grey')
    plt.axvline(x=vertical_line_x)
    plt.plot([N_epsilon + 1], [function(N_epsilon + 1)], 'o', color='green', alpha=0.5, markersize=6)
    
    ax.axhspan(
        function(N_epsilon + 1) + epsilon,
        function(N_epsilon + 1) - epsilon,
        facecolor='gray',
        alpha=0.3,
        label='էպսիլոն միջակայք'
    )
    plt.title('Ֆունդամենտալ հաջորդականություն', fontsize=16)
    ax.set_yticks([function(N_epsilon + 1) + epsilon, function(N_epsilon + 1) - epsilon] + int_y_b)
    epsilon_up = r'$a_{N_\varepsilon + 1} + \varepsilon$'
    epsilon_down = r'$a_{N_\varepsilon + 1} - \varepsilon$'
    ax.set_yticklabels([epsilon_up, epsilon_down] + [r'$a_{'+f'{i+1}'+r'}$' for i in range(len(int_y_b))])
    ax.set_xlabel('N', fontsize=16)
    ax.set_ylabel('R', fontsize=16)
    plt.legend()
    plt.tight_layout()
    
    name = 'boundness_fundamental_sequence_4'
   
    name += '.png'
    directory = 'home/srohund/Documents/1991/math_analysis_1/images'
    file_name = __file__.split('/')[-1].split('.')[0]
    file_name += '_' + name
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, file_name), dpi=300, transparent=True)


boundness_fundamental_sequence(
    (0, 75),
    my_seq,
    epsilon = 0.3,
    N_epsilon = 11
)
