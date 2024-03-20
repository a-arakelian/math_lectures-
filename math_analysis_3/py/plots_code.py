import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

IGNOREDDIRECTORY = '5b49576b'
DIRECTORY = 'math_analysis_3/images'

def func(n):
    n += 1
    if isinstance(n, (int, np.integer)):
        x = (1 + 1 / n) ** n if n % 2 == 0 else (1 + 1 / n) ** (n + 1)
    else:
        x = np.where(n % 2 == 0, (1 + 1 / n) ** n, (1 + 1 / n) ** (n + 1))
    return x


def dens(n):
    return 1 + np.sin(n)


def save_plot(name):
    dir = os.path.join(DIRECTORY, IGNOREDDIRECTORY)
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, name)
    plt.savefig(path, dpi=300, transparent=True)


def set_epsilon_range(ax, epsilon, sequence_limit):
    up_limit = sequence_limit + epsilon
    down_limit = sequence_limit - epsilon
    ul = ax.axhline(up_limit, color='black', linewidth=0.5)
    dl = ax.axhline(down_limit, color='black', linewidth=0.5)
    sp = ax.axhspan(up_limit, down_limit, facecolor='gray', alpha=0.3, label='Էպսիլոն միջակայք')
    return ul, dl, sp


def color_filterd_sequence(x, y, sequence_limit, epsilon):
    les_epsilon = np.abs(y - sequence_limit) < epsilon
    geq_epsilon = np.abs(y - sequence_limit) >= epsilon
    gp = plt.plot(x[les_epsilon], y[les_epsilon], 'o', color='green', alpha=0.5, markersize=3)
    rg = plt.plot(x[geq_epsilon], y[geq_epsilon], 'o', color='red', alpha=0.5, markersize=3)
    return gp[0], rg[0]


def n_epsilon(x, y, sequence_limit, epsilon):
    geq_epsilon = np.array(np.abs(y - sequence_limit) >= epsilon)
    a = np.array([np.sum(geq_epsilon[i:]) for i in range(len(geq_epsilon))])
    index = np.argmin(a)
    return x[index]


def convergent_sequence(function=func, n_limit=140, sequence_limit=np.e, epsilon=1/10, other_epsilon = 1/30):
    x = np.arange(1, n_limit)
    try:
        y = function(x)
    except ValueError:
        y = np.array([function(n) for n in x])
    fig, ax = plt.subplots()
    ax.set_xlim(1, n_limit)
    plot = plt.plot(x, y, 'o', color='red', alpha=0.3, markersize=3, label='$(a_n)$')
    plt.legend(fontsize=12)
    save_plot('convergent_sequence_0')
    ax.set_yticks([sequence_limit])
    ax.set_yticklabels([r'$\lim_{n \rightarrow \infty}a_n$'], fontsize=14)
    ax.axhline(sequence_limit, label=r'$\lim_{n \rightarrow \infty}a_n = \alpha$')
    plt.legend(fontsize=12)
    save_plot('convergent_sequence_1')
    up_limit = sequence_limit + epsilon
    down_limit = sequence_limit - epsilon
    ul, dl, sp = set_epsilon_range(ax, epsilon, sequence_limit)
    ax.set_yticks([down_limit, sequence_limit, up_limit])
    ax.set_yticklabels([r'$\alpha - \varepsilon$', r'$\lim_{n \rightarrow \infty}a_n$', r'$\alpha + \varepsilon$'], fontsize=14)
    plt.legend(fontsize=12)
    save_plot('convergent_sequence_2')
    plot[0].set_visible(False)
    gp, rg = color_filterd_sequence(x, y, sequence_limit, epsilon)
    save_plot('convergent_sequence_3')
    n_eps = n_epsilon(x, y, sequence_limit, epsilon)
    n_ = ax.axvline(n_eps, color='black')
    ax.set_xticks([n_eps])
    ax.set_xticklabels([r'$N_{\varepsilon}$'], fontsize=14)
    save_plot('convergent_sequence_4')
    ul.set_visible(False)
    dl.set_visible(False)
    sp.set_visible(False)
    n_.set_visible(False)
    gp.set_visible(False)
    rg.set_visible(False)
    color_filterd_sequence(x, y, sequence_limit, other_epsilon)
    set_epsilon_range(ax, other_epsilon, sequence_limit)
    n_eps = n_epsilon(x, y, sequence_limit, other_epsilon)
    n_ = ax.axvline(n_eps, color='black')
    ax.set_xticks([n_eps])
    ax.set_xticklabels([r'$N_{\varepsilon}$'], fontsize=14)
    ax.set_yticks([sequence_limit - other_epsilon, sequence_limit + other_epsilon])
    ax.set_yticklabels([r'$\alpha - \varepsilon$', r'$\alpha + \varepsilon$'], fontsize=14)
    save_plot('convergent_sequence_5')


def convergent_cauchy():
    fig, ax = plt.subplots()
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    alpha = (ymax - ymin) / 2
    delta = 1/7
    apd = alpha + delta
    amd = alpha - delta
    n_e = 0.33333333333
    p1 = (1.5 * n_e, alpha + 0.66666 * delta)
    p2 = (2.5 * n_e, alpha - 0.33333 * delta)
    labels = {
        'LIM': r'$\lim_{n \rightarrow \infty}a_n = \alpha$',
        'E_R': r'Էպսիլոն միջակայք',
        'lim': r'$\lim_{n \rightarrow \infty}a_n$',
        'n_e': r'$N_{\frac{\varepsilon}{2}}$',
        'a+e': r'$\alpha + \frac{\varepsilon}{2}$',
        'a-e': r'$\alpha - \frac{\varepsilon}{2}$',
        'alp': r'$\alpha$'
    }
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axhline(alpha, alpha=0.5, label=labels['LIM'])
    ax.set_yticks([alpha])
    ax.set_yticklabels([labels['lim']], fontsize=14)
    ax.set_xticks([])
    plt.legend(fontsize=12)
    save_plot('convergent_cauchy_0')
    ax.axhline(apd, color='black', linewidth=0.5)
    ax.axhline(amd, color='black', linewidth=0.5)
    ax.axhspan(amd, apd, facecolor='gray', alpha=0.3, label=labels['E_R'])
    ax.set_yticks([amd, alpha, apd])
    ax.set_yticklabels([labels['a-e'], labels['alp'], labels['a+e']], fontsize=14)
    plt.legend(fontsize=12)
    save_plot('convergent_cauchy_1')
    ax.axhspan(amd, apd, n_e, facecolor='green', alpha=0.3)
    ax.axvline(n_e, alpha=0.5, color='black')
    ax.set_xticks([n_e])
    ax.set_xticklabels([labels['n_e']], fontsize=12)
    save_plot('convergent_cauchy_2')
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'o')
    ax.annotate(r'$(n, a_{n})$', xy=(p1[0]-0.01, p1[1]+0.025), xytext=(p1[0]-0.16, p1[1]+0.12),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=14)
    ax.annotate(r'(m, $a_{m})$', xy=(p2[0]+0.005, p2[1]-0.025), xytext=(p2[0]+0.01, p2[1]-0.16),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)
    save_plot('convergent_cauchy_3')
    ax.axhline(p1[1], color='black', linewidth=0.5, linestyle='dashed', c='gray')
    ax.axhline(p2[1], color='black', linewidth=0.5, linestyle='dashed', c='gray')
    plt.annotate(
        '', xy=(p1[0]+(p2[0]-p1[0])/3, p1[1]), xycoords='data',
        xytext=(p1[0]+(p2[0]-p1[0])/3, p2[1]), textcoords='data',
        arrowprops={'arrowstyle': '<->'})
    plt.annotate(
        r'$|a_n - a_m|$', xy=(p1[0]+(p2[0]-p1[0])/3+0.025, 0.515), xycoords='data',
        xytext=(0.125, 0.515), textcoords='offset points', fontsize=12)
    save_plot('convergent_cauchy_4')
    plt.annotate(
        '', xy=(0.1, amd), xycoords='data',
        xytext=(0.1, apd), textcoords='data',
        arrowprops={'arrowstyle': '<->'})
    plt.annotate(
        r'$\varepsilon = \frac{\varepsilon}{2} + \frac{\varepsilon}{2}$', xy=(0.125, 0.515), xycoords='data',
        xytext=(0.125, 0.515), textcoords='offset points', fontsize=12)
    save_plot('convergent_cauchy_5')
    
def pointsonstage(n, stage):
    if stage <= 0:
        return 1, 1
    if stage == 1:
        return n, n
    k = pointsonstage(n, stage-1)[0]
    return k * (n+1 - stage), k * (n - stage + 2) - 1

def semifactorial(n, m):
    if n == m:
        return m
    s = 1
    for i in range(n, m+1):
        s *= i
    return i

def plot_line_segment(p1, p2):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='grey', alpha=0.3)


class PermuTree:
    def __init__(self, n, dx = 0.1, dy=0.5):
        self._dx = dx
        self._dy = dy
        self._n = n
        self._start = np.array([])
        self._dict = {}
        self._constract()
    
    def __getitem__(self, stage):
        if stage <= 0:
            return []
        elif stage <= self._n:
            return self._dict[stage]
        else:
            return []
        
    def __call__(self, key, gap=None):
        cascade = self[key]
        values = []
        for state_list in cascade:
            for state in state_list:
                values.append(state[-1])
            values.append(gap)
        return [values[i] for i in range(len(values) - 1)]

    def _next(self, state: np.ndarray):
        elements = set([i for i in range(1, self._n+1)])
        other_elements = elements - set(state.copy().tolist())
        return [np.array(state.copy().tolist()+[i]) for i in sorted(other_elements)]
    
    def _next_cascade(self, cascade_list):
        cascade = []
        for state_list in cascade_list:
            for state in state_list:
                cascade.append(self._next(state))
        return cascade

    def _constract(self):
        self._dict[0] = [self._start]
        self._dict[1] = [self._next(self._start)]
        for j in range(2, self._n + 1):
            self._dict[j] = self._next_cascade(self._dict[j - 1])

    def get_cords(self, stage, none=False):
        xmax = self._dx * (len(self(self._n)) + 1)
        xmin = 0
        y = self._dy * stage
        points = self(stage)
        dx = (xmax - xmin) / (len(points) + 1)
        resalt = []
        for n, p in enumerate(points, 1):
            if n is None:
                continue
            if p is not None or none:
                resalt.append(
                    {
                        'point': p,
                        'cord': (dx * n, y)
                    }
                )
        return resalt
    

    def get_cedg(self, stage, none=False):
        xmax = self._dx * (len(self(self._n)) + 1)
        xmin = 0
        y = self._dy * stage
        points = self(stage)
        p_points = self(stage - 1)
        dx = (xmax - xmin) / (len(points) + 1)
        resalt = []
        c = self.get_cords(stage, True)
        pc = self.get_cords(stage - 1) if stage != 1 else [{'cord': ((xmax-xmin)/2, -self._dy)}]
        i = 0
        for d in c:
            if d['point'] is None:
                i +=1
                if none:
                    resalt.append(None)
                continue
            resalt.append(
                {
                    'p1': (d['cord'][0], d['cord'][1] - self._dy/10),
                    'p2': (pc[i]['cord'][0], pc[i]['cord'][1] + self._dy/10)
                }
            )
        return resalt

    def get_xlim(self):
        xmax = self._dx * (len(self(self._n)) + 1)
        xmin = 0
        return(xmin, xmax)
    
    def get_ylim(self):
        ymax = self._dy * (self._n + 0.1)
        ymin = -self._dy
        return(ymin, ymax)

def binomial_coefficient(n, stily):
    tree = PermuTree(n)
    _, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(*tree.get_xlim())
    ax.set_ylim(*tree.get_ylim())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["bottom", "left", "right", "top"]].set_visible(False)
    for stage in range(n+1):
        i = 0
        cords = tree.get_cords(stage, True)
        edgs = tree.get_cedg(stage, True)
        for d, e in zip(cords, edgs):
            if d['point'] is None:
                save_plot(f'permutations_s{stage}_e{i}')
                i += 1
            else:
                ax.plot([d['cord'][0]], [d['cord'][1]], stily[d['point']])
                # print([d['cord'][0]], [d['cord'][1]], *stily[d['point']])
                plot_line_segment(e['p1'], e['p2'])
        save_plot(f'permutations_s{stage}_e{i}')


def is_prime(k):
    if (k <= 1):
        return False
    if (k == 2 or k == 3):
        return True
    if (k % 2 == 0 or k % 3 == 0):
        return False
    for i in range(5, 1 + int(k ** 0.5), 6):
        if (k % i == 0 or k % (i + 2) == 0):
            return False
    return True

def permutations(n = 4):
    set([i for i in range(1, n+1)]) 

def permute_l_from_n_number(n, l):
    s = 1
    for i in range (n, n-l, -1):
        s *= i
    return s





# convergent_sequence()
# convergent_cauchy()
d = {
    1: 'bo',
    2: 'g^',
    3: 'sr',
    4: 'kp',
}
binomial_coefficient(4, d)
plt.show()

# a = PermuTree(4)
# for i in range(6):
#     print(a(i))