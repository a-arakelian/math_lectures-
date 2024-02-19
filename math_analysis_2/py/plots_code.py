import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import (BboxConnector,
                                                   BboxPatch)
from matplotlib.transforms import (Bbox, TransformedBbox)
import os

IGNOREDDIRECTORY = '5b49576b'
DIRECTORY = 'math_analysis_2/images'


def save_plot(name):
    dir = os.path.join(DIRECTORY, IGNOREDDIRECTORY)
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, name)
    plt.savefig(path, dpi=300, transparent=True)


def func(x):
    return 3 * (x*0.9271132) ** 3 - (x*0.9271132) * 1.5


def preimage(xy, y_range):
    xy = np.array(xy)
    y_range = np.array(y_range)
    y_range.sort()
    filtered_xy = xy[(xy[:, 1] >= y_range[0]) & (xy[:, 1] <= y_range[1])]
    filtered_xy = filtered_xy[filtered_xy[:, 0].argsort()] # sorted with x
    x_range_list = [[filtered_xy[0, 0], filtered_xy[0, 0]]]
    e = 1.5 * (xy[1][0] - xy[0][0]) # epsilon
    for cord in filtered_xy:
        last_uplimit = x_range_list[-1][1]
        if abs(last_uplimit - cord[0]) < e:
            x_range_list[-1][1] = cord[0]
        else:
            x_range_list.append([cord[0], cord[0]])
    return x_range_list


def image(xy, x_range, continues):
    xy = np.array(xy)
    x_range = np.array(x_range)
    x_range.sort()
    filtered_xy = xy[(xy[:, 0] >= x_range[0]) & (xy[:, 0] <= x_range[1])]
    filtered_xy = filtered_xy[filtered_xy[:, 1].argsort()] # sorted with y
    if continues:
        y_range_list = [[filtered_xy[0, 1], filtered_xy[-1, 1]]]
    else:
        y_range_list = [[filtered_xy[0, 1], filtered_xy[0, 1]]]
        e = 1.5 * (xy[1][0] - xy[0][0]) # epsilon TODO
        for cord in filtered_xy:
            last_uplimit = y_range_list[-1][1]
            if abs(last_uplimit - cord[1]) < e:
                y_range_list[-1][1] = cord[1]
            else:
                y_range_list.append([cord[1], cord[1]])
    return y_range_list


def yset_box(y_range, x_width, color='red', opacity=0.6):
    y_range = np.array(y_range)
    y_range.sort()
    delta = x_width / 200
    vertices = [
        (delta, y_range[0]),
        (delta, y_range[1]),
        (-delta, y_range[1]),
        (-delta, y_range[0])
    ]
    box = patches.Polygon(vertices)
    box.set_facecolor(color)
    box.set_alpha(opacity)
    return box


def xset_box(x_range, y_width, color='gold', opacity=0.9):
    x_range = np.array(x_range)
    x_range.sort()
    delta = y_width / 200
    vertices = [
        (x_range[0], delta),
        (x_range[1], delta),
        (x_range[1], -delta),
        (x_range[0], -delta),
    ]
    box = patches.Polygon(vertices)
    box.set_facecolor(color)
    box.set_alpha(opacity)
    return box


def configure_axis(ax, x_range, y_range, xlabel='$x$', ylabel='$f(x)$'):
    # Move the left and bottom spines to x = 0 and y = 0, respectively.
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.text(-0.25, 0.95, ylabel, transform=ax.get_xaxis_transform(), fontsize=16)
    ax.text(0.95, -0.25, xlabel, transform=ax.get_yaxis_transform(), fontsize=16)
    ax.set_yticks([])
    ax.set_xticks([])


def add_axspan_list(ax, range_list, set_range, axis_range, color='gold', axis='x'):
    x_axis = set(('x', 'X', 'h', 'H'))
    y_axis = set(('y', 'Y', 'v', 'V'))
    if axis not in x_axis and axis not in y_axis:
        raise ValueError('invalid axis, chose "x" or "y"')
    axspan = ax.axvspan if axis in x_axis else ax.axhspan
    set_ticks = ax.set_xticks if axis in x_axis else ax.set_yticks
    set_ticklabels = ax.set_xticklabels if axis in x_axis else ax.set_yticklabels
    set_box = xset_box if axis in x_axis else yset_box
    label = '$x_{' if axis in x_axis else '$y_{'
    ticks = [cord for interval in range_list for cord in interval]
    set_min = set_range.min()
    set_max = set_range.max()
    ru = axis_range[1]
    ld = axis_range[0]
    for interval in range_list:
        mid_point = (interval[0] + interval[1])/2
        x_ru, x_ld = (mid_point, mid_point) if axis in x_axis else (ru/4, ld/4)
        y_ru, y_ld = (mid_point, mid_point) if axis in y_axis else (ru/4, ld/4)
        dx, dy = (0, -0.2) if axis in x_axis else (-0.2, 0)
        # arrows
        arrow_ru = patches.Arrow(x_ld, y_ld, dx, dy, width=.3, facecolor=color, alpha=0.6)
        arrow_ld = patches.Arrow(x_ru, y_ru, dx, dy, width=.3, facecolor=color, alpha=0.6)
        if set_min * set_max < 0:
            ax.add_patch(arrow_ld)
            ax.add_patch(arrow_ru)
        elif set_max <= 0:
            ax.add_patch(arrow_ru)
        else:
            ax.add_patch(arrow_ld)
        axspan(interval[0], interval[1], facecolor=color, alpha=0.3)
        ax.add_patch(set_box(interval, ru - ld))
    set_ticks(ticks)
    set_ticklabels([label+f'{i}'+'}$' for i in range(1, len(ticks) + 1)], fontsize=12)


def func_preimage_plot(func, x_range, yset):
    x = np.linspace(x_range[0], x_range[1], 5000)
    y = func(x)
    xy = np.column_stack((x, y))
    yset = np.array(yset)
    yset.sort()
    preimage_list = preimage(xy, yset)
    # Create a figure and axis
    fig, ax = plt.subplots()
    y_range = (y.min(), y.max())
    configure_axis(ax, x_range, y_range)
    # Create plot
    ax.plot(x, y, label='$f(x)$')
    save_plot('func_preimage_plot_0') # mod 0
    ax.add_patch(yset_box(yset, x_range[1] - x_range[0]))
    ax.set_yticks(yset)
    ax.set_yticklabels(['$y_1$', '$y_2$'], fontsize=12)
    save_plot('func_preimage_plot_1') # mod 1
    ax.axhspan(yset[0], yset[1], alpha=0.3, facecolor='red')
    ax.add_patch(
        patches.Arrow(-x_range[1]/8, (yset[0] + yset[1])/2, -0.2, 0, width=.3, facecolor='red', alpha=0.6)
    )
    ax.add_patch(
        patches.Arrow(x_range[1]/8, (yset[0] + yset[1])/2, 0.2, 0, width=.3, facecolor='red', alpha=0.6)
    )
    save_plot('func_preimage_plot_2') # mod 2
    add_axspan_list(ax, preimage_list, yset, y_range, 'gold', 'x')
    save_plot('func_preimage_plot_3') # mod 3


def func_image_plot(func, x_range, xset, continues=True):
    x = np.linspace(x_range[0], x_range[1], 5000)
    y = func(x)
    xy = np.column_stack((x, y))
    xset = np.array(xset)
    xset.sort()
    image_list = image(xy, xset, continues)
    # Create a figure and axis
    fig, ax = plt.subplots()
    y_range = (y.min(), y.max())
    configure_axis(ax, x_range, y_range)
    # Create plot
    ax.plot(x, y, label='$f(x)$')
    save_plot('func_image_plot_0') # mod 0
    ax.add_patch(xset_box(xset, y_range[1] - y_range[0]))
    ax.set_xticks(xset)
    ax.set_xticklabels(['$x_1$', '$x_2$'], fontsize=12)
    save_plot('func_image_plot_1') # mod 1
    ax.axvspan(xset[0], xset[1], facecolor='gold', alpha=0.3)
    ax.add_patch(
        patches.Arrow((xset[0] + xset[1])/2, y_range[1]/4, 0, 0.2, width=.3, facecolor='gold', alpha=0.6)
    )
    ax.add_patch(
        patches.Arrow((xset[0] + xset[1])/2, y_range[0]/4, 0, -0.2, width=.3, facecolor='gold', alpha=0.6)
    )
    save_plot('func_image_plot_2') # mod 2
    add_axspan_list(ax, image_list, xset, x_range, 'red', 'y')
    save_plot('func_image_plot_3') # mod 3


def span(xmin, xmax, ax, color='red', alpha=0.5, add=True):
    bbox = Bbox([[xmin, 0], [xmax, 1]])
    tr_bbox = TransformedBbox(bbox, ax.get_xaxis_transform())
    vspan = BboxPatch(tr_bbox, alpha=alpha, facecolor=color)
    if add:
        ax.add_patch(vspan)
    return vspan, tr_bbox


def nested_intervals(interval_list):
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.spines[["bottom"]].set_position(("data", 0))
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.plot(1, ax.get_ylim()[0], ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_yticks([])
    ax.set_xticks([])
    a, b = interval_list[0]
    ax.set_xlim(a - (b-a)/20, b + (b-a)/20)
    for n, interval in enumerate(interval_list, 2):
        xlabels = [t+f'{i}'+'}$' for i in range(1, n) for t in ['$a_{', '$b_{']]
        xticks = [x for inter in interval_list[:n-1] for x in inter]
        k = 2 * (n-1) if n < 5 else 6
        ax.set_xticks(xticks[:k])
        ax.set_xticklabels(xlabels[:k])
        if n % 2 == 0:
            color='blue'
            alpha=0.2
        else:
            color='red'
            alpha=0.2
        span(interval[0], interval[1], ax, color, alpha)
        if len(interval_list) > 2:
            interval_list[-2]
        save_plot(f'nested_intervals_{n-1}') # mod n-1

def make_sift():
    a = [0]
    def n_to_r(n):
        n += a[0]
        i = 3
        while n / i >= 1:
            i *= 3
        if n % 3 == 0:
            n += 1
            a[0] += 1
        return float(n)/i
    return n_to_r


n_to_r = make_sift()


def cut_to_3(a, b):
    delta = float(b - a) / 3
    a1, b1 = a, a + delta
    a2, b2 = b1, b1 + delta
    a3, b3 = b2, b
    bbox_1 = Bbox([[a1, 0], [b1, 1]])
    bbox_2 = Bbox([[a2, 0], [b2, 1]])
    bbox_3 = Bbox([[a3, 0], [b3, 1]])
    return bbox_1, bbox_2, bbox_3, a2, a3


def transformbboxes(*args, ax):
    for arg in args:
        yield TransformedBbox(arg, ax.get_xaxis_transform())


def label(char, i):
    return f'${char}' + '_{' + f'{i}' + '}$'



def nested_intervals_special(n):
    axs = plt.figure(figsize=(10, 1)).subplot_mosaic([["zoom"], ["main"]])
    main_ax = axs["main"]
    zoom_ax = axs["zoom"]
    for ax in (main_ax, zoom_ax):
        ax.spines[["bottom"]].set_position(("data", 0))
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.plot(1, ax.get_ylim()[0], ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.set_yticks([])
        ax.set_xticks([])
    vspan_main_list = []
    vspan_zoom_list = []
    aconnector_list = []
    bconnector_list = []
    a = 0
    b = 1
    main_ax.set_xlim(-0.1, 1.1)
    for i in range(1, n+1):
        color, alpha = ('blue', 0.2) if i % 2 == 0 else ('red', 0.2)
        bbox_0 = Bbox([[a, 0], [b, 1]])
        main_tr_box_0 = TransformedBbox(bbox_0, main_ax.get_xaxis_transform())
        zoom_tr_box_0 = TransformedBbox(bbox_0, zoom_ax.get_xaxis_transform())
        bbox_1, bbox_2, bbox_3, a2, a3 = cut_to_3(a, b)
        main_tr_bbox_list = [tr for tr in transformbboxes(bbox_1, bbox_2, bbox_3, ax=main_ax)]
        zoom_tr_bbox_list = [tr for tr in transformbboxes(bbox_1, bbox_2, bbox_3, ax=zoom_ax)]
        vspan_main_list.append(main_tr_box_0)
        vspan_zoom_list.append(zoom_tr_box_0)
        main_ax.add_patch(BboxPatch(main_tr_box_0, alpha=alpha, facecolor=color))
        zoom_ax.add_patch(BboxPatch(zoom_tr_box_0, alpha=alpha, facecolor=color))
        c1 = BboxConnector(main_tr_box_0, zoom_tr_box_0, loc1=2, loc2=3, clip_on=False)
        c2 = BboxConnector(main_tr_box_0, zoom_tr_box_0, loc1=1, loc2=4, clip_on=False)
        for acon, bcon in zip(aconnector_list, bconnector_list):
            acon.set_visible(False)
            bcon.set_visible(False)
        aconnector_list.append(c1)
        bconnector_list.append(c2)
        zoom_ax.add_patch(c1)
        zoom_ax.add_patch(c2)
        x = n_to_r(i)
        main_ax.set_xticks([a, b, x])
        zoom_ax.set_xticks([a, b, x])
        main_ax.set_xticklabels([label('a', i), label('b', i), f'$f({i})$'])
        zoom_ax.set_xticklabels([label('a', i), label('b', i), f'$f({i})$'])
        zoom_ax.set_xlim(a - (b-a)/20, b + (b-a)/20)
        save_plot(f'nested_intervals_special_{i}') # mod 1
        if x < a or x > a2:
            bbox_i = 0
            a, b = a, a2
        elif x < a2 or x > a3:
            bbox_i = 1
            a, b = a2, a3
        elif x < a3 or x > b:
            bbox_i = 2
            a, b = a3, b
        main_box = [BboxPatch(cut3, facecolor=color, alpha=0.75) for cut3, color in zip(main_tr_bbox_list, ('#CDFAD5', '#F6FDC3', '#FFCF96'))]
        zoom_box = [BboxPatch(cut3, facecolor=color, alpha=0.75) for cut3, color in zip(zoom_tr_bbox_list, ('#CDFAD5', '#F6FDC3', '#FFCF96'))]
        for mbox, zbox in zip(main_box, zoom_box):
            main_ax.add_patch(mbox)
            zoom_ax.add_patch(zbox)
        save_plot(f'nested_intervals_special_cut3_{i}') # mod 2
        for mbox, zbox in zip(main_box, zoom_box):
            mbox.set_visible(False)
            zbox.set_visible(False)
        ch_tr_mbbox = BboxPatch(main_tr_bbox_list[bbox_i], facecolor='#9BCF53', alpha=0.85)
        ch_tr_zbbox = BboxPatch(zoom_tr_bbox_list[bbox_i], facecolor='#9BCF53', alpha=0.85)
        main_ax.add_patch(ch_tr_mbbox)
        zoom_ax.add_patch(ch_tr_zbbox)
        save_plot(f'nested_intervals_special_check_{i}') # mod 3
        ch_tr_mbbox.set_visible(False)
        ch_tr_zbbox.set_visible(False)


func_preimage_plot(func, (-1, 1), (0.3, 0.7))
func_image_plot(func, (-1, 1), (0.3, 0.7))
nested_intervals([[0, 1], [0.4, 0.8], [0.45, 0.55], [0.475, 0.5], [0.48, 0.5], [0.485, 0.495]])
nested_intervals_special(10)
plt.show()
