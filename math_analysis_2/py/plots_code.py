import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def func_preimage_plot(func, x_range, yset):
    x = np.linspace(x_range[0], x_range[1], 5000)
    y = func(x)
    xy = np.column_stack((x, y))
    yset = np.array(yset)
    yset.sort()
    preimage_list = preimage(xy, yset)
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Move the left and bottom spines to x = 0 and y = 0, respectively.
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    y_range = (y.min(), y.max())
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.text(-0.25, 0.95, '$f(x)$', transform=ax.get_xaxis_transform(), fontsize=16)
    ax.text(0.95, -0.25, '$x$', transform=ax.get_yaxis_transform(), fontsize=16)
    ax.set_yticks([])
    ax.set_xticks([])
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
    xticks = [cord for interval in preimage_list for cord in interval]
    for interval in preimage_list:
        mid_point = (interval[0] + interval)[1]/2
        # x arrows
        ymin = yset.min()
        ymax = yset.max()
        xarrow_down = patches.Arrow(mid_point, y_range[1]/4, 0, -0.2, width=.3, facecolor='gold', alpha=0.6)
        xarrow_up = patches.Arrow(mid_point, y_range[0]/4, 0, 0.2, width=.3, facecolor='gold', alpha=0.6)
        if ymin * ymax < 0:
            ax.add_patch(xarrow_down)
            ax.add_patch(xarrow_up)
        elif ymax <= 0:
            ax.add_patch(xarrow_up)
        else:
            ax.add_patch(xarrow_down)
        xticks.append(interval[0])
        xticks.append(interval[1])
        ax.axvspan(interval[0], interval[1], facecolor='gold', alpha=0.3)
        ax.add_patch(xset_box(interval, y_range[1] - y_range[0]))
    ax.set_xticks(xticks)
    ax.set_xticklabels(['$x_{'+f'{i}'+'}$' for i in range(1, len(xticks) + 1)], fontsize=12) # mod 3
    ax.add_patch(
        patches.Arrow(-x_range[1]/8, (yset[0] + yset[1])/2, -0.2, 0, width=.3, facecolor='red', alpha=0.3)
    )
    ax.add_patch(
        patches.Arrow(x_range[1]/8, (yset[0] + yset[1])/2, 0.2, 0, width=.3, facecolor='red', alpha=0.3)
    )
    save_plot('func_preimage_plot_3') # mod 3


func_preimage_plot(func, (-1, 1), (0.3, 0.7))
#plt.show()
