# Plots Library
#

import matplotlib.pyplot as plt
import matplotlib.rcsetup as mrc


def pleasing_histogram(df, col):
    ax = plt.axes(axisbg="#E6E6E6")
    ax.set_axisbelow(True)
    plt.grid(color="w", linestyle="solid")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    for tick in ax.get_yticklabels():
        tick.set_color("gray")
    ax.hist(df[col], edgecolor="#E6E6E6", color="#EE6666")
    plt.show()


# NOTE: before calling this, save the default rc params thusly: IPython_default = plt.rcParams.copy()
def change_rc():
    colors = mrc.cycler("color",
                        ["#EE6666", "#3388BB", "#9988DD", "#EECC55", "#88BB44", "#FFBBBB"])
    plt.rc("axes", facecolor="#E6E6E6", edgecolor="none", axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc("grid", color="w", linestyle="solid")
    plt.rc("xtick", direction="out", color="gray")
    plt.rc("ytick", direction="out", color="gray")
    plt.rc("patch", edgecolor="#E6E6E6")
    plt.rc("lines", linewidth=2)
