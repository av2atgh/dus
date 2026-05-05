import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

plt.rcParams.update({'font.size': 16})


def fishplot(
    ax,
    w,
    x=np.array([]),
    y=np.array([]),
    z=np.array([]),
    node_cmap='turbo',
    link_cmap='binary',
    bar_height_ratio=0.7,
    bar_gap_ratio=1.01,
    node_marker='o',
    node_size=5,
):
    
    def wcolor(v):
        # color mapping of link weights
        return np.round(v/(20 + v), 2)
    
    if len(x) == 0:
        n = 1 + np.max([max(k[0], k[1]) for k in w.keys()])
        x = np.arange(n)
    else:
        n = len(x)

    # polar coordinates mapping
    r = (x.max() - x.min()) / 2    
    cos = x / r - 1 
    theta = np.arccos(cos)
    sin = np.sin(theta)

    # sort weights
    w = {k: v for k, v in sorted(w.items(), key=lambda item: item[1])}

    # generations
    if len(y) == 0:
        # show nodes
        _ = ax.scatter(r * cos, r  * sin, c='black', marker=node_marker, s=node_size, zorder=3)      
    else:
        # show bar plot
        cmap = colormaps[node_cmap]
        ycolor = z / np.max(z) if len(z) > 0 else np.ones(n)
        y_ = r * y / y.max()
        for i in range(n):
            r1 = r * bar_gap_ratio
            r2 = r1 + bar_height_ratio * y_[i]
            _ = ax.plot([r1 * cos[i], r2 * cos[i]], [r1 * sin[i], r2 * sin[i]], c=cmap(ycolor[i]))

    # dependencies network
    cmap = colormaps[link_cmap]
    for k, v in w.items():
        i = k[0]
        j = k[1]
        _ = ax.plot([r * cos[i], r * cos[j]], [r * sin[i], r * sin[j]], c=cmap(wcolor(v)))

   
    _ = ax.axis("off")
    _ = ax.set_box_aspect(0.5)
