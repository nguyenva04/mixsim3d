import numpy as np

import matplotlib.pyplot as plt


def display_section(cube, name, cmap="gray"):
    """
    plot middle sections of a 3D data cube (against X Y and Z)
    """

    min = np.min(cube)
    max = np.max(cube)
    plt.figure(name, figsize=(20, 20))
    plt.subplot(131)
    plt.imshow(cube[cube.shape[0] // 2, :, :], cmap=cmap, vmin=min, vmax=max)
    plt.title(name + " / Z")
    plt.subplot(132)
    plt.imshow(cube[:, cube.shape[1] // 2, :], cmap=cmap, vmin=min, vmax=max)
    plt.title(name + " / Y")
    plt.subplot(133)
    plt.imshow(cube[:, :, cube.shape[2] // 2], cmap=cmap, vmin=min, vmax=max)
    plt.title(name + " / X")
    plt.show()


def smooth(liste, beta=0.98):
    avg = 0.0
    threshold = 0.0
    smoothed_list = []
    for i, l in enumerate(liste):
        # Compute the smoothed loss
        avg = beta * avg + (1 - beta) * l
        smoothed = avg / (1 - beta ** (i + 1))
        # Stop if the loss is exploding
        if i > len(liste) // 2 and smoothed >= threshold:
            break
        # Record the best loss
        if i == len(liste) // 3:
            threshold = smoothed
        smoothed_list.append(smoothed)
    return smoothed_list


def lrfind_plot(lr, loss):
    fig, ax = plt.subplots(figsize=(10, 5))
    trace = smooth(loss)
    ax.plot(lr[: len(loss)], loss, color="lightsteelblue", alpha=0.4)
    ax.plot(lr[: len(trace)], trace, color="navy")

    ax.set_title("LR Finder", fontsize=18)
    ax.set_xlabel("learning rate", fontsize=15)
    ax.set_ylabel("Loss", fontsize=15)
    ax.set_xscale("log")
    ax.set_xticks(
        np.array([np.arange(1, 10) * 10 ** (-8 + i) for i in range(1, 10)]).flatten()
    )
    ax.set_ylim(0.95 * min(trace), 1.05 * max(trace))

    return fig
