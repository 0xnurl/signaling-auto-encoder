import functools
import logging
import operator
from collections import defaultdict
from typing import Iterable, List, Text, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def plot_raw(
    data,
    masks: List[List[int]],
    labels: List[Text],
    title: Text,
    colors: Optional[torch.Tensor] = None,
    custom_labels: List[Text] = [],
):
    plot_pca = data.shape[1] > 2

    if data.shape[1] > 1:
        num_plots = 2 if plot_pca else 1
        fig, ax = plt.subplots(1, num_plots, figsize=(8, 4), squeeze=False)

        if plot_pca:
            pca_model = PCA(n_components=2, whiten=True)
            data_pca = pca_model.fit_transform(data)

        for i, (mask, label) in enumerate(zip(masks, labels)):
            # plot first two coordinates
            if colors is not None:
                color = f"C{colors[i]}"
            else:
                color = f"C{i}"

            ax[0, 0].scatter(
                data[mask, 0], data[mask, 1], alpha=0.5, label=label, color=color
            )
            ax[0, 0].axis("equal")
            ax[0, 0].set(
                xlabel="Coordinate 1",
                ylabel="Coordinate 2",
                title="First coordinates" if plot_pca else "",
            )

            if plot_pca:
                ax[0, 1].scatter(
                    data_pca[mask, 0],
                    data_pca[mask, 1],
                    alpha=0.1 if label.lower() == "shifted" else 0.5,
                    label=label,
                    color=color,
                )
                ax[0, 1].axis("equal")
                ax[0, 1].set(xlabel="Component 1", ylabel="Component 2", title="PCA")

    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 2))

        for mask, label in zip(masks, labels):
            # plot first (only) coordinate
            ax.scatter(
                data[mask, 0],
                [0] * len(mask),
                alpha=0.01 if label.lower() == "shifted" else 0.5,
                label=label,
            )
            ax.axis("equal")
            ax.set(xlabel="Coordinate 1", ylabel="Dummy", title="First coordinate")

    fig.suptitle(title)

    if custom_labels:
        information_legend = plt.legend(
            handles=[mpatches.Patch(color="gray", label=x) for x in custom_labels],
            loc="lower left",
            bbox_to_anchor=(-1.1, 1),
        )

        plt.gca().add_artist(information_legend)

    functions_legend = plt.legend(ncol=2, loc="lower right", bbox_to_anchor=(1, 1))
    for lh in functions_legend.legendHandles:
        lh.set_alpha(1)

    plt.show()


def plot_clusters(data, labels, title="Clusters"):
    labels_to_idx = defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_idx[label].append(i)

    labels, masks = zip(*labels_to_idx.items())
    plot_raw(data, masks, labels, title)


def plot_bar_list(L, L_labels=None, transform=True):
    if not (L_labels):
        L_labels = np.arange(len(L))
    index = np.arange(len(L))

    if transform:
        COL = ["blue", "red"]
    else:
        COL = "blue"

    plt.bar(index, [x.item() for x in L], color=COL)
    plt.xticks(index, L_labels, fontsize=5)
    plt.xlabel("Functions", fontsize=5)
    plt.ylabel("MSELoss", fontsize=5)
    plt.title("Loss per function")
    plt.show()


def plot_pca_3d(x, data, xlabel, ylabel, zlabel, title):
    pca = PCA(2)
    predictions_pca = pca.fit_transform(data)

    zs = predictions_pca[:, 0]
    ys = predictions_pca[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title)

    ax.plot(x, ys, zs)
    ax.legend()

    plt.show()


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%d-%m-%Y:%H:%M:%S",
        level=logging.INFO,
    )


def batch_mse(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Assumes 2 tensors of shape (batch_size, sample_size) or `target` of size
    (batch_size, sample_size) and `prediction` of shape (1, sample_size)."""
    return torch.mean((target - prediction) ** 2, dim=1)


def reduce_prod(vals):
    return functools.reduce(operator.mul, vals)


def batch_flatten(x):
    return torch.reshape(x, [-1, reduce_prod(x.shape[1:])])


def join_vals(ints: Iterable[int], s=",") -> Text:
    return s.join(f"{str_val(x)}" for x in ints)


def str_val(val) -> Text:
    if isinstance(val, bool):
        return "1" if val else "0"
    elif isinstance(val, int):
        return str(val)
    elif isinstance(val, tuple) or isinstance(val, list):
        return join_vals(val)
    return str(val)


def kwargs_to_str(kwargs) -> Text:
    return "__".join(f"{key}_{str_val(val)}" for key, val in kwargs.items())
