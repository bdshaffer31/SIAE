import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch
import os
import itertools
from cycler import cycler
import seaborn as sns
import matplotlib.patches as mpatches
import experiments

import utils
import siae


COLUMN = 3.2
SMALLER = 2.4
HALF = 1.6
TALL = 4.8
DOUBLE = 6.2


class Plotter:
    def __init__(self, save_dir):
        self.save_dir = os.path.join("recordings", save_dir)
        shutil.rmtree(self.save_dir, ignore_errors=True)
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, input_plot, file_name):
        rel_filepath = os.path.join(self.save_dir, f"{file_name}.png")
        input_plot.savefig(rel_filepath, bbox_inches="tight", transparent=False)
        input_plot.clf()
        plt.close()

    def save_plots(self, input_plots, file_names):
        for plot, name in zip(input_plots, file_names):
            rel_filepath = os.path.join(self.save_dir, f"{name}.png")
            plot.savefig(rel_filepath, bbox_inches="tight", transparent=False)
            plot.clf()
            plt.close()


def set_styling():
    # plt.style.use('ggplot')
    plt.style.use("seaborn-v0_8-deep")

    plt.rcParams["font.size"] = 8
    plt.rcParams["text.color"] = "dimgrey"
    plt.rcParams["axes.labelcolor"] = "dimgrey"
    plt.rcParams["axes.labelsize"] = 6
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (3.2, 2.4)
    plt.rcParams["figure.titleweight"] = "bold"
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.markerscale"] = 0.75
    plt.rcParams["legend.markerscale"] = 0.75
    plt.rcParams["legend.fontsize"] = 6
    plt.rcParams["legend.framealpha"] = 0.0
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["lines.markersize"] = np.sqrt(8)
    plt.rcParams["lines.linewidth"] = 1.0

    # Define grayscale color cycle with different linestyles
    greyscale_cycler = cycler(color=["black"] * 7) + cycler(
        linestyle=["-", "--", "-.", ":", "-", "--", "-."]
    )

    # Apply the custom color cycle to the Matplotlib rcParams
    plt.rcParams["axes.prop_cycle"] = greyscale_cycler


def plot_eigs_from_operator(operator, plotter, model_name="model"):
    """gen a double eigenvalue plot"""
    eigs, _ = np.linalg.eig(operator)
    plot_eigs(eigs, plotter, model_name=model_name)


def plot_eigs(eigs, plotter, model_name="model"):
    fig, ax = plt.subplots(1, figsize=(COLUMN, COLUMN), dpi=300)
    ax.set_title(r"$\lambda$")

    points = ax.scatter(
        eigs.real,
        eigs.imag,
        s=8,
        marker="o",
        linewidth=0.2,
        edgecolors="k",
        facecolors="k",
        zorder=10,
    )

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    ax.spines[["right", "top"]].set_visible(True)
    ax.text(
        0.1,
        0.9,
        r"$\mathrm{\mathbb{C}}$",
        fontsize=12,
        color="k",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        0, 0, "+", fontsize=12, horizontalalignment="center", verticalalignment="center"
    )

    unit_circle = plt.Circle(
        (0.0, 0.0), 1.0, color="k", fill=False, linestyle="-", linewidth=1.0, zorder=5
    )
    ax.add_artist(unit_circle)

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    ax.grid(False)
    ax.set_aspect("equal")
    fig.tight_layout()

    plotter.save(plt, f"{model_name}_eigs_lambda")

    # ======================
    fig, ax = plt.subplots(1, figsize=(COLUMN, COLUMN), dpi=300)
    ax.set_title(r"$\omega$")

    eigs = np.log(eigs)
    points = ax.scatter(
        eigs.real,
        eigs.imag,
        s=8,
        marker="o",
        linewidth=0.2,
        edgecolors="k",
        facecolors="k",
        zorder=10,
    )

    ax.spines[["right", "top"]].set_visible(True)
    ax.text(
        0.1,
        0.9,
        r"$\mathrm{\mathbb{C}}$",
        fontsize=12,
        color="k",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c="k",
    )
    ax.axvline(x=0, c="k", linestyle="-")

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    ax.grid(False)
    ax.set_box_aspect(1)
    fig.tight_layout()

    plotter.save(plt, f"{model_name}_eigs_omega")


def gen_eig_comp_plot_from_checkpoints(checkpoint_dynamics, plotter):
    eigen_vals = []
    for dyn in checkpoint_dynamics.values():
        eigen_vals.append(np.linalg.eig(dyn)[0])
    names = checkpoint_dynamics.keys()
    eigen_vals = np.array(eigen_vals)
    # ==== copied
    colors = plt.cm.binary(np.linspace(0, 1, eigen_vals.shape[0]))

    fig, ax = plt.subplots(1, figsize=(COLUMN, COLUMN), dpi=300)
    ax.set_title(r"$\lambda$")

    for i, (eigs, name) in enumerate(zip(eigen_vals, names)):
        points = ax.scatter(
            eigs.real,
            eigs.imag,
            s=8,
            linewidth=0.2,
            color=colors[i],
            edgecolors="k",
            zorder=10,
        )

    final_eigs = eigen_vals[-1]
    ax.scatter(
        final_eigs.real,
        final_eigs.imag,
        s=8,
        linewidth=1.0,
        color="r",
        marker="x",
        zorder=100,
    )

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    ax.spines[["right", "top"]].set_visible(True)
    ax.text(
        0.1,
        0.9,
        r"$\mathrm{\mathbb{C}}$",
        fontsize=12,
        color="k",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        0, 0, "+", fontsize=12, horizontalalignment="center", verticalalignment="center"
    )

    unit_circle = plt.Circle(
        (0.0, 0.0), 1.0, color="k", fill=False, linestyle="-", linewidth=1.0, zorder=5
    )
    ax.add_artist(unit_circle)

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    ax.grid(False)
    ax.set_aspect("equal")
    fig.tight_layout()

    plotter.save(plt, f"eig_evolution_lambda")

    # ======================
    fig, ax = plt.subplots(1, figsize=(COLUMN, COLUMN), dpi=300)
    ax.set_title(r"$\omega$")

    for i, (eigs, name) in enumerate(zip(eigen_vals, names)):
        eigs = np.log(eigs)
        points = ax.scatter(
            eigs.real,
            eigs.imag,
            s=8,
            linewidth=0.2,
            color=colors[i],
            edgecolors="k",
            zorder=10,
        )

    final_eigs = np.log(eigen_vals[-1])
    ax.scatter(
        final_eigs.real,
        final_eigs.imag,
        s=8,
        linewidth=1,
        color="r",
        marker="x",
        zorder=100,
    )

    ax.spines[["right", "top"]].set_visible(True)
    ax.text(
        0.1,
        0.9,
        r"$\mathrm{\mathbb{C}}$",
        fontsize=12,
        color="k",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c="k",
    )
    ax.axvline(x=0, c="k", linestyle="-")

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    ax.grid(False)
    # ax.set_aspect('equal')
    ax.set_box_aspect(1)
    fig.tight_layout()

    plotter.save(plt, f"eig_evolution_omega")

    # =============
    final_eigs = eigen_vals[-1]
    eigen_vals = eigen_vals.T
    colors = plt.cm.binary(np.linspace(0, 1, eigen_vals.shape[0]))

    fig, ax = plt.subplots(1, figsize=(COLUMN, COLUMN), dpi=300)
    ax.set_title(r"$\lambda$")

    for i, eigs in enumerate(eigen_vals):
        ax.plot(eigs.real, eigs.imag, linewidth=1, color="k", zorder=10, ls="-")

    ax.scatter(
        final_eigs.real,
        final_eigs.imag,
        s=8,
        linewidth=1,
        color="r",
        marker="x",
        zorder=100,
    )

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    ax.spines[["right", "top"]].set_visible(True)
    ax.text(
        0.1,
        0.9,
        r"$\mathrm{\mathbb{C}}$",
        fontsize=12,
        color="k",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        0, 0, "+", fontsize=12, horizontalalignment="center", verticalalignment="center"
    )

    unit_circle = plt.Circle(
        (0.0, 0.0), 1.0, color="k", fill=False, linestyle="-", linewidth=1.0, zorder=5
    )
    ax.add_artist(unit_circle)

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    ax.grid(False)
    ax.set_aspect("equal")
    fig.tight_layout()

    plotter.save(plt, f"eig_evolution_lambda_lines")


def gen_eig_comp_plot(eigs_list, names, plotter, cycle_markers=False):
    """plot comparing different eigenvalues"""
    if cycle_markers:
        marker = itertools.cycle(("o", "v", "s", "D", "P", "X"))
    else:
        marker = itertools.cycle(("o"))

    fig, ax = plt.subplots(1, figsize=(COLUMN, COLUMN), dpi=300)
    ax.set_title(r"$\lambda$")

    for eigs, name in zip(eigs_list, names):
        points = ax.scatter(
            eigs.real,
            eigs.imag,
            s=8,
            marker=next(marker),
            label=name,
            linewidth=0.2,
            edgecolors="k",
            zorder=10,
        )

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    ax.spines[["right", "top"]].set_visible(True)
    ax.text(
        0.1,
        0.9,
        r"$\mathrm{\mathbb{C}}$",
        fontsize=12,
        color="k",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        0, 0, "+", fontsize=12, horizontalalignment="center", verticalalignment="center"
    )

    unit_circle = plt.Circle(
        (0.0, 0.0), 1.0, color="k", fill=False, linestyle="-", linewidth=1.0, zorder=5
    )
    ax.add_artist(unit_circle)

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    ax.grid(False)
    ax.set_aspect("equal")
    fig.tight_layout()

    if len(names) < 6:
        plt.legend(
            loc="upper right",
            bbox_to_anchor=(1.3, 1.0),
            facecolor="white",
            edgecolor="k",
            shadow=True,
        )

    plotter.save(plt, f"eig_evolution_lambda")

    # ======================
    if cycle_markers:
        marker = itertools.cycle(("o", "v", "s", "D", "P", "X"))
    else:
        marker = itertools.cycle(("o"))

    fig, ax = plt.subplots(1, figsize=(COLUMN, COLUMN), dpi=300)
    ax.set_title(r"$\omega$")

    for eigs, name in zip(eigs_list, names):
        eigs = np.log(eigs)
        points = ax.scatter(
            eigs.real,
            eigs.imag,
            s=8,
            marker=next(marker),
            label=name,
            linewidth=0.2,
            edgecolors="k",
            zorder=10,
        )

    # ax.set_xlim((-1.1, 1.1))
    # ax.set_ylim((-1.1, 1.1))

    ax.spines[["right", "top"]].set_visible(True)
    ax.text(
        0.1,
        0.9,
        r"$\mathrm{\mathbb{C}}$",
        fontsize=12,
        color="k",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c="k",
    )
    ax.axvline(x=0, c="k", linestyle="-")

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    ax.grid(False)
    # ax.set_aspect('equal')
    ax.set_box_aspect(1)
    fig.tight_layout()

    if len(names) < 6:
        plt.legend(
            loc="upper right",
            bbox_to_anchor=(1.3, 1.0),
            facecolor="white",
            edgecolor="k",
            shadow=True,
        )

    plotter.save(plt, f"eig_evolution_omega")


def plot_eig_f_evolution(checkpoint_dynamics, plotter):
    eigen_freqs = []
    for dyn in checkpoint_dynamics.values():
        eigen_freqs.append(utils.eigen_freq_from_dynamics(dyn))

    plt.figure(figsize=(COLUMN, COLUMN), dpi=300)
    for epoch, eigen_freq in zip(checkpoint_dynamics.keys(), eigen_freqs):
        plt.scatter(eigen_freq, [epoch] * len(eigen_freq), c="k", s=2)
    plt.ylabel("Epoch")
    plt.xlabel("Frequency")
    plotter.save(plt, "eigen_freq_over_checkpoints")

    eigen_freqs = np.array(eigen_freqs)
    epochs = np.array(
        [[epoch] * len(eigen_freq) for epoch in checkpoint_dynamics.keys()]
    )
    colors = plt.cm.plasma(np.linspace(0, 1, eigen_freqs.shape[1]))

    plt.figure(figsize=(COLUMN, COLUMN), dpi=300)
    for i, (epoch, eigen_freq) in enumerate(zip(epochs.T, eigen_freqs.T)):
        plt.plot(eigen_freq, epoch, color=colors[i], ls="-")  # , c='r', marker_size=12
    plt.ylabel("Epoch")
    plt.xlabel("Frequency")
    plotter.save(plt, "eigen_freq_over_checkpoints_connected")


def plot_eig_f_evolution_multi_model():
    pass


def plot_train_hist(metrics, plotter):
    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    # plt.plot(metrics['loss'], label='Train Loss')
    plt.plot(metrics["val_loss"], label="Validation Loss")
    # plt.plot(metrics['recon'], label='Reconstruction Loss')
    plt.plot(metrics["pred"], label="Train Prediction Loss", c="r", ls="--")
    # plt.plot(metrics['linearity'], label='Norm Linearity Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plotter.save(plt, "loss_history_train_val_pred")

    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(metrics["loss"], label="Train Loss")
    # plt.plot(metrics['val_loss'], label='Validation Loss')
    # plt.plot(metrics['recon'], label='Reconstruction Loss')
    # plt.plot(metrics['pred'], label='Train Prediction Loss', c='r', ls='--')
    # plt.plot(metrics['linearity'], label='Norm Linearity Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plotter.save(plt, "loss_history_train")

    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(metrics["loss"], label="Train Loss")
    plt.plot(metrics["reg"], label="Regularization Loss")
    plt.plot(np.array(metrics["loss"]) - np.array(metrics["reg"]), label="Koopman")
    # plt.plot(metrics['recon'], label='Reconstruction Loss')
    # plt.plot(metrics['pred'], label='Train Prediction Loss', c='r', ls='--')
    # plt.plot(metrics['linearity'], label='Norm Linearity Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plotter.save(plt, "loss_history_train_reg")

    return

    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(metrics["loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.plot(metrics["recon"], label="Reconstruction Loss")
    plt.plot(metrics["pred"], label="Prediction Loss")
    plt.plot(metrics["linearity"], label="Norm Linearity Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plotter.save(plt, "loss_history_full")

    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(metrics["loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plotter.save(plt, "loss_history_train")

    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plotter.save(plt, "loss_history_val")

    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(metrics["loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plotter.save(plt, "loss_history_train_val")

    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(metrics["loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.plot(metrics["recon"], label="Reconstruction Loss")
    plt.plot(metrics["pred"], label="Prediction Loss")
    # plt.plot(metrics['linearity'], label='Norm Linearity Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plotter.save(plt, "loss_history")

    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(
        [val / np.max(metrics["loss"]) for val in metrics["loss"]], label="Train Loss"
    )
    plt.plot(
        [val / np.max(metrics["val_loss"]) for val in metrics["val_loss"]],
        label="Validation Loss",
    )
    plt.plot(
        [val / np.max(metrics["recon"]) for val in metrics["recon"]],
        label="Reconstruction Loss",
    )
    plt.plot(
        [val / np.max(metrics["pred"]) for val in metrics["pred"]],
        label="Prediction Loss",
    )
    plt.plot(
        [val / np.max(metrics["linearity"]) for val in metrics["linearity"]],
        label="Linearity Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Loss")
    plt.legend()
    plt.tight_layout()
    plotter.save(plt, "loss_history_normalized")


def plot_dynamics(dynamics, plotter, save_fn_append="initial"):
    plt.figure(figsize=(COLUMN, COLUMN), dpi=300)
    plt.imshow(np.abs(dynamics), cmap="plasma")
    plt.colorbar()
    plt.grid(False)
    plotter.save(plt, f"linear_dynamics_{save_fn_append}")


def plot_pred_step_eval(mse, std, plotter, save_fn_append="train"):
    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.errorbar(range(1, 1 + len(mse)), mse, 2 * np.array(std))
    plt.xlabel("Prediction Step")
    plt.ylabel("MSE")
    plotter.save(plt, f"pred_step_eval_{save_fn_append}")


def plot_pred_step_comp_eval(
    mse, std, per_mse, per_std, plotter, save_fn_append="train"
):
    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.errorbar(range(1, 1 + len(mse)), mse, 2 * np.array(std), label="Prediction")
    # plt.errorbar(range(1,1+len(per_mse)),per_mse,2*np.array(per_std), label='Persistence')
    plt.plot(range(1, 1 + len(per_mse)), per_mse, label="Persistence")
    plt.xlabel("Prediction Step")
    plt.ylabel("MSE")
    plt.legend()
    plotter.save(plt, f"pred_step_eval_per_{save_fn_append}")


def plot_pred_step_model_comp_eval(
    base_mse, base_std, siae_mse, siae_std, per_mse, per_std, plotter, save_fn_append=""
):
    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.errorbar(
        range(1, 1 + len(base_mse)), base_mse, 2 * np.array(base_std), label="Baseline"
    )
    plt.errorbar(
        range(1, 1 + len(siae_mse)),
        siae_mse,
        2 * np.array(siae_std),
        label="SIAE",
        c="r",
        ls="-",
    )
    # plt.errorbar(range(1,1+len(per_mse)),per_mse,2*np.array(per_std), label='Persistence')
    plt.plot(range(1, 1 + len(per_mse)), per_mse, label="Persistence")
    plt.xlabel("Prediction Step")
    plt.ylabel("MSE")
    plt.legend()
    plotter.save(plt, f"pred_step_eval_model_comp{save_fn_append}")


def plot_dynamics_spectral_norm_series(checkpoint_dynamics, plotter):
    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    spectral_norms = []
    for dyn in checkpoint_dynamics.values():
        spectral_norms.append(utils.spectral_norm(torch.tensor(dyn)))
    plt.plot(list(checkpoint_dynamics.keys()), spectral_norms)
    plt.xlabel("Epoch")
    plt.ylabel("Spectral Norm")
    plotter.save(plt, "spectral_norm_dynamics")


def plot_model_spectral_norm_series(checkpoint_models, plotter):
    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    encoder_norms_ts = []
    dynamics_norms = []
    decoder_norms_ts = []
    for model in checkpoint_models.values():
        encoder_norms = []
        for name, param in model.encoder.named_parameters():
            if "weight" in name:
                encoder_norms.append(torch.linalg.norm(param, ord=2).item())
        encoder_norms_ts.append(encoder_norms)
        dynamics_norms.append(
            torch.linalg.norm(model.linear_dynamics.weight, ord=2).item()
        )
        decoder_norms = []
        for name, param in model.decoder.named_parameters():
            if "weight" in name:
                decoder_norms.append(torch.linalg.norm(param, ord=2).item())
        decoder_norms_ts.append(decoder_norms)
    encoder_norms_ts = np.array(encoder_norms_ts)
    decoder_norms_ts = np.array(decoder_norms_ts)
    encoder_norms_ts = np.swapaxes(encoder_norms_ts, 0, 1)
    decoder_norms_ts = np.swapaxes(decoder_norms_ts, 0, 1)

    for ts in encoder_norms_ts:
        plt.plot(list(checkpoint_models.keys()), ts, c="k", ls="-", alpha=0.7)
    for ts in decoder_norms_ts:
        plt.plot(list(checkpoint_models.keys()), ts, c="k", ls="--", alpha=0.7)
    plt.plot(list(checkpoint_models.keys()), dynamics_norms, c="r", ls="-")
    plt.plot([], [], label="Encoder", c="k", ls="-", alpha=0.7)
    plt.plot([], [], label="Decoder", c="k", ls="--", alpha=0.7)
    plt.plot([], [], label="Dynamics", c="r", ls="-")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Spectral Norm")
    plotter.save(plt, "spectral_norm_all_layers")

    for ts in encoder_norms_ts:
        plt.plot(
            list(checkpoint_models.keys()), ts / np.max(ts), c="k", ls="-", alpha=0.7
        )
    for ts in decoder_norms_ts:
        plt.plot(
            list(checkpoint_models.keys()), ts / np.max(ts), c="k", ls="--", alpha=0.7
        )
    plt.plot(
        list(checkpoint_models.keys()),
        np.array(dynamics_norms) / np.max(dynamics_norms),
        c="r",
        ls="-",
    )
    plt.plot([], [], label="Encoder", c="k", ls="-", alpha=0.7)
    plt.plot([], [], label="Decoder", c="k", ls="--", alpha=0.7)
    plt.plot([], [], label="Dynamics", c="r", ls="-")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Spectral Norm")
    plotter.save(plt, "spectral_norm_normalized_all_layers")


def plot_spatial_psd_comp(
    data, preds, errors, orig_shape, plotter, sample_freq=1, save_fn_append="train"
):
    # # Plot the mean and standard deviation
    # plt.errorbar(radius, mean_spectrum, yerr=std_spectrum, label='Mean Spectrum with Std Dev')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Radial Frequency')
    # plt.ylabel('Power Spectrum')
    # plt.title('Radially Averaged Power Spectrum')
    # plt.legend()
    # plt.show()

    # return

    f, psd = utils.avg_spatial_psd(data, orig_shape, sample_freq=1)
    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(f[1:], psd[1:])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency")
    plt.ylabel("Power")  # sigma squared
    # plt.ylim( (10**-4,10**0) )
    plotter.save(plt, "data_spatial_psd")

    pred_f, pred_psd = utils.avg_spatial_psd(preds, orig_shape, sample_freq=1)
    err_f, err_psd = utils.avg_spatial_psd(errors, orig_shape, sample_freq=1)
    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(f[1:], psd[1:], label="Data")
    plt.plot(pred_f[1:], pred_psd[1:], label="Prediction", c="r", ls="--")
    plt.plot(err_f[1:], err_psd[1:], label="Error", c="b", ls="-.")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency")
    plt.ylabel("Power")  # sigma squared
    plt.legend()
    plotter.save(plt, f"data_spatial_psd_comp_{save_fn_append}")


def plot_psd(data, plotter, sample_freq=1):
    f, psd = utils.avg_temporal_psd(data, sample_freq=1)
    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(f, psd)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency")
    plt.ylabel("Power")  # sigma squared
    plotter.save(plt, "data_psd")


def plot_psd_comp(data, dynamics, avg_coeffs, plotter, sample_freq=1, eig_exponent=1):
    f, psd = utils.avg_temporal_psd(data, sample_freq=1)
    eigen_f = utils.eigen_freq_from_dynamics(dynamics)
    eigen_vals, eigen_vecs = np.linalg.eig(dynamics)
    eigen_coeffs = np.abs(np.dot(avg_coeffs, eigen_vecs)) ** 2
    eigen_coeffs *= np.exp(eig_exponent * np.abs(eigen_vals))

    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(f, psd)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency")
    plt.ylabel("Power")  # sigma squared
    plotter.save(plt, "data_psd")

    # eigen_s = utils.eigen_stability_from_dynamics(dynamics)
    f /= np.max(f)
    eigen_f /= np.pi
    eig_y_coord = []
    for eig_f in eigen_f:
        f_index = f.searchsorted(eig_f) - 1
        eig_mag = psd[f_index]
        eig_y_coord.append(eig_mag)
    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(f, psd)
    plt.scatter(eigen_f, eig_y_coord, s=8, c="r", marker="x", zorder=10)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency")
    plt.ylabel("Power")  # sigma squared
    plotter.save(plt, "data_psd_eigenf_comp")

    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(f, psd / np.max(psd))
    plt.scatter(
        eigen_f, eigen_coeffs / np.max(eigen_coeffs), s=8, c="r", marker="x", zorder=10
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency")
    plt.ylabel("Power")  # sigma squared
    plotter.save(plt, "data_psd_eigenf_comp_coeffs")


def plot_dynamics_psd_comp(
    model, data, pred_steps, plotter, save_fn, fft_stride=1, power=None
):
    # data = data[:pred_steps]
    with torch.no_grad():
        target_fft = torch.zeros_like(torch.abs(torch.fft.rfft(data[:, 0])))
    for i in range(0, data.shape[1], fft_stride):
        fft = torch.abs(torch.fft.rfft(data[:, i]))
        target_fft += fft

    # dyn_fft = utils.encoded_psd(data, model, pred_steps, fft_stride)
    dyn_fft = utils.encoded_psd(data, model, len(data), fft_stride)

    if power is None:
        target_fft = target_fft
    else:
        target_fft = torch.pow(
            torch.arange(1, len(dyn_fft) + 1, dtype=torch.float64), -2
        )

    dyn_fft /= torch.sum(dyn_fft[1:])
    target_fft /= torch.sum(target_fft[1:])

    error = target_fft - dyn_fft

    # plt.plot([0, len(dyn_fft)], [0, 0], c='k', ls='--')
    plt.plot(target_fft[1:], c="k", ls="-", label="Data")
    plt.plot(dyn_fft[1:], c="r", ls="-", label="Dynamics")
    # plt.plot(torch.abs(error), c='k', ls='--', alpha=0.5, label='Abs Error')
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Power")
    plt.xlabel("Frequency")
    # plt.xscale('log')
    # plt.yscale('log')
    plotter.save(plt, save_fn)


def plot_recon_psd_comp(
    model, data, pred_steps, plotter, save_fn, fft_stride=1, power=None
):
    data = data[:pred_steps]
    with torch.no_grad():
        target_fft = torch.zeros_like(torch.abs(torch.fft.rfft(data[:, 0])))
    for i in range(0, data.shape[1], fft_stride):
        fft = torch.abs(torch.fft.rfft(data[:, i]))
        target_fft += fft

    recon_fft = utils.recon_psd(data, model, pred_steps, fft_stride)
    dyn_fft = utils.encoded_psd(data, model, pred_steps, fft_stride)

    if power is None:
        target_fft = target_fft
    else:
        target_fft = torch.pow(
            torch.arange(1, len(recon_fft) + 1, dtype=torch.float64), -2
        )

    recon_fft /= torch.sum(recon_fft[1:])
    dyn_fft /= torch.sum(dyn_fft[1:])
    target_fft /= torch.sum(target_fft[1:])

    error = target_fft - recon_fft

    # plt.plot([0, len(recon_fft)], [0, 0], c='k', ls='--')
    plt.plot(target_fft[1:], c="k", ls="-", label="Data")
    plt.plot(recon_fft[1:], c="r", ls="-", label="Recon")
    # plt.plot(torch.abs(error), c='k', ls='--', alpha=0.5, label='Abs Error')
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Power")
    plt.xlabel("Frequency")
    # plt.xscale('log')
    # plt.yscale('log')
    plotter.save(plt, save_fn)

    # plt.plot([0, len(recon_fft)], [0, 0], c='k', ls='--')
    plt.plot(target_fft[1:], c="k", ls="-", label="Data")
    plt.plot(recon_fft[1:], c="r", ls="-", label="Recon")
    plt.plot(dyn_fft[1:], c="b", ls="-", label="Dynamics")
    # plt.plot(torch.abs(error), c='k', ls='--', alpha=0.5, label='Abs Error')
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Power")
    plt.xlabel("Frequency")
    # plt.xscale('log')
    # plt.yscale('log')
    plotter.save(plt, "dynamics_recon_psd_comp")


def spectral_bias_plot(checkpoint_models, data, orig_shape, plotter, pred_step=1):
    data_f, data_psd = utils.avg_spatial_psd(data, orig_shape, sample_freq=1)

    error_psds = []
    mses = []
    stds = []
    for epoch, model in checkpoint_models.items():
        # TODO make sure this is a one step prediction for pred_step = 1
        preds, errors = utils.evaluated_preds(model, data, max_steps=pred_step)
        mse, std = utils.evaluated_preds_metrics(errors)
        mses.append(mse[-1])
        stds.append(std[-1])
        err_f, err_psd = utils.avg_spatial_psd(errors[-1], orig_shape, sample_freq=1)
        error_psds.append(err_psd)
    error_psds = np.array(error_psds)

    epochs = checkpoint_models.keys()
    # relative_error_spectral = np.abs(1-(error_psds/data_psd))
    relative_error_spectral = np.abs(error_psds / data_psd)

    xs = range(len(data_f))
    ys = range(len(epochs))
    vmin, vmax = np.min(error_psds), np.max(error_psds)
    plt.pcolormesh(
        xs, ys, error_psds, shading="nearest", cmap="magma_r", vmin=vmin, vmax=vmax
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Epoch")
    plt.grid(False)

    cbar = plt.colorbar()
    cbar.set_label("Error")
    plotter.save(plt, f"spectral_error")

    log_error_psds = np.log(error_psds)
    vmin, vmax = np.min(log_error_psds), np.max(log_error_psds)
    plt.pcolormesh(
        xs, ys, log_error_psds, shading="nearest", cmap="magma_r", vmin=vmin, vmax=vmax
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Epoch")
    plt.grid(False)
    cbar = plt.colorbar()
    cbar.set_label("Log Error")
    plotter.save(plt, f"spectral_error_log")

    xs = range(len(data_f))
    ys = range(len(epochs))
    plt.pcolor(
        xs,
        ys,
        relative_error_spectral,
        shading="nearest",
        cmap="magma_r",
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Epoch")
    plt.grid(False)
    cbar = plt.colorbar()
    cbar.set_label("Relative Error")
    plotter.save(plt, f"spectral_relative_error")

    xs = np.log(data_f)
    ys = range(len(epochs))
    plt.pcolor(
        xs,
        ys,
        relative_error_spectral,
        shading="nearest",
        cmap="magma_r",
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Log Frequency (Hz)")
    plt.ylabel("Epoch")
    plt.grid(False)
    cbar = plt.colorbar()
    cbar.set_label("Relative Error")
    plotter.save(plt, f"spectral_relative_error_log")

    # plt.imshow(error_psds)
    # plt.ylabel('Epochs')
    # plt.xlabel('Frequency')
    # plt.colorbar()
    # plotter.save(plt, 'spectral_error')

    # plt.imshow(1-error_psds / data_psd)
    # plt.ylabel('Epochs')
    # plt.xlabel('Frequency')
    # plt.colorbar()
    # plotter.save(plt, 'spectral_error_relative')

    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.errorbar(epochs, mses, 2 * np.array(stds))
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    # plt.legend()
    plotter.save(plt, "pred_step_eval_over_training")


# def simple_spectral_plot(spectral_img, fs, epochs, plotter, save_fn, static_lims=True):
#     xs = range(len(fs))
#     ys = range(len(epochs))
#     if static_lims:
#         vmin, vmax = 0, 1
#     else:
#         vmin, vmax = np.min(spectral_img), np.max(spectral_img)
#     plt.figure(figsize=(COLUMN,COLUMN), dpi=300)
#     plt.pcolor(xs, ys, spectral_img,
#                 shading='nearest', cmap='magma_r', vmin=vmin, vmax=vmax)
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Epoch')
#     plt.grid(False)
#     cbar = plt.colorbar()
#     cbar.set_label('Relative Error')
#     plotter.save(plt, f'spectral_{save_fn}')

# def spectral_plot(psds, baseline_psd, eval_freqs, variable, variable_name, plotter,
#                   save_fn='', semi_log_flag=False, one_shot=True):
#     '''
#     Generate plot of spectral bias from passed in data
#     '''
#     plt.figure(figsize=(COLUMN,COLUMN), dpi=300)
#     relative_psd = torch.abs(1 - psds / baseline_psd)
#     if not one_shot:
#         relative_psd = torch.mean(relative_psd, dim=1)
#     if semi_log_flag:
#         xs = range(len(eval_freqs))
#         ys = range(len(variable))
#         plt.pcolor(xs, ys, relative_psd,
#                    shading='nearest', cmap='magma_r', vmin=0, vmax=1)
#         plt.xticks(xs, eval_freqs.tolist())
#         plt.yticks(ys, variable)
#     else:
#         xs = range(len(eval_freqs))
#         plt.pcolor(xs, variable, relative_psd,
#                    shading='nearest', cmap='magma_r', vmin=0, vmax=1)
#         plt.xticks(xs, eval_freqs.tolist())
#     # plt.title('')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel(variable_name)
#     plt.grid(False)
#     cbar = plt.colorbar()
#     cbar.set_label('Relative Error')
#     plotter.save(plt, f'spectral_{save_fn}')


def plot_frame(
    frame_vec,
    orig_shape,
    plotter,
    vmin=0,
    vmax=1,
    autoscale=True,
    cmap="binary",
    save_fn="frame",
):
    frame_vec = np.array(frame_vec)
    if autoscale:
        vmin, vmax = np.min(frame_vec), np.max(frame_vec)
    image = frame_vec.reshape(orig_shape)
    # x = range(image.shape[0])
    # y = range(image.shape[1])
    plt.figure(figsize=(COLUMN, COLUMN), dpi=300)
    plt.imshow(
        image,
        vmin=vmin,
        vmax=vmax,
        cmap="binary",
        aspect="equal",
        origin="lower",
        interpolation="none",
    )
    plt.colorbar(fraction=0.046, pad=0.04)
    plotter.save(plt, save_fn)


def plot_pred_comp(pred, truth, orig_shape, plotter, save_fn, recon=False):
    error = truth - pred
    error = np.reshape(error, orig_shape)
    truth = np.reshape(truth, orig_shape)
    pred = np.reshape(pred, orig_shape)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, COLUMN))
    vmin = np.min([np.min(pred), np.min(truth), np.min(error)])
    vmax = np.max([np.max(pred), np.max(truth), np.max(error)])
    # cmap = 'binary'
    cmap = sns.color_palette("icefire", as_cmap=True)
    axes[0].imshow(
        pred,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="equal",
        origin="lower",
        interpolation="none",
    )
    if recon:
        axes[0].set_title("Reconstruction")
    else:
        axes[0].set_title("Prediction")
    # axes[1].plot(x, y2, label='cos(x)', color='green')
    axes[1].imshow(
        truth,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="equal",
        origin="lower",
        interpolation="none",
    )
    axes[1].set_title("Truth")
    # axes[2].plot(x, y3, label='tan(x)', color='red')
    cbarim = axes[2].imshow(
        error,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="equal",
        origin="lower",
        interpolation="none",
    )
    axes[2].set_title("Error")
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    # axes[0].set_ylabel('y-axis label')
    cbar = fig.colorbar(
        cbarim, ax=axes, orientation="vertical", fraction=0.046, pad=0.04
    )
    # cbar.set_label('')
    # plt.tight_layout()
    plotter.save(plt, save_fn)


def plot_ssim(ssim_vals, plotter, save_fn=""):
    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(np.mean(ssim_vals, axis=1))
    plt.ylabel("SSIM")
    plt.xlabel("Prediction Step")
    plotter.save(plt, save_fn)


def plot_ssim_comp(ssim_vals, per_ssim_vals, plotter, save_fn=""):
    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.plot(np.mean(ssim_vals, axis=1), label="Prediction")
    plt.plot(np.mean(per_ssim_vals, axis=1), label="Persistence")
    plt.legend()
    plt.ylabel("SSIM")
    plt.xlabel("Prediction Step")
    plotter.save(plt, save_fn)


def plot_ssim_model_comp(
    ssim_vals_base, ssim_vals_siae, ssim_per, plotter, save_fn_append=""
):
    plt.figure(figsize=(COLUMN, SMALLER), dpi=300)
    plt.errorbar(
        range(1, 1 + len(ssim_vals_base)),
        np.mean(ssim_vals_base, axis=1),
        label="Baseline",
    )
    plt.errorbar(
        range(1, 1 + len(ssim_vals_siae)),
        np.mean(ssim_vals_siae, axis=1),
        label="SIAE",
        c="r",
        ls="-",
    )
    # plt.errorbar(range(1,1+len(per_mse)),per_mse,2*np.array(per_std), label='Persistence')
    plt.plot(
        range(1, 1 + len(ssim_per)), np.mean(ssim_per, axis=1), label="Persistence"
    )
    plt.ylabel("SSIM")
    plt.xlabel("Prediction Step")
    plt.legend()
    plotter.save(plt, f"ssim_model_comp{save_fn_append}")


def probe_reg_function(
    plotter,
    disc_lims=[[-1.1, 1.1], [-1.1, 1.1]],
    cont_lims=[[-4.8, 1.5], [-3.15, 3.15]],
    real_grid_steps=110,
    imag_grid_steps=110,
    reg_fn="standard",
    f_scale=0.001,
    s_scale=0.01,
    f_exp=2.0,
    s_exp=1.0,
    annotate=True,
    cmap="plasma",
    annotation_color="w",
    normalize=True,
):
    # TODO move to combo of utils to generate data and plot utils to plot?
    # or just to utils no reason to get complicated.
    # TODO also add continuous version, np.log(???)
    # TODO add unit circle to plots based on flag on input
    if reg_fn == "standard":
        reg = siae.StandardSIAERegularization(
            f_scale=f_scale, s_scale=s_scale, f_exp=f_exp, s_exp=s_exp
        )
    elif reg_fn == "coherent":
        reg = siae.CoherentSIAERegularization(
            f_scale=f_scale, s_scale=s_scale, f_exp=f_exp, s_exp=s_exp
        )
    else:
        print("invalid reg mode")
        return
    x_real = torch.linspace(disc_lims[0][0], disc_lims[0][1], real_grid_steps)
    y_imag = torch.linspace(disc_lims[1][0], disc_lims[1][1], imag_grid_steps)
    loss_grid = torch.zeros((x_real.shape[0], y_imag.shape[0]))
    f_loss_grid = torch.zeros_like((loss_grid))
    s_loss_grid = torch.zeros_like((loss_grid))

    for j in range(len(x_real)):
        for k in range(len(y_imag)):
            x = x_real[j]
            y = y_imag[k]
            eig = x + 1j * y
            cont_eig = torch.log(eig)
            f_loss, s_loss = reg.calc_losses(cont_eig)
            f_reg = f_scale * f_loss
            s_reg = s_scale * s_loss
            reg_loss = f_reg + s_reg
            loss_grid[k][j] = reg_loss
            f_loss_grid[k][j] = f_reg
            s_loss_grid[k][j] = s_reg

    if normalize:
        min_val = torch.min(loss_grid)
        max_val = torch.max(loss_grid)
        loss_grid -= min_val
        loss_grid /= max_val
        f_loss_grid -= min_val
        f_loss_grid /= max_val
        s_loss_grid -= min_val
        s_loss_grid /= max_val

    cont_x_real = torch.linspace(cont_lims[0][0], cont_lims[0][1], real_grid_steps)
    cont_y_imag = torch.linspace(cont_lims[1][0], cont_lims[1][1], imag_grid_steps)
    log_loss_grid = torch.zeros((cont_x_real.shape[0], cont_y_imag.shape[0]))
    log_f_loss_grid = torch.zeros_like((loss_grid))
    log_s_loss_grid = torch.zeros_like((loss_grid))
    for j in range(len(cont_x_real)):
        for k in range(len(cont_y_imag)):
            x = cont_x_real[j]
            y = cont_y_imag[k]
            eig = x + 1j * y
            f_loss, s_loss = reg.calc_losses(eig)
            f_reg = f_scale * f_loss
            s_reg = s_scale * s_loss
            reg_loss = f_reg + s_reg
            log_loss_grid[k][j] = reg_loss
            log_f_loss_grid[k][j] = f_reg
            log_s_loss_grid[k][j] = s_reg

    if normalize:
        min_val = torch.min(log_loss_grid)
        max_val = torch.max(log_loss_grid)
        log_loss_grid -= min_val
        log_loss_grid /= max_val
        log_f_loss_grid -= min_val
        log_f_loss_grid /= max_val
        log_s_loss_grid -= min_val
        log_s_loss_grid /= max_val

    fig, ax = plt.subplots(1, figsize=(COLUMN, COLUMN), dpi=300)
    unit_circle = plt.Circle(
        (0.0, 0.0),
        1.0,
        color=annotation_color,
        fill=False,
        linestyle="-",
        linewidth=1.0,
        zorder=5,
    )
    plt.pcolormesh(x_real, y_imag, loss_grid, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_ticks([0.0, 0.5, 1.0])
    plt.ylabel("Imag")
    plt.xlabel("Real")
    ax.set_aspect("equal")
    ax.add_artist(unit_circle)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    # ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.text(
        0,
        0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c=annotation_color,
    )
    plotter.save(plt, f"total_loss")

    fig, ax = plt.subplots(1, figsize=(COLUMN, COLUMN), dpi=300)
    unit_circle = plt.Circle(
        (0.0, 0.0),
        1.0,
        color=annotation_color,
        fill=False,
        linestyle="-",
        linewidth=1.0,
        zorder=5,
    )
    plt.pcolormesh(x_real, y_imag, loss_grid, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_ticks([0.0, 0.5, 1.0])
    plt.ylabel("Imag")
    plt.xlabel("Real")
    ax.set_aspect("equal")
    ax.add_artist(unit_circle)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    # ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.text(
        0,
        0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c=annotation_color,
    )
    plotter.save(plt, f"total_loss")

    fig, ax = plt.subplots(1, figsize=(COLUMN, COLUMN), dpi=300)
    unit_circle = plt.Circle(
        (0.0, 0.0),
        1.0,
        color=annotation_color,
        fill=False,
        linestyle="-",
        linewidth=1.0,
        zorder=5,
    )
    plt.pcolormesh(x_real, y_imag, f_loss_grid, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_ticks([0.0, 0.5, 1.0])
    plt.ylabel("Imag")
    plt.xlabel("Real")
    ax.set_aspect("equal")
    ax.add_artist(unit_circle)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    # ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.text(
        0,
        0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c=annotation_color,
    )
    plotter.save(plt, f"f_loss")

    fig, ax = plt.subplots(1, figsize=(COLUMN, COLUMN), dpi=300)
    unit_circle = plt.Circle(
        (0.0, 0.0),
        1.0,
        color=annotation_color,
        fill=False,
        linestyle="-",
        linewidth=1.0,
        zorder=5,
    )
    plt.pcolormesh(x_real, y_imag, s_loss_grid, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_ticks([0.0, 0.5, 1.0])
    plt.ylabel("Imag")
    plt.xlabel("Real")
    ax.set_aspect("equal")
    ax.add_artist(unit_circle)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    # ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.text(
        0,
        0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c=annotation_color,
    )
    plotter.save(plt, f"s_loss")

    # plot the continuous plots
    fig, ax = plt.subplots(1, figsize=(COLUMN, COLUMN), dpi=300)
    plt.pcolormesh(cont_x_real, cont_y_imag, log_loss_grid, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_ticks([0.0, 0.5, 1.0])
    plt.ylabel("Imag")
    plt.xlabel("Real")
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_yticks([-np.pi, 0, np.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
    ax.text(
        0.0,
        0.0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c=annotation_color,
    )
    ax.axvline(x=0, c=annotation_color, linestyle="-")
    plotter.save(plt, f"total_loss_cont")

    fig, ax = plt.subplots(1, figsize=(COLUMN, COLUMN), dpi=300)
    plt.pcolormesh(cont_x_real, cont_y_imag, log_f_loss_grid, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_ticks([0.0, 0.5, 1.0])
    plt.ylabel("Imag")
    plt.xlabel("Real")
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_yticks([-np.pi, 0, np.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
    ax.text(
        0.0,
        0.0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c=annotation_color,
    )
    ax.axvline(x=0, c=annotation_color, linestyle="-")
    plotter.save(plt, f"f_loss_cont")

    fig, ax = plt.subplots(1, figsize=(COLUMN, COLUMN), dpi=300)
    plt.pcolormesh(cont_x_real, cont_y_imag, log_s_loss_grid, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_ticks([0.0, 0.5, 1.0])
    plt.ylabel("Imag")
    plt.xlabel("Real")
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_yticks([-np.pi, 0, np.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
    ax.text(
        0.0,
        0.0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c=annotation_color,
    )
    ax.axvline(x=0, c=annotation_color, linestyle="-")
    plotter.save(plt, f"s_loss_cont")

    # fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.6), sharex=True, sharey=True)

    # # Plot loss_grid
    # im1 = axs[0].pcolormesh(loss_grid, cmap='viridis')
    # axs[0].set_title('Total Loss')
    # axs[0].set_aspect('equal')
    # cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', shrink=0.6, pad=0.05)
    # cbar1.set_label('Colorbar Label')

    # # Plot f_loss_grid
    # im2 = axs[1].pcolormesh(f_loss_grid, cmap='viridis')
    # axs[1].set_title('Frequency Loss')
    # axs[1].set_aspect('equal')
    # cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', shrink=0.6, pad=0.05)
    # cbar2.set_label('Colorbar Label')

    # # Plot s_loss_grid
    # im3 = axs[2].pcolormesh(s_loss_grid, cmap='viridis')
    # axs[2].set_title('Stability Loss')
    # axs[2].set_aspect('equal')
    # cbar3 = fig.colorbar(im3, ax=axs[2], orientation='vertical', shrink=0.6, pad=0.05)
    # cbar3.set_label('Colorbar Label')

    # # Show the plot
    # plt.tight_layout()
    # plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 1.6), sharex=True, sharey=True)

    vmin = min(loss_grid.min(), f_loss_grid.min(), s_loss_grid.min())
    vmax = max(loss_grid.max(), f_loss_grid.max(), s_loss_grid.max())

    unit_circle = plt.Circle(
        (0.0, 0.0),
        1.0,
        color=annotation_color,
        fill=False,
        linestyle="-",
        linewidth=1.0,
        zorder=5,
    )
    im1 = axs[0].pcolormesh(x_real, y_imag, loss_grid, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].add_artist(unit_circle)
    axs[0].text(
        0,
        0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c=annotation_color,
    )
    axs[0].set_xticks([-1, 0, 1])
    axs[0].set_yticks([-1, 0, 1])
    # axs[0].xaxis.set_major_locator(plt.MaxNLocator(3))
    # axs[0].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[0].set_title("Total")
    axs[0].set_aspect("equal")

    unit_circle = plt.Circle(
        (0.0, 0.0),
        1.0,
        color=annotation_color,
        fill=False,
        linestyle="-",
        linewidth=1.0,
        zorder=5,
    )
    im2 = axs[1].pcolormesh(
        x_real, y_imag, f_loss_grid, cmap=cmap, vmin=vmin, vmax=vmax
    )
    axs[1].add_artist(unit_circle)
    axs[1].text(
        0,
        0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c=annotation_color,
    )
    axs[1].set_xticks([-1, 0, 1])
    # axs[1].xaxis.set_major_locator(plt.MaxNLocator(3))
    # axs[1].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[1].set_title("Frequency")
    axs[1].set_aspect("equal")

    unit_circle = plt.Circle(
        (0.0, 0.0),
        1.0,
        color=annotation_color,
        fill=False,
        linestyle="-",
        linewidth=1.0,
        zorder=5,
    )
    im3 = axs[2].pcolormesh(
        x_real, y_imag, s_loss_grid, cmap=cmap, vmin=vmin, vmax=vmax
    )
    axs[2].add_artist(unit_circle)
    axs[2].text(
        0,
        0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c=annotation_color,
    )
    axs[2].set_xticks([-1, 0, 1])
    # axs[2].xaxis.set_major_locator(plt.MaxNLocator(3))
    # axs[2].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[2].set_title("Stability")
    axs[2].set_aspect("equal")

    cbar = fig.colorbar(im3, ax=axs, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 0.5, 1.0])
    plotter.save(plt, f"all_losses")

    # Plot continuous loss_grid
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 1.6), sharex=True, sharey=True)

    vmin = min(log_loss_grid.min(), log_f_loss_grid.min(), log_s_loss_grid.min())
    vmax = max(log_loss_grid.max(), log_f_loss_grid.max(), log_s_loss_grid.max())

    im1 = axs[0].pcolormesh(
        cont_x_real, cont_y_imag, log_loss_grid, cmap=cmap, vmin=vmin, vmax=vmax
    )
    axs[0].set_title("Total")
    axs[0].set_aspect("equal")
    axs[0].text(
        0.0,
        0.0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c=annotation_color,
    )
    axs[0].xaxis.set_major_locator(plt.MaxNLocator(3))
    # axs[0].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[0].set_yticks([-np.pi, 0, np.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
    axs[0].axvline(x=0, c=annotation_color, linestyle="-")

    im2 = axs[1].pcolormesh(
        cont_x_real, cont_y_imag, log_f_loss_grid, cmap=cmap, vmin=vmin, vmax=vmax
    )
    axs[1].set_title("Frequency")
    axs[1].set_aspect("equal")
    axs[1].text(
        0.0,
        0.0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c=annotation_color,
    )
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(3))
    # axs[1].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[1].axvline(x=0, c=annotation_color, linestyle="-")

    im3 = axs[2].pcolormesh(
        cont_x_real, cont_y_imag, log_s_loss_grid, cmap=cmap, vmin=vmin, vmax=vmax
    )
    axs[2].set_title("Stability")
    axs[2].set_aspect("equal")
    axs[2].text(
        0.0,
        0.0,
        "+",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        c=annotation_color,
    )
    axs[2].xaxis.set_major_locator(plt.MaxNLocator(3))
    # axs[2].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[2].axvline(x=0, c=annotation_color, linestyle="-")

    cbar = fig.colorbar(im3, ax=axs, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 0.5, 1.0])

    plotter.save(plt, f"all_losses_cont")


def comp_saved_model_preds(
    layers, latent_dim, dir_path1, dir_path2, model_name1, model_name2
):
    # load models from
    pass


def plot_frames():
    """plot for instance 5 frames showing a sample time series with still images"""
    pass


def anim_frames(frames, fps):
    """animate data and save as gif and video"""
    pass


# # TODO make this work
# def make_animation(frames,
#                    shape,
#                    scale=1,
#                    vmin=-1,
#                    vmax=1,
#                    fps=None,
#                    autoscale=False,
#                    labels=None,
#                    cmap='jet',
#                    nan_mask=True,
#                    dir_name='sample',
#                    file_name='sample'):
#     ''' make an animation out of a series of frames using given fps (default~14.3) '''
#     frames = np.array(frames).real
#     if nan_mask:
#         frames[frames==0] = np.nan
#     if autoscale:
#         vmin, vmax = frame_tools.get_nan_min_max(frames)
#     # print(vmin,vmax)
#     shape = frames[0].shape
#     x_lin = np.linspace(-scale, scale, shape[1]+1)
#     y_lin = np.linspace(-scale, scale, shape[0]+1)
#     x_grid, y_grid = np.meshgrid(x_lin, y_lin)
#     fig, ax = plt.subplots(1, figsize=(3.0,2.5), dpi=100) #, gridspec_kw={"width_ratios":[18,1]}

#     norm = mpl.colors.Normalize(vmin, vmax)
#     scalarmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
#     scalarmap.set_array([])
#     cbar = fig.colorbar(scalarmap)
#     cbar.formatter.set_powerlimits((-1, 1))

#     ax.axis('off')
#     fig.tight_layout()

#     plot_frames = []
#     for i, frame in enumerate(frames):
#         if labels == None:
#             title = ax.text(0.5, 1.01, i,
#                 horizontalalignment='center',
#                 verticalalignment='bottom')
#         else:
#             title = ax.text(0.5, 1.01, labels[i],
#                 horizontalalignment='center',
#                 verticalalignment='bottom')
#         plot = ax.pcolor(x_grid, y_grid[::-1,:], frame,
#             cmap=cmap, vmin=vmin, vmax=vmax)
#         # fig.tight_layout() makes plot off-centered :/
#         # fig.colorbar(plot) plots new color bar each frame ahahha
#         ax.set_aspect('equal')
#         # ax.invert_yaxis()
#         plot_frames.append([plot, title])
#     if fps is not None:
#         interval = 1000/fps
#     else:
#         interval = 70 # default value
#     ani = animation.ArtistAnimation(
#         fig,
#         plot_frames,
#         interval=interval,
#         blit=True,
#         repeat=True)
#     show_anim(plt, ani, fps,
#         dir_name=dir_name, file_name=file_name)


def plot_eigenvecs(model, orig_shape, plotter):
    eigen_vals, eigen_vecs = torch.linalg.eig(model.linear_dynamics.weight.detach())
    for i in range(len(eigen_vecs)):
        with torch.no_grad():
            mode = model.decoder(torch.abs(eigen_vecs[i]))
        mode_im = torch.reshape(torch.abs(mode), orig_shape)
        plt.imshow(mode_im, cmap="plasma")
        plt.colorbar()
        plt.title(f"Mode {i}, eig {eigen_vals[i]}")
        plotter.save(plt, f"eigen_vec_{i}")


def plot_results(
    save_dir,
    base_model,
    siae_model,
    train_data,
    data,
    max_steps,
    reg_steps,
    orig_shape,
    pred_steps,
):
    """
    plot the main results
        RMS error over time
        Encoded RMS error over time
        SSIM over time
        Operator Spectrum
        Eigenfrequencies comp?
    """
    plotter = Plotter(save_dir)
    base_eval = experiments.eval(
        base_model, data, train_data, max_steps=max_steps, metrics=False
    )
    base_preds, base_mses, base_stds, base_ssim = base_eval
    siae_eval = experiments.eval(
        siae_model, data, train_data, max_steps=max_steps, metrics=False
    )
    siae_preds, siae_mses, siae_stds, siae_ssim = siae_eval

    plt.figure(figsize=(HALF, SMALLER), dpi=300)
    plt.plot(base_ssim, ls="-", c="k", label="Baseline")
    plt.plot(siae_ssim, ls="-", c="r", label="SIAE")
    # plt.legend()
    plt.ylabel("SSIM")
    plt.xlabel("Prediction Step")
    plotter.save(plt, "ssim_comp")

    plt.figure(figsize=(HALF, SMALLER), dpi=300)
    plt.plot(range(1, len(base_mses) + 1), base_mses, ls="-", c="k", label="Baseline")
    plt.fill_between(
        range(1, len(base_mses) + 1),
        base_mses - base_stds,
        base_mses + base_stds,
        color="k",
        alpha=0.3,
    )
    plt.plot(range(1, len(siae_mses) + 1), siae_mses, ls="-", c="r", label="SIAE")
    plt.fill_between(
        range(1, len(siae_mses) + 1),
        siae_mses - siae_stds,
        siae_mses + siae_stds,
        color="r",
        alpha=0.3,
    )
    # legend or not
    plt.legend()
    plt.ylabel("MSE")
    plt.xlabel("Prediction Step")
    plt.ylim(bottom=0.0)
    plotter.save(plt, "mse_comp")

    # TODO add latency as baseline here
    per_errors = utils.persistence_preds(train_data, max_steps=len(siae_mses))
    per_mse, per_std = utils.evaluated_preds_metrics(per_errors)
    plt.figure(figsize=(HALF, SMALLER), dpi=300)
    plt.plot(range(1, len(base_mses) + 1), base_mses, ls="-", c="k", label="Baseline")
    plt.fill_between(
        range(1, len(base_mses) + 1),
        base_mses - base_stds,
        base_mses + base_stds,
        color="k",
        alpha=0.3,
    )
    plt.plot(range(1, len(siae_mses) + 1), siae_mses, ls="-", c="r", label="SIAE")
    plt.fill_between(
        range(1, len(siae_mses) + 1),
        siae_mses - siae_stds,
        siae_mses + siae_stds,
        color="r",
        alpha=0.3,
    )
    plt.plot(
        range(1, len(per_mse) + 1),
        per_mse,
        ls="--",
        c="k",
        label="Persistence",
        alpha=0.3,
    )
    # plt.fill_between(range(1,len(per_mse)+1), per_mse-per_std, per_mse+per_std, color='g',alpha=0.3)
    # legend or not
    plt.legend()
    plt.ylabel("MSE")
    plt.xlabel("Prediction Step")
    plt.ylim(bottom=0.0)
    plotter.save(plt, "mse_comp_per")

    # TODO add trajectory over time here (how is that meaningful for 2d data?)

    plot_eigs_from_operator(
        siae_model.linear_dynamics.weight.detach().numpy(),
        plotter,
        model_name="siae_eigs",
    )
    plot_eigs_from_operator(
        base_model.linear_dynamics.weight.detach().numpy(),
        plotter,
        model_name="base_eigs",
    )

    plt.imshow(siae_model.linear_dynamics.weight.detach().numpy(), cmap="plasma")
    plt.colorbar()
    plotter.save(plt, "siae_dyn")

    plt.imshow(base_model.linear_dynamics.weight.detach().numpy(), cmap="plasma")
    plt.colorbar()
    plotter.save(plt, "base_dyn")

    fft_stride = 1
    # train_data = data
    with torch.no_grad():
        target_fft = torch.zeros_like(torch.abs(torch.fft.rfft(train_data[:, 0])))
    for i in range(0, train_data.shape[1], fft_stride):
        fft = torch.abs(torch.fft.rfft(train_data[:, i]))
        target_fft += fft

    # orig_target_fft = target_fft.copy()
    target_fft /= torch.sum(target_fft[1:])

    base_dyn_fft = utils.encoded_psd(
        train_data, base_model, len(train_data), fft_stride
    )
    siae_dyn_fft = utils.encoded_psd(
        train_data, siae_model, len(train_data), fft_stride
    )

    fs = np.linspace(0, 1, num=len(target_fft[1:]))
    plt.figure(figsize=(HALF, SMALLER), dpi=300)
    plt.plot(fs, target_fft[1:], c="k", ls="--", label="Data", alpha=0.5)
    plt.plot(fs, base_dyn_fft[1:], c="k", ls="-", label="Baseline")
    plt.plot(fs, siae_dyn_fft[1:], c="r", ls="-", label="SIAE")
    # plt.vlines(base_eigfs/np.pi, ymin=torch.min(base_dyn_fft[1:]), ymax=torch.max(base_dyn_fft[1:]), colors ='k', ls='-')
    # plt.vlines(siae_eigfs/np.pi, ymin=torch.min(siae_dyn_fft[1:]), ymax=torch.max(siae_dyn_fft[1:]), colors ='r', ls='-')
    # plt.plot(torch.abs(error), c='k', ls='--', alpha=0.5, label='Abs Error')
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Power")
    plt.xlabel("Frequency")
    plotter.save(plt, "psd_comp")

    time_steps = pred_steps
    cmap = sns.color_palette("icefire", as_cmap=True)
    fig, axs = plt.subplots(3, 3, figsize=(1.25 * SMALLER, SMALLER), dpi=300)
    global_vmin = torch.min(data)
    global_vmax = torch.max(data)

    for i, timestep in enumerate(time_steps):
        ax = axs[0, i]
        im = ax.imshow(
            torch.reshape(data[timestep], orig_shape),
            cmap=cmap,
            vmin=global_vmin,
            vmax=global_vmax,
        )
        ax.xaxis.set_tick_params(
            which="both", bottom=False, top=False, labelbottom=False
        )
        ax.yaxis.set_tick_params(which="both", left=False, right=False, labelleft=False)

        ax = axs[1, i]
        im = ax.imshow(
            torch.reshape(base_preds[0, timestep], orig_shape),
            cmap=cmap,
            vmin=global_vmin,
            vmax=global_vmax,
        )
        ax.xaxis.set_tick_params(
            which="both", bottom=False, top=False, labelbottom=False
        )
        ax.yaxis.set_tick_params(which="both", left=False, right=False, labelleft=False)

        ax = axs[2, i]
        im = ax.imshow(
            torch.reshape(siae_preds[0, timestep], orig_shape),
            cmap=cmap,
            vmin=global_vmin,
            vmax=global_vmax,
        )
        ax.xaxis.set_tick_params(
            which="both", bottom=False, top=False, labelbottom=False
        )
        ax.yaxis.set_tick_params(which="both", left=False, right=False, labelleft=False)

    axs[0, 0].set_title(f"t = {time_steps[0]}", fontsize=8, pad=4)
    axs[0, 1].set_title(f"t = {time_steps[1]}", fontsize=8, pad=4)
    axs[0, 2].set_title(f"t = {time_steps[2]}", fontsize=8, pad=4)
    axs[0, 0].set_ylabel("Truth", fontsize=8, labelpad=4, rotation=90)
    axs[1, 0].set_ylabel("Baseline", fontsize=8, labelpad=4, rotation=90)
    axs[2, 0].set_ylabel("SIAE", fontsize=8, labelpad=4, rotation=90)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.05)
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7)
    plt.subplots_adjust(right=0.8)
    plotter.save(plt, "pred_samples")

    def interpolate_psd(eig_fs, f, psd):
        interp_psd = torch.zeros_like(eig_fs)
        for i, freq in enumerate(eig_fs):
            idx = torch.argmin(torch.abs(f - freq))
            if idx == 0:
                interp_psd[i] = psd[0]
            elif idx == len(f) - 1:
                interp_psd[i] = psd[-1]
            else:
                w1 = (f[idx] - freq) / (f[idx] - f[idx - 1])
                w2 = 1 - w1
                interp_psd[i] = w1 * psd[idx - 1] + w2 * psd[idx]
        return interp_psd

    def smooth_power_spectrum(power_spectrum, window_size):
        # Define the rolling average kernel
        kernel = torch.ones(window_size) / window_size

        # Apply the rolling average using convolution
        smoothed_spectrum = torch.nn.functional.conv1d(
            power_spectrum.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=(window_size - 1) // 2,
        )

        return smoothed_spectrum.squeeze()

    dynamics = base_model.linear_dynamics.weight
    eigen_vals, eigen_vecs = torch.linalg.eig(dynamics)
    cont_eigen_vals = torch.log(eigen_vals)
    eig_fs = torch.abs(cont_eigen_vals.imag / torch.pi)

    with torch.no_grad():
        _, _, zk = base_model.detailed_forward(train_data)
    # TODO move mean to the next step, after cal eigen coeffs
    avg_zk = torch.mean(zk, axis=0)
    eigen_coeffs = torch.abs(torch.matmul(avg_zk, torch.abs(eigen_vecs))) ** 2
    eigen_coeffs = eigen_coeffs * torch.exp(5 * torch.abs(eigen_vals))
    with torch.no_grad():
        sum_fft = torch.zeros_like(torch.abs(torch.fft.rfft(train_data[:, 0])))
    for i in range(0, data.shape[1], fft_stride):
        fft = torch.abs(torch.fft.rfft(train_data[:, i]))
        sum_fft += fft
    data_psd = sum_fft / data.shape[0]
    # f, data_psd = utils.avg_temporal_psd(train_data, sample_freq=1)
    # data_psd = smooth_power_spectrum(data_psd, 5)
    f = torch.linspace(0, 1, len(data_psd))
    interp_psd = interpolate_psd(eig_fs, f, data_psd)

    f = f.detach().numpy()
    interp_psd = interp_psd.detach().numpy()
    eigen_coeffs = eigen_coeffs.detach().numpy()
    eig_fs = eig_fs.detach().numpy()
    data_psd = data_psd / np.sum(interp_psd)
    interp_psd = interp_psd / np.sum(interp_psd)
    eigen_coeffs = eigen_coeffs / np.sum(eigen_coeffs)

    plt.figure(figsize=(COLUMN, HALF), dpi=300)
    plt.plot([], [], color="darkred", label="Error")
    plt.plot(f, data_psd, ls="-", c="k", alpha=0.3, zorder=1, label=r"$E_{prior}$")
    # plt.scatter(eig_fs, interp_psd, c='k', s=6, marker='o', zorder=3, label='Data')
    plt.scatter(
        eig_fs,
        eigen_coeffs,
        c="r",
        s=6,
        marker="x",
        zorder=4,
        label=r"$E_{\mathbf{\Omega}}$",
    )

    for x, y1, y2 in zip(eig_fs, interp_psd, eigen_coeffs):
        plt.vlines(x, ymin=min(y1, y2), ymax=max(y1, y2), color="darkred", zorder=2)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency")
    plt.ylabel("Normalized Power")
    plotter.save(plt, "base_spectral_loss")

    dynamics = siae_model.linear_dynamics.weight
    eigen_vals, eigen_vecs = torch.linalg.eig(dynamics)
    cont_eigen_vals = torch.log(eigen_vals)
    eig_fs = torch.abs(cont_eigen_vals.imag / torch.pi)

    with torch.no_grad():
        _, _, zk = siae_model.detailed_forward(train_data)
    avg_zk = torch.mean(zk, axis=0)
    eigen_coeffs = torch.abs(torch.matmul(avg_zk, torch.abs(eigen_vecs))) ** 2
    eigen_coeffs = eigen_coeffs * torch.exp(5 * torch.abs(eigen_vals))
    with torch.no_grad():
        sum_fft = torch.zeros_like(torch.abs(torch.fft.rfft(train_data[:, 0])))
    for i in range(0, data.shape[1], fft_stride):
        fft = torch.abs(torch.fft.rfft(train_data[:, i]))
        sum_fft += fft
    data_psd = sum_fft / data.shape[0]
    # data_psd = smooth_power_spectrum(data_psd, 5)
    f = torch.linspace(0, 1, len(data_psd))
    interp_psd = interpolate_psd(eig_fs, f, data_psd)

    f = f.detach().numpy()
    interp_psd = interp_psd.detach().numpy()
    eigen_coeffs = eigen_coeffs.detach().numpy()
    eig_fs = eig_fs.detach().numpy()
    data_psd = data_psd / np.sum(interp_psd)
    interp_psd = interp_psd / np.sum(interp_psd)
    eigen_coeffs = eigen_coeffs / np.sum(eigen_coeffs)

    plt.figure(figsize=(COLUMN, HALF), dpi=300)
    plt.plot([], [], color="darkred", label="Error")
    plt.plot(f, data_psd, ls="-", c="k", alpha=0.3, zorder=1, label=r"$E_{prior}$")
    # plt.scatter(eig_fs, interp_psd, c='k', s=6, marker='o', zorder=3, label='Data')
    plt.scatter(
        eig_fs,
        eigen_coeffs,
        c="r",
        s=6,
        marker="x",
        zorder=4,
        label=r"$E_{\mathbf{\Omega}}$",
    )

    for x, y1, y2 in zip(eig_fs, interp_psd, eigen_coeffs):
        plt.vlines(x, ymin=min(y1, y2), ymax=max(y1, y2), color="darkred", zorder=2)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency")
    plt.ylabel("Normalized Power")
    plotter.save(plt, "siae_spectral_loss")
