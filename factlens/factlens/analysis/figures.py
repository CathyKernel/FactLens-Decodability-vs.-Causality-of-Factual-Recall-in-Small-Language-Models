"""Publication-quality matplotlib figures for every analysis.

Conventions: ``constrained_layout=True`` everywhere (never mixed with
``tight_layout``), colorblind-safe palette, English labels, and consistent
x-axes of residual sites (``embed, blk1..blkL``).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe: importable on servers without displays
import matplotlib.pyplot as plt
import numpy as np

# Colorblind-safe qualitative palette (Okabe-Ito).
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"
C_GRAY = "#666666"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    }
)


def _x_positions(sites: list[str]) -> list[int]:
    return list(range(len(sites)))


def _shorten(sites: list[str]) -> list[str]:
    return ["emb" if s == "embed" else s.replace("block", "blk") for s in sites]


def _style_site_axis(ax, sites: list[str]) -> None:
    """Rotated, collision-free tick labels for residual-stream sites.

    Must be applied to EVERY panel individually: with ``sharex=True`` the
    tick positions and label *strings* propagate to the shared panels, but
    the per-label *rotation* does not — shared panels then render the
    ``emb, blk1..blkL`` labels horizontally and adjacent labels collide
    once the site count exceeds the panel width.
    """
    ax.set_xticks(_x_positions(sites))
    ax.set_xticklabels(_shorten(sites), rotation=90, fontsize=6)


# --- logit lens --------------------------------------------------------------


def plot_lens_trajectory(agg: dict, model_name: str = "", out_path: str | None = None):
    """Two-panel lens figure: answer logit and top-k membership per site."""
    sites = agg["sites"]
    x = _x_positions(sites)
    n = len(sites)
    final_idx = n - 1

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 2.6), constrained_layout=True, sharex=True
    )

    mean = np.array(agg["answer_logit_mean"])
    sem = np.array(agg["answer_logit_sem"])
    axes[0].errorbar(x, mean, yerr=sem, color=C_BLUE, marker="o", ms=3, lw=1.2,
                     capsize=2, label="answer logit")
    axes[0].axvline(final_idx - 0.5, color=C_GRAY, lw=0.8, ls=":")
    axes[0].set_ylabel("answer logit (lens)")
    axes[0].set_title("Answer token logit through the lens")
    axes[0].legend(loc="upper left", frameon=False)

    for k, color in zip(("5", "10", "100"), (C_GREEN, C_ORANGE, C_RED)):
        axes[1].plot(x, agg["in_top_k"][k], color=color, marker=".", ms=3, lw=1.0,
                     label=f"answer in top-{k}")
    axes[1].plot(x, agg["top1_accuracy"], color=C_BLUE, marker="o", ms=3, lw=1.4,
                 label="answer is top-1")
    axes[1].axvline(final_idx - 0.5, color=C_GRAY, lw=0.8, ls=":")
    axes[1].axhline(0.0, color=C_GRAY, lw=0.6)
    axes[1].set_ylabel("fraction of facts")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_title("Answer emergence")
    axes[1].legend(loc="upper left", frameon=False, fontsize=7)

    for ax in axes:
        _style_site_axis(ax, sites)
    fig.suptitle(f"Logit-lens trajectory — {model_name} ({agg['n_facts']} facts)", fontsize=10)
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
        return None
    return fig


# --- probes --------------------------------------------------------------------


def plot_probe_curves(agg: dict, model_name: str = "", out_path: str | None = None):
    """Probe decodability vs. control across sites, one panel per position."""
    positions = list(agg.keys())
    fig, axes = plt.subplots(
        1, len(positions), figsize=(3.4 * len(positions), 2.6),
        constrained_layout=True, squeeze=False,
    )
    for ax, position in zip(axes[0], positions):
        sub = agg[position]
        x = _x_positions(sub["sites"])
        ax.plot(x, sub["task_f1"], color=C_BLUE, marker="o", ms=3, lw=1.4,
                label="task macro-F1")
        ax.plot(x, sub["control_f1"], color=C_ORANGE, marker="s", ms=3, lw=1.2,
                ls="--", label="control macro-F1")
        ax.axhline(sub["chance"], color=C_GRAY, lw=0.8, ls=":", label="chance")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"probe at {position}", fontsize=9)
        _style_site_axis(ax, sub["sites"])
        ax.legend(loc="upper left", frameon=False, fontsize=7)
    axes[0][0].set_ylabel("held-out macro-F1")
    fig.suptitle(f"Decodability curves — {model_name}", fontsize=10)
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
        return None
    return fig


# --- patching --------------------------------------------------------------------


def plot_patch_curves(agg: dict, model_name: str = "", out_path: str | None = None,
                      per_relation: bool = False):
    """Mean restoration per site (optionally with per-relation thin lines)."""
    sites = agg["sites"]
    x = _x_positions(sites)
    fig, ax = plt.subplots(figsize=(5.2, 2.8), constrained_layout=True)
    if per_relation:
        for rel, sub in agg["per_relation"].items():
            ax.plot(x, sub["restoration_mean"], lw=0.8, alpha=0.45, color=C_GRAY)
    mean = np.array(agg["restoration_mean"])
    sem = np.array(agg["restoration_sem"])
    ax.errorbar(x, mean, yerr=sem, color=C_RED, marker="o", ms=3, lw=1.5,
                capsize=2, label="mean restoration")
    ax.axhline(1.0, color=C_GRAY, lw=0.8, ls=":", label="full restoration")
    ax.axhline(0.0, color=C_GRAY, lw=0.8, ls=":")
    ax.set_ylabel("restored logit-diff fraction")
    ax.set_ylim(-0.15, 1.25)
    _style_site_axis(ax, sites)
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax.set_title(f"Activation patching — {model_name} ({agg['n_facts']} facts)", fontsize=10)
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
        return None
    return fig


# --- causal tracing heatmap ---------------------------------------------------------


def plot_trace_heatmap(trace: dict, model_name: str = "", out_path: str | None = None):
    """ROME-style heatmap of single-position restoration (sites x positions)."""
    sites = trace["sites"]
    matrix = np.array(trace["restoration"])  # (n_sites, T)
    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=-0.2, vmax=1.0)
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(_shorten(sites), fontsize=6)
    ax.set_xticks(range(len(trace["token_strs"])))
    ax.set_xticklabels(trace["token_strs"], rotation=90, fontsize=6, family="monospace")
    # Outline the subject positions.
    for p in trace["subject_positions"]:
        ax.add_patch(
            plt.Rectangle((p - 0.5, -0.5), 1, len(sites), fill=False,
                          edgecolor=C_RED, lw=1.2, clip_on=False)
        )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("restored logit-diff fraction", fontsize=8)
    ax.set_title(
        f"Causal trace — {model_name}: '{trace['prompt'][:40]}...'", fontsize=9
    )
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
        return None
    return fig


# --- CPG ---------------------------------------------------------------------------


def plot_cpg(cpg: dict, model_name: str = "", out_path: str | None = None):
    """Three-panel CPG figure: D curves, C curve, and the dormant-knowledge gap.

    Panel 1 — decodability at both probe positions with their controls and
    the chance line. Panel 2 — causality (patch restoration). Panel 3 —
    CPG_subj = D_subj - C (dormant knowledge at the subject token), with the
    readout onset, the routing offset, and the shaded handoff window.
    """
    sites = cpg["sites"]
    x = np.array(_x_positions(sites))

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7), constrained_layout=True,
                             sharex=True)

    axes[0].plot(x, cpg["d_subject"], color=C_BLUE, marker="o", ms=3, lw=1.4,
                 label="D: subject token")
    axes[0].plot(x, cpg["d_read"], color=C_GREEN, marker="o", ms=3, lw=1.4,
                 label="D: readout token")
    axes[0].plot(x, cpg["control_subject"], color=C_ORANGE, ls="--", marker="s",
                 ms=2.5, lw=1.0, alpha=0.8, label="control (subj)")
    axes[0].plot(x, cpg["control_read"], color=C_PURPLE, ls="--", marker="s",
                 ms=2.5, lw=1.0, alpha=0.8, label="control (read)")
    axes[0].axhline(cpg["chance"], color=C_GRAY, ls=":", lw=0.9)
    axes[0].set_ylabel("macro-F1")
    axes[0].set_title("Decodability D(k)")
    axes[0].set_ylim(-0.02, 1.05)
    _panel_handles, _panel_labels = axes[0].get_legend_handles_labels()

    axes[1].plot(x, cpg["causality"], color=C_RED, marker="o", ms=3, lw=1.4)
    axes[1].axhline(0.0, color=C_GRAY, ls=":", lw=0.8)
    axes[1].axhline(1.0, color=C_GRAY, ls=":", lw=0.8)
    axes[1].set_ylabel("restoration")
    axes[1].set_title("Causality C(k)")
    axes[1].set_ylim(-0.15, 1.25)

    colors = [C_GREEN if v >= 0 else C_PURPLE for v in cpg["cpg_subject"]]
    axes[2].bar(x, cpg["cpg_subject"], color=colors, width=0.7)
    axes[2].axhline(0.0, color=C_GRAY, lw=0.8)
    axes[2].set_ylabel("D_subj - C")
    axes[2].set_title("Dormant knowledge")
    _cpg_top = max(float(np.max(cpg["cpg_subject"])), 0.05)
    axes[2].set_ylim(min(0.0, float(np.min(cpg["cpg_subject"])) - 0.05) - 0.02,
                     _cpg_top * 1.28)

    r_on = cpg["readout_onset"]["index"]
    c_off = cpg["routing_offset"]["index"]
    if r_on is not None:
        axes[0].axvline(r_on, color=C_GREEN, lw=1.0, ls="--")
    if c_off is not None:
        axes[1].axvline(c_off, color=C_RED, lw=1.0, ls="--")
    if r_on is not None and c_off is not None:
        for ax in axes:
            ax.axvspan(r_on, c_off, color=C_GREEN, alpha=0.12)
        axes[2].text(
            (r_on + c_off) / 2,
            _cpg_top * 1.03,
            f"handoff\n{c_off - r_on} site(s)",
            ha="center", va="bottom", fontsize=7, color=C_GREEN,
        )
    if c_off is not None:
        for ax in axes:
            ax.axvspan(c_off, len(sites) - 1, color=C_PURPLE, alpha=0.08)

    for ax in axes:
        _style_site_axis(ax, sites)
    fig.legend(_panel_handles, _panel_labels, loc="outside lower center",
               ncols=len(_panel_labels), frameon=False, fontsize=7,
               columnspacing=1.4, handlelength=1.6)
    fig.suptitle(f"Causal-Probe Gap — {model_name}", fontsize=10)
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
        return None
    return fig
