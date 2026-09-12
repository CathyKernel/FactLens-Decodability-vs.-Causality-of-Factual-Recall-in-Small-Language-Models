"""The Causal-Probe Gap (CPG): decodability minus causality, per site.

Motivation
----------
Linear probes report what is *decodable* from a hidden state; activation
patching reports what the model actually *uses*. These can disagree: hidden
states often carry information the model does not read out (Belinkov,
2022; Ravichander et al., 2021; Elazar et al., 2021). For factual recall we
quantify the disagreement site-by-site with two probe positions and one
patching curve:

    D_subj(k)  probe macro-F1 at the last SUBJECT token    (identity/association)
    D_read(k)  probe macro-F1 at the final PROMPT token    (answer availability)
    C(k)       mean subject-swap patch restoration at site k (causal use)

Both D curves and C live in [0, 1], so their differences are directly
interpretable:

    CPG_subj(k) = D_subj(k) - C(k)   dormant knowledge at the subject token
    CPG_read(k) = D_read(k) - C(k)   readout availability minus causal use

Onsets and the handoff window
-----------------------------
Empirically (GPT-2, the reference run): D_subj is near ceiling from the
embedding site — the subject's identity, and hence its associated object,
is trivially linearly present at the subject token. The informative events
are later:

* **readout onset** — the first site where D_read exceeds chance by the
  probe margin (and stays): the fact has become linearly accessible at the
  position the model reads out from.
* **routing offset** — the first site where C falls below the patch margin
  (and stays): the subject token's representation has stopped being
  causally necessary; attention has already moved the fact.

    handoff_window = routing_offset - readout_onset

counts the sites the network spends "in transit". CPG_subj peaks *after*
the routing offset — the fact remains decodable at the subject token long
after the model stopped using it: dormant knowledge.

References
----------
Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). Amnesic
Probing. TACL 9. Ravichander, A., Belinkov, Y., Hovy, E., & Black, A. W.
(2021). Probing the Probe Deployment. EACL 2021. Belinkov, Y. (2022).
Probing Classifiers: Promises, Shortcomings, and Advances. CL 38(1).
"""

from __future__ import annotations

import numpy as np


def first_rise_above(values: list[float], threshold: float, run_length: int = 2) -> int | None:
    """First index where ``values`` *crosses* ``threshold`` upward and stays.

    A *crossing* (not merely a high start) is required: the site before the
    run must sit at or below the threshold, and a run starting at index 0
    does not count. This matters because probe curves can start above
    threshold through confounds — e.g. GPT-2's learned positional
    embeddings leak prompt length into the readout site (subject token
    counts vary), which inflates early-site F1 without any fact routing.
    A curve that is high from the start reports no onset.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    for i in range(1, n):
        window = arr[i : i + run_length]
        if len(window) < run_length:
            break
        if np.all(~np.isnan(window)) and np.all(window > threshold):
            if arr[i - 1] <= threshold or np.isnan(arr[i - 1]):
                return i
    return None


def first_fall_below(values: list[float], threshold: float, run_length: int = 2) -> int | None:
    """First index where ``values`` crosses ``threshold`` downward and stays."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    for i in range(1, n):
        window = arr[i : i + run_length]
        if len(window) < run_length:
            break
        if np.all(~np.isnan(window)) and np.all(window < threshold):
            if arr[i - 1] >= threshold or np.isnan(arr[i - 1]):
                return i
    return None


# Backwards-compatible aliases.
first_above = first_rise_above
first_below = first_fall_below


def compute_cpg(
    sites: list[str],
    d_subject: list[float],
    control_subject: list[float],
    d_read: list[float],
    control_read: list[float],
    chance: float,
    causality: list[float],
    probe_margin: float = 0.15,
    patch_margin: float = 0.2,
    run_length: int = 2,
) -> dict:
    """Compute CPG curves, onsets, and the handoff window.

    Args:
        sites: Residual sites ``embed .. blockL`` (shared axis).
        d_subject/control_subject: Decodability at the last subject token.
        d_read/control_read: Decodability at the final prompt token.
        chance: Uniform-accuracy chance level averaged over relations.
        causality: Mean patch restoration per site.
        probe_margin: D_read must cross ``chance + probe_margin`` upward
            (a crossing, not a high start — see :func:`first_rise_above`).
        patch_margin: C must fall below ``patch_margin`` for the offset.

    Returns:
        Dict with curves, per-site gaps, onsets/offsets, the handoff
        window, and peak dormant knowledge.
    """
    n = len(sites)
    if not (len(d_subject) == len(d_read) == len(causality) == n):
        raise ValueError("Curve lengths must match the site list.")

    cpg_subject = [float(d - c) for d, c in zip(d_subject, causality)]
    cpg_read = [float(d - c) for d, c in zip(d_read, causality)]

    readout_onset = first_rise_above(d_read, chance + probe_margin, run_length)
    routing_offset = first_fall_below(causality, patch_margin, run_length)

    handoff_window = None
    if readout_onset is not None and routing_offset is not None:
        handoff_window = routing_offset - readout_onset

    def site_name(idx: int | None) -> str | None:
        return sites[idx] if idx is not None else None

    peak_idx = int(np.nanargmax(cpg_subject)) if cpg_subject else None
    return {
        "sites": sites,
        "d_subject": d_subject,
        "control_subject": control_subject,
        "d_read": d_read,
        "control_read": control_read,
        "causality": causality,
        "cpg_subject": cpg_subject,
        "cpg_read": cpg_read,
        "chance": chance,
        "margins": {"probe": probe_margin, "patch": patch_margin},
        "readout_onset": {
            "index": readout_onset,
            "site": site_name(readout_onset),
            "threshold": chance + probe_margin,
        },
        "routing_offset": {
            "index": routing_offset,
            "site": site_name(routing_offset),
            "threshold": patch_margin,
        },
        "handoff_window_sites": handoff_window,
        "dormant_sites_after_offset": (
            n - routing_offset if routing_offset is not None else None
        ),
        "peak_dormant": {
            "value": float(cpg_subject[peak_idx]) if peak_idx is not None else None,
            "site": site_name(peak_idx),
        },
    }


def cpg_to_markdown(cpg: dict, model_name: str = "") -> str:
    """Render a CPG summary as a GitHub-flavored markdown table."""
    lines = [
        f"### Causal-Probe Gap — {model_name}".rstrip(),
        "",
        "| site | D_subj (F1) | ctrl_subj | D_read (F1) | ctrl_read | causality | CPG_subj |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, site in enumerate(cpg["sites"]):
        lines.append(
            f"| {site} | {cpg['d_subject'][i]:.3f} | {cpg['control_subject'][i]:.3f} "
            f"| {cpg['d_read'][i]:.3f} | {cpg['control_read'][i]:.3f} "
            f"| {cpg['causality'][i]:.3f} | {cpg['cpg_subject'][i]:+.3f} |"
        )
    lines += [
        "",
        f"- Readout onset (D_read > {cpg['readout_onset']['threshold']:.2f}): "
        f"**{cpg['readout_onset']['site']}**",
        f"- Routing offset (C < {cpg['routing_offset']['threshold']:.2f}): "
        f"**{cpg['routing_offset']['site']}**",
        f"- Handoff window: **{cpg['handoff_window_sites']} site(s)**",
    ]
    if cpg["peak_dormant"]["value"] is not None:
        lines.append(
            f"- Peak dormant knowledge (CPG_subj): {cpg['peak_dormant']['value']:.3f} "
            f"at {cpg['peak_dormant']['site']}"
        )
    if cpg["dormant_sites_after_offset"] is not None:
        lines.append(
            f"- Dormant tail: {cpg['dormant_sites_after_offset']} site(s) where the "
            f"fact stays decodable at the subject token but is no longer causal"
        )
    return "\n".join(lines)
