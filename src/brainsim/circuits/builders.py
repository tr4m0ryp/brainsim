"""Parametric stage builders (T6/T8, Architecture). SMALL SCALES ONLY:
hundreds-to-thousands of neurons, seeded generators, statistics named in the
design doc (coding level f~0.03 default, log-normal weights quantised to a
26-state ceiling via quantile binning, F5). Full-scale parameterisation
(measured wiring, census counts in the 1e5-1e6 range) lives in composite/full.
"""
from __future__ import annotations

import numpy as np

from brainsim.schema.tables import ConnectivityTable

DEFAULT_DATASET = "brainsim-synthetic-smoke"
DEFAULT_VERSION = "v0"


def quantize_to_states(samples: np.ndarray, n_states: int) -> np.ndarray:
    """Quantile-bin `samples` into `n_states` levels, each represented by the
    mean of the samples that fall in it (F5: 26-state Bartol ceiling).
    """
    samples = np.asarray(samples, dtype=np.float64)
    edges = np.quantile(samples, np.linspace(0.0, 1.0, n_states + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    bin_idx = np.clip(np.digitize(samples, edges[1:-1]), 0, n_states - 1)
    quantized = samples.copy()
    for k in range(n_states):
        mask = bin_idx == k
        if np.any(mask):
            quantized[mask] = samples[mask].mean()
    return quantized


def _convergent_block(
    rng: np.random.Generator,
    n_pre: int,
    n_post: int,
    convergence: int,
    pre_offset: int,
    post_offset: int,
    weight_mu: float,
    weight_sigma: float,
    n_states: int,
    sign: int,
    cell_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each post neuron draws `convergence` presynaptic partners without
    replacement; weights are log-normal, quantile-quantised.
    """
    convergence = max(1, min(convergence, n_pre))
    pre_chunks = [rng.choice(n_pre, size=convergence, replace=False) for _ in range(n_post)]
    pre_id = np.concatenate(pre_chunks) + pre_offset
    post_id = np.repeat(np.arange(n_post), convergence) + post_offset
    n_syn = pre_id.size
    raw_w = rng.lognormal(weight_mu, weight_sigma, size=n_syn)
    weight = quantize_to_states(raw_w, n_states)
    signs = np.full(n_syn, sign, dtype=np.int8)
    cell_types = np.full(n_syn, cell_type, dtype=object)
    return pre_id, post_id, weight, signs, cell_types


def _make_table(pre_id, post_id, weight, sign, cell_type, dataset: str, version: str) -> ConnectivityTable:
    n = pre_id.size
    return ConnectivityTable(
        pre_id=pre_id,
        post_id=post_id,
        weight=weight,
        sign=sign,
        cell_type=cell_type,
        dataset=np.full(n, dataset, dtype=object),
        version=np.full(n, version, dtype=object),
    )


def _empty_table(dataset: str, version: str) -> ConnectivityTable:
    z_i = np.empty(0, dtype=np.int64)
    return ConnectivityTable(
        pre_id=z_i,
        post_id=z_i,
        weight=np.empty(0, dtype=np.float32),
        sign=np.empty(0, dtype=np.int8),
        cell_type=np.empty(0, dtype=object),
        dataset=np.empty(0, dtype=object),
        version=np.empty(0, dtype=object),
    )


def make_separation(
    n_in: int,
    n_exp: int,
    f: float = 0.03,
    inhibition: bool = True,
    convergence: int | None = None,
    seed: int = 0,
    weight_mu: float = -1.0,
    weight_sigma: float = 0.5,
    n_states: int = 26,
    dataset: str = DEFAULT_DATASET,
    version: str = DEFAULT_VERSION,
) -> tuple[dict[str, int], ConnectivityTable]:
    """DG-analog: feedforward expansion n_in -> n_exp at coding level f, with
    optional feedforward-inhibition motif (interneuron pool reading the
    expansion layer and inhibiting it back).
    """
    rng = np.random.default_rng(seed)
    convergence = convergence or max(1, int(round(n_in * max(f, 0.05))))
    pre, post, w, s, ct = _convergent_block(
        rng, n_in, n_exp, convergence, 0, n_in, weight_mu, weight_sigma, n_states, +1, "granule"
    )
    pop_sizes = {"separation_in": n_in, "separation_exp": n_exp}

    if inhibition:
        n_inh = max(1, int(round(0.1 * n_exp)))
        inh_conv = min(convergence, n_exp)
        pre_ff, post_ff, w_ff, s_ff, ct_ff = _convergent_block(
            rng, n_exp, n_inh, inh_conv, n_in, n_in + n_exp, weight_mu, weight_sigma, n_states, +1, "interneuron"
        )
        fb_conv = min(inh_conv, n_inh)
        pre_fb, post_fb, w_fb, s_fb, ct_fb = _convergent_block(
            rng, n_inh, n_exp, fb_conv, n_in + n_exp, n_in, weight_mu, weight_sigma, n_states, -1, "interneuron"
        )
        pre = np.concatenate([pre, pre_ff, pre_fb])
        post = np.concatenate([post, post_ff, post_fb])
        w = np.concatenate([w, w_ff, w_fb])
        s = np.concatenate([s, s_ff, s_fb])
        ct = np.concatenate([ct, ct_ff, ct_fb])
        pop_sizes["separation_inh"] = n_inh

    return pop_sizes, _make_table(pre, post, w, s, ct, dataset, version)


def make_formation(
    n: int,
    recurrent_k: int,
    seed: int = 0,
    weight_mu: float = -1.0,
    weight_sigma: float = 0.5,
    n_states: int = 26,
    dataset: str = DEFAULT_DATASET,
    version: str = DEFAULT_VERSION,
) -> tuple[dict[str, int], ConnectivityTable]:
    """CA3-analog: recurrent attractor population, in-degree `recurrent_k`,
    self-loops dropped.
    """
    rng = np.random.default_rng(seed)
    pre, post, w, s, ct = _convergent_block(
        rng, n, n, recurrent_k, 0, 0, weight_mu, weight_sigma, n_states, +1, "pyramidal"
    )
    keep = pre != post
    pre, post, w, s, ct = pre[keep], post[keep], w[keep], s[keep], ct[keep]
    return {"formation": n}, _make_table(pre, post, w, s, ct, dataset, version)


def make_output(
    n: int,
    inhibition_frac: float = 0.1,
    seed: int = 0,
    weight_mu: float = -1.0,
    weight_sigma: float = 0.5,
    n_states: int = 26,
    dataset: str = DEFAULT_DATASET,
    version: str = DEFAULT_VERSION,
) -> tuple[dict[str, int], ConnectivityTable]:
    """CA1-analog: excitatory population plus a local inhibitory pool
    (BTSP one-shot binding onto this population is wired at composite level,
    not intra-stage).
    """
    rng = np.random.default_rng(seed)
    n_inh = max(1, int(round(inhibition_frac * n)))
    conv = max(1, min(n_inh, int(round(0.2 * n))))
    pre, post, w, s, ct = _convergent_block(
        rng, n_inh, n, conv, n, 0, weight_mu, weight_sigma, n_states, -1, "interneuron"
    )
    return {"output_exc": n, "output_inh": n_inh}, _make_table(pre, post, w, s, ct, dataset, version)


def make_relay(n: int, dataset: str = DEFAULT_DATASET, version: str = DEFAULT_VERSION) -> tuple[dict[str, int], ConnectivityTable]:
    """Thin relay stage (T10): population only, no intrinsic wiring."""
    return {"relay": n}, _empty_table(dataset, version)


def make_storage(
    n: int,
    lognormal_params: tuple[float, float],
    n_states: int = 26,
    multi_contact_tail: int = 5,
    seed: int = 0,
    dataset: str = DEFAULT_DATASET,
    version: str = DEFAULT_VERSION,
) -> tuple[dict[str, int], ConnectivityTable]:
    """H01-twin: associative storage population with a rare multi-synaptic
    contact tail up to `multi_contact_tail` redundant contacts per connected
    pair (F20 permanence motif proxy).
    """
    rng = np.random.default_rng(seed)
    mu, sigma = lognormal_params
    base_k = max(1, int(round(0.05 * n)))
    pre, post, w, s, ct = _convergent_block(rng, n, n, base_k, 0, 0, mu, sigma, n_states, +1, "storage")
    keep = pre != post
    pre, post, w, s, ct = pre[keep], post[keep], w[keep], s[keep], ct[keep]

    n_pairs = pre.size
    n_redundant = max(0, int(round(0.01 * n_pairs))) if multi_contact_tail > 1 else 0
    if n_redundant > 0:
        chosen = rng.choice(n_pairs, size=n_redundant, replace=False)
        extra_pre, extra_post, extra_w = [], [], []
        for idx in chosen:
            k_contacts = int(rng.integers(2, multi_contact_tail + 1))
            extra_pre.append(np.full(k_contacts - 1, pre[idx]))
            extra_post.append(np.full(k_contacts - 1, post[idx]))
            extra_w.append(quantize_to_states(rng.lognormal(mu, sigma, size=k_contacts - 1), n_states))
        pre = np.concatenate([pre] + extra_pre)
        post = np.concatenate([post] + extra_post)
        w = np.concatenate([w] + extra_w)
        extra_n = sum(len(e) for e in extra_pre)
        s = np.concatenate([s, np.full(extra_n, 1, dtype=np.int8)])
        ct = np.concatenate([ct, np.full(extra_n, "storage", dtype=object)])

    return {"storage": n}, _make_table(pre, post, w, s, ct, dataset, version)


__all__ = [
    "quantize_to_states",
    "make_separation",
    "make_formation",
    "make_output",
    "make_relay",
    "make_storage",
]
