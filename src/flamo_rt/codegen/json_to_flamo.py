#json_to_flamo
#author: Facundo Franchino
"""
reconstruct a flamo model from a json config dict

consumes the output of flamo_to_json() and produces a flamo model with
the same topology and parameter values. this completes the round-trip:

    flamo model -> json config -> flamo model

each node type (Shell, Series, Parallel, Recursion, Leaf) maps to the
corresponding flamo system module, and each leaf module type maps to a
flamo dsp module. constructor arguments are read from the optional
"flamo" metadata key on each node; when absent, they are inferred from
the parameter shapes and channel counts.

requires flamo and torch to be installed. will raise ImportError if not.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

try:
    import torch
    from flamo.processor import dsp, system
except ImportError as e:
    raise ImportError(
        "json_to_flamo requires flamo and torch. "
        "install them with: pip install flamo torch"
    ) from e


#parameter conversion helpers

def _samples_to_seconds(samples: list[int], fs: float) -> np.ndarray:
    """convert integer delay samples back to seconds for flamo.

    flamo's parallelDelay stores delays in seconds internally
    and converts to samples during forward(). we reverse that here.
    """
    return np.array(samples, dtype=np.float64) / fs


def _denormalise_sos(sos: list) -> np.ndarray:
    """convert normalised 5-coefficient sos back to flamo's 6-coefficient format.

    json stores [b0, b1, b2, a1, a2] per section per channel with a0=1
    implicit. flamo expects shape (n_sections, 6, n_channels) with
    [b0, b1, b2, a0, a1, a2] where a0 is explicit.

    input shape: sos[n_sections][n_channels][5]
    output shape: (n_sections, 6, n_channels)
    """
    n_sections = len(sos)
    n_channels = len(sos[0])

    out = np.zeros((n_sections, 6, n_channels), dtype=np.float64)
    for s in range(n_sections):
        for ch in range(n_channels):
            b0, b1, b2, a1, a2 = sos[s][ch]
            out[s, :, ch] = [b0, b1, b2, 1.0, a1, a2]
    return out


#leaf module construction

def _build_leaf(
    node: dict[str, Any],
    fs: float,
    nfft: int,
    alias_decay_db: float,
    device: str,
) -> Any:
    """create a flamo dsp module from a leaf node and assign its parameters.

    reads constructor arguments from the "flamo" metadata key when
    available, falling back to inference from params and channel counts.
    """
    module_type = node["module_type"]
    params = node.get("params", {})
    meta = node.get("flamo", {})

    #read common constructor args from metadata or use defaults
    m_nfft = meta.get("nfft", nfft)
    m_adb = meta.get("alias_decay_db", alias_decay_db)
    m_rg = meta.get("requires_grad", False)

    if module_type == "parallelDelay":
        return _build_delay(node, params, meta, fs, m_nfft, m_adb, m_rg, device)

    if module_type == "HouseholderMatrix":
        return _build_householder(node, params, meta, m_nfft, m_adb, m_rg, device)

    if module_type in ("Gain", "Matrix"):
        return _build_gain(node, params, meta, m_nfft, m_adb, m_rg, device)

    if module_type == "parallelGain":
        return _build_parallel_gain(node, params, meta, m_nfft, m_adb, m_rg, device)

    if module_type == "parallelSOSFilter":
        return _build_sos_filter(node, params, meta, fs, m_nfft, m_adb, m_rg, device)

    if module_type in ("Biquad", "parallelBiquad"):
        return _build_biquad(node, params, meta, module_type, fs, m_nfft, m_adb, m_rg, device)

    if module_type in ("SVF", "parallelSVF"):
        return _build_svf(node, params, meta, module_type, fs, m_nfft, m_adb, m_rg, device)

    #unknown module type: try to create a Gain as a passthrough
    #this is a best-effort fallback for unrecognised modules
    n_ch = node.get("input_channels", 1)
    mod = dsp.Gain(
        size=(n_ch, n_ch), nfft=m_nfft,
        alias_decay_db=m_adb, device=device,
    )
    mod.assign_value(torch.eye(n_ch, dtype=torch.float32))
    return mod


def _build_delay(node, params, meta, fs, nfft, adb, rg, device):
    """construct a parallelDelay module."""
    samples = params.get("samples", [])
    n_ch = len(samples)
    size = tuple(meta.get("size", (n_ch,)))
    max_len = meta.get("max_len", max(samples) if samples else 2000)
    unit = meta.get("unit", 1)
    isint = meta.get("isint", True)

    mod = dsp.parallelDelay(
        size=size, max_len=max_len, nfft=nfft,
        isint=isint, unit=unit, fs=fs,
        alias_decay_db=adb, device=device,
    )

    if samples:
        delays_sec = _samples_to_seconds(samples, fs)
        mod.assign_value(torch.as_tensor(delays_sec, dtype=torch.float32))

    mod.param.requires_grad_(rg)
    return mod


def _build_gain(node, params, meta, nfft, adb, rg, device):
    """construct a Gain or Matrix module (matrix or diagonal).

    if the flamo metadata contains matrix_type, reconstruct a Matrix
    with the correct type so the map function matches the original.
    otherwise fall back to a plain Gain.
    """
    if "matrix" in params:
        values = np.array(params["matrix"], dtype=np.float64)
        n_out, n_in = values.shape
        size = tuple(meta.get("size", (n_out, n_in)))
    elif "gains" in params:
        gains = np.array(params["gains"], dtype=np.float64)
        #diagonal gain stored as 1d, but Gain expects 2d size
        n = len(gains)
        size = tuple(meta.get("size", (n, 1)))
        values = gains.reshape(size)
    else:
        #no params, identity passthrough
        n_ch = node.get("input_channels", 1)
        size = tuple(meta.get("size", (n_ch, n_ch)))
        values = np.eye(*size, dtype=np.float64)

    matrix_type = meta.get("matrix_type")
    if matrix_type is not None:
        m_iter = meta.get("iter", 1)
        mod = dsp.Matrix(
            size=size, nfft=nfft, matrix_type=matrix_type, iter=m_iter,
            alias_decay_db=adb, device=device,
        )
    else:
        mod = dsp.Gain(
            size=size, nfft=nfft,
            alias_decay_db=adb, device=device,
        )
    mod.assign_value(torch.as_tensor(values, dtype=torch.float32))
    mod.param.requires_grad_(rg)
    return mod


def _build_householder(node, params, meta, nfft, adb, rg, device):
    """construct a HouseholderMatrix module.

    the json stores the unit vector as a column matrix (N, 1).
    HouseholderMatrix internally sets size=(N, 1) and reconstructs
    U = I - 2*u*u^T from the normalised vector.
    """
    if "matrix" in params:
        values = np.array(params["matrix"], dtype=np.float64)
        N = values.shape[0]
    else:
        N = node.get("input_channels", 1)
        values = np.ones((N, 1), dtype=np.float64) / np.sqrt(N)

    mod = dsp.HouseholderMatrix(
        size=(N, N), nfft=nfft,
        alias_decay_db=adb, device=device,
    )
    mod.assign_value(torch.as_tensor(values, dtype=torch.float32))
    mod.param.requires_grad_(rg)
    return mod


def _build_parallel_gain(node, params, meta, nfft, adb, rg, device):
    """construct a parallelGain module."""
    gains = params.get("gains", [1.0])
    n_ch = len(gains)
    size = tuple(meta.get("size", (n_ch,)))

    mod = dsp.parallelGain(
        size=size, nfft=nfft,
        alias_decay_db=adb, device=device,
    )
    mod.assign_value(torch.as_tensor(gains, dtype=torch.float32))
    mod.param.requires_grad_(rg)
    return mod


def _build_sos_filter(node, params, meta, fs, nfft, adb, rg, device):
    """construct a parallelSOSFilter or equivalent biquad cascade.

    the json config stores normalised coefficients [b0, b1, b2, a1, a2]
    with shape sos[n_sections][n_channels][5]. flamo expects the raw
    6-coefficient format (n_sections, 6, n_channels) with a0 explicit.

    since flamo may not have parallelSOSFilter directly, we fall back
    to parallelBiquad which accepts the same coefficient structure.
    """
    sos = params.get("sos", [])
    if not sos:
        n_ch = node.get("input_channels", 1)
        mod = dsp.Gain(size=(n_ch, n_ch), nfft=nfft, device=device)
        mod.assign_value(torch.eye(n_ch, dtype=torch.float32))
        return mod

    n_sections = len(sos)
    n_channels = len(sos[0])
    size = tuple(meta.get("size", (n_channels,)))

    #try parallelSOSFilter first, fall back to parallelBiquad
    sos_6coeff = _denormalise_sos(sos)

    try:
        mod = dsp.parallelSOSFilter(
            size=(n_sections, n_channels), nfft=nfft,
            alias_decay_db=adb, device=device,
        )
    except AttributeError:
        #flamo version without parallelSOSFilter, use parallelBiquad
        mod = dsp.parallelBiquad(
            size=size, n_sections=n_sections,
            nfft=nfft, fs=int(fs),
            alias_decay_db=adb, device=device,
        )

    mod.assign_value(torch.as_tensor(sos_6coeff, dtype=torch.float32))
    mod.param.requires_grad_(rg)
    return mod


def _build_biquad(node, params, meta, module_type, fs, nfft, adb, rg, device):
    """construct a Biquad or parallelBiquad module."""
    n_sections = meta.get("n_sections", 1)
    filter_type = meta.get("filter_type", "lowpass")
    n_ch = node.get("input_channels", 1)

    if module_type == "parallelBiquad":
        size = tuple(meta.get("size", (n_ch,)))
        mod = dsp.parallelBiquad(
            size=size, n_sections=n_sections, filter_type=filter_type,
            nfft=nfft, fs=int(fs), alias_decay_db=adb, device=device,
        )
    else:
        n_out = node.get("output_channels", n_ch)
        size = tuple(meta.get("size", (n_out, n_ch)))
        mod = dsp.Biquad(
            size=size, n_sections=n_sections, filter_type=filter_type,
            nfft=nfft, fs=int(fs), alias_decay_db=adb, device=device,
        )

    #assign raw parameter values if available
    if "raw" in params:
        raw = np.array(params["raw"], dtype=np.float64)
        mod.assign_value(torch.as_tensor(raw, dtype=torch.float32))

    mod.param.requires_grad_(rg)
    return mod


def _build_svf(node, params, meta, module_type, fs, nfft, adb, rg, device):
    """construct an SVF or parallelSVF module."""
    n_sections = meta.get("n_sections", 1)
    filter_type = meta.get("filter_type", None)
    n_ch = node.get("input_channels", 1)

    if module_type == "parallelSVF":
        size = tuple(meta.get("size", (n_ch,)))
        mod = dsp.parallelSVF(
            size=size, n_sections=n_sections, filter_type=filter_type,
            nfft=nfft, fs=int(fs), alias_decay_db=adb, device=device,
        )
    else:
        n_out = node.get("output_channels", n_ch)
        size = tuple(meta.get("size", (n_out, n_ch)))
        mod = dsp.SVF(
            size=size, n_sections=n_sections, filter_type=filter_type,
            nfft=nfft, fs=int(fs), alias_decay_db=adb, device=device,
        )

    if "raw" in params:
        raw = np.array(params["raw"], dtype=np.float64)
        mod.assign_value(torch.as_tensor(raw, dtype=torch.float32))

    mod.param.requires_grad_(rg)
    return mod


#recursive tree traversal

def _build(
    node: dict[str, Any],
    fs: float,
    nfft: int,
    alias_decay_db: float,
    device: str,
) -> Any:
    """recursively reconstruct a flamo model from a json config tree."""
    node_type = node.get("type", "Leaf")

    if node_type == "Shell":
        return _build_shell(node, fs, nfft, alias_decay_db, device)

    if node_type == "Series":
        return _build_series(node, fs, nfft, alias_decay_db, device)

    if node_type == "Parallel":
        return _build_parallel(node, fs, nfft, alias_decay_db, device)

    if node_type == "Recursion":
        return _build_recursion(node, fs, nfft, alias_decay_db, device)

    if node_type == "Leaf":
        return _build_leaf(node, fs, nfft, alias_decay_db, device)

    raise ValueError(f"unknown node type: {node_type}")


def _build_shell(node, fs, nfft, adb, device):
    """reconstruct a Shell with FFT/iFFT io layers.

    the shell wraps a time-domain core with frequency-domain io
    layers. nfft is read from the flamo metadata if available,
    otherwise the function-level default is used.
    """
    meta = node.get("flamo", {})
    shell_nfft = meta.get("nfft", nfft)

    children = node.get("children", [])
    if children:
        core = _build(children[0], fs, shell_nfft, adb, device)
    else:
        #empty shell, create a single-channel passthrough
        core = dsp.Gain(size=(1, 1), nfft=shell_nfft, device=device)
        core.assign_value(torch.ones(1, 1, dtype=torch.float32))

    return system.Shell(
        core=core,
        input_layer=dsp.FFT(shell_nfft),
        output_layer=dsp.iFFT(shell_nfft),
    )


def _build_series(node, fs, nfft, adb, device):
    """reconstruct a Series (ordered sequence of children)."""
    children = node.get("children", [])
    pairs = OrderedDict()
    for child in children:
        name = child.get("name", f"child_{len(pairs)}")
        pairs[name] = _build(child, fs, nfft, adb, device)
    return system.Series(pairs)


def _build_parallel(node, fs, nfft, adb, device):
    """reconstruct a Parallel (two branches, optional output summing)."""
    children = node.get("children", [])
    sum_output = node.get("sum_output", False)

    if len(children) >= 2:
        brA = _build(children[0], fs, nfft, adb, device)
        brB = _build(children[1], fs, nfft, adb, device)
    elif len(children) == 1:
        brA = _build(children[0], fs, nfft, adb, device)
        brB = dsp.Gain(size=(1, 1), nfft=nfft, device=device)
        brB.assign_value(torch.zeros(1, 1, dtype=torch.float32))
    else:
        brA = dsp.Gain(size=(1, 1), nfft=nfft, device=device)
        brA.assign_value(torch.ones(1, 1, dtype=torch.float32))
        brB = dsp.Gain(size=(1, 1), nfft=nfft, device=device)
        brB.assign_value(torch.zeros(1, 1, dtype=torch.float32))

    return system.Parallel(brA=brA, brB=brB, sum_output=sum_output)


def _build_recursion(node, fs, nfft, adb, device):
    """reconstruct a Recursion (feedforward + feedback paths)."""
    ff_node = node.get("fF")
    fb_node = node.get("fB")

    fF = _build(ff_node, fs, nfft, adb, device) if ff_node else None
    fB = _build(fb_node, fs, nfft, adb, device) if fb_node else None

    return system.Recursion(fF=fF, fB=fB)


#public api

def json_to_flamo(
    config: dict[str, Any],
    *,
    nfft: int = 2**16,
    alias_decay_db: float = 0.0,
    device: str = "cpu",
) -> Any:
    """reconstruct a flamo model from a json config dict.

    the config dict is the output of flamo_to_json(). the returned
    model has the same topology and parameter values as the original.

    constructor arguments for each module are read from the optional
    "flamo" metadata key. when absent, sensible defaults are inferred
    from parameter shapes and channel counts.

    parameters
    ----------
    config : dict
        json config dict as produced by flamo_to_json().
    nfft : int
        default fft size for modules without flamo metadata.
    alias_decay_db : float
        default alias decay for modules without flamo metadata.
    device : str
        torch device string ("cpu" or "cuda").

    returns
    -------
    model : flamo module
        reconstructed Shell, Series, Parallel, Recursion, or leaf module.
    """
    fs = config.get("fs", 48000)
    return _build(config, float(fs), nfft, alias_decay_db, device)
