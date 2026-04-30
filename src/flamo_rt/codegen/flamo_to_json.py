#flamo_to_json
#author: Facundo Franchino
"""
traverse a flamo model graph and export a json-serialisable config dict

builds on flamo_model_to_nodes() from pyFDN. extends it with:
-parameter extraction (detach to numpy)
-module type detection (parallelDelay, Gain, Matrix, parallelGain, parallelSOSFilter, etc.)
-delay quantisation (seconds to integer samples)
-sos coefficient normalisation (a0 = 1)
-channel count propagation

the returned dict mirrors the json config schema in FLAMO_RT_SPEC.md section 6
and serves as the input to json_to_faust().
"""


from __future__ import annotations

from typing import Any

import numpy as np


#type introspection helpers
#no flamo import needed at module level

def _typename(module: Any) -> str:
    """return the class name of a module without importing its package."""
    return type(module).__name__


def _is_shell(m: Any) -> bool:
    return _typename(m) == "Shell"


def _is_series(m: Any) -> bool:
    return _typename(m) == "Series"


def _is_parallel(m: Any) -> bool:
    return _typename(m) == "Parallel"


def _is_recursion(m: Any) -> bool:
    return _typename(m) == "Recursion"


#known leaf module types and how their parameters map to json
#the key is type(module).__name__,
#the value is the canonical name used in the json config.
_MODULE_TYPE_MAP = {
    "parallelDelay": "parallelDelay",
    "Delay": "Delay",
    "Gain": "Gain",
    "parallelGain": "parallelGain",
    "Matrix": "Matrix",
    "parallelFilter": "parallelFilter",
    "Biquad": "Biquad",
    "SVF": "SVF",
    "GEQ": "GEQ",
    "PEQ": "PEQ",
    "AccurateGEQ": "AccurateGEQ",
    #sos filters may appear under different class names depending on flamo version
    "parallelSOSFilter": "parallelSOSFilter",
    "SOSFilter": "parallelSOSFilter",
}


def _detect_module_type(module: Any) -> str:
    """determine the canonical module type string for a leaf dsp module."""
    name = _typename(module)
    return _MODULE_TYPE_MAP.get(name, name)


#flamo constructor metadata extraction
#these fields are ignored by json_to_faust but are needed
#by json_to_flamo to reconstruct the original model without
#ambiguity. they live under a "flamo" key on each node.

#constructor arguments shared by all flamo dsp modules
_COMMON_ATTRS = ("nfft", "alias_decay_db")

#constructor arguments specific to each module type.
#each entry lists attribute names to read from the module.
_MODULE_ATTRS: dict[str, tuple[str, ...]] = {
    "parallelDelay": ("max_len", "unit", "isint"),
    "Gain": (),
    "parallelGain": (),
    "Matrix": (),
    "Biquad": ("n_sections", "filter_type"),
    "parallelBiquad": ("n_sections", "filter_type"),
    "SVF": ("n_sections", "filter_type"),
    "parallelSVF": ("n_sections", "filter_type"),
    "parallelSOSFilter": (),
    "SOSFilter": (),
    "GEQ": (),
    "PEQ": (),
    "AccurateGEQ": (),
}


def _to_native(val: Any) -> Any:
    """convert a value to a json-serialisable python native type.

    handles numpy scalars, torch tensors, and other numeric types
    that flamo modules may store as attributes.
    """
    #torch tensors (check by attribute to avoid importing torch)
    if hasattr(val, "item") and callable(val.item):
        return val.item()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float, str)):
        return val
    return None


def _extract_flamo_meta(module: Any, module_type: str) -> dict[str, Any]:
    """extract constructor arguments from a flamo module for round-tripping.

    reads the common attributes (nfft, alias_decay_db) and any
    module-specific attributes listed in _MODULE_ATTRS. also
    captures requires_grad from the parameter tensor if present.
    """
    meta: dict[str, Any] = {}

    #common attributes shared by all dsp modules
    for attr in _COMMON_ATTRS:
        val = getattr(module, attr, None)
        if val is not None:
            native = _to_native(val)
            if native is not None:
                meta[attr] = native

    #module-specific attributes
    for attr in _MODULE_ATTRS.get(module_type, ()):
        val = getattr(module, attr, None)
        if val is not None:
            native = _to_native(val)
            if native is not None:
                meta[attr] = native

    #requires_grad from the parameter tensor
    param = getattr(module, "param", None)
    if param is not None and hasattr(param, "requires_grad"):
        meta["requires_grad"] = bool(param.requires_grad)

    #size: read from the module if available, otherwise infer from
    #input/output channels. this is the constructor size argument.
    size = getattr(module, "size", None)
    if size is not None:
        if hasattr(size, "__iter__"):
            meta["size"] = [_to_native(s) or s for s in size]
        else:
            meta["size"] = [_to_native(size) or size]

    return meta


#parameter extraction

def _extract_param(module: Any) -> np.ndarray | None:
    """extract the raw parameter tensor from a flamo dsp module as a numpy array.

    flamo stores parameters in module.param (an nn.Parameter).
    returns None if the module has no param attribute.
    """
    param = getattr(module, "param", None)
    if param is None:
        return None
    #detach from autograd, move to cpu, convert to float64 numpy
    return param.detach().cpu().numpy().astype(np.float64)


def _get_fs_from_delay(module: Any) -> float | None:
    """attempt to read the sampling rate stored on a parallelDelay module."""
    return getattr(module, "fs", None)


#delay quantisation

def _quantise_delays(delays_sec: np.ndarray, fs: float) -> list[int]:
    """convert delay lengths from seconds to integer samples.

    rounds to nearest integer. fractional delay error is accepted
    (see FLAMO_RT_SPEC.md section 11, known limitation 1).
    """
    samples = np.round(delays_sec * fs).astype(int)
    return samples.ravel().tolist()


#sos normalisation

def _normalise_sos(sos: np.ndarray) -> list:
    """normalise sos coefficients so that a0 = 1 for every section and channel.

    input shape: (n_sections, 6, n_channels)
    each section stores [b0, b1, b2, a0, a1, a2].

    output: nested list with shape (n_sections, n_channels, 5)
    storing [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0] per channel.
    faust fi.tf2 expects (b0, b1, b2, a1, a2) with a0 = 1 implicit.
    """
    n_sections, six, n_channels = sos.shape
    assert six == 6, f"expected 6 coefficients per section, got {six}"

    normalised = []
    for s in range(n_sections):
        section = []
        for ch in range(n_channels):
            b0, b1, b2, a0, a1, a2 = sos[s, :, ch]
            if abs(a0) < 1e-15:
                raise ValueError(
                    f"a0 is near zero for section {s}, channel {ch}, "
                    f"filter is degenerate"
                )
            section.append([
                float(b0 / a0),
                float(b1 / a0),
                float(b2 / a0),
                float(a1 / a0),
                float(a2 / a0),
            ])
        normalised.append(section)
    return normalised


#gain shape classification

def _classify_gain(param: np.ndarray) -> str:
    """determine whether a gain parameter represents a matrix or diagonal gains.

    returns "matrix" for any 2d param that mixes or routes channels
    (including row vectors N to 1 and column vectors 1 to N).
    returns "diagonal" for 1d params and scalar (1,1) gains.
    """
    if param.ndim == 1:
        return "diagonal"
    if param.ndim == 2:
        n_out, n_in = param.shape
        if n_out == 1 and n_in == 1:
            #scalar gain, no routing needed
            return "diagonal"
        return "matrix"
    return "matrix"


#leaf node serialisation

def _serialise_leaf(module: Any, name: str, fs: float) -> dict[str, Any]:
    """extract parameters from a leaf dsp module and return a json-serialisable dict."""
    module_type = _detect_module_type(module)
    param = _extract_param(module)

    node: dict[str, Any] = {
        "type": "Leaf",
        "name": name,
        "module_type": module_type,
    }

    #channel counts
    in_ch = getattr(module, "input_channels", None)
    out_ch = getattr(module, "output_channels", None)
    if in_ch is not None:
        node["input_channels"] = int(in_ch)
    if out_ch is not None:
        node["output_channels"] = int(out_ch)

    #flamo constructor metadata for round-tripping via json_to_flamo
    flamo_meta = _extract_flamo_meta(module, module_type)
    if flamo_meta:
        node["flamo"] = flamo_meta

    if param is None:
        node["params"] = {}
        return node

    #dispatch by module type
    if module_type == "parallelDelay":
        node["params"] = {
            "samples": _quantise_delays(param, fs),
        }

    elif module_type in ("Gain", "Matrix"):
        shape_class = _classify_gain(param)
        if shape_class == "matrix":
            node["params"] = {
                "matrix": param.tolist(),
            }
        else:
            node["params"] = {
                "gains": param.ravel().tolist(),
            }

    elif module_type == "parallelGain":
        node["params"] = {
            "gains": param.ravel().tolist(),
        }

    elif module_type == "parallelSOSFilter":
        #sos shape: (n_sections, 6, n_channels)
        if param.ndim == 3 and param.shape[1] == 6:
            node["params"] = {
                "sos": _normalise_sos(param),
            }
        else:
            #unexpected shape, store raw for debugging
            node["params"] = {
                "raw": param.tolist(),
            }

    else:
        #unknown module type, store raw parameter values
        node["params"] = {
            "raw": param.tolist(),
        }

    return node


#series children extraction (same logic as pyFDN/flamo_graph.py)

def _series_children(module: Any) -> list[tuple[str, Any]]:
    """return (name, submodule) pairs for a Series (nn.Sequential) module."""
    modules = getattr(module, "_modules", None)
    if modules is None:
        modules = getattr(module, "modules", None)
    if modules is None:
        return []
    if hasattr(modules, "items"):
        return list(modules.items())
    if hasattr(modules, "__iter__") and not isinstance(modules, (str, bytes)):
        return [(str(i), m) for i, m in enumerate(modules)]
    return []


#shell core extraction

def _get_shell_core(model: Any) -> Any | None:
    """extract the core module from a Shell, trying multiple access patterns."""
    if callable(getattr(model, "get_core", None)):
        core = model.get_core()
        if core is not None:
            return core
    for attr in ("core", "_Shell__core"):
        core = getattr(model, attr, None)
        if core is not None:
            return core
    return None


#recursive graph traversal to json dict

def _traverse(model: Any, name: str, fs: float) -> dict[str, Any]:
    """recursively traverse a flamo model and produce a json-serialisable tree.

    handles Shell, Series, Parallel, Recursion, and leaf dsp modules.
    """
    #shell: skip fft/ifft io layers, descend into core
    if _is_shell(model):
        node: dict[str, Any] = {
            "type": "Shell",
            "name": name,
        }
        core = _get_shell_core(model)
        if core is not None:
            node["children"] = [_traverse(core, "core", fs)]
        else:
            node["children"] = []

        #capture nfft from the io layers for round-tripping
        flamo_meta: dict[str, Any] = {}
        input_layer = getattr(model, "input_layer", None)
        if input_layer is not None:
            nfft = getattr(input_layer, "nfft", None)
            if nfft is not None:
                flamo_meta["nfft"] = int(nfft)
        if flamo_meta:
            node["flamo"] = flamo_meta

        return node

    #series: ordered sequence of children
    if _is_series(model):
        pairs = _series_children(model)
        return {
            "type": "Series",
            "name": name,
            "children": [_traverse(sub, nm, fs) for nm, sub in pairs],
        }

    #parallel: two branches (brA, brB)
    if _is_parallel(model):
        node = {
            "type": "Parallel",
            "name": name,
            "children": [],
        }
        #check if parallel sums outputs (for faust :> vs ,)
        sum_output = getattr(model, "sum_output", None)
        if sum_output is not None:
            node["sum_output"] = bool(sum_output)

        for attr in ("brA", "brB", "branchA", "branchB"):
            branch = getattr(model, attr, None)
            if branch is not None:
                #normalise name to brA/brB
                br_name = "brA" if "A" in attr else "brB"
                node["children"].append(_traverse(branch, br_name, fs))
        return node

    #recursion: forward path fF and feedback path fB
    if _is_recursion(model):
        node = {
            "type": "Recursion",
            "name": name,
        }
        fF = getattr(model, "fF", None) or getattr(model, "feedforward", None)
        fB = getattr(model, "fB", None) or getattr(model, "feedback", None)
        node["fF"] = _traverse(fF, "fF", fs) if fF is not None else None
        node["fB"] = _traverse(fB, "fB", fs) if fB is not None else None
        return node

    #leaf: dsp processing module
    return _serialise_leaf(model, name, fs)


#public api

def flamo_to_json(
    model: Any,
    fs: float,
    *,
    name: str = "FDN",
) -> dict[str, Any]:
    """extract a json-serialisable config dict from a flamo model.

    traverses the full model graph (Shell, Series, Parallel, Recursion, Leaf),
    extracts parameter values from each leaf module, detects module types, and
    returns a nested dict matching the schema in FLAMO_RT_SPEC.md section 6.

    parameters
    ----------
    model : flamo model
        a Shell, Recursion, Series, Parallel, or leaf dsp module.
    fs : float
        sampling rate in hz. used to convert delay lengths from seconds
        to integer samples.
    name : str
        name for the root node in the config (default "FDN").

    returns
    -------
    config : dict
        json-serialisable nested dict. pass to json.dumps() or to json_to_faust().
    """
    config = _traverse(model, name, fs)
    config["fs"] = int(fs)
    return config
