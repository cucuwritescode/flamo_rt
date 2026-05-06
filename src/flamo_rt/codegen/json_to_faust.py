#json_to_faust
#author: Facundo Franchino
"""
generate valid faust dsp code from a json config dict

consumes the output of flamo_to_json() and produces a complete .dsp file.
each node type (Shell, Series, Parallel, Recursion, Leaf) maps to a
faust composition operator, and each leaf module type maps to a faust
dsp expression as defined in FLAMO_RT_SPEC.md section 5.

the generated code is deterministic: same config always yields same output.

note on fdn topology:
    faust's ~ operator introduces exactly one sample of implicit delay
    in the feedback path. when generating delay lines that participate
    in a recursive structure, we must subtract one from the requested
    delay length to compensate. the _emit_parallel_delay function
    receives a flag indicating whether it sits inside a recursion.
"""

from __future__ import annotations

from typing import Any


#number formatting

def _fmt(x: float) -> str:
    """format a numeric value for faust source code.

    integers are written without decimal points (1000 not 1000.0).
    floats use ten significant figures, which eliminates representation
    noise (0.30000000000000004 becomes 0.3) while preserving more than
    enough precision for audio coefficients.
    """
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return f"{x:.10g}"


#matrix row arithmetic

def _build_matrix_row(row: list[float], n_in: int) -> str:
    """build a faust arithmetic expression for one row of a mixing matrix.

    handles sign properly so output reads e.g. 0.5*x0 - 0.5*x1
    rather than 0.5*x0 + -0.5*x1. this makes the generated code
    more readable and matches hand-written faust style.
    """
    parts: list[str] = []
    for j in range(n_in):
        coeff = row[j]
        if coeff == 0.0:
            #zero coefficients contribute nothing, skip entirely
            continue
        #format the term without its sign
        abs_coeff = abs(coeff)
        if abs_coeff == 1.0:
            term = f"x{j}"
        else:
            term = f"{_fmt(abs_coeff)}*x{j}"
        #first non-zero term carries its sign directly
        if not parts:
            if coeff < 0:
                parts.append(f"-{term}")
            else:
                parts.append(term)
        else:
            if coeff < 0:
                parts.append(f"- {term}")
            else:
                parts.append(f"+ {term}")
    if not parts:
        #entire row is zeros, output literal zero
        return "0.0"
    return " ".join(parts)


#leaf module emitters
#each function takes a node dict and returns a faust expression string.
#the expression operates on N parallel signal channels.

def _emit_parallel_delay(node: dict[str, Any], in_recursion: bool = False) -> str:
    """parallel delay lines: one @(samples) per channel.

    faust @(n) delays a signal by n samples.
    channels are composed in parallel with the , operator.

    when in_recursion is true, we subtract one sample from each delay
    to compensate for the implicit one-sample delay introduced by
    the ~ operator. this ensures the total delay matches the original
    flamo specification. we clamp to zero to avoid negative delays.
    """
    samples = node["params"]["samples"]
    n = len(samples)

    #compute effective delays, accounting for recursion offset
    if in_recursion:
        effective = [max(0, s - 1) for s in samples]
    else:
        effective = list(samples)

    if n == 1:
        return f"@({effective[0]})"
    channels = [f"@({s})" for s in effective]
    return "(" + " , ".join(channels) + ")"


def _emit_variable_delay(node: dict[str, Any], in_recursion: bool = False) -> str:
    """variable delay using de.delay for runtime-variable delay times.

    this is used when delays may change at runtime or when fractional
    delays are needed. for fixed delays, @(n) is more efficient.

    the maxdelay is set to the next power of two above the requested
    delay, as faust delay buffers must be powers of two.
    """
    samples = node["params"]["samples"]
    n = len(samples)

    #find the maximum delay to size the buffer
    max_delay = max(samples)
    #round up to next power of two for efficient modulo
    buffer_size = 1
    while buffer_size < max_delay:
        buffer_size *= 2

    #compute effective delays, accounting for recursion offset
    if in_recursion:
        effective = [max(0, s - 1) for s in samples]
    else:
        effective = list(samples)

    if n == 1:
        return f"de.delay({buffer_size}, {effective[0]})"
    channels = [f"de.delay({buffer_size}, {s})" for s in effective]
    return "(" + " , ".join(channels) + ")"


def _emit_fractional_delay(node: dict[str, Any], in_recursion: bool = False) -> str:
    """fractional delay using de.fdelay with linear interpolation.

    used when delay times are not integer samples. faust's fdelay
    interpolates between adjacent samples for smooth delay modulation.
    """
    samples = node["params"].get("samples_fractional", node["params"].get("samples", []))
    n = len(samples)

    #find the maximum delay to size the buffer
    max_delay = max(samples) if samples else 1
    max_delay_int = int(max_delay) + 2  #headroom for interpolation
    #round up to next power of two
    buffer_size = 1
    while buffer_size < max_delay_int:
        buffer_size *= 2

    #compute effective delays, accounting for recursion offset
    if in_recursion:
        effective = [max(0.0, s - 1.0) for s in samples]
    else:
        effective = list(samples)

    if n == 1:
        return f"de.fdelay({buffer_size}, {_fmt(effective[0])})"
    channels = [f"de.fdelay({buffer_size}, {_fmt(s)})" for s in effective]
    return "(" + " , ".join(channels) + ")"


def _emit_diagonal_gain(node: dict[str, Any]) -> str:
    """diagonal (per-channel) gains: *(g0), *(g1), ...

    each channel is multiplied by its own gain coefficient.
    this is the efficient form when no cross-channel mixing occurs.
    """
    gains = node["params"]["gains"]
    n = len(gains)
    if n == 1:
        return f"*({_fmt(gains[0])})"
    channels = [f"*({_fmt(g)})" for g in gains]
    return "(" + " , ".join(channels) + ")"


def _emit_matrix_as_function(node: dict[str, Any]) -> tuple[str, str]:
    """emit a matrix as a separate faust function definition.

    returns (function_name, function_definition_string).
    the caller places the definition at the top of the dsp file
    and uses the function name inline.

    the function signature is: name(x0, x1, ...) = row0, row1, ...;
    this expands to explicit sum-of-products for each output channel,
    which faust compiles efficiently.
    """
    matrix = node["params"]["matrix"]
    n_out = len(matrix)
    n_in = len(matrix[0])
    name = _safe_name(node.get("name", "matrix"))

    args = ", ".join(f"x{j}" for j in range(n_in))
    rows = [_build_matrix_row(matrix[i], n_in) for i in range(n_out)]
    body = ",\n    ".join(rows)
    definition = f"{name}({args}) =\n    {body};"
    return name, definition


def _emit_sos_filter(node: dict[str, Any]) -> str:
    """second-order section (biquad) filters per channel.

    each section is a fi.tf2(b0, b1, b2, a1, a2) cascaded in series.
    channels are composed in parallel.

    the sos data is already normalised (a0 = 1) by flamo_to_json.
    shape: sos[section][channel] = [b0, b1, b2, a1, a2].

    cascading multiple biquads in series builds higher-order filters
    while maintaining numerical stability.
    """
    sos = node["params"]["sos"]
    n_sections = len(sos)
    n_channels = len(sos[0])

    channels = []
    for ch in range(n_channels):
        #cascade sections in series for this channel
        sections = []
        for s in range(n_sections):
            coeffs = sos[s][ch]
            b0, b1, b2, a1, a2 = coeffs
            sections.append(
                f"fi.tf2({_fmt(b0)}, {_fmt(b1)}, {_fmt(b2)}, "
                f"{_fmt(a1)}, {_fmt(a2)})"
            )
        if len(sections) == 1:
            channels.append(sections[0])
        else:
            #cascade: section0 : section1 : ...
            channels.append("(" + " : ".join(sections) + ")")

    if n_channels == 1:
        return channels[0]
    return "(" + " , ".join(channels) + ")"


def _emit_biquad_filter(node: dict[str, Any]) -> str:
    """single biquad filter (second-order iir).

    expects params with keys: b0, b1, b2, a1, a2 (normalised, a0=1)
    or a coeffs list [b0, b1, b2, a1, a2].
    """
    params = node["params"]

    #handle both dict and list coefficient formats
    if "coeffs" in params:
        b0, b1, b2, a1, a2 = params["coeffs"]
    else:
        b0 = params.get("b0", 1.0)
        b1 = params.get("b1", 0.0)
        b2 = params.get("b2", 0.0)
        a1 = params.get("a1", 0.0)
        a2 = params.get("a2", 0.0)

    return f"fi.tf2({_fmt(b0)}, {_fmt(b1)}, {_fmt(b2)}, {_fmt(a1)}, {_fmt(a2)})"


def _emit_svf_filter(node: dict[str, Any]) -> str:
    """state variable filter.

    svf provides simultaneous lowpass, highpass, bandpass outputs.
    we emit the appropriate fi.svf variant based on the mode parameter.
    defaults to lowpass if mode is unspecified.
    """
    params = node["params"]
    fc = params.get("fc", 1000.0)
    q = params.get("q", 0.707)
    mode = params.get("mode", "lowpass")

    #map mode to faust svf function
    mode_map = {
        "lowpass": "fi.svf.lp",
        "highpass": "fi.svf.hp",
        "bandpass": "fi.svf.bp",
        "notch": "fi.svf.notch",
        "allpass": "fi.svf.ap",
    }
    svf_func = mode_map.get(mode, "fi.svf.lp")
    return f"{svf_func}({_fmt(fc)}, {_fmt(q)})"


def _emit_lowpass_filter(node: dict[str, Any]) -> str:
    """butterworth lowpass filter.

    uses fi.lowpass(order, cutoff_freq).
    """
    params = node["params"]
    order = params.get("order", 2)
    fc = params.get("fc", 1000.0)
    return f"fi.lowpass({order}, {_fmt(fc)})"


def _emit_highpass_filter(node: dict[str, Any]) -> str:
    """butterworth highpass filter.

    uses fi.highpass(order, cutoff_freq).
    """
    params = node["params"]
    order = params.get("order", 2)
    fc = params.get("fc", 1000.0)
    return f"fi.highpass({order}, {_fmt(fc)})"


def _emit_bandpass_filter(node: dict[str, Any]) -> str:
    """butterworth bandpass filter.

    uses fi.bandpass(order, low_freq, high_freq).
    """
    params = node["params"]
    order = params.get("order", 2)
    fl = params.get("fl", 100.0)
    fh = params.get("fh", 1000.0)
    return f"fi.bandpass({order}, {_fmt(fl)}, {_fmt(fh)})"


def _emit_peak_eq(node: dict[str, Any]) -> str:
    """parametric peaking equaliser.

    uses fi.peak_eq(gain_db, centre_freq, bandwidth).
    """
    params = node["params"]
    gain_db = params.get("gain_db", 0.0)
    fc = params.get("fc", 1000.0)
    bw = params.get("bandwidth", 100.0)
    return f"fi.peak_eq({_fmt(gain_db)}, {_fmt(fc)}, {_fmt(bw)})"


def _emit_allpass_comb(node: dict[str, Any]) -> str:
    """allpass comb filter (schroeder allpass).

    used in reverb networks for diffusion without colouration.
    uses fi.allpass_comb(maxdelay, delay, feedback).
    """
    params = node["params"]
    delay = params.get("delay", 100)
    feedback = params.get("feedback", 0.5)
    #buffer size must be power of two and larger than delay
    buffer_size = 1
    while buffer_size <= delay:
        buffer_size *= 2
    return f"fi.allpass_comb({buffer_size}, {delay}, {_fmt(feedback)})"


def _emit_dc_blocker(node: dict[str, Any]) -> str:
    """dc blocking filter to remove dc offset.

    uses fi.dcblocker which is a one-pole highpass at very low frequency.
    """
    return "fi.dcblocker"


def _emit_onepole(node: dict[str, Any]) -> str:
    """one-pole lowpass filter.

    simple exponential smoothing: y[n] = x[n] + p*y[n-1]
    uses fi.pole(coefficient).
    """
    params = node["params"]
    p = params.get("pole", 0.9)
    return f"fi.pole({_fmt(p)})"


#safe naming for faust identifiers

def _safe_name(name: str) -> str:
    """convert a node name to a valid faust identifier.

    replaces non-alphanumeric characters with underscores.
    faust identifiers must start with a letter or underscore,
    so we prepend underscore if the name starts with a digit.
    """
    result = []
    for c in name:
        if c.isalnum() or c == "_":
            result.append(c)
        else:
            result.append("_")
    #faust identifiers must start with a letter or underscore
    if result and result[0].isdigit():
        result.insert(0, "_")
    return "".join(result) or "_unnamed"


#channel count inference for recursion routing

def _get_channel_count(node: dict[str, Any] | None) -> int | None:
    """infer the output channel count from a json config node.

    checks output_channels on leaf nodes, and recurses into
    container nodes to find a leaf with channel information.
    this is necessary for constructing the correct number of
    adders at the recursion input.
    """
    if node is None:
        return None
    #leaf nodes carry channel counts directly
    out_ch = node.get("output_channels")
    if out_ch is not None:
        return int(out_ch)
    #for matrices, infer from params
    params = node.get("params", {})
    if "matrix" in params:
        return len(params["matrix"])
    if "gains" in params:
        return len(params["gains"])
    if "samples" in params:
        return len(params["samples"])
    #for container nodes, check children or fF/fB
    children = node.get("children", [])
    if children:
        #last child's output is the container's output
        return _get_channel_count(children[-1])
    ff = node.get("fF")
    if ff is not None:
        return _get_channel_count(ff)
    return None


#recursive code generation from the json config tree

class _FaustEmitter:
    """walks a json config tree and collects faust code.

    separates concerns: leaf emitters produce expressions,
    the emitter handles composition and collects top-level definitions
    (like matrix functions) that need to be hoisted.

    the in_recursion flag tracks whether we are inside a recursion
    node, which affects delay offset calculation.
    """

    def __init__(self):
        #top-level function definitions collected during traversal
        self.definitions: list[str] = []
        #track recursion depth for delay offset
        self._in_recursion: bool = False

    def emit(self, node: dict[str, Any]) -> str:
        """dispatch to the appropriate handler based on node type."""
        node_type = node.get("type", "Leaf")

        if node_type == "Shell":
            return self._emit_shell(node)
        if node_type == "Series":
            return self._emit_series(node)
        if node_type == "Parallel":
            return self._emit_parallel(node)
        if node_type == "Recursion":
            return self._emit_recursion(node)
        if node_type == "Leaf":
            return self._emit_leaf(node)

        raise ValueError(f"unknown node type: {node_type}")

    def _emit_shell(self, node: dict[str, Any]) -> str:
        """shell: skip the fft/ifft wrapper, emit the core only.

        flamo's shell wraps a time-domain core with frequency-domain
        io layers. for faust we emit only the core, as faust operates
        purely in the time domain.
        """
        children = node.get("children", [])
        if not children:
            return "_"
        #shell has exactly one child: the core
        return self.emit(children[0])

    def _emit_series(self, node: dict[str, Any]) -> str:
        """series composition: a : b : c

        sequential chaining where the output of each stage feeds
        the input of the next. the : operator in faust.
        """
        children = node.get("children", [])
        if not children:
            return "_"
        parts = [self.emit(child) for child in children]
        if len(parts) == 1:
            return parts[0]
        return "(" + " : ".join(parts) + ")"

    def _emit_parallel(self, node: dict[str, Any]) -> str:
        """parallel composition: a , b or a , b :> _ (if summing).

        side-by-side composition where inputs and outputs are
        concatenated. if sum_output is true, we add :> _ to
        sum all outputs to a single channel.
        """
        children = node.get("children", [])
        if not children:
            return "_"
        parts = [self.emit(child) for child in children]
        if len(parts) == 1:
            return parts[0]
        parallel_expr = " , ".join(parts)
        sum_output = node.get("sum_output", False)
        if sum_output:
            return f"({parallel_expr} :> _)"
        return f"({parallel_expr})"

    def _emit_recursion(self, node: dict[str, Any]) -> str:
        """recursion (feedback): (interleave : adders : fF) ~ fB

        faust's ~ operator feeds fB's m outputs back to the first m
        inputs of A, with the remaining n inputs as external inputs.
        we prepend par(i,N,+) so that each adder sums one feedback
        signal with one external signal.

        for N>1 channels, the input layout of par(i,N,+) is:
            [add0_a, add0_b, add1_a, add1_b, ...]
        but ~ delivers feedback to the first N inputs contiguously:
            [fb0, fb1, ..., fbN-1, ext0, ext1, ..., extN-1]
        without interleaving, adder 0 would get (fb0 + fb1) instead
        of (fb0 + ext0). ro.interleave(N,2) reshuffles inputs so
        each adder receives its corresponding feedback and external
        signal as a pair.

        for N=1, the single + has two inputs [fb, ext] and no
        interleaving is needed.

        the ~ operator introduces exactly one sample of implicit delay
        in the feedback path. we set _in_recursion=True so that delay
        emitters compensate by subtracting one from their delay times.
        """
        ff_node = node.get("fF")
        fb_node = node.get("fB")

        #mark that we are inside a recursion so delays compensate
        old_in_recursion = self._in_recursion
        self._in_recursion = True

        ff_expr = self.emit(ff_node) if ff_node else "_"

        #feedback path is not affected by the delay offset
        self._in_recursion = old_in_recursion
        fb_expr = self.emit(fb_node) if fb_node else "_"

        #determine the feedback channel count from fB's output or fF's input
        n_fb = _get_channel_count(fb_node)

        if n_fb is not None and n_fb > 0:
            #prepend adders so external inputs can enter the feedback loop
            if n_fb == 1:
                adders = "+"
            else:
                #interleave feedback and external signals so each adder
                #receives one of each: [fb0,ext0, fb1,ext1, ...]
                adders = f"ro.interleave({n_fb}, 2) : par(i, {n_fb}, +)"
            return f"(({adders} : {ff_expr}) ~ {fb_expr})"

        return f"({ff_expr} ~ {fb_expr})"

    def _emit_leaf(self, node: dict[str, Any]) -> str:
        """dispatch to the correct leaf emitter based on module_type.

        each module type has its own emitter function that produces
        the appropriate faust expression. unknown types become
        passthrough wires with a warning comment.
        """
        module_type = node.get("module_type", "")
        params = node.get("params", {})

        #delay modules
        if module_type == "parallelDelay":
            #check if fractional delays are present
            if "samples_fractional" in params:
                return _emit_fractional_delay(node, self._in_recursion)
            return _emit_parallel_delay(node, self._in_recursion)

        if module_type == "Delay":
            #single-channel delay, treat as parallel with one channel
            return _emit_parallel_delay(node, self._in_recursion)

        if module_type == "variableDelay":
            return _emit_variable_delay(node, self._in_recursion)

        if module_type == "fractionalDelay":
            return _emit_fractional_delay(node, self._in_recursion)

        #gain modules
        if module_type in ("Gain", "Matrix"):
            if "matrix" in params:
                #hoist the matrix as a top-level function definition
                func_name, func_def = _emit_matrix_as_function(node)
                self.definitions.append(func_def)
                return func_name
            if "gains" in params:
                return _emit_diagonal_gain(node)
            #no params, pass through
            return "_"

        if module_type == "parallelGain":
            return _emit_diagonal_gain(node)

        #filter modules
        if module_type == "parallelSOSFilter":
            return _emit_sos_filter(node)

        if module_type == "Biquad":
            return _emit_biquad_filter(node)

        if module_type == "SVF":
            return _emit_svf_filter(node)

        if module_type == "lowpass":
            return _emit_lowpass_filter(node)

        if module_type == "highpass":
            return _emit_highpass_filter(node)

        if module_type == "bandpass":
            return _emit_bandpass_filter(node)

        if module_type in ("PEQ", "peakEQ"):
            return _emit_peak_eq(node)

        if module_type == "allpassComb":
            return _emit_allpass_comb(node)

        if module_type == "dcBlocker":
            return _emit_dc_blocker(node)

        if module_type == "onePole":
            return _emit_onepole(node)

        #parallel filter applies the same filter to each channel
        if module_type == "parallelFilter":
            #extract the inner filter type and emit it for each channel
            inner_type = params.get("filter_type", "lowpass")
            n_ch = node.get("output_channels", 1)
            inner_node = {
                "module_type": inner_type,
                "params": params,
            }
            inner_expr = self._emit_leaf(inner_node)
            if n_ch == 1:
                return inner_expr
            channels = [inner_expr] * n_ch
            return "(" + " , ".join(channels) + ")"

        #unknown module type with no specific handler
        #emit a wire (passthrough) and leave a comment in the definitions
        self.definitions.append(
            f"//warning: no codegen for module type '{module_type}' "
            f"(node '{node.get('name', '?')}')"
        )
        return "_"


#topology description

def _collect_nodes(node: dict[str, Any], type_filter: str) -> list[dict]:
    """collect all nodes of a given type from the config tree."""
    found = []
    if node.get("type") == type_filter or node.get("module_type") == type_filter:
        found.append(node)
    for child in node.get("children", []):
        found.extend(_collect_nodes(child, type_filter))
    for key in ("fF", "fB"):
        sub = node.get(key)
        if sub is not None:
            found.extend(_collect_nodes(sub, type_filter))
    return found


def _describe_topology(config: dict[str, Any], fs: int) -> list[str]:
    """generate human-readable topology comments from the config tree.

    returns a list of comment strings (without the // prefix).
    returns an empty list if the config has no interesting structure.
    """
    desc: list[str] = []

    #check for recursion (fdn feedback loops)
    recursions = _collect_nodes(config, "Recursion")
    if not recursions:
        return desc

    desc.append("fdn topology: (adders : delays : filters) ~ feedback_matrix")
    desc.append("the ~ operator adds one sample implicit delay in feedback")
    desc.append("delay lengths are adjusted to compensate for this")

    #delay summary
    delays = _collect_nodes(config, "parallelDelay")
    if delays:
        all_samples = []
        for d in delays:
            all_samples.extend(d.get("params", {}).get("samples", []))
        if all_samples:
            n_ch = len(all_samples)
            d_min = min(all_samples)
            d_max = max(all_samples)
            ms_min = d_min / fs * 1000
            ms_max = d_max / fs * 1000
            desc.append(f"{n_ch} delay lines: {d_min}-{d_max} samples "
                        f"({ms_min:.1f}-{ms_max:.1f} ms)")

    #filter summary
    sos_nodes = _collect_nodes(config, "parallelSOSFilter")
    if sos_nodes:
        total_sections = 0
        for s in sos_nodes:
            sos = s.get("params", {}).get("sos", [])
            total_sections += len(sos)
        desc.append(f"absorption: {total_sections} biquad section"
                    f"{'s' if total_sections != 1 else ''} per channel")

    #feedback matrix summary
    for rec in recursions:
        fb = rec.get("fB")
        if fb and "matrix" in fb.get("params", {}):
            matrix = fb["params"]["matrix"]
            n = len(matrix)
            desc.append(f"feedback matrix: {n}x{n}")

    return desc


#public api

def json_to_faust(config: dict[str, Any]) -> str:
    """generate a complete faust .dsp source file from a json config dict.

    the config dict is the output of flamo_to_json(). the returned string
    is valid faust code ready to be written to a .dsp file or passed to
    the faust interpreter.

    parameters
    ----------
    config : dict
        json config dict as produced by flamo_to_json().

    returns
    -------
    faust_code : str
        complete faust dsp source code.
    """
    name = config.get("name", "untitled")
    fs = config.get("fs", 48000)

    emitter = _FaustEmitter()
    process_expr = emitter.emit(config)

    #assemble the complete dsp file
    lines = []

    #header with metadata
    lines.append(f"//{name}")
    lines.append(f"//sample rate: {fs} hz")
    lines.append("")

    #faust standard library import
    lines.append('import("stdfaust.lib");')
    lines.append("")

    #topology documentation extracted from the config tree.
    #summarises the fdn structure so the generated code is readable
    #without cross-referencing the json config.
    topo = _describe_topology(config, fs)
    if topo:
        for line in topo:
            lines.append(f"//{line}")
        lines.append("")

    #hoisted definitions (matrices, warnings)
    if emitter.definitions:
        for defn in emitter.definitions:
            lines.append(defn)
        lines.append("")

    #process assignment
    lines.append(f"process = {process_expr};")
    lines.append("")

    return "\n".join(lines)
