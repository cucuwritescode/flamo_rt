#json_to_faust
#author: Facundo Franchino
"""
generate valid faust dsp code from a json config dict

consumes the output of flamo_to_json() and produces a complete .dsp file.
each node type (Shell, Series, Parallel, Recursion, Leaf) maps to a
faust composition operator, and each leaf module type maps to a faust
dsp expression as defined in FLAMO_RT_SPEC.md section 5.

the generated code is deterministic: same config always yields same output.
"""

from __future__ import annotations

from typing import Any


#number formatting

def _fmt(x: float) -> str:
    """format a numeric value for faust source code.

    integers are written without decimal points.
    floats keep enough precision to be faithful to the original.
    """
    if isinstance(x, int) or (isinstance(x, float) and x == int(x)):
        return str(int(x))
    return f"{x:.15g}"


#matrix row arithmetic

def _build_matrix_row(row: list[float], n_in: int) -> str:
    """build a faust arithmetic expression for one row of a mixing matrix.

    handles sign properly so output reads e.g. 0.5*x0 - 0.5*x1
    rather than 0.5*x0 + -0.5*x1.
    """
    parts: list[str] = []
    for j in range(n_in):
        coeff = row[j]
        if coeff == 0.0:
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
        return "0.0"
    return " ".join(parts)


#leaf module emitters
#each function takes a node dict and returns a faust expression string.
#the expression operates on N parallel signal channels.

def _emit_parallel_delay(node: dict[str, Any]) -> str:
    """parallel delay lines: one @(samples) per channel.

    faust @(n) delays a signal by n samples.
    channels are composed in parallel with the , operator.
    """
    samples = node["params"]["samples"]
    n = len(samples)
    if n == 1:
        return f"@({samples[0]})"
    channels = [f"@({s})" for s in samples]
    return "(" + " , ".join(channels) + ")"


def _emit_diagonal_gain(node: dict[str, Any]) -> str:
    """diagonal (per-channel) gains: *(g0), *(g1), ...

    each channel is multiplied by its own gain coefficient.
    """
    gains = node["params"]["gains"]
    n = len(gains)
    if n == 1:
        return f"*({_fmt(gains[0])})"
    channels = [f"*({_fmt(g)})" for g in gains]
    return "(" + " , ".join(channels) + ")"


def _emit_matrix_gain(node: dict[str, Any]) -> str:
    """dense mixing matrix: sum-of-products per output channel.

    for an NxM matrix A, output channel i = sum_j(A[i][j] * input_j).
    in faust this is expressed as a function taking M inputs and
    producing N outputs using the si.bus routing.

    the matrix is emitted as a named local expression that takes
    n_in signal arguments and produces n_out signals.
    """
    matrix = node["params"]["matrix"]
    n_out = len(matrix)
    n_in = len(matrix[0])

    args = ", ".join(f"x{j}" for j in range(n_in))
    rows = [_build_matrix_row(matrix[i], n_in) for i in range(n_out)]
    body = " , ".join(rows)
    return f"si.bus({n_in}) : _,_ -> ({args}) {{ {body} }}"


def _emit_matrix_as_function(node: dict[str, Any]) -> tuple[str, str]:
    """emit a matrix as a separate faust function definition.

    returns (function_name, function_definition_string).
    the caller can place the definition at the top of the dsp file
    and use the function name inline.

    the function signature is: name(x0, x1, ...) = row0, row1, ...;
    """
    matrix = node["params"]["matrix"]
    n_out = len(matrix)
    n_in = len(matrix[0])
    name = _safe_name(node.get("name", "matrix"))

    args = ", ".join(f"x{j}" for j in range(n_in))
    rows = [_build_matrix_row(matrix[i], n_in) for i in range(n_out)]
    body = ", ".join(rows)
    definition = f"{name}({args}) = {body};"
    return name, definition


def _emit_sos_filter(node: dict[str, Any]) -> str:
    """second-order section (biquad) filters per channel.

    each section is a fi.tf2(b0, b1, b2, a1, a2) cascaded in series.
    channels are composed in parallel.

    the sos data is already normalised (a0 = 1) by flamo_to_json.
    shape: sos[section][channel] = [b0, b1, b2, a1, a2].
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
            channels.append(" : ".join(sections))

    if n_channels == 1:
        return channels[0]
    return "(" + " , ".join(channels) + ")"


#safe naming for faust identifiers

def _safe_name(name: str) -> str:
    """convert a node name to a valid faust identifier.

    replaces non-alphanumeric characters with underscores.
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


#recursive code generation from the json config tree

class _FaustEmitter:
    """walks a json config tree and collects faust code.

    separates concerns: leaf emitters produce expressions,
    the emitter handles composition and collects top-level definitions
    (like matrix functions) that need to be hoisted.
    """

    def __init__(self):
        #top-level function definitions collected during traversal
        self.definitions: list[str] = []

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
        """shell: skip the fft/ifft wrapper, emit the core only."""
        children = node.get("children", [])
        if not children:
            return "_"
        #shell has exactly one child: the core
        return self.emit(children[0])

    def _emit_series(self, node: dict[str, Any]) -> str:
        """series composition: a : b : c"""
        children = node.get("children", [])
        if not children:
            return "_"
        parts = [self.emit(child) for child in children]
        if len(parts) == 1:
            return parts[0]
        return "(" + " : ".join(parts) + ")"

    def _emit_parallel(self, node: dict[str, Any]) -> str:
        """parallel composition: a , b or a , b :> _ (if summing)."""
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
        """recursion (feedback): fF ~ fB"""
        ff_node = node.get("fF")
        fb_node = node.get("fB")

        ff_expr = self.emit(ff_node) if ff_node else "_"
        fb_expr = self.emit(fb_node) if fb_node else "_"

        return f"({ff_expr} ~ {fb_expr})"

    def _emit_leaf(self, node: dict[str, Any]) -> str:
        """dispatch to the correct leaf emitter based on module_type."""
        module_type = node.get("module_type", "")
        params = node.get("params", {})

        if module_type == "parallelDelay":
            return _emit_parallel_delay(node)

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

        if module_type == "parallelSOSFilter":
            return _emit_sos_filter(node)

        #unknown module type with no specific handler
        #emit a wire (passthrough) and leave a comment in the definitions
        self.definitions.append(
            f"// warning: no codegen for module type '{module_type}' "
            f"(node '{node.get('name', '?')}')"
        )
        return "_"


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

    #header
    lines.append(f"// {name}")
    lines.append(f"// sample rate: {fs} hz")
    lines.append("")
    lines.append('import("stdfaust.lib");')
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
