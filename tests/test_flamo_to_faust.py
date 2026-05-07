#test_flamo_to_faust
#author: Facundo Franchino
"""
integration tests for the full pipeline: flamo model to faust code.

tests the convenience function flamo_to_faust() end-to-end,
verifying that a mock flamo model produces correct faust output
with all parameters preserved through the json intermediate step.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

import numpy as np
import pytest

from flamo_rt.codegen.flamo_to_faust import flamo_to_faust
from flamo_rt.codegen.flamo_to_json import flamo_to_json
from flamo_rt.codegen.json_to_faust import json_to_faust


#mock flamo modules (same as test_flamo_to_json.py)

class _MockParam:
    """stand-in for torch nn.Parameter, supports detach().cpu().numpy()."""

    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values.copy()


class parallelDelay:
    def __init__(self, delays_sec: np.ndarray, fs: float):
        self.param = _MockParam(delays_sec)
        self.fs = fs
        n = len(delays_sec)
        self.input_channels = n
        self.output_channels = n


class Gain:
    def __init__(self, values: np.ndarray):
        self.param = _MockParam(values)
        if values.ndim == 2:
            self.output_channels, self.input_channels = values.shape
        else:
            self.input_channels = len(values)
            self.output_channels = len(values)


class parallelGain:
    def __init__(self, values: np.ndarray):
        self.param = _MockParam(values.ravel())
        self.input_channels = len(values)
        self.output_channels = len(values)


class parallelSOSFilter:
    def __init__(self, sos: np.ndarray):
        self.param = _MockParam(sos)
        self.input_channels = sos.shape[2]
        self.output_channels = sos.shape[2]


class Series:
    def __init__(self, children: OrderedDict):
        self._modules = children
        last = list(children.values())[-1] if children else None
        self.input_channels = getattr(list(children.values())[0], "input_channels", None) if children else None
        self.output_channels = getattr(last, "output_channels", None)


class Parallel:
    def __init__(self, brA: Any, brB: Any, sum_output: bool = False):
        self.brA = brA
        self.brB = brB
        self.sum_output = sum_output
        self.input_channels = getattr(brA, "input_channels", None)
        self.output_channels = getattr(brA, "output_channels", None)


class Recursion:
    def __init__(self, fF: Any, fB: Any):
        self.fF = fF
        self.fB = fB
        self.input_channels = getattr(fF, "input_channels", None)
        self.output_channels = getattr(fF, "output_channels", None)


class Shell:
    def __init__(self, core: Any):
        self._Shell__core = core
        self.input_channels = getattr(core, "input_channels", None)
        self.output_channels = getattr(core, "output_channels", None)

    def get_core(self):
        return self._Shell__core


#test fixtures

def _build_fdn_model(N: int = 4, fs: float = 48000.0):
    """build a standard N-channel fdn model matching dss_to_flamo structure.

    returns (model, fs, params_dict) where params_dict contains
    the original numpy arrays for verification.
    """
    #prime delay lengths for good diffusion
    delay_samples = {
        4: np.array([1103, 1447, 1811, 2137]),
        8: np.array([1103, 1447, 1811, 2137, 2473, 2693, 2999, 3319]),
    }[N]
    delays_sec = delay_samples / fs

    #hadamard feedback matrix (normalised, orthogonal)
    if N == 4:
        A = 0.5 * np.array([
            [1,  1,  1,  1],
            [1, -1,  1, -1],
            [1,  1, -1, -1],
            [1, -1, -1,  1],
        ], dtype=np.float64)
    else:
        #generic orthogonal for N=8 via normalised hadamard
        from scipy.linalg import hadamard
        A = hadamard(N).astype(np.float64) / np.sqrt(N)

    B = np.ones((N, 1), dtype=np.float64)
    C = np.ones((1, N), dtype=np.float64) / N
    D = np.zeros((1, 1), dtype=np.float64)

    #absorption: simple one-pole lowpass per channel
    sos = np.array([[[0.998]*N, [0.0]*N, [0.0]*N,
                     [1.0]*N, [-0.002]*N, [0.0]*N]])

    delay_mod = parallelDelay(delays_sec, fs)
    absorption_mod = parallelSOSFilter(sos)
    feedback_matrix = Gain(A)
    input_gain = Gain(B)
    output_gain = Gain(C)
    direct_gain = Gain(D)

    delay_chain = Series(OrderedDict({
        "delay": delay_mod,
        "filter": absorption_mod,
    }))
    feedback_loop = Recursion(fF=delay_chain, fB=feedback_matrix)
    fdn_branch = Series(OrderedDict({
        "input_gain": input_gain,
        "feedback_loop": feedback_loop,
        "output_gain": output_gain,
    }))
    core = Parallel(brA=fdn_branch, brB=direct_gain, sum_output=True)
    model = Shell(core=core)

    params = {
        "delay_samples": delay_samples.tolist(),
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "sos": sos,
    }
    return model, fs, params


#convenience function tests

class TestFlamoToFaust:
    """verify flamo_to_faust produces the same output as the two-step pipeline."""

    def test_equivalent_to_two_step(self):
        """flamo_to_faust(model, fs) == json_to_faust(flamo_to_json(model, fs))."""
        model, fs, _ = _build_fdn_model()
        direct = flamo_to_faust(model, fs, name="EquivTest")
        two_step = json_to_faust(flamo_to_json(model, fs, name="EquivTest"))
        assert direct == two_step

    def test_name_propagates(self):
        model, fs, _ = _build_fdn_model()
        code = flamo_to_faust(model, fs, name="MyReverb")
        assert "//MyReverb" in code

    def test_fs_propagates(self):
        model, fs, _ = _build_fdn_model(fs=44100.0)
        code = flamo_to_faust(model, 44100.0, name="Test")
        assert "44100" in code


#end-to-end parameter preservation tests

class TestParameterPreservation:
    """verify that parameters survive the full pipeline: model to faust code."""

    @pytest.fixture
    def fdn_4ch(self):
        return _build_fdn_model(N=4)

    def test_all_delays_in_output(self, fdn_4ch):
        """every delay length must appear as @(samples) in the faust code.

        delays inside a recursion are decremented by 1 to compensate for
        the implicit one-sample delay introduced by the ~ operator.
        """
        model, fs, params = fdn_4ch
        code = flamo_to_faust(model, fs)
        for d in params["delay_samples"]:
            #subtract 1 because these delays are inside recursion
            assert f"@({d - 1})" in code, f"delay @({d - 1}) missing from output"

    def test_feedback_matrix_coefficients(self, fdn_4ch):
        """hadamard matrix coefficients must appear in the hoisted function."""
        model, fs, params = fdn_4ch
        code = flamo_to_faust(model, fs)
        #the 4x4 hadamard has entries of +/- 0.5
        assert "0.5*x0" in code
        #matrix is hoisted as fB (the node name from the recursion path)
        assert "fB(x0, x1, x2, x3)" in code

    def test_feedback_matrix_orthogonality(self, fdn_4ch):
        """verify through the json intermediate that A^T A = I."""
        model, fs, params = fdn_4ch
        config = flamo_to_json(model, fs)

        #navigate to feedback matrix
        core = config["children"][0]  #Parallel
        br_a = core["children"][0]    #Series
        rec = br_a["children"][1]     #Recursion
        fb = rec["fB"]               #Leaf (feedback matrix)

        A_extracted = np.array(fb["params"]["matrix"])
        np.testing.assert_allclose(
            A_extracted.T @ A_extracted,
            np.eye(4),
            atol=1e-10,
            err_msg="orthogonality lost in extraction"
        )

    def test_sos_filter_normalised(self, fdn_4ch):
        """sos coefficients in faust should be normalised (a0 = 1 implicit)."""
        model, fs, params = fdn_4ch
        code = flamo_to_faust(model, fs)
        #fi.tf2 calls must be present
        assert "fi.tf2(" in code
        #4 channels, 1 section each
        assert code.count("fi.tf2(") == 4

    def test_sos_coefficients_correct(self, fdn_4ch):
        """verify actual sos values through the json intermediate."""
        model, fs, params = fdn_4ch
        config = flamo_to_json(model, fs)

        core = config["children"][0]
        br_a = core["children"][0]
        rec = br_a["children"][1]
        ff = rec["fF"]
        filt = ff["children"][1]

        sos = filt["params"]["sos"]
        #one section, four channels
        assert len(sos) == 1
        for ch in range(4):
            b0, b1, b2, a1, a2 = sos[0][ch]
            #original: b0=0.998, a0=1.0, a1=-0.002 (after normalisation)
            assert abs(b0 - 0.998) < 1e-10
            assert abs(a1 - (-0.002)) < 1e-10

    def test_input_output_gains(self, fdn_4ch):
        """B and C gains should appear as matrix functions in faust."""
        model, fs, params = fdn_4ch
        code = flamo_to_faust(model, fs)
        #B is (4,1) matrix, hoisted as a function
        assert "input_gain" in code
        #C is (1,4) matrix, hoisted as a function
        assert "output_gain" in code

    def test_sum_output_generates_merge(self, fdn_4ch):
        """Parallel with sum_output=True must produce :> in faust."""
        model, fs, _ = fdn_4ch
        code = flamo_to_faust(model, fs)
        assert ":>" in code

    def test_recursion_generates_tilde(self, fdn_4ch):
        """Recursion must produce ~ operator in faust."""
        model, fs, _ = fdn_4ch
        code = flamo_to_faust(model, fs)
        assert "~" in code


#structural correctness tests

class TestStructuralCorrectness:
    """verify the generated faust code has correct structural properties."""

    def test_valid_faust_syntax_markers(self):
        """basic syntax check: balanced parens, ends with semicolon."""
        model, fs, _ = _build_fdn_model()
        code = flamo_to_faust(model, fs)
        #process line must end with ;
        process_lines = [l for l in code.split("\n") if l.startswith("process")]
        assert len(process_lines) == 1
        assert process_lines[0].strip().endswith(";")

        #balanced parentheses in the process expression
        process_expr = process_lines[0]
        assert process_expr.count("(") == process_expr.count(")")

    def test_import_present(self):
        model, fs, _ = _build_fdn_model()
        code = flamo_to_faust(model, fs)
        assert 'import("stdfaust.lib");' in code

    def test_no_shell_in_output(self):
        """shell is a flamo wrapper, should not appear in faust code."""
        model, fs, _ = _build_fdn_model()
        code = flamo_to_faust(model, fs)
        #only in comments is acceptable
        for line in code.split("\n"):
            if not line.strip().startswith("//"):
                assert "Shell" not in line

    def test_json_roundtrip_stability(self):
        """json serialise/deserialise should not change the faust output."""
        model, fs, _ = _build_fdn_model()
        config = flamo_to_json(model, fs, name="RoundTrip")
        #serialise and deserialise through json
        config_rt = json.loads(json.dumps(config))
        code_direct = json_to_faust(config)
        code_roundtrip = json_to_faust(config_rt)
        assert code_direct == code_roundtrip

    def test_deterministic_output(self):
        """same model, same fs should always produce identical code."""
        model, fs, _ = _build_fdn_model()
        code1 = flamo_to_faust(model, fs, name="Det")
        code2 = flamo_to_faust(model, fs, name="Det")
        assert code1 == code2


#edge cases

class TestEdgeCases:
    def test_single_channel_fdn(self):
        """1-channel fdn should produce valid faust."""
        fs = 48000.0
        delay_mod = parallelDelay(np.array([0.023]), fs)
        gain_mod = parallelGain(np.array([0.7]))
        rec = Recursion(fF=delay_mod, fB=gain_mod)
        model = Shell(core=rec)
        code = flamo_to_faust(model, fs, name="Mono")
        assert "~" in code
        #0.023 * 48000 = 1104 samples, minus 1 for implicit ~ delay = 1103
        assert "@(1103)" in code
        assert "*(0.7)" in code

    def test_bare_leaf_module(self):
        """a raw gain module (no wrapping) should produce valid faust."""
        g = Gain(np.eye(3))
        code = flamo_to_faust(g, 48000.0, name="NakedGain")
        assert "process = " in code
