#tests for flamo_to_json: parameter extraction and graph traversal
#
#uses lightweight mock objects that mimic flamo's module structure
#(Shell, Series, Parallel, Recursion, and leaf dsp modules) so the
#tests run without torch or flamo installed.
#
#author: Facundo Franchino

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

import numpy as np
import pytest

from flamo_rt.codegen.flamo_to_json import (
    flamo_to_json,
    _classify_gain,
    _detect_module_type,
    _normalise_sos,
    _quantise_delays,
)


#mock flamo modules (no torch dependency)

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
    """mock flamo parallelDelay module."""

    def __init__(self, delays_sec: np.ndarray, fs: float):
        self.param = _MockParam(delays_sec)
        self.fs = fs
        n = len(delays_sec)
        self.input_channels = n
        self.output_channels = n


class Gain:
    """mock flamo Gain module (matrix or diagonal)."""

    def __init__(self, values: np.ndarray):
        self.param = _MockParam(values)
        if values.ndim == 2:
            self.output_channels, self.input_channels = values.shape
        else:
            self.input_channels = len(values)
            self.output_channels = len(values)


class parallelGain:
    """mock flamo parallelGain module."""

    def __init__(self, values: np.ndarray):
        self.param = _MockParam(values.ravel())
        self.input_channels = len(values)
        self.output_channels = len(values)


class Matrix(Gain):
    """mock flamo Matrix module."""
    pass


class parallelSOSFilter:
    """mock flamo parallelSOSFilter module."""

    def __init__(self, sos: np.ndarray):
        #sos shape: (n_sections, 6, n_channels)
        self.param = _MockParam(sos)
        self.input_channels = sos.shape[2]
        self.output_channels = sos.shape[2]


class Series:
    """mock flamo Series (nn.Sequential)."""

    def __init__(self, children: OrderedDict):
        self._modules = children
        #propagate channel info from last child
        last = list(children.values())[-1] if children else None
        self.input_channels = getattr(list(children.values())[0], "input_channels", None) if children else None
        self.output_channels = getattr(last, "output_channels", None)


class Parallel:
    """mock flamo Parallel module."""

    def __init__(self, brA: Any, brB: Any, sum_output: bool = False):
        self.brA = brA
        self.brB = brB
        self.sum_output = sum_output
        self.input_channels = getattr(brA, "input_channels", None)
        self.output_channels = getattr(brA, "output_channels", None)


class Recursion:
    """mock flamo Recursion module."""

    def __init__(self, fF: Any, fB: Any):
        self.fF = fF
        self.fB = fB
        self.input_channels = getattr(fF, "input_channels", None)
        self.output_channels = getattr(fF, "output_channels", None)


class Shell:
    """mock flamo Shell module (wraps core with fft/ifft io layers)."""

    def __init__(self, core: Any):
        self._Shell__core = core
        self.input_channels = getattr(core, "input_channels", None)
        self.output_channels = getattr(core, "output_channels", None)

    def get_core(self):
        return self._Shell__core


#delay quantisation tests

class TestQuantiseDelays:
    def test_basic_conversion(self):
        delays_sec = np.array([0.023, 0.030, 0.038, 0.045])
        fs = 48000.0
        samples = _quantise_delays(delays_sec, fs)
        expected = [1104, 1440, 1824, 2160]
        assert samples == expected

    def test_exact_integer_samples(self):
        #1000 samples at 48000 hz = 0.020833... seconds
        delays_sec = np.array([1000.0 / 48000.0])
        samples = _quantise_delays(delays_sec, 48000.0)
        assert samples == [1000]

    def test_rounding(self):
        #0.5 sample boundary, rounds to nearest even (banker's rounding via np.round)
        delays_sec = np.array([1.5 / 48000.0, 2.5 / 48000.0])
        samples = _quantise_delays(delays_sec, 48000.0)
        assert samples == [2, 2]  #numpy rounds 0.5 to even


#sos normalisation tests

class TestNormaliseSOS:
    def test_identity_when_a0_is_one(self):
        #single section, single channel, a0 = 1
        #shape: (1, 6, 1)
        sos = np.array([[[1.0], [0.5], [0.0], [1.0], [-0.9], [0.0]]])
        result = _normalise_sos(sos)
        #expect [b0, b1, b2, a1, a2] = [1.0, 0.5, 0.0, -0.9, 0.0]
        assert len(result) == 1  #one section
        assert len(result[0]) == 1  #one channel
        np.testing.assert_allclose(result[0][0], [1.0, 0.5, 0.0, -0.9, 0.0])

    def test_normalisation_with_a0_not_one(self):
        #a0 = 2.0, all coefficients should be halved
        sos = np.array([[[2.0], [1.0], [0.0], [2.0], [-1.8], [0.4]]])
        result = _normalise_sos(sos)
        np.testing.assert_allclose(result[0][0], [1.0, 0.5, 0.0, -0.9, 0.2])

    def test_degenerate_a0_raises(self):
        sos = np.array([[[1.0], [0.5], [0.0], [0.0], [-0.9], [0.0]]])
        with pytest.raises(ValueError, match="a0 is near zero"):
            _normalise_sos(sos)

    def test_multiple_sections_and_channels(self):
        #2 sections, 2 channels
        sos = np.array([
            [[1.0, 0.5], [0.2, 0.1], [0.0, 0.0], [1.0, 1.0], [-0.5, -0.3], [0.1, 0.05]],
            [[0.8, 0.6], [0.4, 0.2], [0.1, 0.0], [1.0, 2.0], [-0.6, -0.4], [0.2, 0.1]],
        ])
        result = _normalise_sos(sos)
        assert len(result) == 2
        assert len(result[0]) == 2
        #section 0, channel 0: a0=1, no change
        np.testing.assert_allclose(result[0][0], [1.0, 0.2, 0.0, -0.5, 0.1])
        #section 1, channel 1: a0=2, divide all by 2
        np.testing.assert_allclose(result[1][1], [0.3, 0.1, 0.0, -0.2, 0.05])


#gain classification tests

class TestClassifyGain:
    def test_1d_is_diagonal(self):
        assert _classify_gain(np.array([1.0, 2.0, 3.0])) == "diagonal"

    def test_square_matrix(self):
        assert _classify_gain(np.eye(4)) == "matrix"

    def test_row_vector_is_matrix(self):
        #(1,3) row vector changes channel count (3 to 1), so it is a matrix
        assert _classify_gain(np.array([[1.0, 2.0, 3.0]])) == "matrix"

    def test_column_vector_is_matrix(self):
        #(3,1) column vector changes channel count (1 to 3), so it is a matrix
        assert _classify_gain(np.array([[1.0], [2.0], [3.0]])) == "matrix"

    def test_rectangular_matrix(self):
        assert _classify_gain(np.ones((3, 4))) == "matrix"


#module type detection tests

class TestDetectModuleType:
    def test_known_types(self):
        assert _detect_module_type(parallelDelay(np.array([0.01]), 48000)) == "parallelDelay"
        assert _detect_module_type(Gain(np.eye(2))) == "Gain"
        assert _detect_module_type(parallelGain(np.array([1.0]))) == "parallelGain"
        assert _detect_module_type(Matrix(np.eye(2))) == "Matrix"

    def test_unknown_type_returns_class_name(self):
        class FancyProcessor:
            pass
        assert _detect_module_type(FancyProcessor()) == "FancyProcessor"


#full model traversal tests

class TestFlamoToJson:
    """test full graph traversal on a mock fdn model matching the pyFDN dss_to_flamo structure."""

    @pytest.fixture
    def mock_fdn_model(self):
        """build a 4-channel fdn model matching dss_to_flamo output structure.

        signal flow: Shell(Parallel(Series(B, Recursion(delays, A), C), D))
        """
        fs = 48000.0
        N = 4

        #delay lengths: prime numbers for good diffusion
        delays_sec = np.array([1103, 1447, 1811, 2137]) / fs

        #hadamard feedback matrix (4x4, orthogonal)
        A = 0.5 * np.array([
            [1,  1,  1,  1],
            [1, -1,  1, -1],
            [1,  1, -1, -1],
            [1, -1, -1,  1],
        ], dtype=np.float64)

        #input/output gains
        B = np.ones((N, 1), dtype=np.float64)
        C = np.ones((1, N), dtype=np.float64) / N
        D = np.zeros((1, 1), dtype=np.float64)

        #absorption: 1 sos section, 4 channels, near-unity passthrough
        sos_4ch = np.array([[[0.998]*N, [0.0]*N, [0.0]*N,
                             [1.0]*N, [-0.002]*N, [0.0]*N]])

        delay_mod = parallelDelay(delays_sec, fs)
        absorption_mod = parallelSOSFilter(sos_4ch)
        feedback_matrix = Gain(A)
        input_gain = Gain(B)
        output_gain = Gain(C)
        direct_gain = Gain(D)

        #build graph matching dss_to_flamo structure
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

        return model, fs

    def test_root_structure(self, mock_fdn_model):
        model, fs = mock_fdn_model
        config = flamo_to_json(model, fs, name="TestFDN")

        assert config["type"] == "Shell"
        assert config["name"] == "TestFDN"
        assert config["fs"] == 48000

    def test_json_serialisable(self, mock_fdn_model):
        """the entire config must be json-serialisable."""
        model, fs = mock_fdn_model
        config = flamo_to_json(model, fs)
        #this will raise if any value is not serialisable
        json_str = json.dumps(config)
        assert isinstance(json_str, str)
        #round-trip
        parsed = json.loads(json_str)
        assert parsed["type"] == "Shell"

    def test_delay_extraction(self, mock_fdn_model):
        """delays should appear as integer samples, not seconds."""
        model, fs = mock_fdn_model
        config = flamo_to_json(model, fs)

        #navigate: Shell > core (Parallel) > brA (Series) > feedback_loop (Recursion) > fF (Series) > delay
        core = config["children"][0]
        assert core["type"] == "Parallel"
        br_a = core["children"][0]
        assert br_a["type"] == "Series"
        feedback_loop = br_a["children"][1]
        assert feedback_loop["type"] == "Recursion"
        ff = feedback_loop["fF"]
        assert ff["type"] == "Series"
        delay_node = ff["children"][0]

        assert delay_node["module_type"] == "parallelDelay"
        assert delay_node["params"]["samples"] == [1103, 1447, 1811, 2137]

    def test_feedback_matrix_extraction(self, mock_fdn_model):
        """feedback matrix should be extracted as a 4x4 nested list."""
        model, fs = mock_fdn_model
        config = flamo_to_json(model, fs)

        core = config["children"][0]
        br_a = core["children"][0]
        feedback_loop = br_a["children"][1]
        fb = feedback_loop["fB"]

        assert fb["module_type"] == "Gain"
        assert "matrix" in fb["params"]
        matrix = np.array(fb["params"]["matrix"])
        assert matrix.shape == (4, 4)
        #verify orthogonality preserved
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(4), atol=1e-10)

    def test_sos_normalisation(self, mock_fdn_model):
        """sos coefficients should be normalised (a0 = 1, dropped from output)."""
        model, fs = mock_fdn_model
        config = flamo_to_json(model, fs)

        core = config["children"][0]
        br_a = core["children"][0]
        feedback_loop = br_a["children"][1]
        ff = feedback_loop["fF"]
        filter_node = ff["children"][1]

        assert filter_node["module_type"] == "parallelSOSFilter"
        assert "sos" in filter_node["params"]
        sos = filter_node["params"]["sos"]
        #1 section, 4 channels, 5 coefficients each
        assert len(sos) == 1
        assert len(sos[0]) == 4
        assert len(sos[0][0]) == 5

    def test_sum_output_flag(self, mock_fdn_model):
        """parallel node should carry sum_output flag."""
        model, fs = mock_fdn_model
        config = flamo_to_json(model, fs)

        core = config["children"][0]
        assert core["type"] == "Parallel"
        assert core["sum_output"] is True

    def test_input_output_gains(self, mock_fdn_model):
        """B and C gains should be classified as matrix (they change channel count)."""
        model, fs = mock_fdn_model
        config = flamo_to_json(model, fs)

        core = config["children"][0]
        br_a = core["children"][0]
        input_gain = br_a["children"][0]
        output_gain = br_a["children"][2]

        #B is (4,1), a column vector that expands 1 to 4 channels
        assert "matrix" in input_gain["params"]
        #C is (1,4), a row vector that mixes 4 channels to 1
        assert "matrix" in output_gain["params"]

    def test_direct_path(self, mock_fdn_model):
        """direct path D should be present as brB."""
        model, fs = mock_fdn_model
        config = flamo_to_json(model, fs)

        core = config["children"][0]
        br_b = core["children"][1]
        assert br_b["name"] == "brB"
        assert br_b["module_type"] == "Gain"


#edge cases

class TestEdgeCases:
    def test_single_delay_line(self):
        """single channel fdn should work."""
        fs = 48000.0
        delay_mod = parallelDelay(np.array([0.01]), fs)
        config = flamo_to_json(delay_mod, fs, name="single")
        assert config["type"] == "Leaf"
        assert config["params"]["samples"] == [480]

    def test_bare_gain(self):
        """gain module without any wrapping structure."""
        g = Gain(np.eye(3))
        config = flamo_to_json(g, 48000.0, name="naked_gain")
        assert config["module_type"] == "Gain"
        assert "matrix" in config["params"]

    def test_empty_series(self):
        """series with no children."""
        s = Series(OrderedDict())
        config = flamo_to_json(s, 48000.0)
        assert config["type"] == "Series"
        assert config["children"] == []

    def test_deeply_nested(self):
        """three levels of series nesting."""
        leaf = parallelGain(np.array([0.5, 0.5]))
        inner = Series(OrderedDict({"g": leaf}))
        middle = Series(OrderedDict({"inner": inner}))
        outer = Series(OrderedDict({"middle": middle}))
        config = flamo_to_json(outer, 48000.0)

        #drill down
        assert config["type"] == "Series"
        assert config["children"][0]["type"] == "Series"
        assert config["children"][0]["children"][0]["type"] == "Series"
        assert config["children"][0]["children"][0]["children"][0]["module_type"] == "parallelGain"
