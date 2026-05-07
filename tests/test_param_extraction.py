#test_param_extraction
#author: Facundo Franchino
"""
unit tests for parameter extraction from flamo leaf modules

focuses on _extract_param, _serialise_leaf, and boundary conditions
not covered by test_flamo_to_json.py. uses the same mock objects.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from flamo_rt.codegen.flamo_to_json import (
    flamo_to_json,
    _classify_gain,
    _extract_param,
    _normalise_sos,
    _quantise_delays,
    _serialise_leaf,
)


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
    def __init__(self, delays_sec, fs):
        self.param = _MockParam(np.asarray(delays_sec))
        self.fs = fs
        n = np.asarray(delays_sec).size
        self.input_channels = n
        self.output_channels = n


class Gain:
    def __init__(self, values):
        values = np.asarray(values)
        self.param = _MockParam(values)
        if values.ndim == 2:
            self.output_channels, self.input_channels = values.shape
        else:
            self.input_channels = values.size
            self.output_channels = values.size


class parallelGain:
    def __init__(self, values):
        values = np.asarray(values).ravel()
        self.param = _MockParam(values)
        self.input_channels = values.size
        self.output_channels = values.size


class parallelSOSFilter:
    def __init__(self, sos):
        sos = np.asarray(sos)
        self.param = _MockParam(sos)
        self.input_channels = sos.shape[2]
        self.output_channels = sos.shape[2]


class Series:
    def __init__(self, children: OrderedDict):
        self._modules = children


class Shell:
    def __init__(self, core):
        self._Shell__core = core
    def get_core(self):
        return self._Shell__core


#_extract_param tests

class TestExtractParam:
    def test_returns_numpy_array(self):
        g = Gain(np.array([1.0, 2.0]))
        result = _extract_param(g)
        assert isinstance(result, np.ndarray)

    def test_returns_none_when_no_param(self):
        class Bare:
            pass
        assert _extract_param(Bare()) is None

    def test_float32_converted_to_float64(self):
        #flamo may store params as float32 internally
        class Float32Module:
            def __init__(self):
                self.param = _MockParam(np.array([1.0], dtype=np.float32))
        result = _extract_param(Float32Module())
        assert result.dtype == np.float64

    def test_returns_copy_not_reference(self):
        #modifying the extracted array must not affect the module
        g = Gain(np.array([1.0, 2.0]))
        a = _extract_param(g)
        b = _extract_param(g)
        a[0] = 999.0
        assert b[0] == 1.0

    def test_preserves_shape(self):
        g = Gain(np.eye(3))
        result = _extract_param(g)
        assert result.shape == (3, 3)

    def test_high_dimensional_param(self):
        #sos filters have 3d params
        sos = np.ones((2, 6, 4))
        f = parallelSOSFilter(sos)
        result = _extract_param(f)
        assert result.shape == (2, 6, 4)


#_serialise_leaf tests

class TestSerialiseLeaf:
    def test_module_with_no_param_gets_empty_dict(self):
        class NoParam:
            input_channels = 2
            output_channels = 2
        node = _serialise_leaf(NoParam(), "test", 48000.0)
        assert node["params"] == {}
        assert node["type"] == "Leaf"

    def test_unknown_module_stores_raw(self):
        class WeirdModule:
            def __init__(self):
                self.param = _MockParam(np.array([42.0, 43.0]))
                self.input_channels = 2
                self.output_channels = 2
        node = _serialise_leaf(WeirdModule(), "weird", 48000.0)
        assert "raw" in node["params"]
        assert node["params"]["raw"] == [42.0, 43.0]

    def test_channel_counts_omitted_when_absent(self):
        class Minimal:
            def __init__(self):
                self.param = _MockParam(np.array([1.0]))
        node = _serialise_leaf(Minimal(), "m", 48000.0)
        assert "input_channels" not in node
        assert "output_channels" not in node

    def test_channel_counts_present_when_available(self):
        d = parallelDelay([0.01], 48000.0)
        node = _serialise_leaf(d, "d", 48000.0)
        assert node["input_channels"] == 1
        assert node["output_channels"] == 1

    def test_sos_unexpected_shape_stores_raw(self):
        #wrong second dimension (should be 6, give it 5)
        class BadSOS:
            def __init__(self):
                self.param = _MockParam(np.ones((2, 5, 3)))
                self.input_channels = 3
                self.output_channels = 3
        #override type name to match
        BadSOS.__name__ = "parallelSOSFilter"
        node = _serialise_leaf(BadSOS(), "bad", 48000.0)
        assert "raw" in node["params"]

    def test_gain_matrix_values_correct(self):
        A = np.array([[0.7, 0.3], [0.3, 0.7]])
        g = Gain(A)
        node = _serialise_leaf(g, "mix", 48000.0)
        assert node["params"]["matrix"] == A.tolist()

    def test_gain_diagonal_values_correct(self):
        vals = np.array([0.1, 0.9, 0.5])
        g = parallelGain(vals)
        node = _serialise_leaf(g, "g", 48000.0)
        np.testing.assert_allclose(node["params"]["gains"], [0.1, 0.9, 0.5])


#delay quantisation boundary cases

class TestQuantiseDelaysBoundary:
    def test_zero_delay(self):
        samples = _quantise_delays(np.array([0.0]), 48000.0)
        assert samples == [0]

    def test_sub_sample_delay(self):
        #less than one sample at 48khz
        delays_sec = np.array([0.5 / 48000.0])
        samples = _quantise_delays(delays_sec, 48000.0)
        #0.5 rounds to 0 (banker's rounding)
        assert samples == [0]

    def test_large_delay(self):
        #2 second delay at 48khz = 96000 samples
        samples = _quantise_delays(np.array([2.0]), 48000.0)
        assert samples == [96000]

    def test_2d_array_ravelled(self):
        #delays stored as column vector should still work
        delays_sec = np.array([[0.01], [0.02]])
        samples = _quantise_delays(delays_sec, 48000.0)
        assert samples == [480, 960]

    def test_different_sample_rates(self):
        delays_sec = np.array([0.01])
        assert _quantise_delays(delays_sec, 44100.0) == [441]
        assert _quantise_delays(delays_sec, 48000.0) == [480]
        assert _quantise_delays(delays_sec, 96000.0) == [960]

    def test_many_delays(self):
        #16-channel fdn
        n = 16
        delays_sec = np.arange(1, n + 1) * 0.001
        samples = _quantise_delays(delays_sec, 48000.0)
        assert len(samples) == n
        assert samples[0] == 48
        assert samples[-1] == 768


#sos normalisation boundary cases

class TestNormaliseSOSBoundary:
    def test_negative_a0(self):
        #negative a0 is unusual but valid, coefficients should flip sign
        sos = np.array([[[-1.0], [0.5], [0.0], [-1.0], [0.9], [0.0]]])
        result = _normalise_sos(sos)
        np.testing.assert_allclose(result[0][0], [1.0, -0.5, 0.0, -0.9, 0.0])

    def test_very_small_a0_raises(self):
        #a0 = 1e-16 is effectively zero
        sos = np.array([[[1.0], [0.0], [0.0], [1e-16], [0.0], [0.0]]])
        with pytest.raises(ValueError, match="a0 is near zero"):
            _normalise_sos(sos)

    def test_a0_just_above_threshold(self):
        #a0 = 1e-14 should not raise
        sos = np.array([[[1.0], [0.0], [0.0], [1e-14], [0.0], [0.0]]])
        result = _normalise_sos(sos)
        assert len(result) == 1

    def test_large_coefficients(self):
        #verify numerical stability with large values
        sos = np.array([[[1e6], [1e6], [0.0], [1e6], [-1e6], [0.0]]])
        result = _normalise_sos(sos)
        np.testing.assert_allclose(result[0][0], [1.0, 1.0, 0.0, -1.0, 0.0])

    def test_all_zero_numerator(self):
        #b0 = b1 = b2 = 0, filter outputs silence
        sos = np.array([[[0.0], [0.0], [0.0], [1.0], [-0.5], [0.1]]])
        result = _normalise_sos(sos)
        np.testing.assert_allclose(result[0][0], [0.0, 0.0, 0.0, -0.5, 0.1])

    def test_many_sections(self):
        #8 cascaded biquad sections, single channel
        n_sections = 8
        sos = np.zeros((n_sections, 6, 1))
        for s in range(n_sections):
            sos[s, :, 0] = [1.0, 0.1 * s, 0.0, 1.0, -0.1 * s, 0.01 * s]
        result = _normalise_sos(sos)
        assert len(result) == n_sections


#gain classification boundary cases

class TestClassifyGainBoundary:
    def test_scalar_0d(self):
        #0d array has ndim=0, falls through to matrix default
        #in practice flamo never produces 0d gain params
        assert _classify_gain(np.array(1.0)) == "matrix"

    def test_3d_is_matrix(self):
        assert _classify_gain(np.ones((2, 3, 4))) == "matrix"

    def test_1x1_is_diagonal(self):
        assert _classify_gain(np.array([[1.0]])) == "diagonal"

    def test_square_2x2(self):
        assert _classify_gain(np.array([[1.0, 0.0], [0.0, 1.0]])) == "matrix"


#sample rate propagation

class TestSampleRate:
    def test_fs_stored_as_integer(self):
        d = parallelDelay([0.01], 48000.0)
        config = flamo_to_json(d, 48000.0)
        assert config["fs"] == 48000
        assert isinstance(config["fs"], int)

    def test_fs_affects_delay_quantisation(self):
        d = parallelDelay([0.01], 44100.0)
        config_44 = flamo_to_json(d, 44100.0)
        config_48 = flamo_to_json(d, 48000.0)
        #same delay in seconds, different sample counts
        assert config_44["params"]["samples"] == [441]
        assert config_48["params"]["samples"] == [480]

    def test_fs_96k(self):
        d = parallelDelay([0.01], 96000.0)
        config = flamo_to_json(d, 96000.0)
        assert config["params"]["samples"] == [960]
        assert config["fs"] == 96000


#negative and zero gain values

class TestGainValues:
    def test_negative_diagonal_gains(self):
        g = parallelGain(np.array([-0.5, -1.0, -0.1]))
        node = _serialise_leaf(g, "g", 48000.0)
        np.testing.assert_allclose(node["params"]["gains"], [-0.5, -1.0, -0.1])

    def test_zero_diagonal_gains(self):
        g = parallelGain(np.array([0.0, 0.0]))
        node = _serialise_leaf(g, "g", 48000.0)
        assert node["params"]["gains"] == [0.0, 0.0]

    def test_negative_matrix_entries(self):
        A = np.array([[-1.0, 0.0], [0.0, -1.0]])
        g = Gain(A)
        node = _serialise_leaf(g, "neg", 48000.0)
        assert node["params"]["matrix"] == [[-1.0, 0.0], [0.0, -1.0]]

    def test_rectangular_matrix(self):
        #3 inputs, 2 outputs
        A = np.ones((2, 3)) * 0.333
        g = Gain(A)
        node = _serialise_leaf(g, "rect", 48000.0)
        assert len(node["params"]["matrix"]) == 2
        assert len(node["params"]["matrix"][0]) == 3

    def test_large_channel_count(self):
        #16x16 fdn feedback matrix
        n = 16
        A = np.eye(n) * 0.5
        g = Gain(A)
        node = _serialise_leaf(g, "big", 48000.0)
        matrix = node["params"]["matrix"]
        assert len(matrix) == n
        assert len(matrix[0]) == n
        #check diagonal
        for i in range(n):
            assert matrix[i][i] == 0.5
            for j in range(n):
                if i != j:
                    assert matrix[i][j] == 0.0
