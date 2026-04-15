#test_json_to_faust
#author: Facundo Franchino
"""
faust code generation from json config dicts

each test builds a minimal json config (matching flamo_to_json output)
and verifies that the generated faust code is correct and compilable.
"""

from __future__ import annotations

import pytest

from flamo_rt.codegen.json_to_faust import json_to_faust


#helpers

def _leaf(name: str, module_type: str, params: dict, n_ch: int = 4) -> dict:
    """shorthand for building a leaf node dict."""
    return {
        "type": "Leaf",
        "name": name,
        "module_type": module_type,
        "params": params,
        "input_channels": n_ch,
        "output_channels": n_ch,
    }


def _wrap_config(node: dict, name: str = "test", fs: int = 48000) -> dict:
    """wrap a node in a minimal root config with fs and name."""
    config = dict(node)
    config["fs"] = fs
    config["name"] = name
    return config


#header and structure tests

class TestHeader:
    def test_includes_import(self):
        config = _wrap_config({"type": "Shell", "children": []})
        code = json_to_faust(config)
        assert 'import("stdfaust.lib");' in code

    def test_includes_name(self):
        config = _wrap_config({"type": "Shell", "children": []}, name="MyReverb")
        code = json_to_faust(config)
        assert "// MyReverb" in code

    def test_includes_sample_rate(self):
        config = _wrap_config({"type": "Shell", "children": []}, fs=44100)
        code = json_to_faust(config)
        assert "44100" in code

    def test_has_process(self):
        config = _wrap_config({"type": "Shell", "children": []})
        code = json_to_faust(config)
        assert "process = " in code

    def test_empty_shell_is_wire(self):
        config = _wrap_config({"type": "Shell", "children": []})
        code = json_to_faust(config)
        assert "process = _;" in code


#delay tests

class TestDelayCodegen:
    def test_single_delay(self):
        node = _leaf("d", "parallelDelay", {"samples": [1000]}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "@(1000)" in code

    def test_multiple_delays(self):
        node = _leaf("d", "parallelDelay", {"samples": [1103, 1447, 1811, 2137]})
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "@(1103)" in code
        assert "@(1447)" in code
        assert "@(1811)" in code
        assert "@(2137)" in code

    def test_delays_joined_parallel(self):
        #four delays should be composed with ,
        node = _leaf("d", "parallelDelay", {"samples": [100, 200, 300, 400]})
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "@(100) , @(200) , @(300) , @(400)" in code


#diagonal gain tests

class TestDiagonalGainCodegen:
    def test_single_gain(self):
        node = _leaf("g", "parallelGain", {"gains": [0.5]}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "*(0.5)" in code

    def test_multiple_gains(self):
        node = _leaf("g", "parallelGain", {"gains": [0.1, 0.2, 0.3]}, n_ch=3)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "*(0.1)" in code
        assert "*(0.2)" in code
        assert "*(0.3)" in code

    def test_gain_module_diagonal(self):
        #Gain with "gains" key (classified as diagonal by flamo_to_json)
        node = _leaf("g", "Gain", {"gains": [1.0, 0.5]}, n_ch=2)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "*(1)" in code
        assert "*(0.5)" in code


#matrix gain tests

class TestMatrixGainCodegen:
    def test_identity_matrix(self):
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        node = _leaf("m", "Gain", {"matrix": matrix}, n_ch=2)
        config = _wrap_config(node, name="m")
        code = json_to_faust(config)
        #should produce a function definition
        assert "m(x0, x1)" in code
        #identity: row0 = x0, row1 = x1
        assert "x0" in code
        assert "x1" in code

    def test_hadamard_matrix(self):
        matrix = [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
        ]
        node = _leaf("feedback", "Gain", {"matrix": matrix})
        config = _wrap_config(node, name="feedback")
        code = json_to_faust(config)
        #function should be hoisted as a definition
        assert "feedback(x0, x1, x2, x3)" in code
        #check that it appears as a definition line ending with ;
        assert "feedback(x0, x1, x2, x3) =" in code

    def test_zero_row_emits_zero(self):
        matrix = [[0.0, 0.0], [1.0, 0.0]]
        node = _leaf("m", "Gain", {"matrix": matrix}, n_ch=2)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "0.0" in code

    def test_matrix_module_type(self):
        #Matrix type should behave identically to Gain with matrix params
        matrix = [[0.7, 0.3], [0.3, 0.7]]
        node = _leaf("mix", "Matrix", {"matrix": matrix}, n_ch=2)
        config = _wrap_config(node, name="mix")
        code = json_to_faust(config)
        assert "mix(x0, x1)" in code


#sos filter tests

class TestSOSFilterCodegen:
    def test_single_section_single_channel(self):
        #one section, one channel, 5 coefficients (already normalised)
        sos = [[[1.0, 0.0, 0.0, -0.9, 0.0]]]
        node = _leaf("f", "parallelSOSFilter", {"sos": sos}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "fi.tf2(" in code
        assert "-0.9" in code

    def test_multiple_sections_cascaded(self):
        #two sections should be composed in series with :
        sos = [
            [[0.998, 0.0, 0.0, -0.002, 0.0]],
            [[0.5, 0.1, 0.0, -0.3, 0.05]],
        ]
        node = _leaf("f", "parallelSOSFilter", {"sos": sos}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        #two fi.tf2 calls joined by :
        assert code.count("fi.tf2(") == 2

    def test_multiple_channels_parallel(self):
        #one section, two channels should be composed with ,
        sos = [[[0.9, 0.0, 0.0, -0.1, 0.0], [0.8, 0.0, 0.0, -0.2, 0.0]]]
        node = _leaf("f", "parallelSOSFilter", {"sos": sos}, n_ch=2)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert code.count("fi.tf2(") == 2
        assert "," in code


#composition tests

class TestComposition:
    def test_series(self):
        delay = _leaf("d", "parallelDelay", {"samples": [500]}, n_ch=1)
        gain = _leaf("g", "parallelGain", {"gains": [0.5]}, n_ch=1)
        series = {
            "type": "Series",
            "name": "chain",
            "children": [delay, gain],
        }
        config = _wrap_config(series)
        code = json_to_faust(config)
        #series uses : operator
        assert ":" in code
        assert "@(500)" in code
        assert "*(0.5)" in code

    def test_parallel_no_sum(self):
        g1 = _leaf("a", "parallelGain", {"gains": [0.5]}, n_ch=1)
        g2 = _leaf("b", "parallelGain", {"gains": [0.3]}, n_ch=1)
        par = {
            "type": "Parallel",
            "name": "split",
            "children": [g1, g2],
            "sum_output": False,
        }
        config = _wrap_config(par)
        code = json_to_faust(config)
        assert "*(0.5)" in code
        assert "*(0.3)" in code
        #parallel without sum should not have :>
        assert ":>" not in code

    def test_parallel_with_sum(self):
        g1 = _leaf("a", "parallelGain", {"gains": [0.5]}, n_ch=1)
        g2 = _leaf("b", "parallelGain", {"gains": [0.3]}, n_ch=1)
        par = {
            "type": "Parallel",
            "name": "merge",
            "children": [g1, g2],
            "sum_output": True,
        }
        config = _wrap_config(par)
        code = json_to_faust(config)
        assert ":> _" in code

    def test_recursion(self):
        delay = _leaf("d", "parallelDelay", {"samples": [1000]}, n_ch=1)
        gain = _leaf("g", "parallelGain", {"gains": [0.5]}, n_ch=1)
        rec = {
            "type": "Recursion",
            "name": "loop",
            "fF": delay,
            "fB": gain,
        }
        config = _wrap_config(rec)
        code = json_to_faust(config)
        #recursion uses ~ operator
        assert "~" in code
        assert "@(1000)" in code
        assert "*(0.5)" in code

    def test_shell_unwraps(self):
        gain = _leaf("g", "parallelGain", {"gains": [0.8]}, n_ch=1)
        shell = {
            "type": "Shell",
            "name": "wrapper",
            "children": [gain],
        }
        config = _wrap_config(shell)
        code = json_to_faust(config)
        #shell should not appear in the output, just its core
        assert "Shell" not in code.split("//")[-1]
        assert "*(0.8)" in code


#full fdn integration test

class TestFullFDN:
    """test code generation for a complete 4-channel fdn structure.

    this mirrors the mock_fdn_model from test_flamo_to_json.py but
    works from the json config dict directly.
    """

    @pytest.fixture
    def fdn_config(self) -> dict:
        """a 4-channel fdn config matching FLAMO_RT_SPEC.md section 6."""
        return {
            "type": "Shell",
            "name": "TestFDN",
            "fs": 48000,
            "children": [{
                "type": "Parallel",
                "name": "core",
                "sum_output": True,
                "children": [
                    {
                        "type": "Series",
                        "name": "brA",
                        "children": [
                            _leaf("input_gain", "Gain",
                                  {"gains": [1.0, 1.0, 1.0, 1.0]}),
                            {
                                "type": "Recursion",
                                "name": "feedback_loop",
                                "fF": {
                                    "type": "Series",
                                    "name": "fF",
                                    "children": [
                                        _leaf("delay", "parallelDelay",
                                              {"samples": [1103, 1447, 1811, 2137]}),
                                        _leaf("filter", "parallelSOSFilter", {
                                            "sos": [
                                                [
                                                    [0.998, 0.0, 0.0, -0.002, 0.0],
                                                    [0.998, 0.0, 0.0, -0.002, 0.0],
                                                    [0.998, 0.0, 0.0, -0.002, 0.0],
                                                    [0.998, 0.0, 0.0, -0.002, 0.0],
                                                ]
                                            ]
                                        }),
                                    ],
                                },
                                "fB": _leaf("feedback_matrix", "Gain", {
                                    "matrix": [
                                        [0.5, 0.5, 0.5, 0.5],
                                        [0.5, -0.5, 0.5, -0.5],
                                        [0.5, 0.5, -0.5, -0.5],
                                        [0.5, -0.5, -0.5, 0.5],
                                    ]
                                }),
                            },
                            _leaf("output_gain", "Gain",
                                  {"gains": [0.25, 0.25, 0.25, 0.25]}),
                        ],
                    },
                    _leaf("direct", "Gain", {"gains": [0.0]}, n_ch=1),
                ],
            }],
        }

    def test_generates_valid_structure(self, fdn_config):
        code = json_to_faust(fdn_config)
        #must have import, process, and the key dsp elements
        assert 'import("stdfaust.lib");' in code
        assert "process = " in code
        assert "~" in code  #recursion
        assert ":>" in code  #parallel sum

    def test_all_delays_present(self, fdn_config):
        code = json_to_faust(fdn_config)
        for d in [1103, 1447, 1811, 2137]:
            assert f"@({d})" in code

    def test_feedback_matrix_hoisted(self, fdn_config):
        code = json_to_faust(fdn_config)
        #the feedback matrix should appear as a top-level definition
        assert "feedback_matrix(x0, x1, x2, x3) =" in code

    def test_filters_present(self, fdn_config):
        code = json_to_faust(fdn_config)
        assert "fi.tf2(" in code
        #4 channels, 1 section each = 4 fi.tf2 calls
        assert code.count("fi.tf2(") == 4

    def test_no_trailing_whitespace(self, fdn_config):
        code = json_to_faust(fdn_config)
        for line in code.split("\n"):
            assert line == line.rstrip(), f"trailing whitespace: {line!r}"


#edge cases

class TestEdgeCases:
    def test_unknown_module_emits_wire(self):
        node = _leaf("mystery", "FancyProcessor", {"raw": [1, 2, 3]})
        config = _wrap_config(node)
        code = json_to_faust(config)
        #unknown modules become passthrough wires
        assert "process = _;" in code

    def test_gain_with_no_params(self):
        node = _leaf("g", "Gain", {})
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "process = _;" in code

    def test_recursion_with_missing_fb(self):
        delay = _leaf("d", "parallelDelay", {"samples": [100]}, n_ch=1)
        rec = {
            "type": "Recursion",
            "name": "loop",
            "fF": delay,
            "fB": None,
        }
        config = _wrap_config(rec)
        code = json_to_faust(config)
        #should still produce valid code with _ as feedback
        assert "~" in code
        assert "_" in code

    def test_deeply_nested_series(self):
        leaf = _leaf("g", "parallelGain", {"gains": [0.5, 0.5]}, n_ch=2)
        inner = {"type": "Series", "name": "inner", "children": [leaf]}
        outer = {"type": "Series", "name": "outer", "children": [inner]}
        config = _wrap_config(outer)
        code = json_to_faust(config)
        assert "*(0.5)" in code

    def test_integer_gain_no_decimal(self):
        #integer values should not have decimal points in faust output
        node = _leaf("g", "parallelGain", {"gains": [1.0]}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "*(1)" in code

    def test_negative_coefficient(self):
        node = _leaf("g", "parallelGain", {"gains": [-0.707]}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "*(-0.707)" in code
