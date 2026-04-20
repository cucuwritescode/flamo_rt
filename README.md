<div align="center">

# flamo_rt

**real-time deployment of FLAMO audio graphs via FAUST**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/licence-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-120%20passing-brightgreen.svg)](#testing)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](FLAMO_RT_SPEC.md)

*bridge the gap between differentiable audio research and deployable real-time plugins*

</div>

---

## the problem

researchers design and optimise FDNs using [FLAMO](https://github.com/gdalsanto/flamo)'s differentiable audio framework, but deploying these as real-time plugins requires manual reimplementation. this is error-prone and creates a gap between research prototypes and usable tools.

```
before:   FLAMO model (PyTorch)  →  ???  →  real-time plugin
                                     ↑
                                manual rewrite

after:    FLAMO model (PyTorch)  →  flamo_rt  →  FAUST  →  plugin
```

## how it works

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│    FLAMO     │       │     JSON     │       │    FAUST     │
│    model     │  ───▶ │    config    │  ───▶ │    code      │
│  (PyTorch)   │       │              │       │   (.dsp)     │
└──────────────┘       └──────────────┘       └──────────────┘
         flamo_to_json()        json_to_faust()
         ╰───────────── flamo_to_faust() ──────────────╯
```

the pipeline traverses a FLAMO model graph, extracts all parameters (delays, gains, matrices, filters), serialises them to a JSON intermediate representation, and generates valid FAUST DSP code. the JSON stage decouples extraction from codegen and enables future backends.

## installation

```bash
pip install -e .
```

for full FLAMO model support (requires PyTorch):

```bash
pip install -e ".[full]"
```

## quick start

```python
from flamo_rt import flamo_to_faust

#given a trained FLAMO model and sample rate
faust_code = flamo_to_faust(model, fs=48000, name="MyReverb")

#write to file
with open("reverb.dsp", "w") as f:
    f.write(faust_code)
```

or use the two-step pipeline for inspection:

```python
from flamo_rt import flamo_to_json, json_to_faust

#step 1: extract parameters to json-serialisable dict
config = flamo_to_json(model, fs=48000, name="MyReverb")

#inspect, modify, serialise...
import json
print(json.dumps(config, indent=2))

#step 2: generate faust code
faust_code = json_to_faust(config)
```

## supported modules

| FLAMO module | FAUST output | description |
|---|---|---|
| `parallelDelay` | `@(n)` | integer sample delays |
| `Gain` / `Matrix` | sum-of-products function | mixing matrices (hoisted) |
| `parallelGain` | `*(g)` | per-channel diagonal gains |
| `parallelSOSFilter` | `fi.tf2(...)` | cascaded biquad filters |
| `Series` | `:` | sequential composition |
| `Parallel` | `,` / `:>` | side-by-side or summing |
| `Recursion` | `~` | feedback loops (FDN core) |
| `Biquad` / `SVF` | `fi.tf2` / `fi.svf.*` | single-channel filters |
| `Shell` | *(unwrapped)* | FFT wrapper skipped |

## testing

```bash
#unit tests (no external dependencies)
pytest tests/ -q

#integration tests (requires flamo venv + faust compiler)
pytest tests/integration/ -v
```

120 unit tests validate the full pipeline: parameter extraction, delay quantisation, SOS normalisation, gain classification, graph traversal, code generation, and end-to-end equivalence.

integration tests compare impulse responses between FLAMO (frequency domain) and generated FAUST (time domain) sample-by-sample.

## project structure

```
src/flamo_rt/
  codegen/
    flamo_to_json.py     parameter extraction and graph traversal
    json_to_faust.py     FAUST code generation from JSON config
    flamo_to_faust.py    convenience wrapper (both steps)
tests/
    test_flamo_to_json.py
    test_json_to_faust.py
    test_flamo_to_faust.py
    test_param_extraction.py
    integration/
        test_ir_comparison.py
        generate_flamo_ir.py
```

## related projects

- [FLAMO](https://github.com/gdalsanto/flamo) — differentiable audio processing framework
- [pyFDN](https://github.com/gdalsanto/pyfdn) — python feedback delay networks
- [FAUST](https://faust.grame.fr/) — functional audio stream

## licence

MIT — see [LICENSE](LICENSE) for details.

**Author:** Facundo Franchino

