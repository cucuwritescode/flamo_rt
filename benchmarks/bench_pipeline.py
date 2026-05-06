#bench_pipeline
#author: Facundo Franchino
"""
benchmark the flamo-rt translation pipeline across fdn sizes

measures wall clock time for three stages:
  1. flamo_to_json: model graph traversal and parameter extraction
  2. json_to_faust: faust code generation from json config
  3. faust compilation: faust compiler translating .dsp to c++

sweeps N (number of delay lines) from 2 to 64 and reports timing
for each stage. also measures generated code size and faust
compilation diagnostics.

usage:
    python benchmarks/bench_pipeline.py [--max-n 64] [--repeats 10] [--csv results.csv]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.linalg import hadamard


def _build_fdn_config(N: int, fs: int = 48000) -> dict:
    """build a json config dict for an N-channel fdn.

    uses prime delay lengths, hadamard feedback matrix, and
    simple one-pole absorption filters. this mirrors the standard
    fdn topology used throughout flamo-rt.
    """
    #generate N prime delay lengths spread across 20-45ms
    from sympy import nextprime
    delays = []
    d = 960  #starting point ~20ms at 48k
    for _ in range(N):
        d = nextprime(d)
        delays.append(d)

    #hadamard feedback matrix scaled for ~2s T60
    avg_delay = np.mean(delays)
    n_trips = 2.0 * fs / avg_delay
    gain = 10**(-3.0 / n_trips)

    #build hadamard for powers of two, pad for others
    n_had = 1
    while n_had < N:
        n_had *= 2
    H = hadamard(n_had).astype(float)[:N, :N] / np.sqrt(n_had)
    A = (gain * H).tolist()

    #absorption: one-pole lowpass per channel
    pole = 0.35
    coeffs = [1.0 - pole, 0.0, 0.0, -pole, 0.0]
    sos = [[coeffs for _ in range(N)]]

    #input gain: mono to N channels
    B = [[1.0]] * N

    #output gain: N channels to mono
    C = [[1.0 / N] * N]

    return {
        "type": "Shell",
        "name": f"BenchFDN_{N}",
        "fs": fs,
        "children": [{
            "type": "Parallel",
            "name": "core",
            "sum_output": True,
            "children": [
                {
                    "type": "Series",
                    "name": "reverb",
                    "children": [
                        {
                            "type": "Leaf",
                            "name": "input_gain",
                            "module_type": "Gain",
                            "input_channels": 1,
                            "output_channels": N,
                            "params": {"matrix": B},
                        },
                        {
                            "type": "Recursion",
                            "name": "fdn",
                            "fF": {
                                "type": "Series",
                                "name": "fF",
                                "children": [
                                    {
                                        "type": "Leaf",
                                        "name": "delay",
                                        "module_type": "parallelDelay",
                                        "input_channels": N,
                                        "output_channels": N,
                                        "params": {"samples": delays},
                                    },
                                    {
                                        "type": "Leaf",
                                        "name": "absorption",
                                        "module_type": "parallelSOSFilter",
                                        "input_channels": N,
                                        "output_channels": N,
                                        "params": {"sos": sos},
                                    },
                                ],
                            },
                            "fB": {
                                "type": "Leaf",
                                "name": "fB",
                                "module_type": "Gain",
                                "input_channels": N,
                                "output_channels": N,
                                "params": {"matrix": A},
                            },
                        },
                        {
                            "type": "Leaf",
                            "name": "output_gain",
                            "module_type": "Gain",
                            "input_channels": N,
                            "output_channels": 1,
                            "params": {"matrix": C},
                        },
                    ],
                },
                {
                    "type": "Leaf",
                    "name": "direct",
                    "module_type": "parallelGain",
                    "input_channels": 1,
                    "output_channels": 1,
                    "params": {"gains": [0.0]},
                },
            ],
        }],
    }


def _bench_json_to_faust(config: dict, repeats: int) -> tuple[float, str]:
    """benchmark json_to_faust, return (median_time_ms, generated_code)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from flamo_rt.codegen.json_to_faust import json_to_faust

    #warmup
    json_to_faust(config)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        code = json_to_faust(config)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.median(times), code


def _bench_faust_compile(code: str) -> tuple[float, bool]:
    """benchmark faust compilation to c++, return (time_ms, success)."""
    with tempfile.NamedTemporaryFile(suffix=".dsp", mode="w", delete=False) as f:
        f.write(code)
        dsp_path = f.name

    try:
        t0 = time.perf_counter()
        result = subprocess.run(
            ["faust", "-lang", "cpp", dsp_path, "-o", "/dev/null"],
            capture_output=True, text=True, timeout=120,
        )
        t1 = time.perf_counter()
        return (t1 - t0) * 1000, result.returncode == 0
    finally:
        Path(dsp_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="benchmark flamo-rt pipeline")
    parser.add_argument("--max-n", type=int, default=64, help="max fdn order")
    parser.add_argument("--repeats", type=int, default=20, help="repeats for json_to_faust timing")
    parser.add_argument("--csv", type=str, default=None, help="output csv path")
    parser.add_argument("--skip-compile", action="store_true", help="skip faust compilation benchmark")
    args = parser.parse_args()

    #sweep N: 2, 4, 8, 16, 32, 64
    sizes = [n for n in [2, 4, 8, 16, 32, 64] if n <= args.max_n]

    print(f"{'N':>4}  {'json_to_faust':>14}  {'faust_compile':>14}  {'code_lines':>10}  {'code_bytes':>10}  {'compiles':>8}")
    print(f"{'':>4}  {'(ms)':>14}  {'(ms)':>14}  {'':>10}  {'':>10}  {'':>8}")

    rows = []
    for N in sizes:
        config = _build_fdn_config(N)

        #stage 1: json_to_faust
        t_codegen, code = _bench_json_to_faust(config, args.repeats)

        n_lines = code.count("\n")
        n_bytes = len(code.encode("utf-8"))

        #stage 2: faust compilation
        if not args.skip_compile:
            t_compile, compiles = _bench_faust_compile(code)
        else:
            t_compile, compiles = 0.0, True

        print(f"{N:>4}  {t_codegen:>14.3f}  {t_compile:>14.1f}  {n_lines:>10}  {n_bytes:>10}  {'yes' if compiles else 'FAIL':>8}")

        rows.append({
            "N": N,
            "json_to_faust_ms": round(t_codegen, 4),
            "faust_compile_ms": round(t_compile, 1),
            "code_lines": n_lines,
            "code_bytes": n_bytes,
            "compiles": compiles,
        })

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")

    #print json for easy consumption
    print(f"\n{json.dumps(rows, indent=2)}")


if __name__ == "__main__":
    main()
