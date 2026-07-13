#!/usr/bin/env python
"""
Standalone test for synth_mixtures.py — run manually, NOT through Claude Code.

Usage:
    cd /gpfs0/bgu-rgilad/users/orelgr/deep2
    /gpfs0/bgu-rgilad/users/orelgr/env/deep2_env/bin/python scripts/test_synth.py

Output saved to: outputs/test_synth_results.txt
"""
import sys
import os
import traceback
import tracemalloc

tracemalloc.start()

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Redirect output to both file and stdout
os.makedirs("outputs", exist_ok=True)
OUT = "outputs/test_synth_results.txt"

class Tee:
    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
    def flush(self):
        self.file.flush()
        self.stdout.flush()

sys.stdout = Tee(OUT)
sys.stderr = sys.stdout

try:
    print("=" * 60)
    print("STEP 1: Import modules")
    print("=" * 60)
    import numpy as np
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  After numpy: current={cur/1e6:.1f} MB, peak={peak/1e6:.1f} MB")

    import pandas as pd
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  After pandas: current={cur/1e6:.1f} MB, peak={peak/1e6:.1f} MB")

    import torch
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  After torch: current={cur/1e6:.1f} MB, peak={peak/1e6:.1f} MB")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    print()
    print("=" * 60)
    print("STEP 2: Load ChemicalPool")
    print("=" * 60)
    from src.data.synth_mixtures import ChemicalPool, SyntheticMixtures, SynthConfig
    pool = ChemicalPool.load()
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  Pool: {len(pool.chemicals)} chemicals, "
          f"{sum(len(v) for v in pool.by_chem.values())} spectra")
    print(f"  Memory: current={cur/1e6:.1f} MB, peak={peak/1e6:.1f} MB")

    print()
    print("=" * 60)
    print("STEP 3: Train/holdout split")
    print("=" * 60)
    train, hold = pool.split(holdout_frac=0.2, seed=0)
    print(f"  Train: {len(train.chemicals)} chemicals")
    print(f"  Holdout: {len(hold.chemicals)} chemicals")
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  Memory: current={cur/1e6:.1f} MB, peak={peak/1e6:.1f} MB")

    print()
    print("=" * 60)
    print("STEP 4: Generate 3 synthetic samples")
    print("=" * 60)
    ds = SyntheticMixtures(train, SynthConfig(seed=1))
    it = iter(ds)
    for i in range(3):
        s = next(it)
        c_np = s["c"].numpy()
        print(f"  Sample {i}: K={s['K']} M={s['M']} SNR={s['snr_db']:.1f} dB "
              f"coeffs(>0)={c_np[c_np > 0].round(3)}")
        print(f"    y shape={s['y'].shape}, R shape={s['R'].shape}")
        print(f"    ref_names={s['ref_names']}")
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  Memory: current={cur/1e6:.1f} MB, peak={peak/1e6:.1f} MB")

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print(f"Final memory: current={cur/1e6:.1f} MB, peak={peak/1e6:.1f} MB")
    print("=" * 60)

except Exception:
    traceback.print_exc()
    print("\nTEST FAILED — see traceback above")

tracemalloc.stop()
