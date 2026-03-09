#!/usr/bin/env python3
"""Debug all sampling files to understand their characteristics"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from signal_features import load_current_data, extract_features

samplings_dir = Path("samplings")
files = sorted(samplings_dir.glob("*.csv"))

print("Analyzing all sampling files:")
print("=" * 100)
print(f"{'File':<50} {'front/back':>10} {'non_sin':>8} {'asym':>8} {'sine_last':>10}")
print("=" * 100)

for file_path in files:
    try:
        signal = load_current_data(str(file_path))
        if len(signal) < 100:
            continue
        
        features = extract_features(signal, 3200)
        
        change_last_first = features.get('change_last_first', 0)
        has_non_sinusoidal = features.get('has_non_sinusoidal_cycles', False)
        has_asymmetric = features.get('has_asymmetric_cycles', False)
        sine_ratio_last = features.get('sine_ratio_last', 0)
        
        print(f"{file_path.name:<50} {change_last_first:>10.2f} {str(has_non_sinusoidal):>8} {str(has_asymmetric):>8} {sine_ratio_last:>10.3f}")
    
    except Exception as e:
        print(f"{file_path.name:<50} ERROR: {str(e)[:30]}")

print("=" * 100)
