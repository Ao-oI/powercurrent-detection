#!/usr/bin/env python3
"""Debug the last misclassified sample"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from signal_features import load_current_data, extract_features

file_path = Path("samplings/C0322025090003921_2026-03-09 04_54_23.csv")

signal = load_current_data(str(file_path))
features = extract_features(signal, 3200)

print(f"File: {file_path.name}")
print(f"\nCycle Analysis:")
print(f"  has_non_sinusoidal_cycles: {features.get('has_non_sinusoidal_cycles', False)}")
print(f"  has_asymmetric_cycles: {features.get('has_asymmetric_cycles', False)}")

beginning_analysis = features.get('beginning_cycles_analysis', [])
print(f"\nBeginning cycles ({len(beginning_analysis)}):")
for i, analysis in enumerate(beginning_analysis):
    print(f"  Cycle {i}: sine_ratio={analysis['sine_ratio']:.3f}, "
          f"symmetry={analysis['symmetry_score']:.3f}, "
          f"non_sin={analysis['is_non_sinusoidal']}, "
          f"asym={analysis['is_asymmetric']}")

end_analysis = features.get('end_cycles_analysis', [])
print(f"\nEnd cycles ({len(end_analysis)}):")
for i, analysis in enumerate(end_analysis):
    print(f"  Cycle {i}: sine_ratio={analysis['sine_ratio']:.3f}, "
          f"symmetry={analysis['symmetry_score']:.3f}, "
          f"non_sin={analysis['is_non_sinusoidal']}, "
          f"asym={analysis['is_asymmetric']}")

print(f"\nRMS Features:")
print(f"  is_first_near_zero: {features.get('is_first_near_zero', False)}")
print(f"  is_last_near_zero: {features.get('is_last_near_zero', False)}")
print(f"  rms_first: {features.get('rms_first', 0):.2f}")
print(f"  rms_last: {features.get('rms_last', 0):.2f}")
