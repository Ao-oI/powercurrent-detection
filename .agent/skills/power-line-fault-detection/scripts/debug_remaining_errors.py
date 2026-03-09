#!/usr/bin/env python3
"""Debug remaining misclassified samples"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from signal_features import load_current_data, extract_features
from detect_fault import detect_fault

remaining_errors = [
    "C0072020010200061_2026-03-08 11_54_14.csv",
    "C0322023040000172_2026-03-09 07_57_48.csv",
    "C0322025090003901_2026-03-09 05_09_15.csv",
    "C0322025090003921_2026-03-09 04_54_23.csv",
    "C0322025090003951_2026-03-09 04_40_23.csv",
]

for filename in remaining_errors:
    file_path = Path("samplings") / filename
    if not file_path.exists():
        print(f"File not found: {filename}")
        continue
    
    print(f"\n{'='*80}")
    print(f"File: {filename}")
    print('='*80)
    
    signal = load_current_data(str(file_path))
    features = extract_features(signal, 3200)
    result = detect_fault(str(file_path))
    
    print(f"Detected: {result['fault_type']} (confidence: {result['confidence']:.2f})")
    print(f"\nKey Features:")
    print(f"  change_last_first: {features.get('change_last_first', 0):.2f}")
    print(f"  is_first_near_zero: {features.get('is_first_near_zero', False)}")
    print(f"  is_last_near_zero: {features.get('is_last_near_zero', False)}")
    print(f"  rms_first: {features.get('rms_first', 0):.2f}")
    print(f"  rms_last: {features.get('rms_last', 0):.2f}")
    print(f"  num_segments: {features.get('num_segments', 0)}")
    print(f"  sine_ratio_first: {features.get('sine_ratio_first', 0):.3f}")
    print(f"  sine_ratio_last: {features.get('sine_ratio_last', 0):.3f}")
    
    print(f"\nCycle Analysis:")
    print(f"  has_non_sinusoidal_cycles: {features.get('has_non_sinusoidal_cycles', False)}")
    print(f"  has_asymmetric_cycles: {features.get('has_asymmetric_cycles', False)}")
    print(f"  avg_sine_ratio: {features.get('avg_sine_ratio', 0):.3f}")
    print(f"  avg_symmetry_score: {features.get('avg_symmetry_score', 0):.3f}")
    
    print(f"\nScores:")
    for fault_type, score in sorted(result['scores'].items(), key=lambda x: -x[1])[:3]:
        print(f"  {fault_type:20s}: {score:.3f}")
