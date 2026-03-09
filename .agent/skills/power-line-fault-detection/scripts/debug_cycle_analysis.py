#!/usr/bin/env python3
"""Debug cycle analysis to understand why disturbances are still misclassified"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from signal_features import load_current_data, extract_features, analyze_beginning_end_cycles

# Test on first sampling file
sampling_file = Path("samplings/C0072019060300132_2026-03-09 01_27_26.csv")

if sampling_file.exists():
    print(f"Analyzing: {sampling_file.name}")
    signal = load_current_data(str(sampling_file))
    print(f"Signal length: {len(signal)}")
    
    # Extract features
    features = extract_features(signal, 3200)
    
    # Print cycle analysis results
    print("\nCycle Analysis Results:")
    print(f"  has_non_sinusoidal_cycles: {features.get('has_non_sinusoidal_cycles', False)}")
    print(f"  has_asymmetric_cycles: {features.get('has_asymmetric_cycles', False)}")
    print(f"  avg_sine_ratio: {features.get('avg_sine_ratio', 0):.3f}")
    print(f"  avg_symmetry_score: {features.get('avg_symmetry_score', 0):.3f}")
    print(f"  num_beginning_cycles: {features.get('num_beginning_cycles', 0)}")
    print(f"  num_end_cycles: {features.get('num_end_cycles', 0)}")
    
    # Print beginning cycles analysis
    beginning_analysis = features.get('beginning_cycles_analysis', [])
    print(f"\nBeginning cycles ({len(beginning_analysis)}):")
    for i, analysis in enumerate(beginning_analysis):
        print(f"  Cycle {i}: sine_ratio={analysis['sine_ratio']:.3f}, "
              f"symmetry={analysis['symmetry_score']:.3f}, "
              f"non_sin={analysis['is_non_sinusoidal']}, "
              f"asym={analysis['is_asymmetric']}")
    
    # Print end cycles analysis
    end_analysis = features.get('end_cycles_analysis', [])
    print(f"\nEnd cycles ({len(end_analysis)}):")
    for i, analysis in enumerate(end_analysis):
        print(f"  Cycle {i}: sine_ratio={analysis['sine_ratio']:.3f}, "
              f"symmetry={analysis['symmetry_score']:.3f}, "
              f"non_sin={analysis['is_non_sinusoidal']}, "
              f"asym={analysis['is_asymmetric']}")
    
    # Print RMS features
    print(f"\nRMS Features:")
    print(f"  change_last_first: {features.get('change_last_first', 0):.2f}")
    print(f"  is_first_near_zero: {features.get('is_first_near_zero', False)}")
    print(f"  is_last_near_zero: {features.get('is_last_near_zero', False)}")
    
else:
    print(f"File not found: {sampling_file}")
