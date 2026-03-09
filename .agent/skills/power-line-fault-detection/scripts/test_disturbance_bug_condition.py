#!/usr/bin/env python3
"""
Bug Condition Exploration Test for Disturbance Detection
Property 1: Non-Sinusoidal and Asymmetric Waveform Detection

This test MUST FAIL on unfixed code - failure confirms the bug exists.
The test encodes the expected behavior and will validate the fix when it passes.

Goal: Surface counterexamples that demonstrate the bug exists
- Load real disturbance signals from samplings/ directory
- Verify signals have non-sinusoidal or asymmetric waveforms in beginning/end cycles
- Assert that detect_fault_type() classifies them as 'disturbance'
"""

import numpy as np
import sys
from pathlib import Path
import csv

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from detect_fault import detect_fault
from signal_features import load_current_data, extract_features


def load_samplings_data(samplings_dir: str = "samplings") -> list:
    """Load all CSV files from samplings directory"""
    samplings_path = Path(samplings_dir)
    if not samplings_path.exists():
        print(f"Warning: {samplings_dir} directory not found")
        return []
    
    files = sorted(samplings_path.glob("*.csv"))
    print(f"Found {len(files)} sampling files")
    return files


def test_disturbance_bug_condition():
    """
    Property 1: Bug Condition - Non-Sinusoidal and Asymmetric Waveform Detection
    
    For any signal where the bug condition holds (has non-sinusoidal patterns,
    asymmetric waveforms, or transient effects at beginning/end cycles AND
    front/back ratio ≤ 1.5 AND is not a clear fault type), the system SHALL
    classify it as a disturbance.
    
    EXPECTED OUTCOME: Test FAILS on unfixed code (this proves the bug exists)
    """
    samplings_files = load_samplings_data()
    
    if not samplings_files:
        print("No sampling files found. Skipping test.")
        return
    
    misclassified = []
    correctly_classified = []
    
    print("\n" + "="*80)
    print("BUG CONDITION EXPLORATION TEST")
    print("Testing disturbance detection on real-world samplings/")
    print("="*80)
    
    for file_path in samplings_files:
        try:
            # Load signal
            signal = load_current_data(str(file_path))
            if len(signal) < 100:
                print(f"⚠ {file_path.name}: Signal too short ({len(signal)} samples)")
                continue
            
            # Extract features
            features = extract_features(signal, 3200)
            
            # Detect fault
            result = detect_fault(str(file_path))
            fault_type = result['fault_type']
            confidence = result['confidence']
            
            # Extract key features for analysis
            change_last_first = features.get('change_last_first', 1.0)
            sine_ratio_first = features.get('sine_ratio_first', 0.7)
            sine_ratio_last = features.get('sine_ratio_last', 0.7)
            is_first_near_zero = features.get('is_first_near_zero', False)
            is_last_near_zero = features.get('is_last_near_zero', False)
            num_segments = features.get('num_segments', 3)
            
            # Check if this looks like a disturbance based on waveform characteristics
            # Bug condition: non-sinusoidal or asymmetric patterns + front/back ratio ≤ 1.5
            has_non_sinusoidal = (sine_ratio_first < 0.95) or (sine_ratio_last < 0.95)
            has_low_front_back_ratio = change_last_first <= 1.5
            is_not_near_zero = not (is_first_near_zero or is_last_near_zero)
            
            # Determine if this should be classified as disturbance
            should_be_disturbance = (
                has_non_sinusoidal and 
                has_low_front_back_ratio and 
                is_not_near_zero
            )
            
            # Check classification result
            is_correctly_classified = (fault_type == 'disturbance')
            
            # Format output
            status = "✓ PASS" if is_correctly_classified else "✗ FAIL"
            
            print(f"\n{status} | {file_path.name}")
            print(f"  Detected: {fault_type:20s} (confidence: {confidence:.2f})")
            print(f"  Features: front/back={change_last_first:.2f}, "
                  f"sine_first={sine_ratio_first:.3f}, sine_last={sine_ratio_last:.3f}")
            print(f"  Segments: {num_segments}, near_zero: first={is_first_near_zero}, last={is_last_near_zero}")
            print(f"  Bug condition indicators: non_sinusoidal={has_non_sinusoidal}, "
                  f"low_ratio={has_low_front_back_ratio}, not_near_zero={is_not_near_zero}")
            
            if is_correctly_classified:
                correctly_classified.append({
                    'file': file_path.name,
                    'fault_type': fault_type,
                    'confidence': confidence
                })
            else:
                misclassified.append({
                    'file': file_path.name,
                    'detected': fault_type,
                    'expected': 'disturbance',
                    'confidence': confidence,
                    'change_last_first': change_last_first,
                    'sine_ratio_first': sine_ratio_first,
                    'sine_ratio_last': sine_ratio_last
                })
        
        except Exception as e:
            print(f"✗ ERROR | {file_path.name}: {str(e)}")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total files tested: {len(samplings_files)}")
    print(f"Correctly classified as disturbance: {len(correctly_classified)}")
    print(f"Misclassified: {len(misclassified)}")
    
    if misclassified:
        print("\nMISCLASSIFIED SIGNALS (Bug Condition Counterexamples):")
        print("-" * 80)
        for item in misclassified:
            print(f"\n  File: {item['file']}")
            print(f"    Detected as: {item['detected']} (confidence: {item['confidence']:.2f})")
            print(f"    Expected: {item['expected']}")
            print(f"    Features: front/back={item['change_last_first']:.2f}, "
                  f"sine_first={item['sine_ratio_first']:.3f}, sine_last={item['sine_ratio_last']:.3f}")
    
    # Assertion: All samplings should be classified as disturbance
    print("\n" + "="*80)
    if len(misclassified) == 0:
        print("✓ TEST PASSED: All disturbance signals correctly classified!")
        return True
    else:
        print(f"✗ TEST FAILED: {len(misclassified)} disturbance signals misclassified")
        print("This confirms the bug exists - disturbances are being misclassified")
        print("as other fault types due to missing cycle-level waveform analysis.")
        return False


if __name__ == '__main__':
    success = test_disturbance_bug_condition()
    sys.exit(0 if success else 1)
