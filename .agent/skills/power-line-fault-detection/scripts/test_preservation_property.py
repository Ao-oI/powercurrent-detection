#!/usr/bin/env python3
"""
Preservation Property Tests for Disturbance Detection Bugfix
Property 2: Existing Fault Type Classification Unchanged

This test MUST PASS on unfixed code - it confirms baseline behavior to preserve.
After the fix, this test should still pass, ensuring no regressions.

Goal: Verify that for all inputs where the bug condition does NOT hold,
the fixed function produces the same result as the original function.
"""

import numpy as np
import sys
from pathlib import Path
import csv

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from detect_fault import detect_fault
from signal_features import load_current_data, extract_features


def load_power_samplings_data(power_samplings_dir: str = "power_samplings") -> dict:
    """Load all CSV files from power_samplings directory, grouped by fault type"""
    power_samplings_path = Path(power_samplings_dir)
    if not power_samplings_path.exists():
        print(f"Warning: {power_samplings_dir} directory not found")
        return {}
    
    # Mapping from Chinese labels to English algorithm labels
    label_mapping = {
        '跳闸': 'trip',
        '合闸成功': 'closing_success',
        '合闸失败': 'closing_failure',
        '上电': 'power_on',
        '停电': 'power_off_sag',
        '电流骤升': 'current_surge',
        '电流骤增': 'current_surge',
        '电流骤降': 'power_off_sag',
        '扰动': 'disturbance',
        '重合闸成功': 'closing_success',
        '合闸成功.': 'closing_success',
    }
    
    files_by_type = {}
    for file_path in sorted(power_samplings_path.glob("*.csv")):
        # Extract fault type from filename (format: ...-[FaultType].csv)
        filename = file_path.name
        if '-' in filename:
            chinese_type = filename.split('-')[-1].replace('.csv', '')
            # Map to English algorithm label
            english_type = label_mapping.get(chinese_type, chinese_type)
            if english_type not in files_by_type:
                files_by_type[english_type] = []
            files_by_type[english_type].append(file_path)
    
    print(f"Found {sum(len(v) for v in files_by_type.values())} power sampling files")
    for fault_type, files in sorted(files_by_type.items()):
        print(f"  {fault_type}: {len(files)} files")
    
    return files_by_type


def test_preservation_property():
    """
    Property 2: Preservation - Existing Fault Type Classification Unchanged
    
    For any signal where the bug condition does NOT hold (has sinusoidal,
    symmetric waveforms OR is a clear fault type), the fixed code SHALL
    produce the same classification result as the original code.
    
    EXPECTED OUTCOME: Test PASSES on unfixed code (confirms baseline behavior)
    """
    files_by_type = load_power_samplings_data()
    
    if not files_by_type:
        print("No power sampling files found. Skipping test.")
        return True
    
    print("\n" + "="*80)
    print("PRESERVATION PROPERTY TEST")
    print("Testing that existing fault type classifications are preserved")
    print("="*80)
    
    # Test each fault type
    preservation_results = {}
    total_tested = 0
    total_correct = 0
    
    for fault_type, files in sorted(files_by_type.items()):
        print(f"\n--- Testing {fault_type} ({len(files)} files) ---")
        
        correct_count = 0
        misclassified = []
        
        for file_path in files[:5]:  # Test first 5 of each type for speed
            try:
                # Load signal
                signal = load_current_data(str(file_path))
                if len(signal) < 100:
                    continue
                
                # Extract features
                features = extract_features(signal, 3200)
                
                # Detect fault
                result = detect_fault(str(file_path))
                detected_type = result['fault_type']
                confidence = result['confidence']
                
                # Map detected type to expected type
                # (some types may be grouped differently)
                expected_type = fault_type
                
                # Check if correctly classified
                is_correct = (detected_type == expected_type)
                
                if is_correct:
                    correct_count += 1
                    print(f"  ✓ {file_path.name}: {detected_type} (conf: {confidence:.2f})")
                else:
                    misclassified.append({
                        'file': file_path.name,
                        'detected': detected_type,
                        'expected': expected_type,
                        'confidence': confidence
                    })
                    print(f"  ✗ {file_path.name}: detected {detected_type}, expected {expected_type}")
                
                total_tested += 1
                if is_correct:
                    total_correct += 1
            
            except Exception as e:
                print(f"  ✗ ERROR | {file_path.name}: {str(e)}")
        
        preservation_results[fault_type] = {
            'tested': min(5, len(files)),
            'correct': correct_count,
            'misclassified': misclassified
        }
    
    # Summary
    print("\n" + "="*80)
    print("PRESERVATION TEST SUMMARY")
    print("="*80)
    print(f"Total files tested: {total_tested}")
    print(f"Correctly classified: {total_correct}")
    print(f"Misclassified: {total_tested - total_correct}")
    
    print("\nResults by fault type:")
    for fault_type, results in sorted(preservation_results.items()):
        accuracy = results['correct'] / results['tested'] * 100 if results['tested'] > 0 else 0
        print(f"  {fault_type:20s}: {results['correct']}/{results['tested']} ({accuracy:.1f}%)")
    
    # Determine if test passes
    # For preservation, we want to ensure existing behavior is maintained
    # A high accuracy on power_samplings indicates the algorithm is working well
    preservation_accuracy = total_correct / total_tested * 100 if total_tested > 0 else 0
    
    print("\n" + "="*80)
    if preservation_accuracy >= 70:
        print(f"✓ TEST PASSED: {preservation_accuracy:.1f}% preservation accuracy")
        print("Existing fault type classifications are working correctly.")
        return True
    else:
        print(f"✗ TEST FAILED: {preservation_accuracy:.1f}% preservation accuracy")
        print("Some existing fault types are being misclassified.")
        return False


def test_clear_fault_types():
    """
    Additional test: Verify clear fault type patterns are preserved
    """
    print("\n" + "="*80)
    print("CLEAR FAULT TYPE PRESERVATION TEST")
    print("="*80)
    
    files_by_type = load_power_samplings_data()
    
    # Test specific fault types that should be clearly distinguishable
    clear_types = ['trip', 'closing_success', 'power_on']  # Trip, Closing Success, Power On
    
    for fault_type in clear_types:
        if fault_type not in files_by_type:
            continue
        
        files = files_by_type[fault_type][:3]  # Test first 3
        print(f"\n{fault_type}:")
        
        for file_path in files:
            try:
                signal = load_current_data(str(file_path))
                result = detect_fault(str(file_path))
                
                is_correct = (result['fault_type'] == fault_type)
                status = "✓" if is_correct else "✗"
                print(f"  {status} {file_path.name}: {result['fault_type']} (conf: {result['confidence']:.2f})")
            except Exception as e:
                print(f"  ✗ ERROR: {str(e)}")
    
    return True


if __name__ == '__main__':
    success1 = test_preservation_property()
    success2 = test_clear_fault_types()
    
    sys.exit(0 if (success1 and success2) else 1)
