#!/usr/bin/env python3
"""Debug preservation test errors"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from signal_features import load_current_data, extract_features
from detect_fault import detect_fault

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

power_samplings_path = Path("power_samplings")
files = sorted(power_samplings_path.glob("*.csv"))

print("Checking which fault types are being misclassified as disturbance:")
print("=" * 100)

misclassified_as_disturbance = {}

for file_path in files[:50]:  # Check first 50
    filename = file_path.name
    if '-' not in filename:
        continue
    
    chinese_type = filename.split('-')[-1].replace('.csv', '')
    expected_type = label_mapping.get(chinese_type, chinese_type)
    
    try:
        signal = load_current_data(str(file_path))
        if len(signal) < 100:
            continue
        
        result = detect_fault(str(file_path))
        detected_type = result['fault_type']
        
        if detected_type == 'disturbance' and expected_type != 'disturbance':
            if expected_type not in misclassified_as_disturbance:
                misclassified_as_disturbance[expected_type] = []
            
            features = extract_features(signal, 3200)
            misclassified_as_disturbance[expected_type].append({
                'file': filename,
                'confidence': result['confidence'],
                'has_non_sinusoidal': features.get('has_non_sinusoidal_cycles', False),
                'has_asymmetric': features.get('has_asymmetric_cycles', False),
                'rms_first': features.get('rms_first', 0),
                'rms_last': features.get('rms_last', 0),
                'is_first_near_zero': features.get('is_first_near_zero', False),
                'is_last_near_zero': features.get('is_last_near_zero', False),
            })
    
    except Exception as e:
        pass

print("\nFault types being misclassified as disturbance:")
for fault_type, samples in sorted(misclassified_as_disturbance.items()):
    print(f"\n{fault_type}: {len(samples)} samples")
    for sample in samples[:2]:
        print(f"  {sample['file']}")
        print(f"    non_sin={sample['has_non_sinusoidal']}, asym={sample['has_asymmetric']}")
        print(f"    rms_first={sample['rms_first']:.2f}, rms_last={sample['rms_last']:.2f}")
        print(f"    near_zero: first={sample['is_first_near_zero']}, last={sample['is_last_near_zero']}")
