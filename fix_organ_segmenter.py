"""
Fix OrganSegmenter to properly detect all organs by adding --roi_subset parameter
"""
import sys

# Read the file
with open('src/organ_segmenter.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the command building section
old_code = """        cmd = [
            'TotalSegmentator',
            '-i', input_path,
            '-o', output_dir,
            '--ml',  # Use multilabel output for efficiency
        ]"""

new_code = """        cmd = [
            'TotalSegmentator',
            '-i', input_path,
            '-o', output_dir,
            '--ml',  # Use multilabel output for efficiency
            '--roi_subset'
        ]
        
        # Add all target organs to ensure they're segmented
        cmd.extend(target_organs)"""

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✓ Fixed: Added --roi_subset parameter with target organs")
else:
    print("✗ Could not find the code to replace")
    sys.exit(1)

# Write back
with open('src/organ_segmenter.py', 'w', encoding='utf-8') as f:
    f.write(content)
    
print("✓ Updated src/organ_segmenter.py")
print("\nNow TotalSegmentator will explicitly request all 16 target organs:")
print("  - liver, spleen, kidney_right, kidney_left")
print("  - pancreas, stomach, gallbladder, urinary_bladder")
print("  - heart, esophagus, small_bowel, colon")
print("  - spinal_cord, aorta, adrenal_gland_left, adrenal_gland_right")
