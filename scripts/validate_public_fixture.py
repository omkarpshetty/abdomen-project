"""Run upstream real-image integration; never label it clinical validation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    source = 'https://raw.githubusercontent.com/wasserth/TotalSegmentator/v2.8.0/tests/reference_files/'
    hashes = {}
    for name in ['example_ct_sm.nii.gz', 'example_seg_roi_subset.nii.gz']:
        with urllib.request.urlopen(source + name, timeout=60) as response:
            data = response.read()
        (root / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    # The upstream fixture's age metadata is not supplied. Adult confirmation here
    # is only to exercise the adult pipeline using the upstream integration fixture.
    subprocess.run([sys.executable, '-m', 'ct_dose', 'predict', str(root / 'example_ct_sm.nii.gz'),
                    '--output', str(root / 'run'), '--organs', 'liver', '--adult-confirmed'], check=True)
    import numpy as np
    import SimpleITK as sitk
    from ct_dose.imaging import load_scan, write_mask
    from ct_dose.evaluation import evaluate_masks
    scan = load_scan(root / 'example_ct_sm.nii.gz')
    reference = sitk.ReadImage(str(root / 'example_seg_roi_subset.nii.gz'))
    # Official total-task multilabel ID for liver is 5 in the pinned model.
    from totalsegmentator.map_to_binary import class_map
    liver_id = next(key for key, value in class_map['total'].items() if value == 'liver')
    mask = sitk.GetArrayFromImage(reference) == liver_id
    if not mask.any():
        raise ValueError('Upstream reference lacks liver label')
    target = root / 'reference_masks'; target.mkdir()
    write_mask(mask, reference, target / 'liver.nii.gz')
    metrics = evaluate_masks(scan, root / 'run' / 'masks', target, ['liver'])
    (root / 'validation.json').write_text(json.dumps({
        'purpose': 'software integration and upstream regression comparison',
        'independent_clinical_accuracy': False, 'source': source, 'sha256': hashes,
        'population_eligibility': 'upstream software fixture; not independently verified', 'metrics': metrics,
    }, indent=2))


if __name__ == '__main__':
    main()
