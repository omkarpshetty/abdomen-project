"""Convert official CT-ORG labels to binary references without inventing laterality."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct_dose.imaging import write_mask

LABELS = {1: "liver", 2: "urinary_bladder", 3: "lungs", 4: "kidneys", 5: "bones", 6: "brain"}


def convert(labels_path, output):
    image = sitk.ReadImage(str(labels_path))
    values = sitk.GetArrayFromImage(image)
    if image.GetDimension() != 3 or not np.isin(values, list(range(7))).all():
        raise ValueError("Expected CT-ORG integer labels 0–6")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    for label, name in LABELS.items():
        write_mask(values == label, image, output / f"{name}.nii.gz")
    (output / 'provenance.json').write_text(json.dumps({
        'dataset': 'CT-ORG-v1', 'source': 'https://www.cancerimagingarchive.net/wp-content/uploads/ctorg_README.txt',
        'label_sha256': hashlib.sha256(Path(labels_path).read_bytes()).hexdigest(),
        'label_mapping': LABELS, 'pretraining_overlap': 'unresolved',
        'warnings': ['Kidneys and lungs are combined labels; no laterality inferred',
                     'Bone/lung annotation provenance differs by split; audit the official reference documentation'],
    }, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    convert(args.labels, args.output)
