"""Compatibility entry point for annotation: python -m ct_dose annotate."""
from ct_dose.compat import legacy


def main():
    return legacy('annotate', ['--dicom-dir'], ['--out', '--output'])


if __name__ == '__main__':
    raise SystemExit(main())
