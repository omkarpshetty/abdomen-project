"""Compatibility entry point for prediction: python -m ct_dose predict."""
from ct_dose.compat import legacy


def main():
    return legacy('predict', ['--patient', '-p'], ['--output', '-o'])


if __name__ == '__main__':
    raise SystemExit(main())
