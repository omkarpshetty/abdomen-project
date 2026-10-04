"""Small legacy flag adapters; all computation goes through the supported CLI."""
import argparse
import sys
from .cli import main


def legacy(command, input_flags, output_flags, argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if '--help' in args or '-h' in args:
        return main([command, '--help'])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(*input_flags, dest='source', required=True)
    parser.add_argument(*output_flags, dest='output', required=True)
    known, remaining = parser.parse_known_args(args)
    return main([command, known.source, '--output', known.output, *remaining])
