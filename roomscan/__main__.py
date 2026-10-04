"""Enables `python -m roomscan run <input_path> --tier {...} --out <dir>` --
the literal command contract from the case study brief (CLAUDE.md)."""
import sys

from roomscan.cli import main

if __name__ == "__main__":
    sys.exit(main())
