"""Compatibility entry point for the original nested monitor location."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    current = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(current))
    runpy.run_path(str(current / 'intent_monitor.py'), run_name='__main__')
