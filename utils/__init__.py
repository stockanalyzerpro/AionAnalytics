import sys, os
from pathlib import Path

stub_path = Path(__file__).resolve().parent / "stubs"
if stub_path.exists() and str(stub_path) not in sys.path:
    sys.path.insert(0, str(stub_path))
