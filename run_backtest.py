"""Run from a downloaded repository, without requiring an editable install."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from sp500_pairs.cli import main

if __name__ == "__main__":
    main()
