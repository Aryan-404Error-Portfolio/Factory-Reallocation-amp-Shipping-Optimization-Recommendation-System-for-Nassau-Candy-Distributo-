"""Train and save model artifacts without pickling classes from __main__."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import main

if __name__ == "__main__":
    main()
