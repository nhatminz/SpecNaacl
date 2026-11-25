"""Use the frozen-pair harness with this checkout's EAGLE-3 runtime."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(1,str(root))
from scripts.benchmark_rebuttal import main

if __name__ == '__main__':
    main(['--engine','eagle3', *sys.argv[1:]])
