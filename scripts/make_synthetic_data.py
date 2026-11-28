"""Write a small synthetic dataset to data/ so the planner can be smoke-tested:

    python scripts/make_synthetic_data.py data
    groundwatch-plan configs/experiments/phase1_surface_seg.yaml --stats
"""

import sys

from groundwatch.data.synthetic import make_synthetic_dataset

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data"
    print(make_synthetic_dataset(out, n_drives=3, frames_per_drive=5))
