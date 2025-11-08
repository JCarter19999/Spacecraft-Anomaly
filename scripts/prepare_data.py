import argparse, yaml
from pathlib import Path
from src.data import build_windowed_datasets

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = build_windowed_datasets(cfg)
    print("Prepared:", {k: (len(v) if hasattr(v, '__len__') else v) for k, v in out.items()})
