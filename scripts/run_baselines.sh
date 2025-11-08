#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import yaml
from src.data import load_and_split
from src.baselines.zscore import ZScoreBaseline
from src.baselines.pca import PCABaseline
from src.metrics import evaluate_pointwise

cfg = yaml.safe_load(open("configs/tcn.yaml"))
train, val, test, sensors = load_and_split(cfg)

# Z-score
z = ZScoreBaseline().fit(train[sensors])
val_scores = z.score(val[sensors])
print("Z-SCORE VAL:", evaluate_pointwise(val["label"].values, val_scores))

test_scores = z.score(test[sensors])
print("Z-SCORE TEST:", evaluate_pointwise(test["label"].values, test_scores))

# PCA
pca = PCABaseline(n_components=20).fit(train[sensors])
val_scores = pca.score(val[sensors])
print("PCA VAL:", evaluate_pointwise(val["label"].values, val_scores))

test_scores = pca.score(test[sensors])
print("PCA TEST:", evaluate_pointwise(test["label"].values, test_scores))
PY
