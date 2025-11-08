#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import yaml, torch
from src.data import windowed_dataloaders
from src.models.tcn import TCNClassifier
from src.trainer import Trainer

cfg = yaml.safe_load(open("configs/tcn.yaml"))
train_loader, val_loader, sensors = windowed_dataloaders(cfg)

model = TCNClassifier(num_inputs=len(sensors),
                      channels=cfg["MODEL"]["TCN_CHANNELS"],
                      kernel_size=cfg["MODEL"]["KERNEL_SIZE"],
                      dropout=cfg["MODEL"]["DROPOUT"])

trainer = Trainer(cfg)
trainer.fit(model, train_loader, val_loader)
PY
