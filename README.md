# Spacecraft Anomaly Detection (ESA ADB)

This project evaluates whether deep sequence models (TCN / Transformer) can outperform classical baselines on multivariate spacecraft telemetry anomaly detection.

## Quickstart

```bash
# 0) (Optional) create env
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 1) Put Kaggle ESA ADB files into data/raw/
#    Expected (edit as needed):
#    - data/raw/telemetry_train.csv
#    - data/raw/labels_train.csv   (start_time,end_time,event_type)

# 2) Prepare data windows
python scripts/prepare_data.py --config configs/tcn.yaml

# 3) Run baselines
bash scripts/run_baselines.sh

# 4) Train TCN
bash scripts/train_tcn.sh
```

## Research question
> Do sequence-based deep models (TCN/Transformer) trained on sliding windows achieve higher event-level F1 and lower false alarm rates than z-score and PCA reconstruction baselines on the ESA ADB telemetry?

## Repo layout
- `src/data.py`: loading, normalization, windowing
- `src/metrics.py`: point- and event-level metrics
- `src/models/`: TCN, Transformer, LSTM-AE
- `src/baselines/`: z-score, PCA reconstruction
- `src/trainer.py`: config-driven train/eval loop
- `scripts/`: dataset prep and simple runners

