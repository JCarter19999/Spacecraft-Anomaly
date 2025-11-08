from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import time_split

# --- CSV loading & label alignment -----------------------------------------

def _load_raw(cfg):
    telem = pd.read_csv(cfg["DATA"]["TELEMETRY_CSV"], parse_dates=[cfg["DATA"]["TIMESTAMP_COL"]])
    labels = pd.read_csv(cfg["DATA"]["LABELS_CSV"], parse_dates=["start_time","end_time"])
    if "event_type" in labels.columns:
        use = set(cfg["DATA"].get("USE_EVENTS", []))
        labels = labels[labels["event_type"].isin(use)]
    telem = telem.sort_values(cfg["DATA"]["TIMESTAMP_COL"]).reset_index(drop=True)
    return telem, labels

def _mark_point_labels(telem: pd.DataFrame, labels: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    telem = telem.copy()
    telem["label"] = 0
    for _, r in labels.iterrows():
        m = (telem[ts_col] >= r.start_time) & (telem[ts_col] <= r.end_time)
        telem.loc[m, "label"] = 1
    return telem

def _normalize(train_df: pd.DataFrame, df: pd.DataFrame, sensor_cols: List[str]):
    mu = train_df[sensor_cols].mean()
    sd = train_df[sensor_cols].std().replace(0, 1.0)
    out = df.copy()
    out[sensor_cols] = (df[sensor_cols] - mu) / (sd + 1e-8)
    return out, mu, sd

def load_and_split(cfg):
    ts_col = cfg["DATA"]["TIMESTAMP_COL"]
    telem, labels = _load_raw(cfg)
    sensors = [c for c in telem.columns if c not in [ts_col]]
    telem = _mark_point_labels(telem, labels, ts_col)
    train, val, test = time_split(telem, cfg["DATA"]["TRAIN_SPLIT"], cfg["DATA"]["VAL_SPLIT"])
    train, mu, sd = _normalize(train, train, sensors)
    val, _, _  = _normalize(train, val, sensors)
    test, _, _ = _normalize(train, test, sensors)
    return train, val, test, sensors

# --- Windowed dataset -------------------------------------------------------

class TelemetryWindowed(Dataset):
    def __init__(self, df: pd.DataFrame, sensors: List[str], window: int = 128, stride: int = 4):
        self.df = df.reset_index(drop=True)
        self.sensors = sensors
        self.window = window
        self.stride = stride
        self.indices = []
        n = len(df)
        for start in range(0, n - window, stride):
            end = start + window
            label = int(self.df["label"].iloc[start:end].max())
            self.indices.append((start, end, label))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s, e, y = self.indices[idx]
        x = self.df[self.sensors].iloc[s:e].values.astype(np.float32)
        x = torch.from_numpy(x)  # (L, D)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

def windowed_dataloaders(cfg):
    train, val, test, sensors = load_and_split(cfg)
    W = cfg["DATA"]["WINDOW"]; S = cfg["DATA"]["STRIDE"]
    train_ds = TelemetryWindowed(train, sensors, W, S)
    val_ds   = TelemetryWindowed(val,   sensors, W, S)
    train_loader = DataLoader(train_ds, batch_size=cfg["TRAIN"]["BATCH_SIZE"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["TRAIN"]["BATCH_SIZE"], shuffle=False)
    return train_loader, val_loader, sensors

def build_windowed_datasets(cfg):
    train, val, test, sensors = load_and_split(cfg)
    W = cfg["DATA"]["WINDOW"]; S = cfg["DATA"]["STRIDE"]
    return {
        "train": TelemetryWindowed(train, sensors, W, S),
        "val":   TelemetryWindowed(val,   sensors, W, S),
        "test":  TelemetryWindowed(test,  sensors, W, S),
        "sensors": sensors,
    }
