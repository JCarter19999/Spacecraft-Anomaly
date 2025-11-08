from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .metrics import evaluate_pointwise
from .models.transformer import TransformerClassifier
from .models.tcn import TCNClassifier

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg["TRAIN"]["DEVICE"] if torch.cuda.is_available() else "cpu")
        self.crit = nn.BCELoss()
        self.lr = cfg["TRAIN"]["LR"]
        self.epochs = cfg["TRAIN"]["EPOCHS"]
        self.cls_w = cfg["TRAIN"].get("CLASS_WEIGHT", 1.0)

    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.lr)

    def fit(self, model, train_loader: DataLoader, val_loader: DataLoader):
        model = model.to(self.device)
        opt = self.build_optimizer(model)
        best_f1 = -1.0
        for ep in range(1, self.epochs+1):
            model.train()
            pbar = tqdm(train_loader, desc=f"epoch {ep}")
            for xb, yb in pbar:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = model(xb)
                # upweight positive class
                w = torch.ones_like(yb)
                w[yb==1] = self.cls_w
                loss = (self.crit(pred, yb) * w).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                pbar.set_postfix(loss=float(loss))

            # validation
            model.eval()
            ys, ss = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    pred = model(xb).detach().cpu()
                    ys.append(yb.numpy()); ss.append(pred.numpy())
            import numpy as np
            y = np.concatenate(ys); s = np.concatenate(ss)
            metrics = evaluate_pointwise(y, s, thr=self.cfg["EVAL"]["THRESHOLD"])
            print("VAL:", metrics)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(model.state_dict(), "best.pt")
                print("Saved best.pt")
