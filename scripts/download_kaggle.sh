#!/usr/bin/env bash
# Requires: Kaggle CLI configured
set -euo pipefail
kaggle competitions download -c esa-adb-challenge -p data/raw
unzip -o data/raw/esa-adb-challenge.zip -d data/raw
