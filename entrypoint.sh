#!/bin/sh
set -e

if [ ! -f "models/ncf_model.keras" ]; then
  echo "No trained model found, training now (this can take a few minutes)..."
  python train.py
fi

streamlit run app.py --server.address=0.0.0.0 --server.port=8501
