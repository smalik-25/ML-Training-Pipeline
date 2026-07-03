---
title: Sneaker ML Platform
emoji: "◼️"
colorFrom: gray
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# Sneaker ML platform — live demo

A live dashboard over an offline ML training pipeline that predicts sneaker resale
price premium from StockX sales. Interactive predictor (the real trained model),
model card, pipeline topology, and the real-run monitoring artifacts.

The model is deliberately simple; the pipeline around it (ingest, PySpark features,
a Pandera data contract, Ray Train + PyTorch, an MLflow registry, batch + online
serving, and PSI drift monitoring) is the point.

Source: https://github.com/smalik-25/ML-Training-Pipeline

## Files

- `app.py` — the Streamlit dashboard (Sam Malik design system).
- `net.py` — self-contained model + inference (vendored from the main repo).
- `model.pt` — the trained model (weights + preprocessing stats).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
