# Feature Drift Detection

End-to-end project to detect feature drift in tabular data. It covers data preparation, feature engineering, baseline model training, evaluation, and monitoring experiments. The workflow is implemented in Python and documented in notebooks.

## Why drift detection
When the data distribution changes between training and production, model performance can degrade. Drift detection helps you:
- monitor changes in feature distributions
- identify unstable or shifted features
- decide when to retrain or investigate data quality

## Applied math from scratch (core ideas)
This project focuses on practical, interpretable drift metrics. Common choices include:

### 1) Summary statistics
Compute baseline statistics on a reference dataset and compare them to a new batch.
- Mean: $\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$
- Variance: $\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2$
- Standard deviation: $\sigma = \sqrt{\sigma^2}$

Large changes in these values indicate potential drift.


### 2) Kolmogorov-Smirnov (KS) test (continuous)
KS measures the maximum distance between cumulative distributions:

$$D = \sup_x |F_{ref}(x) - F_{cur}(x)|$$

A large $D$ suggests the distributions differ.


## Project structure
- data/
  - raw/: raw source data
  - processed/: train/val/test splits
  - monitoring/: monitoring batches
- notebooks/: EDA, training, evaluation, monitoring experiments
- src/: pipeline code
  - data_loader.py: data ingestion and split logic
  - features.py: feature engineering
  - train.py: model training
  - evaluate.py: model evaluation
  - monitoring/: drift detection utilities
- test/: drift tests

## Setup
1) Create and activate a Python environment
2) Install dependencies (example):

```bash
pip install -r requirements.txt
```

If you do not have a requirements file, install the basics:

```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

## Typical workflow
1) Explore data in notebooks/00_eda.ipynb
2) Train baseline model in notebooks/01_train_baseline.ipynb
3) Evaluate in notebooks/02_model_evaluation.ipynb
4) Run monitoring experiments in notebooks/03_monitoring_experiments.ipynb

## Run training and evaluation (scripts)
From the project root:

```bash
python -m src.train
python -m src.evaluate
```

## Monitoring / drift detection
Drift utilities live under src/monitoring/. Use them to compare reference data with new batches. See the monitoring notebook for example usage.

## Tests
Run the drift tests:

```bash
pytest -q
```

## Data
This project uses a raw retail dataset located at data/raw/online__retail_II.csv. Processed splits are stored under data/processed/.

## Notes
- The notebooks are the main source of experimentation and results.
- The code under src/ is organized for reproducible runs and reuse.
