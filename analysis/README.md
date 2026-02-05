# Analysis Workflow: NBA Shot Analytics

Python scripts for NBA shot prediction and advanced analytics.

## Folder Structure

```
analysis/
├── data/           # Parsed data and CSV outputs
├── figures/        # Generated plots, heatmaps, and PDPs
├── models/         # Trained models (.joblib, .pkl)
├── reports/        # Rmarkdown reports
├── utils/          # Helper modules
│   └── court_utils.py
├── expected_points_analysis.py   # Logistic Regression xFG model
├── gam_analysis.py               # GAM model with partial dependence plots
├── advanced_analytics.py         # Residuals, SDI, Clustering
├── player_performance_analysis.py
├── shot_density.py
├── salary_collector.py
└── value_analysis.py
```

## Quick Start

```bash
python analysis/expected_points_analysis.py  # Train xFG model (Logistic Regression)
python analysis/gam_analysis.py              # Train GAM model + generate PDPs
python analysis/advanced_analytics.py        # Residuals, SDI, Clusters
python analysis/salary_collector.py          # Fetch salaries
python analysis/value_analysis.py            # Value rankings
```

## Models

| Model | Type | Accuracy | AUC-ROC |
|-------|------|----------|---------|
| xFG (Logistic Regression) | `sklearn` | 62.9% | 0.653 |
| GAM (PyGAM) | `pygam` | 62.9% | **0.655** |

## Key Metrics

- **xFG**: Expected FG% from model
- **POE**: Points Over Expected
- **SDI**: Shot Difficulty Index
- **POE/$M**: Value efficiency (POE per $1M salary)

## GAM Partial Dependence Plots

The GAM model generates interpretable visualizations showing how each feature affects shot probability:

- `gam_effect_distance.png` - Rim shots easiest, mid-range hardest
- `gam_effect_angle.png` - Corner 3s easier than straight-on
- `gam_effect_period.png` - Shooting drops off in OT (fatigue)
- `gam_spatial_probability.png` - Court heatmap of make probability
