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
python analysis/player_performance_analysis.py  # Player summaries + shot charts
python analysis/shot_density.py              # League shot density heatmaps
python analysis/salary_collector.py          # Fetch salaries
python analysis/value_analysis.py            # POE per $1M + FG residual vs salary
```

Notes:
- `value_analysis.py` expects `analysis/data/player_salaries_2024-25.csv` from `salary_collector.py`.

## Models

| Model | Type | Metrics |
|-------|------|---------|
| xFG (Logistic Regression) | `sklearn` | `analysis/data/model_metrics_xfg.csv` |
| GAM (PyGAM) | `pygam` | `analysis/data/model_metrics_gam.csv` |

## Key Metrics

- **xFG**: Expected FG% from model
- **POE**: Points Over Expected
- **SDI**: Shot Difficulty Index
- **POE/$M**: Value efficiency (POE per $1M salary)
- **FG Residual**: Actual FG% − Expected FG% (shot-making only)

## GAM Partial Dependence Plots

The GAM model generates interpretable visualizations showing how each feature affects shot probability:

- `gam_effect_distance.png` - Rim shots easiest, mid-range hardest
- `gam_effect_angle.png` - Corner 3s easier than straight-on
- `gam_effect_period.png` - Shooting drops off in OT (fatigue)
- `gam_spatial_probability.png` - Court heatmap of make probability

## Value Plots

- `poe_vs_salary_scatter.png` - POE vs salary
- `fg_residual_vs_salary_scatter.png` - FG% residual vs salary

## Correlation Plot

- `player_metric_correlations.png` - Player metric correlation heatmap (xFG, SDI, distance, usage, zone mix)
