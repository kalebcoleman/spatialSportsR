# Analysis Workflow: NBA Shot Analytics

Python scripts for NBA shot prediction and advanced analytics.

## Files

| File | Description |
|------|-------------|
| `court_utils.py` | Shared court drawing utilities |
| `expected_points_analysis.py` | xFG model training and POE calculation |
| `player_performance_analysis.py` | Player-level POE rankings |
| `shot_density.py` | Full-court hexbin density heatmaps |
| `advanced_analytics.py` | **NEW** Residuals, SDI, Clustering |

## Quick Start

```bash
python expected_points_analysis.py    # Train xFG model
python advanced_analytics.py          # Run all 3 advanced analytics
```

## Advanced Analytics

### 1. Residual Analysis
- Compute: `actual - expected` per shot
- Top overperformer: Nikola Jokić (+12.9%)
- Output: `player_residuals.csv`, residual heatmaps

### 2. Shot Difficulty Index (SDI)
```
SDI = 0.30×distance + 0.20×clock + 0.20×type + 0.15×zone + 0.15×angle
```
- Higher = harder shot
- Output: `sdi_vs_xfg_scatter.png`

### 3. Player Archetypes
- K-Means clustering on shot profile features
- Archetypes: Rim Pressure Slasher, Off-Dribble Shooter, Balanced Scorer
- Output: `player_clusters.csv`, `player_archetypes_scatter.png`

## Model Performance

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 62.9% | 0.653 |

## Outputs → `outputs/`

All visualizations and data files saved automatically.
