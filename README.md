# spatialSportsR

R package + Python analytics for NBA shot prediction.

## Quick Start

```bash
# R package
devtools::load_all()
devtools::test()

# Python analysis
python analysis/expected_points_analysis.py  # Train xFG model
python analysis/gam_analysis.py              # Train GAM + PDPs
python analysis/advanced_analytics.py        # Residuals, SDI, Clusters
python analysis/player_performance_analysis.py  # Player summaries + shot charts
python analysis/shot_density.py              # League shot density
python analysis/salary_collector.py          # Fetch 467 player salaries
python analysis/value_analysis.py            # Salary value + FG residual plots

# Streamlit App
streamlit run app/streamlit_app.py
```

## Features

### xFG Model (Logistic Regression)
- **True Out-of-Sample Evaluation**: Trained on 5-season rolling window (2020-21 to 2024-25), predicts on current season (2025-26) to eliminate data leakage.
- Metrics: Accuracy, AUC-ROC, Log Loss, and **Brier Score** (calibration).
- Saved to `analysis/data/model_metrics_xfg.csv`.
- Features: location, distance, shot type, zone, game clock.

### Advanced Analytics
1. **Residual Analysis** - Over/underperformance vs expectations (Green=Over, Red=Under).
2. **Shot Difficulty Index (SDI)** - Quantifies shot complexity based on 5-year historical difficulty.
3. **Player Archetypes** - role-aware GMM clustering (e.g., "Guard - Perimeter-Focused (High Usage)", "Big - Paint-Dominant").
4. **Value Analysis** - POE per $1M identifying underpaid stars and efficient role players.
5. **GAM PDPs** - Nonlinear effects visualizing the "zone of death" and shot angle impacts.

### Streamlit Dashboard
Interactive web app for exploring shot data and metrics.

- **Shot Map**: Filter by season, team, player. View shot charts with zones, accuracy, and detailed xFG metrics per shot.
- **SDI Explorer**: Interactive scatter plot of Shot Difficulty vs Actual Efficiency.
  - Highlights elite finishers (high efficiency, low difficulty) vs elite shot-makers (high efficiency, high difficulty).
  - Dynamic filtering by season and shot volume.
  - Seamless navigation to player shot charts.

### Visualizations
- **Shot Difficulty vs Actual Efficiency**: Y-axis = Actual FG%, Color = FG% Residual. Identifies elite shot-makers who convert difficult attempts.
- **Residual Heatmaps**: Spatial overperformance maps for individual players.
- **Archetype Scatter**: Clusters players by shot diet and difficulty.

## Key Findings (2025-26 Out-of-Sample)

- **Elite Shot Makers**: Shai Gilgeous-Alexander, Luke Kennard, Nikola Jokić.
- **Top Value**: Collin Gillespie (+37 POE/$M), Cam Spencer (+35 POE/$M).
- **Model Accuracy**: ~63% on unseen data.

## Data

- SQLite: `data/parsed/nba.sqlite`
- **Training Data**: ~1.1M shots (2020-21 to 2024-25)
- **Evaluation Data**: ~136K shots (2025-26 Regular Season)
- 467 player salaries from Basketball-Reference
- Model metrics: `analysis/data/model_metrics_xfg.csv` and `analysis/data/model_metrics_gam.csv`

## Data Updates

Manual run:

```bash
Rscript scripts/update_season.R --season=2026 --season-type=regular --backfill-days=3 --sources=espn,nba_stats
```

Install the daily launchd job (4:00 AM local time):

```bash
launchctl bootstrap gui/$(id -u) /Users/itzjuztmya/Kaleb/spatialSportsR/scripts/launchd/com.spatialSportsR.update.plist
```

Note: the launchd job runs `scripts/update_season.sh`, which executes the data update and then refreshes analysis figures. To skip analysis, set `SKIP_ANALYSIS=1` in your environment before running the script.

Normalize ESPN seasons to `YYYY-YY` (SQLite + RDS):

```bash
Rscript scripts/normalize_espn_seasons.R
```

Uninstall the launchd job:

```bash
launchctl bootout gui/$(id -u)/com.spatialSportsR.update
```

Logs:
- `logs/update_season.out`
- `logs/update_season.err`
- `logs/update_season_runs.csv`

## Structure

```
spatialSportsR/
├── R/                  # R package
├── analysis/           # Python scripts
│   ├── expected_points_analysis.py
│   ├── gam_analysis.py
│   ├── advanced_analytics.py
│   ├── player_performance_analysis.py
│   ├── shot_density.py
│   ├── salary_collector.py
│   ├── value_analysis.py
│   └── outputs/
├── data/parsed/        # SQLite database
└── tests/              # R unit tests
```
