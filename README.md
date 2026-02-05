# spatialSportsR

R package + Python analytics for NBA shot prediction.

## Quick Start

```bash
# R package
devtools::load_all()
devtools::test()

# Python analysis
python analysis/expected_points_analysis.py  # Train xFG model
python analysis/advanced_analytics.py        # Residuals, SDI, Clusters
python analysis/salary_collector.py          # Fetch 467 player salaries
python analysis/value_analysis.py            # POE per $1M rankings
```

## Features

### xFG Model (Logistic Regression)
- Predicts shot success probability
- 63% accuracy, 0.653 AUC-ROC
- Features: location, distance, shot type, zone, game clock

### Advanced Analytics
1. **Residual Analysis** - Over/underperformance vs expectations
2. **Shot Difficulty Index (SDI)** - Quantifies shot complexity
3. **Player Archetypes** - K-means clustering into play styles
4. **Value Analysis** - POE per $1M salary

## Key Findings

| Player | Residual | POE/$M |
|--------|----------|--------|
| Nikola Jokić | +12.9% | +2.67 |
| SGA | +8.9% | +4.32 |
| Stephen Curry | +6.5% | +1.77 |

### Archetypes
- **Rim Pressure Slasher**: Gobert, Ayton
- **Off-Dribble Shooter**: Kennard, Bridges
- **Balanced Scorer**: Jokić, Holmgren

## Data

- SQLite: `data/parsed/nba.sqlite`
- 124k+ shots from 2025-26 season
- 467 player salaries from Basketball-Reference

## Structure

```
spatialSportsR/
├── R/                  # R package
├── analysis/           # Python scripts
│   ├── expected_points_analysis.py
│   ├── advanced_analytics.py
│   ├── salary_collector.py
│   ├── value_analysis.py
│   └── outputs/
├── data/parsed/        # SQLite database
└── tests/              # R unit tests
```
