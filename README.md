# spatialSportsR

R package + Python analytics for NBA shot prediction.

## Quick Start

```bash
# R package
devtools::load_all()
devtools::test()

# Python analysis
python analysis/expected_points_analysis.py  # Train xFG model
python analysis/advanced_analytics.py        # Run all analytics
```

## Features

### xFG Model (Logistic Regression)
- Predicts shot success probability
- 63% accuracy, 0.653 AUC-ROC
- Features: location, distance, shot type, zone, game clock

### Advanced Analytics
1. **Residual Analysis** - Who over/underperforms expectations?
2. **Shot Difficulty Index (SDI)** - How hard are their shots?
3. **Player Archetypes** - K-means clustering into play styles

## Key Findings

| Player | Residual | Insight |
|--------|----------|---------|
| Nikola Jokić | +12.9% | Top overperformer |
| Luke Kennard | +12.6% | Shot-making god |
| SGA | +8.9% | Elite efficiency |

### Archetypes
- **Rim Pressure Slasher**: Gobert, Ayton, Mark Williams
- **Off-Dribble Shooter**: Kennard, Bridges, Hachimura
- **Balanced Scorer**: Jokić, Holmgren, Johnson

## Data

- SQLite: `data/parsed/nba.sqlite`
- 124k+ shots from 2025-26 season
- Full R pipeline: collect → parse → validate → write

## Structure

```
spatialSportsR/
├── R/                  # R package
├── analysis/           # Python scripts
│   ├── expected_points_analysis.py
│   ├── advanced_analytics.py
│   └── outputs/
├── data/parsed/        # SQLite database
└── tests/              # R unit tests
```

## Installation

```r
install.packages(c("devtools", "renv"))
devtools::load_all()
```

```bash
pip install -r requirements.txt
```
