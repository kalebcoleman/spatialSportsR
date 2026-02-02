# spatialSportsR

An R package for reproducible NBA data analysis.

`spatialSportsR` provides a reproducible data pipeline for NBA data, focusing on data from `stats.nba.com`.

## Shot Density Heatmaps

Here are examples of shot density hexbin plots that can be generated:

### Dark Theme
![Shot Density Hexbin Plot (Dark Theme)](analysis/shot_density_hexbin_2025-26_dark.png)




## Core pipeline

The core data processing pipeline is as follows:

```
collect_raw() → parse_raw() → validate_tables() → write_tables()
```

A batch helper is also available:

```
collect_parse_write()
```

## Key conventions

- Parsed outputs are stored by season and source: `data/parsed/nba/nba_stats/<season>/`
- Playoffs are stored under a `playoffs/` subfolder: `data/parsed/nba/nba_stats/<season>/playoffs/`
- A bundled file is written when `bundle = TRUE`: `nba_stats_all.rds`
- All tables include `season_type` (normalized to `regular` / `playoffs`).
- `games` always includes `season` and `source`.
- NBA Stats `games` includes `home_score`, `away_score`, `home_margin`, `away_margin`, `winner`.

## Common workflows

### Backfill NBA Stats

```r
seasons <- 2020:2025

collect_parse_write(
  league = "nba",
  seasons = seasons,
  source = "nba_stats",
  season_type = "Regular Season",
  format = "rds",
  bundle = TRUE,
  skip_existing = TRUE,
  keep_tables = FALSE,
  force = FALSE,
  progress = TRUE,
  quiet = TRUE,
  validate = TRUE,
  workers = 2,
  rate_sleep = c(0.4, 1.0),
  max_retries = 2L,
  log_path = "logs/collect_parse_errors.csv"
)
```

## SQLite export (from RDS bundles)

```r
write_sqlite_from_rds(
  root_dir = "data/parsed",
  league = "nba",
  sources = c("nba_stats"),
  seasons = c("2026"),
  season_type = "regular",
  db_path = "data/parsed/nba.sqlite",
  mode = "append",
  bundle = TRUE,
  table_prefix = "source",
  debug = TRUE
)
```

This reads bundles or per-table RDS and writes into one SQLite DB. Date columns are stored as ISO strings.

## Debugging

- Per-season errors are written to `logs/collect_parse_errors.csv` if `log_path` is provided.
- NBA Stats raw fetch failures are logged to `data/raw/nba/<season>/nba_stats/collect_failures.csv`.
