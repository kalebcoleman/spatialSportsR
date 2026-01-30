# spatialSportsR

Reproducible multi-sport data pipeline in R. Current focus is NBA with two sources:
- **ESPN summaries** (games, events, teams, team/player box, manifest)
- **NBA Stats** (games, pbp, shots, team/player box variants, manifest)

Outputs follow a contract and are written per season/source to `data/parsed/`, with optional SQLite export.

## Core pipeline

```
collect_raw() → parse_raw() → validate_tables() → write_tables()
```

Batch helper:

```
collect_parse_write()
```

## Key conventions

- Parsed outputs are stored by season and source:
  - `data/parsed/nba/espn/<season>/`
  - `data/parsed/nba/nba_stats/<season>/`
- Playoffs are stored under a `playoffs/` subfolder:
  - `data/parsed/nba/espn/<season>/playoffs/`
  - `data/parsed/nba/nba_stats/<season>/playoffs/`
- Bundles are written when `bundle = TRUE`:
  - ESPN: `espn_all.rds`
  - NBA Stats: `nba_stats_all.rds`
- `games` always includes `season` and `source`.
- All tables include `season_type` (normalized to `regular` / `playoffs`).
- NBA Stats `games` includes `home_score`, `away_score`, `home_margin`, `away_margin`, `winner`.

## NBA Stats notes

- Season directory uses the NBA Stats season string (e.g., `2025-26`).
- `pbp` is de-duplicated on `(league, source, season, game_id, event_num)`.
- `team_box_usage` is intentionally dropped (player usage remains).

## ESPN notes

- Event coordinates are normalized to `[0,1]`.
- `events$meta` is stored as JSON string.

## Common workflows

### Backfill NBA Stats (fast + safe defaults)

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

### Backfill ESPN

```r
collect_parse_write(
  league = "nba",
  seasons = seasons,
  source = "espn",
  season_type = "regular",
  format = "rds",
  bundle = TRUE,
  skip_existing = TRUE,
  keep_tables = FALSE,
  force = FALSE,
  progress = TRUE,
  quiet = TRUE,
  validate = TRUE,
  log_path = "logs/collect_parse_errors.csv"
)
```

## SQLite export (from RDS bundles)

```r
write_sqlite_from_rds(
  root_dir = "data/parsed",
  league = "nba",
  sources = c("espn", "nba_stats"),
  seasons = c("2026", "2025-26"),
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
