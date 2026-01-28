# AGENTS.md

## Project mission
- Build a reproducible multi-sport data pipeline in R.
- Support NBA/NHL shot charts + NFL play-by-play (field-position events) and optional MLB later.
- Standardize outputs into contract tables to enable shared modeling + visualization across leagues.

## Core pipeline
collect_raw() → parse_raw() → validate_tables() → write_tables()

- core/* = orchestration + standardization
- source_* = league-specific API knowledge only (no business logic duplication)

## The CONTRACT

### 1) games (required)
Columns:
- league
- season
- game_id
- game_date
- home_team
- away_team
- optional: home_team_id, away_team_id

Key:
- unique key: (league, game_id)

### 2) events (required)
Columns:
- league
- season
- game_id
- event_id
- event_type
- x_unit
- y_unit
- label
- meta

Rules:
- x_unit, y_unit can be NA if league has no true XY.
- if present, x_unit/y_unit must be in [0,1].

Key:
- unique key: (league, game_id, event_id)

Foreign key:
- events$game_id must exist in games$game_id for the same league.

### 3) teams (optional but preferred)
Columns:
- league
- team_id
- team_abbrev
- team_name

### 4) manifest (required when raw data exists)
Columns:
- league
- season
- game_id
- downloaded_at
- source_url
- status_code
- bytes
- optional: hash

## Normalization spec
normalize_xy():
- Converts league-native coordinates into x_unit/y_unit in [0,1].
- Purpose: consistent plotting + cross-league comparisons.
- Per-league geometry lives ONLY in source_* normalize function; core code treats [0,1] as universal.

## Validation rules
validate_tables() must check:
- required columns exist
- uniqueness of keys
- FK integrity
- x_unit/y_unit bounds if present
- friendly error messages listing missing columns and bad rows counts

## Implementation plan (Phases A–F)
Phase A: repo skeleton stable, core API stubs + exports
Phase B: contract validation + tests
Phase C: implement NBA source (index + raw + parse + normalize)
Phase D: implement NHL source
Phase E: implement NFL source (play-by-play to events; x_unit from yardline)
Phase F: vignettes + examples + CI

## Working rules for AI contributors
- Always add/maintain tests when changing core behavior
- Avoid adding heavy dependencies; use Suggests for league-specific packages
- Keep source_* adapters thin; parsing should output contract tables

## Current state (Jan 2026)

### NBA ESPN summary pipeline
- Summary JSON ingestion is implemented end-to-end:
  - `collect_raw()` downloads ESPN summary JSONs to `data/raw/nba/<season>/summary_<season>_<YYYYMMDD>_<GAMEID>.json`.
  - `parse_raw()` parses all summaries into tables: `games`, `events`, `teams`, `team_box`, `player_box`, `manifest`.
  - `validate_tables()` checks core contract for `games` + `events` (plus teams/manifest if present).
  - `write_tables()` writes **all tables** (dynamic names) to rds/csv/sqlite.
- ESPN standings integration was **removed** (not historically accurate). No standings fetch, no standings merge.

### NBA parsing details
- `games` includes:
  - core fields + `home_record_total`, `away_record_total`.
  - `home_record_at_home` and `away_record_on_road` from summary `competitors$record`.
  - Other sub-records (home road vsconf for opposite side) removed due to inconsistent availability.
- `teams` may still contain duplicates across games; de-dupe if needed.
- `team_box` / `player_box`:
  - dynamic stats parsed from ESPN boxscore.
  - compound stats split into separate columns (e.g. `fieldGoalsMade`, `fieldGoalsAttempted`) while keeping composite `fieldGoalsMade-fieldGoalsAttempted`.
  - team_box drops low-coverage columns:
    - `leadChanges`, `leadPercentage`, `streak`, `Last Ten Games`,
    - `avgPointsAgainst`, `avgPoints`, `avgRebounds`, `avgAssists`, `avgBlocks`, `avgSteals`,
    - `avgTeamTurnovers`, `avgTotalTurnovers`.
- `events` parsing coerces numeric columns to avoid bind failures (`sequence_number`, `period`, `points_attempted`, `score_value`, `x_raw`, `y_raw`).

### Manifest flags
- `manifest` now includes `has_events`, `has_team_box`, `has_player_box`, `has_teams` for easy filtering.

### Write utilities
- `write_tables()` now writes **all data frames** in `tables` (not just core ones).
- Added `skip_existing` to avoid overwriting rds/csv if files already exist.

### Batch pipeline
- `collect_parse_write()` now lives in `R/core-write.R` and runs:
  `collect_raw() → parse_raw() → validate_tables() → write_tables()` per season.
- Use for backfill, e.g.:
  `collect_parse_write("nba", 2006:2025, format = "rds", skip_existing = TRUE)`
- SQLite mode:
  - `append` adds rows.
  - `upsert` currently overwrites table (not true key-based merge).

### Known data realities / modeling notes
- ESPN summaries for future/unplayed games exist but have empty boxscore/events.
- For completed games: team_box has full stats; player_box has NAs for DNPs.
- Win%/streak/last10 are NOT in summary JSON, only in standings (removed).

### Testing
- `test-nba-source.R` was deleted after standings rollback.
- `parse_raw()` uses `bind_rows()` to avoid column mismatch errors.
