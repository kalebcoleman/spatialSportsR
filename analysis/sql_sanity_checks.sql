-- SQLite sanity checks for spatialSportsR NBA data (fast version)

-- List tables
SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;

-- Row counts by key tables
SELECT 'espn_games' AS table_name, COUNT(*) AS rows FROM espn_games;
SELECT 'nba_stats_games' AS table_name, COUNT(*) AS rows FROM nba_stats_games;

-- Counts by season + season_type
SELECT season, season_type, COUNT(*) AS games
FROM espn_games
GROUP BY season, season_type
ORDER BY season, season_type;

SELECT season, season_type, COUNT(*) AS games
FROM nba_stats_games
GROUP BY season, season_type
ORDER BY season, season_type;

-- Duplicate key checks (games only)
SELECT season, season_type, game_id, COUNT(*) AS n
FROM espn_games
GROUP BY season, season_type, game_id
HAVING COUNT(*) > 1
ORDER BY n DESC;

SELECT season, season_type, game_id, COUNT(*) AS n
FROM nba_stats_games
GROUP BY season, season_type, game_id
HAVING COUNT(*) > 1
ORDER BY n DESC;

-- Null / missing critical fields
SELECT
  SUM(game_id IS NULL OR game_id='') AS missing_game_id,
  SUM(game_date IS NULL OR game_date='') AS missing_game_date,
  SUM(home_team_id IS NULL OR home_team_id='') AS missing_home_team_id,
  SUM(away_team_id IS NULL OR away_team_id='') AS missing_away_team_id
FROM espn_games;

SELECT
  SUM(game_id IS NULL OR game_id='') AS missing_game_id,
  SUM(game_date IS NULL OR game_date='') AS missing_game_date,
  SUM(home_team_id IS NULL OR home_team_id='') AS missing_home_team_id,
  SUM(away_team_id IS NULL OR away_team_id='') AS missing_away_team_id
FROM nba_stats_games;

-- Shots sanity
SELECT season, season_type, COUNT(*) AS shots
FROM nba_stats_shots
GROUP BY season, season_type
ORDER BY season, season_type;

SELECT COUNT(*) AS shots_missing_game_id
FROM nba_stats_shots
WHERE game_id IS NULL OR game_id = '';

-- Manifest sanity
SELECT season, season_type, COUNT(*) AS manifest_rows
FROM espn_manifest
GROUP BY season, season_type
ORDER BY season, season_type;

SELECT season, season_type, COUNT(*) AS manifest_rows
FROM nba_stats_manifest
GROUP BY season, season_type
ORDER BY season, season_type;

-- Missing data checks (qualified columns to avoid ambiguity)
SELECT g.season, g.season_type, COUNT(*) AS games_without_events
FROM espn_games g
LEFT JOIN espn_events e
  ON g.game_id = e.game_id AND g.season = e.season AND g.season_type = e.season_type
WHERE e.game_id IS NULL
GROUP BY g.season, g.season_type
ORDER BY g.season, g.season_type;

SELECT g.season, g.season_type, COUNT(*) AS games_without_player_box
FROM espn_games g
LEFT JOIN espn_player_box p
  ON g.game_id = p.game_id AND g.season_type = p.season_type
WHERE p.game_id IS NULL
GROUP BY g.season, g.season_type
ORDER BY g.season, g.season_type;

-- ESPN status breakdown
SELECT season, season_type, status, COUNT(*) AS n
FROM espn_games
GROUP BY season, season_type, status
ORDER BY season, season_type, n DESC;

-- ESPN future games (scheduled or missing scores)
SELECT season, season_type, COUNT(*) AS future_games
FROM espn_games
WHERE (status LIKE '%Scheduled%' OR status LIKE '%Pre-Game%')
   OR (home_score IS NULL OR home_score = '' OR away_score IS NULL OR away_score = '')
GROUP BY season, season_type
ORDER BY season, season_type;

-- ESPN 2026 focus
SELECT status, COUNT(*) AS n
FROM espn_games
WHERE season = 2026 AND season_type = 'regular'
GROUP BY status
ORDER BY n DESC;

SELECT COUNT(*) AS scheduled_no_events
FROM espn_games g
LEFT JOIN espn_events e ON g.game_id = e.game_id AND g.season = e.season AND g.season_type = e.season_type
WHERE g.season = 2026 AND g.season_type = 'regular'
  AND (g.status LIKE '%Scheduled%' OR g.status LIKE '%Pre-Game%')
  AND e.game_id IS NULL;
