# Extracted from test-nba-source.R:62

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "spatialSportsR", path = "..")
attach(test_env, warn.conflicts = FALSE)

# test -------------------------------------------------------------------------
path <- testthat::test_path("..", "..", "data/raw/nba/2026/summary_2026_20251021_401809243.json")
if (!file.exists(path)) {
    skip("NBA summary fixture not available")
  }
src <- source_nba()
fake_standings <- data.frame(
    team_id = c("7", "21"),
    wins = c(31, 27),
    losses = c(15, 19),
    winPercent = c(0.6739, 0.587),
    gamesBehind = c(5.5, 1.5),
    streak = c("W2", "L2"),
    lasttengames = c("6-4", "5-5"),
    playoffSeed = c(3, 8),
    home = c("14-7", "12-9"),
    road = c("17-8", "15-10"),
    vsdiv = c("7-4", "6-5"),
    vsconf = c("21-11", "18-14"),
    avgPointsFor = c(113.2, 112.1),
    avgPointsAgainst = c(108.4, 110.9),
    differential = c(4.8, 1.2),
    pointsFor = c(5200, 5150),
    pointsAgainst = c(4985, 5090),
    pointDifferential = c(215, 60),
    leagueWinPercent = c(0.674, 0.587),
    divisionWinPercent = c(0.636, 0.545),
    stringsAsFactors = FALSE
  )
parsed <- src$parse_game_raw(
    path,
    include_events = FALSE,
    include_boxes = FALSE,
    include_players = FALSE,
    include_manifest = FALSE,
    include_standings = TRUE,
    standings = fake_standings,
    standings_fetch = FALSE
  )
expect_true(is.data.frame(parsed$games))
expect_equal(nrow(parsed$games), 1)
expected_cols <- c(
    "home_wins", "home_losses", "home_win_percent", "home_streak", "home_games_behind",
    "away_wins", "away_losses", "away_win_percent", "away_streak", "away_games_behind",
    "home_last10", "away_last10",
    "home_playoff_seed", "away_playoff_seed",
    "home_record_home", "away_record_road",
    "home_record_vsdiv", "away_record_vsconf",
    "home_avg_points_for", "away_avg_points_against",
    "home_point_differential", "away_point_differential",
    "home_league_win_percent", "away_division_win_percent"
  )
expect_true(all(expected_cols %in% names(parsed$games)))
expect_false(all(is.na(parsed$games$home_wins)))
expect_false(all(is.na(parsed$games$away_wins)))
expect_equal(parsed$games$home_last10, "6-4")
expect_equal(parsed$games$away_last10, "5-5")
