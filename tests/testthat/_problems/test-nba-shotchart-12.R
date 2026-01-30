# Extracted from test-nba-shotchart.R:12

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "spatialSportsR", path = "..")
attach(test_env, warn.conflicts = FALSE)

# test -------------------------------------------------------------------------
raw <- list(
    resultSets = list(
      list(
        name = "Shot_Chart_Detail",
        headers = c("GAME_ID", "PLAYER_ID", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "SHOT_ATTEMPTED_FLAG", "SHOT_DISTANCE", "PERIOD", "GAME_DATE"),
        rowSet = list()
      )
    )
  )
out <- parse_nba_shotchart(raw, season = 2024)
