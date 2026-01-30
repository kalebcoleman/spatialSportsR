# Extracted from test-nba-shotchart.R:58

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "spatialSportsR", path = "..")
attach(test_env, warn.conflicts = FALSE)

# test -------------------------------------------------------------------------
tmp_dir <- tempdir()
dir.create(file.path(tmp_dir, "nba_shotchart"), showWarnings = FALSE, recursive = TRUE)
season <- "2024-25"
path <- file.path(tmp_dir, "nba_shotchart", paste0(season, ".json"))
jsonlite::write_json(list(resultSets = list()), path, auto_unbox = TRUE)
out_path <- collect_nba_shotchart(season = 2024, raw_dir = tmp_dir, force = FALSE)
testthat::expect_equal(out_path, path)
