testthat::test_that("manifest() uses postseason index for ESPN playoffs", {
  tmp_dir <- tempdir()

  # Regular season
  reg_dir <- dirname(cache_path(
    raw_dir = tmp_dir,
    league = "nba",
    season = 2026,
    source = "espn",
    season_type = "regular",
    filename = "x.json"
  ))
  dir.create(reg_dir, recursive = TRUE, showWarnings = FALSE)
  writeLines("{}", file.path(reg_dir, "summary_2026_20251021_401809243.json"))

  reg <- manifest(league = "nba", season = 2026, source = "espn", raw_dir = tmp_dir, season_type = "regular")
  testthat::expect_true("401809243" %in% reg$present)
  testthat::expect_true(grepl("nba_index_2026_regular\\.json$", reg$index_path))

  # Playoffs (ESPN uses 'postseason' naming for the index)
  po_dir <- dirname(cache_path(
    raw_dir = tmp_dir,
    league = "nba",
    season = 2026,
    source = "espn",
    season_type = "playoffs",
    filename = "x.json"
  ))
  dir.create(po_dir, recursive = TRUE, showWarnings = FALSE)
  writeLines("{}", file.path(po_dir, "summary_2026_20251021_401809244.json"))

  po <- manifest(league = "nba", season = 2026, source = "espn", raw_dir = tmp_dir, season_type = "playoffs")
  testthat::expect_true("401809244" %in% po$present)
  testthat::expect_true(grepl("nba_index_2026_postseason\\.json$", po$index_path))
})
