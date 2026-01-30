testthat::test_that("write_sqlite_from_rds writes tables from bundles", {
  if (!requireNamespace("DBI", quietly = TRUE) || !requireNamespace("RSQLite", quietly = TRUE)) {
    testthat::skip("DBI/RSQLite not installed")
  }

  tmp_dir <- tempdir()
  root_dir <- file.path(tmp_dir, "parsed")
  season_dir <- file.path(root_dir, "nba", "espn", "2026")
  dir.create(season_dir, recursive = TRUE, showWarnings = FALSE)

  tables <- list(
    games = data.frame(
      league = "nba",
      season = 2026,
      game_id = "1",
      game_date = "2026-01-01",
      home_team = "A",
      away_team = "B",
      source = "espn",
      stringsAsFactors = FALSE
    ),
    events = data.frame(
      league = "nba",
      season = 2026,
      game_id = "1",
      event_id = "1",
      event_type = "shot",
      x_unit = 0.5,
      y_unit = 0.5,
      label = "test",
      meta = "{}",
      source = "espn",
      stringsAsFactors = FALSE
    )
  )

  saveRDS(tables, file.path(season_dir, "espn_all.rds"))

  db_path <- file.path(tmp_dir, "nba.sqlite")
  write_sqlite_from_rds(
    root_dir = root_dir,
    league = "nba",
    sources = "espn",
    seasons = "2026",
    db_path = db_path,
    mode = "append",
    bundle = TRUE,
    debug = FALSE
  )

  con <- DBI::dbConnect(RSQLite::SQLite(), db_path)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  testthat::expect_true(DBI::dbExistsTable(con, "nba_espn_games"))
  testthat::expect_true(DBI::dbExistsTable(con, "nba_espn_events"))
  testthat::expect_equal(DBI::dbGetQuery(con, "select count(*) as n from nba_espn_games")$n, 1)
  testthat::expect_equal(DBI::dbGetQuery(con, "select count(*) as n from nba_espn_events")$n, 1)
})
