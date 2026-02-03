testthat::test_that("write_tables() writes rds/csv outputs and bundles", {
  tmp_dir <- tempdir()
  out_dir <- file.path(tmp_dir, "parsed_out")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  tables <- list(
    games = data.frame(
      league = "nba",
      source = "espn",
      season = "2026",
      game_id = "1",
      game_date = as.Date("2026-01-01"),
      home_team = "AAA",
      away_team = "BBB",
      stringsAsFactors = FALSE
    ),
    events = data.frame(
      league = "nba",
      source = "espn",
      season = "2026",
      game_id = "1",
      event_id = "e1",
      event_type = "shot",
      x_unit = 0.5,
      y_unit = 0.5,
      label = "test",
      meta = "{}",
      stringsAsFactors = FALSE
    )
  )

  # RDS + bundle
  testthat::expect_true(write_tables(
    tables = tables,
    format = "rds",
    out_dir = out_dir,
    bundle = TRUE,
    bundle_name = "bundle_test"
  ))
  testthat::expect_true(file.exists(file.path(out_dir, "games.rds")))
  testthat::expect_true(file.exists(file.path(out_dir, "events.rds")))
  testthat::expect_true(file.exists(file.path(out_dir, "bundle_test.rds")))

  bundle <- readRDS(file.path(out_dir, "bundle_test.rds"))
  testthat::expect_true(all(c("games", "events") %in% names(bundle)))

  # CSV
  csv_dir <- file.path(tmp_dir, "csv_out")
  dir.create(csv_dir, recursive = TRUE, showWarnings = FALSE)
  testthat::expect_true(write_tables(
    tables = tables,
    format = "csv",
    out_dir = csv_dir
  ))
  testthat::expect_true(file.exists(file.path(csv_dir, "games.csv")))
  testthat::expect_true(file.exists(file.path(csv_dir, "events.csv")))
})

testthat::test_that("write_tables() sqlite mode supports append and overwrite (upsert alias)", {
  if (!requireNamespace("DBI", quietly = TRUE) || !requireNamespace("RSQLite", quietly = TRUE)) {
    testthat::skip("DBI/RSQLite not installed")
  }

  tmp_dir <- tempdir()
  db_path <- file.path(tmp_dir, "out.sqlite")

  tables1 <- list(
    games = data.frame(
      id = 1,
      value = "a",
      stringsAsFactors = FALSE
    )
  )
  tables2 <- list(
    games = data.frame(
      id = 2,
      value = "b",
      stringsAsFactors = FALSE
    )
  )

  write_tables(tables = tables1, format = "sqlite", db_path = db_path, mode = "overwrite")
  write_tables(tables = tables2, format = "sqlite", db_path = db_path, mode = "append")

  con <- DBI::dbConnect(RSQLite::SQLite(), db_path)
  testthat::expect_equal(DBI::dbGetQuery(con, "select count(*) as n from games")$n, 2)
  DBI::dbDisconnect(con)

  testthat::expect_warning(
    write_tables(tables = tables2, format = "sqlite", db_path = db_path, mode = "upsert"),
    "deprecated"
  )
  con <- DBI::dbConnect(RSQLite::SQLite(), db_path)
  testthat::expect_equal(DBI::dbGetQuery(con, "select count(*) as n from games")$n, 1)
  DBI::dbDisconnect(con)
})
