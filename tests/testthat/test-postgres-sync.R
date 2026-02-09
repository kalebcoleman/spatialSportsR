# ---- Unit tests (no Postgres required) ----

testthat::test_that(".sqlite_type_to_pg maps types correctly", {
  expect_equal(.sqlite_type_to_pg("INTEGER"), "BIGINT")
  expect_equal(.sqlite_type_to_pg("INT"), "BIGINT")
  expect_equal(.sqlite_type_to_pg("integer"), "BIGINT")
  expect_equal(.sqlite_type_to_pg("REAL"), "DOUBLE PRECISION")
  expect_equal(.sqlite_type_to_pg("TEXT"), "TEXT")
  expect_equal(.sqlite_type_to_pg("BLOB"), "BYTEA")
  expect_equal(.sqlite_type_to_pg("NUMERIC"), "NUMERIC")
  expect_equal(.sqlite_type_to_pg(""), "TEXT")
  expect_equal(.sqlite_type_to_pg(NULL), "TEXT")
  expect_equal(.sqlite_type_to_pg("  "), "TEXT")
})

testthat::test_that(".r_class_to_pg maps R classes to Postgres types", {
  expect_equal(.r_class_to_pg("integer"), "BIGINT")
  expect_equal(.r_class_to_pg("numeric"), "DOUBLE PRECISION")
  expect_equal(.r_class_to_pg("character"), "TEXT")
  expect_equal(.r_class_to_pg("logical"), "BOOLEAN")
  expect_equal(.r_class_to_pg("Date"), "DATE")
  expect_equal(.r_class_to_pg("POSIXct"), "TIMESTAMPTZ")
  expect_equal(.r_class_to_pg("raw"), "BYTEA")
  expect_equal(.r_class_to_pg("unknown_class"), "TEXT")
})

testthat::test_that(".default_conflict_keys returns keys for all 17 tables", {
  keys <- .default_conflict_keys()
  expected_tables <- c(
    "espn_games", "espn_events", "espn_manifest", "espn_player_box",
    "espn_team_box", "espn_teams",
    "nba_stats_games", "nba_stats_manifest", "nba_stats_pbp", "nba_stats_shots",
    "nba_stats_player_box_traditional", "nba_stats_player_box_advanced",
    "nba_stats_player_box_fourfactors", "nba_stats_player_box_usage",
    "nba_stats_team_box_traditional", "nba_stats_team_box_advanced",
    "nba_stats_team_box_fourfactors"
  )

  expect_equal(sort(names(keys)), sort(expected_tables))

  for (tbl in names(keys)) {
    expect_true(length(keys[[tbl]]) >= 1, info = paste("table:", tbl))
    expect_true(all(nzchar(keys[[tbl]])), info = paste("table:", tbl))
  }
})

testthat::test_that(".default_sync_tables matches .default_conflict_keys", {
  expect_equal(sort(.default_sync_tables()), sort(names(.default_conflict_keys())))
})

testthat::test_that("pg_connect requires RPostgres and connection params", {
  if (!requireNamespace("RPostgres", quietly = TRUE)) {
    expect_error(pg_connect(), "Install the RPostgres package")
  } else {
    withr::with_envvar(
      c(POSTGRES_HOST = "", POSTGRES_DB = "", POSTGRES_USER = ""),
      expect_error(pg_connect(), "host is required")
    )
  }
})

testthat::test_that("sync_sqlite_to_postgres errors on missing SQLite file", {
  expect_error(
    sync_sqlite_to_postgres(sqlite_path = "/tmp/nonexistent.sqlite"),
    "not found"
  )
})

# ---- Integration tests (require a real Postgres instance) ----

testthat::test_that("full sync round-trip works against Postgres", {
  testthat::skip_if_not(
    nzchar(Sys.getenv("POSTGRES_TEST")),
    "Set POSTGRES_TEST=true to run Postgres integration tests"
  )
  testthat::skip_if_not_installed("RPostgres")
  testthat::skip_if_not_installed("DBI")
  testthat::skip_if_not_installed("RSQLite")

  # Create a small test SQLite database
  sl_path <- tempfile(fileext = ".sqlite")
  sl_con <- DBI::dbConnect(RSQLite::SQLite(), sl_path)
  on.exit(DBI::dbDisconnect(sl_con), add = TRUE)

  DBI::dbWriteTable(sl_con, "test_games", data.frame(
    game_id = c("G1", "G2", "G3"),
    season = c("2025-26", "2025-26", "2024-25"),
    season_type = c("regular", "regular", "regular"),
    home_team = c("LAL", "BOS", "MIA"),
    away_team = c("GSW", "NYK", "CHI"),
    home_score = c(110L, 105L, 98L),
    stringsAsFactors = FALSE
  ))

  # Connect to Postgres
  pg_con <- pg_connect()
  on.exit(DBI::dbDisconnect(pg_con), add = TRUE)

  # Clean up test table
  tryCatch(DBI::dbExecute(pg_con, "DROP TABLE IF EXISTS test_games"), error = function(e) NULL)

  keys <- list(test_games = "game_id")

  # Test 1: Initial full sync
  result <- sync_sqlite_to_postgres(
    sqlite_path = sl_path,
    pg_con = pg_con,
    tables = "test_games",
    conflict_keys = keys,
    verbose = FALSE
  )

  expect_equal(result$rows_synced, 3L)

  pg_rows <- DBI::dbGetQuery(pg_con, "SELECT * FROM test_games ORDER BY game_id")
  expect_equal(nrow(pg_rows), 3L)
  expect_true("_synced_at" %in% names(pg_rows))

  # Test 2: Upsert updates existing + inserts new
  DBI::dbExecute(sl_con, "UPDATE test_games SET home_score = 120 WHERE game_id = 'G1'")
  DBI::dbExecute(sl_con, "INSERT INTO test_games VALUES ('G4', '2025-26', 'regular', 'PHX', 'DEN', 115)")

  result2 <- sync_sqlite_to_postgres(
    sqlite_path = sl_path,
    pg_con = pg_con,
    tables = "test_games",
    conflict_keys = keys,
    verbose = FALSE
  )

  pg_rows2 <- DBI::dbGetQuery(pg_con, "SELECT * FROM test_games ORDER BY game_id")
  expect_equal(nrow(pg_rows2), 4L)
  expect_equal(pg_rows2$home_score[pg_rows2$game_id == "G1"], 120)

  # Test 3: Season filter
  result3 <- sync_sqlite_to_postgres(
    sqlite_path = sl_path,
    pg_con = pg_con,
    tables = "test_games",
    conflict_keys = keys,
    seasons = "2025-26",
    verbose = FALSE
  )

  expect_equal(result3$rows_synced, 3L)

  # Test 4: Chunk boundaries
  result4 <- sync_sqlite_to_postgres(
    sqlite_path = sl_path,
    pg_con = pg_con,
    tables = "test_games",
    conflict_keys = keys,
    chunk_size = 2L,
    verbose = FALSE
  )

  pg_rows4 <- DBI::dbGetQuery(pg_con, "SELECT COUNT(*) AS n FROM test_games")
  expect_equal(pg_rows4$n, 4L)

  # Cleanup
  DBI::dbExecute(pg_con, "DROP TABLE IF EXISTS test_games")
  unlink(sl_path)
})

testthat::test_that("schema evolution adds new columns", {
  testthat::skip_if_not(
    nzchar(Sys.getenv("POSTGRES_TEST")),
    "Set POSTGRES_TEST=true to run Postgres integration tests"
  )
  testthat::skip_if_not_installed("RPostgres")
  testthat::skip_if_not_installed("DBI")
  testthat::skip_if_not_installed("RSQLite")

  sl_path <- tempfile(fileext = ".sqlite")
  sl_con <- DBI::dbConnect(RSQLite::SQLite(), sl_path)
  on.exit(DBI::dbDisconnect(sl_con), add = TRUE)

  DBI::dbWriteTable(sl_con, "test_evolve", data.frame(
    game_id = "G1", value = 10L, stringsAsFactors = FALSE
  ))

  pg_con <- pg_connect()
  on.exit(DBI::dbDisconnect(pg_con), add = TRUE)

  tryCatch(DBI::dbExecute(pg_con, "DROP TABLE IF EXISTS test_evolve"), error = function(e) NULL)
  keys <- list(test_evolve = "game_id")

  sync_sqlite_to_postgres(sl_path, pg_con, "test_evolve", keys, verbose = FALSE)

  # Add a new column in SQLite
  DBI::dbExecute(sl_con, "ALTER TABLE test_evolve ADD COLUMN new_col TEXT")
  DBI::dbExecute(sl_con, "UPDATE test_evolve SET new_col = 'hello' WHERE game_id = 'G1'")

  sync_sqlite_to_postgres(sl_path, pg_con, "test_evolve", keys, verbose = FALSE)

  pg_cols <- DBI::dbListFields(pg_con, "test_evolve")
  expect_true("new_col" %in% pg_cols)

  pg_data <- DBI::dbGetQuery(pg_con, "SELECT new_col FROM test_evolve WHERE game_id = 'G1'")
  expect_equal(pg_data$new_col, "hello")

  DBI::dbExecute(pg_con, "DROP TABLE IF EXISTS test_evolve")
  unlink(sl_path)
})
