testthat::test_that("collect_raw parallelizes nba_stats per game and logs failures", {
  if (!requireNamespace("future", quietly = TRUE) || !requireNamespace("future.apply", quietly = TRUE)) {
    testthat::skip("future/future.apply not installed")
  }

  tmp_dir <- tempdir()
  raw_dir <- file.path(tmp_dir, "raw")
  season <- "2023-24"
  game_ids <- c("001", "002", "003")
  game_dates <- as.Date(c("2023-10-01", "2023-10-02", "2023-10-03"))

  calls <- new.env(parent = emptyenv())
  calls$count <- 0L
  calls$endpoint_requests <- 0L

  fake_collect_index <- function(season, raw_dir, force = FALSE, ...) {
    data.frame(
      game_id = game_ids,
      game_date = game_dates,
      stringsAsFactors = FALSE
    )
  }

  fake_collect_game_raw <- function(game_id, season, raw_dir, force = FALSE, ...) {
    calls$count <- calls$count + 1L
    game_dir <- file.path(raw_dir, paste0(format(game_dates[match(game_id, game_ids)], "%Y%m%d"), "_", game_id))
    dir.create(game_dir, recursive = TRUE, showWarnings = FALSE)
    endpoint_path <- file.path(game_dir, "boxscore_traditional.json")
    if (!file.exists(endpoint_path)) {
      calls$endpoint_requests <- calls$endpoint_requests + 1L
      jsonlite::write_json(list(ok = TRUE), endpoint_path, auto_unbox = TRUE)
    }
    if (game_id == "002") stop("boom")
    endpoint_path
  }

  fake_src <- list(
    league = "nba",
    source = "nba_stats",
    collect_game_index = fake_collect_index,
    collect_game_raw = fake_collect_game_raw
  )

  testthat::local_mocked_bindings(
    get_source = function(league, source = NULL) fake_src,
    .resolve_sources = function(league, source = NULL) "nba_stats"
  )

  out <- collect_raw(
    league = "nba",
    season = season,
    source = "nba_stats",
    raw_dir = raw_dir,
    workers = 2,
    progress = FALSE
  )

  testthat::expect_equal(calls$count, 3L)
  testthat::expect_equal(calls$endpoint_requests, 3L)
  testthat::expect_true(file.exists(file.path(out$league_dir, "collect_failures.csv")))

  # re-run should not recreate endpoint files (cache check inside fake function)
  calls$count <- 0L
  calls$endpoint_requests <- 0L
  collect_raw(
    league = "nba",
    season = season,
    source = "nba_stats",
    raw_dir = raw_dir,
    workers = 2,
    progress = FALSE
  )
  testthat::expect_equal(calls$count, 3L)
  testthat::expect_equal(calls$endpoint_requests, 0L)
})
