testthat::test_that("parse_nba_shotchart handles empty result set", {
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
  testthat::expect_true(is.data.frame(out))
  testthat::expect_equal(nrow(out), 0)
  testthat::expect_true(all(c("season", "season_type") %in% names(out)))
})

testthat::test_that("parse_nba_shotchart parses shot rows", {
  raw <- list(
    resultSets = list(
      list(
        name = "Shot_Chart_Detail",
        headers = c(
          "GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_NAME",
          "PERIOD", "LOC_X", "LOC_Y", "SHOT_ATTEMPTED_FLAG", "SHOT_MADE_FLAG",
          "SHOT_TYPE", "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
          "SHOT_DISTANCE", "GAME_DATE"
        ),
        rowSet = list(
          c(
            "0022400001", "201939", "Steph Curry", "1610612744", "Warriors",
            1, 10, 20, 1, 1,
            "2PT Field Goal", "Mid-Range", "Center(C)", "16-24 ft.",
            18, "2024-10-22"
          )
        )
      )
    )
  )

  out <- parse_nba_shotchart(raw, season = 2024, season_type = "Regular Season")
  testthat::expect_equal(nrow(out), 1)
  testthat::expect_equal(out$season[[1]], "2023-24")
  testthat::expect_equal(out$season_type[[1]], "Regular Season")
  testthat::expect_true(is.numeric(out$LOC_X))
  testthat::expect_true(is.integer(out$SHOT_MADE_FLAG))
  testthat::expect_true(inherits(out$GAME_DATE, "Date"))
})

testthat::test_that("collect_nba_shotchart respects existing cache", {
  tmp_dir <- tempdir()
  dir.create(file.path(tmp_dir, "nba_shotchart"), showWarnings = FALSE, recursive = TRUE)
  season <- "2023-24"
  path <- file.path(tmp_dir, "nba_shotchart", paste0(season, ".json"))
  jsonlite::write_json(list(resultSets = list()), path, auto_unbox = TRUE)

  out_path <- collect_nba_shotchart(season = 2024, raw_dir = tmp_dir, force = FALSE)
  testthat::expect_equal(out_path, path)
})

testthat::test_that("collect_nba_shotchart skips date chunking for playoffs", {
  tmp_dir <- tempdir()
  dir.create(file.path(tmp_dir, "nba_shotchart"), showWarnings = FALSE, recursive = TRUE)
  captured <- NULL

  testthat::local_mocked_bindings(
    request_with_proxy = function(url, params, proxy = NULL) {
      captured <<- params
      list(
        resultSets = list(
          list(
            name = "Shot_Chart_Detail",
            headers = c("GAME_ID"),
            rowSet = list(list("1"))
          )
        )
      )
    }
  )

  out_path <- collect_nba_shotchart(
    season = "2019-20",
    season_type = "Playoffs",
    raw_dir = tmp_dir,
    chunk = "month",
    force = TRUE
  )

  testthat::expect_true(file.exists(out_path))
  testthat::expect_false("DateFrom" %in% names(captured))
  testthat::expect_false("DateTo" %in% names(captured))
  testthat::expect_equal(captured$SeasonType, "Playoffs")
})
