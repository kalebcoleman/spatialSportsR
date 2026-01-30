testthat::test_that("nba_stats parsing and bundling works", {
  tmp_dir <- tempdir()
  raw_dir <- file.path(tmp_dir, "nba_stats_raw")
  dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)

  season <- "2024-25"
  game_id <- "0022400001"

  index_raw <- list(
    resultSets = list(
      list(
        name = "LeagueGameFinder",
        headers = c("GAME_ID", "GAME_DATE", "MATCHUP", "TEAM_ID", "TEAM_ABBREVIATION"),
        rowSet = list(
          c(game_id, "2024-10-22", "LAL vs. BOS", "1610612747", "LAL"),
          c(game_id, "2024-10-22", "BOS @ LAL", "1610612738", "BOS")
        )
      )
    )
  )
  jsonlite::write_json(index_raw, file.path(raw_dir, paste0("index_", season, ".json")), auto_unbox = TRUE)

  game_dir <- file.path(raw_dir, paste0("20241022_", game_id))
  dir.create(game_dir, recursive = TRUE, showWarnings = FALSE)

  box_raw <- list(
    resultSets = list(
      list(
        name = "TeamStats",
        headers = c("GAME_ID", "TEAM_ID", "TEAM_NAME", "PTS"),
        rowSet = list(
          c(game_id, "1610612747", "Lakers", 110),
          c(game_id, "1610612738", "Celtics", 102)
        )
      ),
      list(
        name = "PlayerStats",
        headers = c("GAME_ID", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "PTS"),
        rowSet = list(
          c(game_id, "1610612747", "2544", "LeBron James", 28),
          c(game_id, "1610612738", "201939", "Steph Curry", 30)
        )
      )
    )
  )
  jsonlite::write_json(box_raw, file.path(game_dir, "boxscore_traditional.json"), auto_unbox = TRUE)
  jsonlite::write_json(box_raw, file.path(game_dir, "boxscore_advanced.json"), auto_unbox = TRUE)
  jsonlite::write_json(box_raw, file.path(game_dir, "boxscore_fourfactors.json"), auto_unbox = TRUE)
  jsonlite::write_json(box_raw, file.path(game_dir, "boxscore_usage.json"), auto_unbox = TRUE)

  pbp_raw <- list(
    resultSets = list(
      list(
        name = "PlayByPlay",
        headers = c("gameId", "eventNum", "eventMsgType", "description"),
        rowSet = list(
          c(game_id, 1, 1, "Jump ball"),
          c(game_id, 2, 2, "Shot made")
        )
      )
    )
  )
  jsonlite::write_json(pbp_raw, file.path(game_dir, "playbyplay.json"), auto_unbox = TRUE)

  shots_raw <- list(
    resultSets = list(
      list(
        name = "Shot_Chart_Detail",
        headers = c("GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "TEAM_ID", "LOC_X", "LOC_Y"),
        rowSet = list(
          c(game_id, 2, "2544", "1610612747", 10, 15)
        )
      )
    )
  )
  shotchart_dir <- file.path(raw_dir, "nba_shotchart")
  dir.create(shotchart_dir, recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(shots_raw, file.path(shotchart_dir, paste0(season, ".json")), auto_unbox = TRUE)

  tables <- nba_stats_parse_raw_dir(season = season, raw_dir = raw_dir)

  expected_tables <- c(
    "games",
    "team_box_traditional",
    "player_box_traditional",
    "team_box_advanced",
    "player_box_advanced",
    "team_box_fourfactors",
    "player_box_fourfactors",
    "player_box_usage",
    "pbp",
    "shots",
    "manifest"
  )
  testthat::expect_true(all(expected_tables %in% names(tables)))
  testthat::expect_true(all(c("league", "source", "season", "game_id") %in% names(tables$games)))
  testthat::expect_true(all(c("team_id") %in% names(tables$team_box_traditional)))
  testthat::expect_true(all(c("player_id") %in% names(tables$player_box_traditional)))
  testthat::expect_true(all(c("team_id") %in% names(tables$team_box_advanced)))
  testthat::expect_true(all(c("player_id") %in% names(tables$player_box_advanced)))
  testthat::expect_true(all(c("team_id") %in% names(tables$team_box_fourfactors)))
  testthat::expect_true(all(c("player_id") %in% names(tables$player_box_fourfactors)))
  testthat::expect_true(all(c("player_id") %in% names(tables$player_box_usage)))
  testthat::expect_true(all(c("event_num") %in% names(tables$pbp)))
  testthat::expect_true(all(c("event_num", "shot_id") %in% names(tables$shots)))
  testthat::expect_true(all(c("league", "source", "season", "game_id") %in% names(tables$shots)))
  testthat::expect_true(all(c("game_id", "has_pbp", "has_shots") %in% names(tables$manifest)))
  testthat::expect_true(all(c("home_score", "away_score") %in% names(tables$games)))
  testthat::expect_true(all(c("home_margin", "away_margin", "winner") %in% names(tables$games)))
  testthat::expect_true(all(c("season_type") %in% names(tables$games)))
  testthat::expect_equal(unique(tables$games$season_type), "regular")
  testthat::expect_equal(tables$games$home_score[[1]], 110)
  testthat::expect_equal(tables$games$away_score[[1]], 102)
  testthat::expect_equal(tables$games$home_margin[[1]], 8)
  testthat::expect_equal(tables$games$away_margin[[1]], -8)
  testthat::expect_equal(tables$games$winner[[1]], "home")

  testthat::expect_silent(validate_tables(tables, "nba", source = "nba_stats"))

  out_dir <- file.path(tmp_dir, "nba_stats_parsed")
  write_tables(
    tables = tables,
    format = "rds",
    out_dir = out_dir,
    bundle = TRUE,
    bundle_name = "nba_stats_all"
  )

  bundle_path <- file.path(out_dir, "nba_stats_all.rds")
  testthat::expect_true(file.exists(bundle_path))
  bundle <- readRDS(bundle_path)
  testthat::expect_true(all(expected_tables %in% names(bundle)))
})

testthat::test_that("nba_stats v3 headers map to standard ids", {
  tmp_dir <- tempdir()
  game_id <- "0022409999"

  v3_raw <- list(
    resultSets = list(
      list(
        name = "TeamStats",
        headers = c("gameId", "teamId", "teamName", "pts"),
        rowSet = list(
          c(game_id, "1610612747", "Lakers", 110)
        )
      ),
      list(
        name = "PlayerStats",
        headers = c("gameId", "teamId", "personId", "name", "points"),
        rowSet = list(
          c(game_id, "1610612747", "2544", "LeBron James", 28)
        )
      )
    )
  )

  json_path <- file.path(tmp_dir, "boxscore_advanced.json")
  jsonlite::write_json(v3_raw, json_path, auto_unbox = TRUE)

  parsed <- nba_stats_parse_boxscore_advanced(json_path, season = "2023-24", game_id = game_id)
  testthat::expect_true("team" %in% names(parsed))
  testthat::expect_true("player" %in% names(parsed))
  testthat::expect_equal(parsed$team$game_id[[1]], game_id)
  testthat::expect_equal(parsed$team$team_id[[1]], "1610612747")
  testthat::expect_equal(parsed$player$player_id[[1]], "2544")
})
