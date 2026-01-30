#' NBA ESPN source helpers (internal)
#'
#' @description
#' Internal ESPN NBA source implementation returned by `get_source("nba", source = "espn")`.
#'
#' @keywords internal
#' @importFrom dplyr as_tibble bind_rows
#' @importFrom jsonlite fromJSON

source_nba_espn <- function() {
  list(
    league = "nba",
    source = "espn",
    # Collect NBA game index (ESPN scoreboard)
    collect_game_index = function(season, season_type = c("regular", "postseason"), raw_dir = "data/raw", force = FALSE, date_from = NULL, date_to = NULL, progress = TRUE, quiet = FALSE, game_ids = NULL, save_index = FALSE, ...) {
      season_type <- match.arg(season_type)
      season <- as.integer(season)

      if (!requireNamespace("httr", quietly = TRUE)) {
        stop("Install httr to use ESPN NBA scoreboard", call. = FALSE)
      }
      if (!requireNamespace("jsonlite", quietly = TRUE)) {
        stop("Install jsonlite to use ESPN NBA scoreboard", call. = FALSE)
      }

      dates <- .espn_nba_season_dates(season, season_type, date_from, date_to)
      total <- length(dates)
      if (total == 0) {
        return(.espn_nba_empty_games_index())
      }

      if (isTRUE(progress)) {
        pb <- utils::txtProgressBar(min = 0, max = total, style = 3)
        on.exit(close(pb), add = TRUE)
      }

      rows <- vector("list", total)
      row_idx <- 0L
      error_count <- 0L

      for (i in seq_along(dates)) {
        date_val <- dates[[i]]
        season_type_num <- if (season_type == "regular") 2 else 3
        scoreboard <- espn_nba_scoreboard(format(as.Date(date_val), "%Y%m%d"))
        if (is.null(scoreboard)) {
          error_count <- error_count + 1L
          if (isTRUE(progress)) {
            utils::setTxtProgressBar(pb, i)
          } else if (!isTRUE(quiet)) {
            message(sprintf("[%s/%s] skip %s (scoreboard error)", i, total, format(date_val, "%Y-%m-%d")))
          }
          next
        }
        day_rows <- scoreboard
        if (nrow(day_rows) > 0) {
          day_rows <- day_rows[day_rows$season == season & day_rows$season_type == season_type_num, , drop = FALSE]
        }

        if (nrow(day_rows) > 0) {
          row_idx <- row_idx + 1L
          rows[[row_idx]] <- day_rows
        }

        if (isTRUE(progress)) {
          utils::setTxtProgressBar(pb, i)
        } else if (!isTRUE(quiet)) {
          message(sprintf("[%s/%s] %s", i, total, format(date_val, "%Y-%m-%d")))
        }
      }

      rows <- rows[seq_len(row_idx)]
      games <- if (length(rows) == 0) .espn_nba_empty_games_index() else do.call(rbind, rows)

      if (nrow(games) > 0) {
        games <- games[!duplicated(games$game_id), , drop = FALSE]
      }

      if (!is.null(game_ids) && length(game_ids) > 0 && nrow(games) > 0) {
        game_ids <- as.character(game_ids)
        games <- games[games$game_id %in% game_ids, , drop = FALSE]
      }

      if (nrow(games) == 0) {
        warning(sprintf("No games found for season %s (%s). Scoreboard errors: %s", season, season_type, error_count))
      }

      if (isTRUE(save_index) && nrow(games) > 0) {
        index_path <- file.path(raw_dir, paste0("nba_index_", season, "_", season_type, ".json"))
        dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
        jsonlite::write_json(games, index_path, auto_unbox = TRUE, pretty = TRUE)
        return(list(games = games, index_path = index_path))
      }

      games
    },
    # Collect raw NBA game summary JSON (ESPN)
    collect_game_raw = function(game_id, season, raw_dir = "data/raw", force = FALSE, game_date = NULL, ...) {
      season <- as.character(season)
      league_dir <- file.path(raw_dir)
      if (!dir.exists(league_dir)) {
        dir.create(league_dir, recursive = TRUE, showWarnings = FALSE)
      }

      game_id <- as.character(game_id)
      date_key <- .espn_nba_clean_date_key(game_date)
      if (!is.null(date_key)) {
        file_path <- file.path(league_dir, paste0("summary_", season, "_", date_key, "_", game_id, ".json"))
      } else {
        file_path <- file.path(league_dir, paste0("summary_", season, "_", game_id, ".json"))
      }
      legacy_path <- file.path(league_dir, paste0("summary_", season, "_", game_id, ".json"))

      if (!isTRUE(force) && file.exists(file_path)) {
        return(file_path)
      }
      if (!isTRUE(force) && file.exists(legacy_path)) {
        return(legacy_path)
      }

      result <- espn_nba_summary_raw_safe(
        game_id = game_id,
        save_raw = TRUE,
        raw_dir = league_dir,
        file_path = file_path
      )

      if (!is.null(result$error)) {
        log_failed_scrape(
          game_id = game_id,
          game_date = NA,
          status_code = result$status_code,
          error_message = result$error,
          dir = league_dir
        )
        return(NA_character_)
      }

      result$file_path
    },
    parse_game_raw = function(path,
                              include_games = TRUE,
                              include_events = TRUE,
                              include_teams = TRUE,
                              include_boxes = TRUE,
                              include_players = TRUE,
                              include_manifest = TRUE,
                              season_type = "regular",
                              ...) {
      if (!requireNamespace("jsonlite", quietly = TRUE)) {
        stop("Install jsonlite to parse ESPN NBA summaries", call. = FALSE)
      }

      raw_json <- fromJSON(path, simplifyVector = FALSE)
      parsed <- list()

      stype <- tolower(as.character(season_type))
      game_type <- suppressWarnings(as.integer(.espn_nba_pluck(raw_json, "header", "season", "type", default = NA_integer_)))
      if (!is.na(game_type)) {
        if (stype == "regular" && game_type != 2L) {
          # Skip non-regular games when running regular-season pulls.
          parsed$games <- .espn_nba_empty_games()
          parsed$events <- .espn_nba_empty_events()
          parsed$teams <- .espn_nba_empty_teams()
          parsed$team_box <- .espn_nba_empty_team_box()
          parsed$player_box <- .espn_nba_empty_player_box()
          parsed$manifest <- .espn_nba_manifest_from_file(
            path,
            raw_json,
            has_events = FALSE,
            has_team_box = FALSE,
            has_player_box = FALSE,
            has_teams = FALSE
          )
        } else if (stype %in% c("playoffs", "postseason") && game_type == 2L) {
          # Skip regular-season games when running playoff pulls.
          parsed$games <- .espn_nba_empty_games()
          parsed$events <- .espn_nba_empty_events()
          parsed$teams <- .espn_nba_empty_teams()
          parsed$team_box <- .espn_nba_empty_team_box()
          parsed$player_box <- .espn_nba_empty_player_box()
          parsed$manifest <- .espn_nba_manifest_from_file(
            path,
            raw_json,
            has_events = FALSE,
            has_team_box = FALSE,
            has_player_box = FALSE,
            has_teams = FALSE
          )
        }
      }

      if (length(parsed) > 0) {
        stype_norm <- if (stype == "postseason") "playoffs" else stype
        for (nm in names(parsed)) {
          parsed[[nm]] <- .espn_nba_add_source(parsed[[nm]], source = "espn")
          if (is.data.frame(parsed[[nm]])) {
            parsed[[nm]]$season_type <- rep(stype_norm, nrow(parsed[[nm]]))
          }
        }
        return(parsed)
      }

      if (isTRUE(include_games)) {
        parsed$games <- .espn_nba_parse_game_header(raw_json)
      }
      if (isTRUE(include_events)) {
        parsed$events <- .espn_nba_parse_plays(raw_json)
      }
      if (isTRUE(include_teams)) {
        parsed$teams <- .espn_nba_parse_teams(raw_json)
      }
      if (isTRUE(include_boxes)) {
        parsed$team_box <- .espn_nba_parse_team_box(raw_json)
      }
      if (isTRUE(include_players)) {
        parsed$player_box <- .espn_nba_parse_player_box(raw_json)
      }
      if (isTRUE(include_manifest)) {
        parsed$manifest <- .espn_nba_manifest_from_file(
          path,
          raw_json,
          has_events = isTRUE(include_events) && !is.null(parsed$events) && nrow(parsed$events) > 0,
          has_team_box = isTRUE(include_boxes) && !is.null(parsed$team_box) && nrow(parsed$team_box) > 0,
          has_player_box = isTRUE(include_players) && !is.null(parsed$player_box) && nrow(parsed$player_box) > 0,
          has_teams = isTRUE(include_teams) && !is.null(parsed$teams) && nrow(parsed$teams) > 0
        )
      }

      stype <- tolower(as.character(season_type))
      if (stype == "postseason") stype <- "playoffs"
      for (nm in names(parsed)) {
        parsed[[nm]] <- .espn_nba_add_source(parsed[[nm]], source = "espn")
        if (is.data.frame(parsed[[nm]])) {
          parsed[[nm]]$season_type <- rep(stype, nrow(parsed[[nm]]))
        }
      }

      parsed
    },
    normalize_xy = function(x, y) {
      x <- suppressWarnings(as.numeric(x))
      y <- suppressWarnings(as.numeric(y))

      x_unit <- x
      y_unit <- y

      if (length(x_unit) > 0 || length(y_unit) > 0) {
        max_x <- if (all(is.na(x_unit))) NA_real_ else max(x_unit, na.rm = TRUE)
        max_y <- if (all(is.na(y_unit))) NA_real_ else max(y_unit, na.rm = TRUE)
        min_x <- if (all(is.na(x_unit))) NA_real_ else min(x_unit, na.rm = TRUE)
        min_y <- if (all(is.na(y_unit))) NA_real_ else min(y_unit, na.rm = TRUE)

        if (!is.na(min_x) && min_x < 0) {
          # Centered coords (e.g., -47..47, -25..25)
          x_unit <- (x_unit + 47) / 94
          y_unit <- (y_unit + 25) / 50
        } else if (!is.na(max_x) && !is.na(max_y) && max_x <= 100 && max_y <= 100) {
          # ESPN uses 0-100 style coordinates
          x_unit <- x_unit / 100
          y_unit <- y_unit / 100
        } else {
          # Fallback to feet-based court
          x_unit <- x_unit / 94
          y_unit <- y_unit / 50
        }
      }

      x_unit <- pmin(pmax(x_unit, 0), 1)
      y_unit <- pmin(pmax(y_unit, 0), 1)

      data.frame(x_unit = x_unit, y_unit = y_unit)
    },
    schemas = function() {
      list(
        events = system.file("extdata/schemas/nba_spatial_events.json", package = "spatialSportsR"),
        games = system.file("extdata/schemas/common_games.json", package = "spatialSportsR"),
        teams = system.file("extdata/schemas/common_teams.json", package = "spatialSportsR")
      )
    },
    raw_pattern = "^summary_.*\\.json$"
  )
}

.espn_nba_add_source <- function(df, source = "espn") {
  if (!is.data.frame(df)) return(df)
  df$source <- rep(source, nrow(df))
  df
}

.espn_nba_season_dates <- function(season, season_type, date_from = NULL, date_to = NULL) {
  if (!is.null(date_from) && !is.null(date_to)) {
    return(seq.Date(as.Date(date_from), as.Date(date_to), by = "day"))
  }

  if (season_type == "regular") {
    start_date <- as.Date(sprintf("%d-10-01", season - 1))
    end_date <- as.Date(sprintf("%d-06-30", season))
    # Pandemic-era regular seasons extended into July/August.
    if (season %in% c(2020, 2021)) {
      end_date <- as.Date(sprintf("%d-08-31", season))
    }
  } else if (season_type == "postseason") {
    start_date <- as.Date(sprintf("%d-04-01", season))
    end_date <- as.Date(sprintf("%d-06-30", season))
    if (season %in% c(2020, 2021)) {
      end_date <- as.Date(sprintf("%d-10-31", season))
    }
  } else {
    start_date <- as.Date(sprintf("%d-04-01", season))
    end_date <- as.Date(sprintf("%d-07-31", season))
  }

  seq.Date(start_date, end_date, by = "day")
}

.espn_nba_empty_games_index <- function() {
  data.frame(
    game_id = character(),
    game_date = as.Date(character()),
    season = integer(),
    season_type = integer(),
    home_team_id = character(),
    away_team_id = character(),
    home_team = character(),
    away_team = character(),
    stringsAsFactors = FALSE
  )
}

.espn_nba_or <- function(a, b) {
  if (is.null(a) || length(a) == 0) b else a
}

.espn_nba_pluck <- function(x, ..., default = NULL) {
  keys <- list(...)
  for (key in keys) {
    if (is.null(x)) return(default)
    if (is.list(x)) {
      if (is.numeric(key)) {
        if (length(x) < key) return(default)
        x <- x[[key]]
      } else {
        if (is.null(x[[key]])) return(default)
        x <- x[[key]]
      }
    } else {
      return(default)
    }
  }
  if (is.null(x)) default else x
}

.espn_nba_as_df <- function(x) {
  as_tibble(x)
}

.espn_nba_bind_rows_fill <- function(rows) {
  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (length(rows) == 0) return(.espn_nba_as_df(data.frame()))
  bind_rows(rows)
}

.espn_nba_parse_game_header <- function(x) {
  header <- .espn_nba_or(x$header, list())
  comp <- .espn_nba_pluck(header, "competitions", 1, default = list())
  competitors <- .espn_nba_or(comp$competitors, list())

  home <- list()
  away <- list()
  if (length(competitors) > 0) {
    for (entry in competitors) {
      side <- .espn_nba_pluck(entry, "homeAway", default = NA_character_)
      if (identical(side, "home")) home <- entry
      if (identical(side, "away")) away <- entry
    }
  }

  game_id <- .espn_nba_pluck(comp, "id", default = .espn_nba_pluck(header, "id", default = NA_character_))
  season <- .espn_nba_pluck(header, "season", "year", default = NA_integer_)
  season_type <- .espn_nba_pluck(header, "season", "type", default = NA_integer_)
  game_date <- .espn_nba_pluck(comp, "date", default = NA_character_)
  status <- .espn_nba_pluck(comp, "status", "type", "description", default = NA_character_)
  neutral_site <- .espn_nba_pluck(comp, "neutralSite", default = NA)
  attendance <- .espn_nba_pluck(comp, "attendance", default = NA_integer_)
  home_team_id <- as.character(.espn_nba_pluck(home, "team", "id", default = NA_character_))
  away_team_id <- as.character(.espn_nba_pluck(away, "team", "id", default = NA_character_))

  record_home <- .espn_nba_record_summary(home)
  record_away <- .espn_nba_record_summary(away)

  .espn_nba_as_df(list(
    league = "nba",
    season = season,
    season_type = season_type,
    game_id = as.character(game_id),
    game_date = game_date,
    home_team_id = home_team_id,
    away_team_id = away_team_id,
    home_team = .espn_nba_pluck(home, "team", "displayName", default = NA_character_),
    away_team = .espn_nba_pluck(away, "team", "displayName", default = NA_character_),
    home_score = .espn_nba_pluck(home, "score", default = NA_character_),
    away_score = .espn_nba_pluck(away, "score", default = NA_character_),
    status = status,
    neutralSite = neutral_site,
    attendance = attendance,
    home_record_total = record_home$total,
    home_record_at_home = record_home$home,
    away_record_total = record_away$total,
    away_record_on_road = record_away$road
  ))
}

.espn_nba_parse_teams <- function(x) {
  header <- .espn_nba_or(x$header, list())
  comp <- .espn_nba_pluck(header, "competitions", 1, default = list())
  competitors <- .espn_nba_or(comp$competitors, list())

  if (length(competitors) == 0) return(.espn_nba_empty_teams())

  rows <- lapply(competitors, function(entry) {
    team <- .espn_nba_or(entry$team, list())
    data.frame(
      league = "nba",
      team_id = as.character(.espn_nba_pluck(team, "id", default = NA_character_)),
      team_abbrev = .espn_nba_pluck(team, "abbreviation", default = NA_character_),
      team_name = .espn_nba_pluck(team, "displayName", default = NA_character_),
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  })

  unique(.espn_nba_bind_rows_fill(rows))
}

.espn_nba_record_summary <- function(competitor) {
  out <- list(total = NA_character_, home = NA_character_, road = NA_character_, vsconf = NA_character_)
  records <- .espn_nba_or(competitor$record, list())
  if (length(records) == 0) return(out)
  for (rec in records) {
    rec_type <- .espn_nba_pluck(rec, "type", default = NA_character_)
    summary <- .espn_nba_pluck(rec, "summary", default = .espn_nba_pluck(rec, "displayValue", default = NA_character_))
    if (is.na(rec_type)) next
    if (rec_type == "total") out$total <- summary
    if (rec_type == "home") out$home <- summary
    if (rec_type == "road") out$road <- summary
    if (rec_type == "vsconf") out$vsconf <- summary
  }
  if (!is.na(out$total)) {
    if (is.na(out$home)) out$home <- "0-0"
    if (is.na(out$road)) out$road <- "0-0"
    if (is.na(out$vsconf)) out$vsconf <- "0-0"
  }
  out
}


.espn_nba_parse_team_box <- function(x) {
  teams <- .espn_nba_or(.espn_nba_pluck(x, "boxscore", "teams", default = list()), list())
  if (length(teams) == 0) return(.espn_nba_empty_team_box())

  game_id <- as.character(.espn_nba_pluck(x, "header", "competitions", 1, "id", default = NA_character_))
  rows <- lapply(teams, function(team) {
    stats <- .espn_nba_or(team$statistics, list())
    stat_names <- vapply(stats, function(s) .espn_nba_pluck(s, "name", default = NA_character_), character(1))
    stat_values <- lapply(stats, function(s) {
      .espn_nba_pluck(s, "displayValue", default = .espn_nba_pluck(s, "value", default = NA))
    })
    if (length(stat_values) == 0 && length(stat_names) > 0) {
      stat_values <- as.list(rep(NA, length(stat_names)))
    }
    if (length(stat_values) == length(stat_names)) {
      names(stat_values) <- stat_names
    }
    stat_values <- .espn_nba_split_compound_stats(stat_values, drop_compound = FALSE)
    drop_cols <- c(
      "leadChanges",
      "leadPercentage",
      "streak",
      "Last Ten Games",
      "avgPointsAgainst",
      "avgPoints",
      "avgRebounds",
      "avgAssists",
      "avgBlocks",
      "avgSteals",
      "avgTeamTurnovers",
      "avgTotalTurnovers"
    )
    stat_values[drop_cols] <- NULL

    base <- list(
      game_id = game_id,
      team_id = as.character(.espn_nba_pluck(team, "team", "id", default = NA_character_)),
      home_away = .espn_nba_pluck(team, "homeAway", default = NA_character_)
    )

    .espn_nba_as_df(c(base, stat_values))
  })

  .espn_nba_bind_rows_fill(rows)
}

.espn_nba_parse_player_box <- function(x) {
  teams <- .espn_nba_or(.espn_nba_pluck(x, "boxscore", "players", default = list()), list())
  if (length(teams) == 0) return(.espn_nba_empty_player_box())

  game_id <- as.character(.espn_nba_pluck(x, "header", "competitions", 1, "id", default = NA_character_))
  rows <- list()
  idx <- 0L

  for (team_block in teams) {
    team_id <- as.character(.espn_nba_pluck(team_block, "team", "id", default = NA_character_))
    categories <- .espn_nba_or(team_block$statistics, list())

    for (cat in categories) {
      keys <- .espn_nba_or(cat$keys, cat$labels)
      athletes <- .espn_nba_or(cat$athletes, list())

      for (ath in athletes) {
        idx <- idx + 1L
        stats <- .espn_nba_or(ath$stats, list())
        stat_values <- stats
        if (length(stat_values) == 0 && length(keys) > 0) {
          stat_values <- as.list(rep(NA, length(keys)))
        }
        if (length(stat_values) == length(keys)) {
          names(stat_values) <- keys
        }
        stat_values <- .espn_nba_split_compound_stats(stat_values, drop_compound = FALSE)

        rows[[idx]] <- .espn_nba_as_df(c(
          list(
            game_id = game_id,
            team_id = team_id,
            player_id = as.character(.espn_nba_pluck(ath, "athlete", "id", default = NA_character_)),
            player_display_name = .espn_nba_pluck(ath, "athlete", "displayName", default = NA_character_),
            starter = .espn_nba_pluck(ath, "starter", default = NA),
            did_not_play = .espn_nba_pluck(ath, "didNotPlay", default = NA),
            position = .espn_nba_pluck(ath, "athlete", "position", "abbreviation", default = NA_character_)
          ),
          stat_values
        ))
      }
    }
  }

  if (length(rows) == 0) return(.espn_nba_empty_player_box())
  .espn_nba_bind_rows_fill(rows)
}

.espn_nba_split_compound_stats <- function(stat_values, drop_compound = FALSE) {
  out <- list()
  for (nm in names(stat_values)) {
    val <- stat_values[[nm]]
    if (is.character(val) && grepl("^\\d+-\\d+$", val)) {
      parts <- strsplit(val, "-", fixed = TRUE)[[1]]
      out[[nm]] <- val
      out[[paste0(nm, "Made")]] <- as.integer(parts[[1]])
      out[[paste0(nm, "Attempted")]] <- as.integer(parts[[2]])
      if (isTRUE(drop_compound)) out[[nm]] <- NULL
    } else {
      out[[nm]] <- val
    }
  }
  out
}

.espn_nba_parse_plays <- function(x) {
  plays <- .espn_nba_or(x$plays, list())
  if (length(plays) == 0) return(.espn_nba_empty_events())

  game_id <- as.character(.espn_nba_pluck(x, "header", "competitions", 1, "id", default = NA_character_))
  season <- .espn_nba_pluck(x, "header", "season", "year", default = NA_integer_)

  rows <- lapply(plays, function(play) {
    coord <- .espn_nba_or(play$coordinate, play$coordinates)
    x_raw <- .espn_nba_pluck(coord, "x", default = NA_real_)
    y_raw <- .espn_nba_pluck(coord, "y", default = NA_real_)
    norm <- source_nba_espn()$normalize_xy(x_raw, y_raw)

    participants <- .espn_nba_or(play$participants, list())
    participant_ids <- NULL
    if (length(participants) > 0) {
      participant_ids <- vapply(participants, function(p) {
        as.character(.espn_nba_pluck(p, "athlete", "id", default = NA_character_))
      }, character(1))
    }

    meta <- list(
      sequence_number = suppressWarnings(as.integer(.espn_nba_pluck(play, "sequenceNumber", default = NA_integer_))),
      period = suppressWarnings(as.integer(.espn_nba_pluck(play, "period", "number", default = NA_integer_))),
      clock = .espn_nba_pluck(play, "clock", "displayValue", default = NA_character_),
      wallclock = .espn_nba_pluck(play, "wallclock", default = NA_character_),
      scoring_play = .espn_nba_pluck(play, "scoringPlay", default = NA),
      shooting_play = .espn_nba_pluck(play, "shootingPlay", default = NA),
      points_attempted = suppressWarnings(as.integer(.espn_nba_pluck(play, "pointsAttempted", default = NA_integer_))),
      score_value = suppressWarnings(as.integer(.espn_nba_pluck(play, "scoreValue", default = NA_integer_))),
      team_id = as.character(.espn_nba_pluck(play, "team", "id", default = NA_character_)),
      participants = participant_ids
    )

    .espn_nba_as_df(list(
      league = "nba",
      season = season,
      game_id = game_id,
      event_id = as.character(.espn_nba_pluck(play, "id", default = .espn_nba_pluck(play, "sequenceNumber", default = NA_character_))),
      event_type = .espn_nba_pluck(play, "type", "text", default = NA_character_),
      x_unit = norm$x_unit,
      y_unit = norm$y_unit,
      label = .espn_nba_pluck(play, "text", default = .espn_nba_pluck(play, "shortDescription", default = NA_character_)),
      meta = as.character(jsonlite::toJSON(meta, auto_unbox = TRUE))
    ))
  })

  .espn_nba_bind_rows_fill(rows)
}

.espn_nba_manifest_from_file <- function(path, x, has_events = NA, has_team_box = NA, has_player_box = NA, has_teams = NA) {
  season <- .espn_nba_pluck(x, "header", "season", "year", default = NA_integer_)
  game_id <- as.character(.espn_nba_pluck(x, "header", "competitions", 1, "id", default = NA_character_))

  .espn_nba_as_df(list(
    league = "nba",
    season = season,
    game_id = game_id,
    downloaded_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    source_url = path,
    status_code = 200L,
    bytes = file.info(path)$size,
    has_events = has_events,
    has_team_box = has_team_box,
    has_player_box = has_player_box,
    has_teams = has_teams
  ))
}

.espn_nba_empty_events <- function() {
  .espn_nba_as_df(data.frame(
    league = character(),
    season = integer(),
    game_id = character(),
    event_id = character(),
    event_type = character(),
    x_unit = numeric(),
    y_unit = numeric(),
    label = character(),
    meta = character(),
    stringsAsFactors = FALSE
  ))
}

.espn_nba_empty_teams <- function() {
  .espn_nba_as_df(data.frame(
    league = character(),
    team_id = character(),
    team_abbrev = character(),
    team_name = character(),
    stringsAsFactors = FALSE
  ))
}

.espn_nba_empty_team_box <- function() {
  .espn_nba_as_df(data.frame(
    game_id = character(),
    team_id = character(),
    home_away = character(),
    stringsAsFactors = FALSE
  ))
}

.espn_nba_empty_player_box <- function() {
  .espn_nba_as_df(data.frame(
    game_id = character(),
    team_id = character(),
    player_id = character(),
    player_display_name = character(),
    starter = logical(),
    did_not_play = logical(),
    position = character(),
    stringsAsFactors = FALSE
  ))
}
