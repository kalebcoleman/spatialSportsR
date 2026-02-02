#' NBA Stats source helpers (internal)
#'
#' @description
#' NBA Stats source implementation returned by `get_source("nba", source = "nba_stats")`.
#'
#' @keywords internal
#' @importFrom dplyr as_tibble bind_rows
#' @importFrom jsonlite fromJSON

source_nba_stats <- function() {
  list(
    league = "nba",
    source = "nba_stats",
    collect_game_index = function(season, raw_dir = "data/raw", force = FALSE, season_type = "Regular Season", proxy = NULL, ...) {
      nba_stats_collect_game_index(
        season = season,
        raw_dir = raw_dir,
        force = force,
        season_type = season_type,
        proxy = proxy,
        ...
      )
    },
    collect_game_raw = function(game_id, season, raw_dir = "data/raw", force = FALSE, proxy = NULL, verbose = FALSE, ...) {
      nba_stats_collect_game_raw(
        game_id = game_id,
        season = season,
        raw_dir = raw_dir,
        force = force,
        proxy = proxy,
        verbose = verbose,
        ...
      )
    },
    parse_raw_dir = function(season, raw_dir = "data/raw", season_type = "Regular Season", ...) {
      nba_stats_parse_raw_dir(
        season = season,
        raw_dir = raw_dir,
        season_type = season_type,
        ...
      )
    }
  )
}

#' Collect an NBA Stats season (index + per-game endpoints)
#'
#' @param season Season identifier (e.g., 2024 or "2024-25").
#' @param raw_dir Base directory for raw data.
#' @param force Force re-download even if cached.
#' @param progress Show progress while collecting.
#' @param quiet Suppress progress messages.
#' @param season_type Season type string (e.g., "Regular Season").
#' @param proxy Optional proxy string or list for NBA Stats requests.
#' @return A list with games, raw_paths, league_dir, and index_path.
#'
#' @export
nba_stats_collect_season <- function(season,
                                    raw_dir = "data/raw",
                                    force = FALSE,
                                    progress = TRUE,
                                    quiet = FALSE,
                                    season_type = "Regular Season",
                                    proxy = NULL,
                                    ...) {
  season_str <- .nba_stats_season_string(season)
  out <- collect_raw(
    league = "nba",
    source = "nba_stats",
    season = season,
    raw_dir = raw_dir,
    force = force,
    progress = progress,
    quiet = quiet,
    season_type = season_type,
    proxy = proxy,
    ...
  )

  shotchart_path <- tryCatch(
    collect_nba_shotchart(
      season = season_str,
      season_type = season_type,
      raw_dir = out$league_dir,
      force = force,
      proxy = proxy,
      ...
    ),
    error = function(e) NULL
  )

  out$shotchart_path <- shotchart_path
  out
}

nba_stats_collect_game_index <- function(season,
                                         raw_dir = "data/raw",
                                         force = FALSE,
                                         season_type = "Regular Season",
                                         proxy = NULL,
                                         save_index = TRUE,
                                         ...) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Install jsonlite to parse NBA Stats index", call. = FALSE)
  }

  season_str <- .nba_stats_season_string(season)
  dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
  index_path <- file.path(raw_dir, paste0("index_", season_str, ".json"))

  query <- list(
    LeagueID = "00",
    Season = season_str,
    SeasonType = season_type,
    PlayerOrTeam = "T",
    DateFrom = "",
    DateTo = ""
  )

  res <- nba_api_get_with_backoff(
    url = nba_endpoint("leaguegamefinder"),
    query = query,
    path = index_path,
    force = force,
    proxy = proxy
  )

  if (!is.null(res$error)) {
    warning("NBA Stats index fetch failed: ", res$error)
  }

  raw <- res$raw
  if (is.null(raw)) {
    return(.nba_stats_empty_games_index())
  }

  games <- nba_stats_parse_game_index(raw, season = season_str)
  shotchart_path <- tryCatch(
    collect_nba_shotchart(
      season = season_str,
      season_type = season_type,
      raw_dir = raw_dir,
      force = force,
      proxy = proxy,
      ...
    ),
    error = function(e) NULL
  )
  if (isTRUE(save_index)) {
    return(list(games = games, index_path = index_path, shotchart_path = shotchart_path))
  }

  games
}

nba_stats_collect_game_raw <- function(game_id,
                                       season,
                                       raw_dir = "data/raw",
                                       force = FALSE,
                                       season_type = "Regular Season",
                                       rate_sleep = c(0.5, 1.2),
                                       max_retries = 2L,
                                       proxy = NULL,
                                       verbose = FALSE,
                                       ...) {
  season_str <- .nba_stats_season_string(season)
  dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
  game_id <- as.character(game_id)
  game_date <- list(...)$game_date
  date_key <- .nba_stats_date_key(game_date)
  dir_name <- if (!is.null(date_key) && nzchar(date_key)) {
    paste0(date_key, "_", game_id)
  } else {
    game_id
  }
  game_dir <- file.path(raw_dir, dir_name)
  dir.create(game_dir, recursive = TRUE, showWarnings = FALSE)

  endpoints <- .nba_stats_game_endpoint_specs(
    season_str = season_str,
    game_id = game_id,
    season_type = season_type
  )
  first_path <- NULL

  sleep_range <- suppressWarnings(as.numeric(rate_sleep))
  if (length(sleep_range) < 2 || any(is.na(sleep_range))) {
    sleep_range <- c(0.5, 1.2)
  }

  if (!isTRUE(force)) {
    endpoint_paths <- vapply(endpoints, function(spec) file.path(game_dir, spec$file), character(1))
    if (length(endpoint_paths) > 0 && all(file.exists(endpoint_paths))) {
      return(endpoint_paths[[1]])
    }
  }

  for (idx in seq_along(endpoints)) {
    spec <- endpoints[[idx]]
    path <- file.path(game_dir, spec$file)
    if (!isTRUE(force) && file.exists(path)) {
      if (is.null(first_path)) first_path <- path
      next
    }
    res <- .nba_stats_fetch_with_retry(
      url = nba_endpoint(spec$endpoint),
      query = spec$query,
      path = path,
      force = force,
      proxy = proxy,
      verbose = verbose,
      label = spec$label,
      max_retries = max_retries
    )

    if (is.null(first_path)) first_path <- path

    if (!is.null(res$error) && !file.exists(path)) {
      jsonlite::write_json(
        list(
          endpoint = spec$endpoint,
          label = spec$label,
          game_id = game_id,
          season = season_str,
          status_code = res$status_code,
          error = res$error,
          query = spec$query
        ),
        path,
        auto_unbox = TRUE,
        pretty = TRUE,
        null = "null"
      )
      .log_collect_failure(
        league_dir = raw_dir,
        league = "nba",
        source = "nba_stats",
        season = season_str,
        game_id = game_id,
        error_message = paste0(spec$label, " ", res$status_code, ": ", res$error)
      )
    }

    if (idx < length(endpoints)) {
      Sys.sleep(stats::runif(1, min = sleep_range[1], max = sleep_range[2]))
    }
  }

  first_path
}

.nba_stats_fetch_with_retry <- function(url,
                                        query,
                                        path,
                                        force,
                                        proxy,
                                        verbose,
                                        label,
                                        max_retries = 2L) {
  attempts <- max(1L, as.integer(max_retries) + 1L)
  last_res <- NULL

  for (i in seq_len(attempts)) {
    res <- nba_api_get_with_backoff(
      url = url,
      query = query,
      path = path,
      force = force,
      proxy = proxy,
      times = 1,
      backoff_base = 1,
      backoff_cap = 8,
      jitter = 0.25,
      verbose = verbose,
      label = label
    )
    last_res <- res

    if (is.null(res$error)) return(res)

    is_rate <- !is.null(res$status_code) && res$status_code %in% c(403, 429)
    is_timeout <- !is.null(res$error) && grepl("timeout", res$error, ignore.case = TRUE)
    if (!(is_rate || is_timeout) || i == attempts) {
      return(res)
    }

    backoff <- min(8, 2 ^ (i - 1)) + stats::runif(1, 0, 0.5)
    Sys.sleep(backoff)
  }

  last_res
}

.nba_stats_game_endpoint_specs <- function(season_str, game_id, season_type = "Regular Season") {
  base_boxscore <- list(
    GameID = game_id,
    StartPeriod = 1,
    EndPeriod = 10,
    StartRange = 0,
    EndRange = 0,
    RangeType = 0
  )

  list(
    list(
      endpoint = "boxscoretraditionalv3",
      label = "boxscore_traditional",
      file = "boxscore_traditional.json",
      query = base_boxscore
    ),
    list(
      endpoint = "boxscoreadvancedv3",
      label = "boxscore_advanced",
      file = "boxscore_advanced.json",
      query = base_boxscore
    ),
    list(
      endpoint = "boxscoreusagev3",
      label = "boxscore_usage",
      file = "boxscore_usage.json",
      query = base_boxscore
    ),
    list(
      endpoint = "boxscorefourfactorsv3",
      label = "boxscore_fourfactors",
      file = "boxscore_fourfactors.json",
      query = base_boxscore
    ),
    list(
      endpoint = "playbyplayv3",
      label = "playbyplay",
      file = "playbyplay.json",
      query = list(
        GameID = game_id,
        StartPeriod = 0,
        EndPeriod = 10
      )
    )
  )
}

.nba_stats_date_key <- function(value) {
  if (is.null(value) || (length(value) == 1 && is.na(value))) return(NULL)
  if (inherits(value, "Date")) return(format(value, "%Y%m%d"))
  date_str <- as.character(value)
  date_str <- sub("Z$", "", date_str)
  date_str <- substr(date_str, 1, 10)
  date_str <- gsub("-", "", date_str)
  if (!nzchar(date_str)) return(NULL)
  date_str
}

#' Parse NBA Stats boxscoretraditionalv2 JSON
#'
#' @param json Parsed JSON list or path to raw JSON.
#' @param season Season identifier string.
#' @param game_id Optional game id to override.
#' @return A named list with team_box and player_box.
#'
#' @export
nba_stats_parse_boxscore <- function(json, season = NA_character_, game_id = NULL) {
  raw <- .nba_stats_read_json(json)
  v3 <- .nba_stats_v3_boxscore_tables(raw, "boxScoreTraditional")
  if (!is.null(v3)) {
    team_box <- v3$team
    player_box <- v3$player
  } else {
    team_rs <- .nba_stats_find_result_set(raw, "TeamStats")
    player_rs <- .nba_stats_find_result_set(raw, "PlayerStats")
    team_box <- .nba_stats_result_set_to_df(team_rs)
    player_box <- .nba_stats_result_set_to_df(player_rs)
  }

  if (!is.null(game_id) && nrow(team_box) > 0) {
    team_box$GAME_ID <- game_id
  }
  if (!is.null(game_id) && nrow(player_box) > 0) {
    player_box$GAME_ID <- game_id
  }

  team_box <- .nba_stats_standardize_box(team_box, season = season)
  player_box <- .nba_stats_standardize_box(player_box, season = season)

  list(team_box = team_box, player_box = player_box)
}

#' Parse NBA Stats boxscoreadvancedv2 JSON
#'
#' @param json Parsed JSON list or path to raw JSON.
#' @param season Season identifier string.
#' @param game_id Optional game id to override.
#' @return A named list with team and player tables.
#'
#' @export
nba_stats_parse_boxscore_advanced <- function(json, season = NA_character_, game_id = NULL) {
  .nba_stats_parse_boxscore_variant(json, season = season, game_id = game_id)
}

#' Parse NBA Stats boxscorefourfactorsv2 JSON
#'
#' @param json Parsed JSON list or path to raw JSON.
#' @param season Season identifier string.
#' @param game_id Optional game id to override.
#' @return A named list with team and player tables.
#'
#' @export
nba_stats_parse_boxscore_fourfactors <- function(json, season = NA_character_, game_id = NULL) {
  .nba_stats_parse_boxscore_variant(json, season = season, game_id = game_id)
}

#' Parse NBA Stats boxscoreusagev2 JSON
#'
#' @param json Parsed JSON list or path to raw JSON.
#' @param season Season identifier string.
#' @param game_id Optional game id to override.
#' @return A named list with team and player tables.
#'
#' @export
nba_stats_parse_boxscore_usage <- function(json, season = NA_character_, game_id = NULL) {
  .nba_stats_parse_boxscore_variant(json, season = season, game_id = game_id)
}

#' Parse NBA Stats playbyplayv2 JSON
#'
#' @param json Parsed JSON list or path to raw JSON.
#' @param season Season identifier string.
#' @param game_id Optional game id to override.
#' @return A data frame of play-by-play rows.
#'
#' @export
nba_stats_parse_pbp <- function(json, season = NA_character_, game_id = NULL) {
  raw <- .nba_stats_read_json(json)
  df <- .nba_stats_v3_pbp_to_df(raw)
  if (is.null(df)) {
    rs <- .nba_stats_find_result_set(raw, "PlayByPlay")
    df <- .nba_stats_result_set_to_df(rs)
  }

  if (!is.null(game_id) && nrow(df) > 0) {
    df$GAME_ID <- game_id
  }

  df <- .nba_stats_standardize_pbp(df, season = season)
  df
}

#' Parse NBA Stats shotchartdetail JSON
#'
#' @param json Parsed JSON list or path to raw JSON.
#' @param season Season identifier string.
#' @param game_id Optional game id to override.
#' @return A data frame of shots.
#'
#' @export
nba_stats_parse_shots <- function(json, season = NA_character_, game_id = NULL) {
  raw <- .nba_stats_read_json(json)
  rs <- .nba_stats_find_result_set(raw, "Shot_Chart_Detail")
  df <- .nba_stats_result_set_to_df(rs)

  if (!is.null(game_id) && nrow(df) > 0) {
    df$GAME_ID <- game_id
  }

  df <- .nba_stats_standardize_shots(df, season = season)
  df
}

#' Parse NBA Stats raw directory into tables
#'
#' @param season Season identifier.
#' @param raw_dir Directory containing raw NBA Stats files.
#' @return A named list of tables including games, box score tables, pbp, and shots.
#'
#' @export
nba_stats_parse_raw_dir <- function(season, raw_dir, season_type = "Regular Season") {
  if (!dir.exists(raw_dir)) {
    stop("Raw directory not found: ", raw_dir, call. = FALSE)
  }

  season_str <- .nba_stats_season_string(season)
  season_shotchart_path <- file.path(raw_dir, "nba_shotchart", paste0(season_str, ".json"))
  if (tolower(as.character(season_type)) %in% c("playoffs", "postseason")) {
    season_shotchart_path <- file.path(raw_dir, "nba_shotchart", paste0(season_str, "_playoffs.json"))
  }

  game_dirs <- list.dirs(raw_dir, full.names = TRUE, recursive = FALSE)
  game_dirs <- setdiff(game_dirs, raw_dir)
  game_dirs <- game_dirs[basename(game_dirs) != ""]

  use_flat <- length(game_dirs) == 0

  team_box_trad_list <- list()
  player_box_trad_list <- list()
  team_box_adv_list <- list()
  player_box_adv_list <- list()
  team_box_ff_list <- list()
  player_box_ff_list <- list()
  team_box_usage_list <- list()
  player_box_usage_list <- list()
  pbp_list <- list()
  shots_list <- list()

  if (isTRUE(use_flat)) {
    box_files <- list.files(raw_dir, pattern = "^boxscoretraditionalv2_.*\\.json$", full.names = TRUE)
    pbp_files <- list.files(raw_dir, pattern = "^playbyplayv2_.*\\.json$", full.names = TRUE)
    shot_files <- list.files(raw_dir, pattern = "^shotchartdetail_.*\\.json$", full.names = TRUE)

    for (path in box_files) {
      game_id <- .nba_stats_game_id_from_path(path)
      parsed <- nba_stats_parse_boxscore(path, season = season_str, game_id = game_id)
      team_box_trad_list[[length(team_box_trad_list) + 1L]] <- parsed$team_box
      player_box_trad_list[[length(player_box_trad_list) + 1L]] <- parsed$player_box
    }

    for (path in pbp_files) {
      game_id <- .nba_stats_game_id_from_path(path)
      pbp_list[[length(pbp_list) + 1L]] <- nba_stats_parse_pbp(path, season = season_str, game_id = game_id)
    }

    for (path in shot_files) {
      game_id <- .nba_stats_game_id_from_path(path)
      shots_list[[length(shots_list) + 1L]] <- nba_stats_parse_shots(path, season = season_str, game_id = game_id)
    }
  } else {
    for (dir_path in game_dirs) {
      game_id <- .nba_stats_game_id_from_path(dir_path)

      trad_path <- file.path(dir_path, "boxscore_traditional.json")
      adv_path <- file.path(dir_path, "boxscore_advanced.json")
      ff_path <- file.path(dir_path, "boxscore_fourfactors.json")
      usage_path <- file.path(dir_path, "boxscore_usage.json")
      pbp_path <- file.path(dir_path, "playbyplay.json")
      shots_path <- file.path(dir_path, "shotchartdetail.json")

      if (file.exists(trad_path)) {
        parsed <- nba_stats_parse_boxscore(trad_path, season = season_str, game_id = game_id)
        team_box_trad_list[[length(team_box_trad_list) + 1L]] <- parsed$team_box
        player_box_trad_list[[length(player_box_trad_list) + 1L]] <- parsed$player_box
      }

      if (file.exists(adv_path)) {
        parsed <- nba_stats_parse_boxscore_advanced(adv_path, season = season_str, game_id = game_id)
        team_box_adv_list[[length(team_box_adv_list) + 1L]] <- parsed$team
        player_box_adv_list[[length(player_box_adv_list) + 1L]] <- parsed$player
      }

      if (file.exists(ff_path)) {
        parsed <- nba_stats_parse_boxscore_fourfactors(ff_path, season = season_str, game_id = game_id)
        team_box_ff_list[[length(team_box_ff_list) + 1L]] <- parsed$team
        player_box_ff_list[[length(player_box_ff_list) + 1L]] <- parsed$player
      }

      if (file.exists(usage_path)) {
        parsed <- nba_stats_parse_boxscore_usage(usage_path, season = season_str, game_id = game_id)
        team_box_usage_list[[length(team_box_usage_list) + 1L]] <- parsed$team
        player_box_usage_list[[length(player_box_usage_list) + 1L]] <- parsed$player
      }

      if (file.exists(pbp_path)) {
        pbp_list[[length(pbp_list) + 1L]] <- nba_stats_parse_pbp(pbp_path, season = season_str, game_id = game_id)
      }

      if (file.exists(shots_path)) {
        shots_list[[length(shots_list) + 1L]] <- nba_stats_parse_shots(shots_path, season = season_str, game_id = game_id)
      }
    }
  }

  team_box_traditional <- .nba_stats_bind_or_empty(team_box_trad_list, .nba_stats_empty_team_box())
  player_box_traditional <- .nba_stats_bind_or_empty(player_box_trad_list, .nba_stats_empty_player_box())
  team_box_advanced <- .nba_stats_bind_or_empty(team_box_adv_list, .nba_stats_empty_team_box())
  player_box_advanced <- .nba_stats_bind_or_empty(player_box_adv_list, .nba_stats_empty_player_box())
  team_box_fourfactors <- .nba_stats_bind_or_empty(team_box_ff_list, .nba_stats_empty_team_box())
  player_box_fourfactors <- .nba_stats_bind_or_empty(player_box_ff_list, .nba_stats_empty_player_box())
  team_box_usage <- .nba_stats_bind_or_empty(team_box_usage_list, .nba_stats_empty_team_box())
  player_box_usage <- .nba_stats_bind_or_empty(player_box_usage_list, .nba_stats_empty_player_box())

  dedupe_box <- function(df, key_cols) {
    if (is.null(df) || nrow(df) == 0) return(df)
    if (!all(key_cols %in% names(df))) return(df)
    df[!duplicated(df[, key_cols]), , drop = FALSE]
  }

  team_box_traditional <- dedupe_box(team_box_traditional, c("league", "source", "season", "game_id", "team_id"))
  player_box_traditional <- dedupe_box(player_box_traditional, c("league", "source", "season", "game_id", "player_id"))
  team_box_advanced <- dedupe_box(team_box_advanced, c("league", "source", "season", "game_id", "team_id"))
  player_box_advanced <- dedupe_box(player_box_advanced, c("league", "source", "season", "game_id", "player_id"))
  team_box_fourfactors <- dedupe_box(team_box_fourfactors, c("league", "source", "season", "game_id", "team_id"))
  player_box_fourfactors <- dedupe_box(player_box_fourfactors, c("league", "source", "season", "game_id", "player_id"))
  player_box_usage <- dedupe_box(player_box_usage, c("league", "source", "season", "game_id", "player_id"))
  pbp <- .nba_stats_bind_or_empty(pbp_list, .nba_stats_empty_pbp())
  if (nrow(pbp) > 0) {
    key_cols <- c("league", "source", "season", "game_id", "event_num")
    if (all(key_cols %in% names(pbp))) {
      pbp <- pbp[!duplicated(pbp[, key_cols]), , drop = FALSE]
    }
  }
  shots <- .nba_stats_bind_or_empty(shots_list, .nba_stats_empty_shots())
  if (nrow(shots) == 0 && file.exists(season_shotchart_path)) {
    shots <- parse_nba_shotchart(
      raw_json_or_path = season_shotchart_path,
      season = season_str,
      season_type = season_type
    )
    shots <- .nba_stats_standardize_shots(shots, season = season_str)
  }

  manifest <- .nba_stats_build_manifest(
    game_dirs = if (isTRUE(use_flat)) character() else game_dirs,
    season = season_str,
    shotchart_path = season_shotchart_path
  )

  games <- .nba_stats_parse_games_from_index(raw_dir, season_str)
  if (nrow(games) == 0) {
    game_ids <- unique(c(
      team_box_traditional$game_id,
      player_box_traditional$game_id,
      team_box_advanced$game_id,
      player_box_advanced$game_id,
      team_box_fourfactors$game_id,
      player_box_fourfactors$game_id,
      team_box_usage$game_id,
      player_box_usage$game_id,
      pbp$game_id,
      shots$game_id
    ))
    games <- .nba_stats_empty_games()
    if (length(game_ids) > 0) {
      games <- data.frame(
        league = "nba",
        source = "nba_stats",
        season = season_str,
        game_id = game_ids,
        stringsAsFactors = FALSE
      )
    }
  }

  if (nrow(games) > 0 && nrow(team_box_traditional) > 0) {
    score_col <- NULL
    if ("PTS" %in% names(team_box_traditional)) {
      score_col <- "PTS"
    } else if ("points" %in% names(team_box_traditional)) {
      score_col <- "points"
    }
    if (!is.null(score_col) && all(c("game_id", "team_id") %in% names(team_box_traditional))) {
      tb <- team_box_traditional
      tb$key <- paste(tb$game_id, tb$team_id)
      score_vals <- suppressWarnings(as.numeric(tb[[score_col]]))
      games$home_score <- score_vals[match(paste(games$game_id, games$home_team_id), tb$key)]
      games$away_score <- score_vals[match(paste(games$game_id, games$away_team_id), tb$key)]
      games$home_margin <- games$home_score - games$away_score
      games$away_margin <- games$away_score - games$home_score
      games$winner <- ifelse(
        is.na(games$home_score) | is.na(games$away_score),
        NA_character_,
        ifelse(games$home_score > games$away_score, "home",
          ifelse(games$away_score > games$home_score, "away", "tie")
        )
      )
    }
  }

  tables <- list(
    games = games,
    team_box_traditional = team_box_traditional,
    player_box_traditional = player_box_traditional,
    team_box_advanced = team_box_advanced,
    player_box_advanced = player_box_advanced,
    team_box_fourfactors = team_box_fourfactors,
    player_box_fourfactors = player_box_fourfactors,
    player_box_usage = player_box_usage,
    pbp = pbp,
    shots = shots,
    manifest = manifest
  )
  stype <- tolower(as.character(season_type))
  if (stype == "regular season") stype <- "regular"
  if (stype == "playoffs") stype <- "playoffs"
  for (nm in names(tables)) {
    tbl <- tables[[nm]]
    if (!is.null(tbl) && is.data.frame(tbl)) {
      tables[[nm]]$season_type <- rep(stype, nrow(tbl))
    }
  }
  tables
}

nba_stats_parse_game_index <- function(raw, season) {
  rs <- .nba_stats_find_result_set(raw, "LeagueGameFinder")
  df <- .nba_stats_result_set_to_df(rs)
  if (nrow(df) == 0) return(.nba_stats_empty_games())

  if ("GAME_DATE" %in% names(df)) {
    df$GAME_DATE <- .nba_safe_date(df$GAME_DATE)
  }

  games <- .nba_stats_build_games_from_finder(df, season = season)
  games
}

# =========================
# NBA Stats shot chart (season-level)
# =========================

#' NBA Stats shot chart collection and parsing
#'
#' @description
#' Helpers to collect season-level shot chart data from the NBA Stats API and
#' parse it into a table for the pipeline.
#'
#' @param season Season identifier (e.g., 2024 or "2024-25").
#' @param season_type Season type string for NBA Stats (e.g., "Regular Season").
#' @param raw_dir Base directory for raw data.
#' @param force Force re-download even if cached.
#' @param chunk Chunking strategy for date ranges ("month" or "none").
#' @param proxy Optional proxy string or list for NBA Stats requests.
#' @return For collect_nba_shotchart(), the file path written.
#'
#' @export
collect_nba_shotchart <- function(season,
                                  season_type = "Regular Season",
                                  raw_dir = "data/raw",
                                  force = FALSE,
                                  chunk = "month",
                                  proxy = NULL) {
  season <- .nba_stats_season_string(season)
  dir_path <- file.path(raw_dir, "nba_shotchart")
  if (!dir.exists(dir_path)) dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)

  is_playoffs <- tolower(as.character(season_type)) %in% c("playoffs", "postseason")
  if (is_playoffs) chunk <- "none"

  suffix <- if (is_playoffs) "_playoffs" else ""
  out_path <- file.path(dir_path, paste0(season, suffix, ".json"))
  if (!isTRUE(force) && file.exists(out_path)) {
    return(out_path)
  }

  url <- "https://stats.nba.com/stats/shotchartdetail"
  base_query <- list(
    LeagueID = "00",
    TeamID = 0,
    PlayerID = 0,
    ContextMeasure = "FGA",
    Season = season,
    SeasonType = season_type,
    SeasonTypeAllStar = season_type,
    PerMode = "Totals",
    LastNGames = 0,
    OpponentTeamID = 0,
    VsTeamID = 0,
    GroupQuantity = 5,
    PORound = 0,
    ShotDistanceRange = "",
    Conference = "",
    Division = "",
    PointDiff = "",
    GameID = "",
    GameSegment = "",
    Month = 0,
    Location = "",
    Outcome = "",
    Period = 0,
    Position = "",
    SeasonSegment = "",
    VsConference = "",
    VsDivision = "",
    PlayerPosition = "",
    StarterBench = "",
    CourtType = "",
    ClutchTime = "",
    AheadBehind = "",
    EndRange = 0,
    StartRange = 0,
    RangeType = 0,
    RookieYear = "",
    AllStarSeason = "",
    ShotClockRange = ""
  )

  ranges <- .nba_shotchart_date_ranges(season, chunk)
  combined <- NULL
  if (length(ranges) == 0) ranges <- list(list(date_from = NULL, date_to = NULL))

  for (rng in ranges) {
    query <- base_query
    if (!is.null(rng$date_from) && !is.null(rng$date_to)) {
      query$DateFrom <- format(rng$date_from, "%m/%d/%Y")
      query$DateTo <- format(rng$date_to, "%m/%d/%Y")
    }

    raw <- tryCatch(
      request_with_proxy(url = url, params = query, proxy = proxy),
      error = function(e) NULL
    )

    if (is.null(raw) || is.null(raw$resultSets)) next

    rs <- .nba_stats_find_result_set(raw, "Shot_Chart_Detail")
    if (!is.null(rs) && is.null(rs$rowSet)) next
    if (!is.null(rs) && length(rs$rowSet) == 0) next

    combined <- .nba_shotchart_merge_raw(combined, raw)
  }

  if (is.null(combined)) {
    combined <- .nba_shotchart_empty_raw()
    message("NBA shot chart: empty combined result for ", season, " ", season_type)
  }

  jsonlite::write_json(combined, out_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
  out_path
}

#' Parse NBA shot chart raw JSON into a table
#'
#' @param raw_json_or_path Parsed JSON list or path to raw JSON.
#' @param season Season identifier (e.g., 2024 or "2024-25").
#' @param season_type Season type string for NBA Stats.
#' @return A tibble of shot chart rows.
#'
#' @export
parse_nba_shotchart <- function(raw_json_or_path, season, season_type = "Regular Season") {
  season <- .nba_stats_season_string(season)
  season_type <- as.character(season_type)

  raw <- NULL
  if (is.character(raw_json_or_path) && length(raw_json_or_path) == 1) {
    if (!file.exists(raw_json_or_path)) {
      return(.nba_shotchart_empty_table(season, season_type))
    }
    raw <- tryCatch(
      jsonlite::fromJSON(raw_json_or_path, simplifyVector = FALSE),
      error = function(e) NULL
    )
  } else {
    raw <- raw_json_or_path
  }

  if (is.null(raw)) {
    return(.nba_shotchart_empty_table(season, season_type))
  }

  rs <- .nba_stats_find_result_set(raw, "Shot_Chart_Detail")
  if (is.null(rs) || is.null(rs$headers)) {
    return(.nba_shotchart_empty_table(season, season_type))
  }

  headers <- as.character(rs$headers)
  row_set <- rs$rowSet

  if (is.null(row_set) || length(row_set) == 0) {
    return(.nba_shotchart_empty_table(season, season_type, headers = headers))
  }

  mat <- tryCatch(do.call(rbind, row_set), error = function(e) NULL)
  if (is.null(mat)) {
    return(.nba_shotchart_empty_table(season, season_type, headers = headers))
  }

  df <- as.data.frame(mat, stringsAsFactors = FALSE)
  if (length(headers) == ncol(df)) {
    names(df) <- headers
  }

  df <- .nba_shotchart_coerce_types(df)
  df$season <- season
  df$season_type <- season_type

  dplyr::as_tibble(df)
}

# ----------------------
# NBA Stats helpers
# ----------------------

.nba_stats_season_string <- function(season) {
  season <- as.character(season)
  if (grepl("^\\d{4}-\\d{2}$", season)) return(season)
  if (grepl("^\\d{4}-\\d{4}$", season)) {
    parts <- strsplit(season, "-")[[1]]
    return(sprintf("%s-%02d", parts[1], as.integer(substr(parts[2], 3, 4))))
  }
  if (grepl("^\\d{4}$", season)) {
    end_year <- as.integer(season)
    start_year <- end_year - 1L
    return(sprintf("%d-%02d", start_year, end_year %% 100))
  }
  season
}

.nba_stats_season_years <- function(season) {
  season <- .nba_stats_season_string(season)
  if (!grepl("^\\d{4}-\\d{2}$", season)) return(c(NA_integer_, NA_integer_))
  start_year <- as.integer(substr(season, 1, 4))
  end_year <- start_year + 1L
  c(start_year, end_year)
}

.nba_stats_read_json <- function(json) {
  if (is.character(json) && length(json) == 1) {
    if (!file.exists(json)) return(NULL)
    return(jsonlite::fromJSON(json, simplifyVector = FALSE))
  }
  json
}

.nba_stats_find_result_set <- function(raw, target) {
  if (!is.null(raw$resultSets)) {
    idx <- .nba_result_set_index(raw$resultSets, target)
    if (!is.na(idx)) return(raw$resultSets[[idx]])
  }
  if (!is.null(raw$resultSet)) {
    rs <- raw$resultSet
    if (!is.null(rs$name) && .nba_result_set_match(rs$name, target)) return(rs)
  }
  if (!is.null(raw$resultSets) && length(raw$resultSets) >= 1) {
    return(raw$resultSets[[1]])
  }
  NULL
}

.nba_stats_result_set_to_df <- function(result_set) {
  if (is.null(result_set) || is.null(result_set$headers)) return(data.frame())
  headers <- as.character(result_set$headers)
  row_set <- result_set$rowSet
  if (is.null(row_set) || length(row_set) == 0) {
    df <- as.data.frame(matrix(nrow = 0, ncol = length(headers)), stringsAsFactors = FALSE)
    names(df) <- headers
    return(df)
  }
  mat <- tryCatch(do.call(rbind, row_set), error = function(e) NULL)
  if (is.null(mat)) return(data.frame())
  df <- as.data.frame(mat, stringsAsFactors = FALSE, check.names = FALSE)
  if (length(headers) == ncol(df)) {
    names(df) <- headers
  }
  df
}

.nba_stats_v3_boxscore_key <- function(raw) {
  keys <- c("boxScoreTraditional", "boxScoreAdvanced", "boxScoreFourFactors", "boxScoreUsage")
  for (key in keys) {
    if (!is.null(raw[[key]])) return(key)
  }
  NULL
}

.nba_stats_bind_rows_list <- function(rows) {
  if (length(rows) == 0) return(data.frame())
  dplyr::bind_rows(rows)
}

.nba_stats_v3_boxscore_tables <- function(raw, key) {
  if (is.null(raw) || is.null(raw[[key]])) return(NULL)
  bs <- raw[[key]]
  game_id <- bs$gameId
  teams <- list(homeTeam = bs$homeTeam, awayTeam = bs$awayTeam)
  team_rows <- list()
  player_rows <- list()

  for (side in names(teams)) {
    team <- teams[[side]]
    if (is.null(team)) next
    home_away <- if (identical(side, "homeTeam")) "home" else "away"
    team_stats <- team$statistics
    if (is.null(team_stats)) team_stats <- list()

    team_rows[[length(team_rows) + 1L]] <- c(
      list(
        gameId = game_id,
        teamId = team$teamId,
        teamCity = team$teamCity,
        teamName = team$teamName,
        teamTricode = team$teamTricode,
        teamSlug = team$teamSlug,
        homeAway = home_away
      ),
      team_stats
    )

    players <- team$players
    if (is.null(players)) next
    for (player in players) {
      player_stats <- player$statistics
      if (is.null(player_stats)) player_stats <- list()
      player_rows[[length(player_rows) + 1L]] <- c(
        list(
          gameId = game_id,
          teamId = team$teamId,
          teamTricode = team$teamTricode,
          homeAway = home_away,
          personId = player$personId,
          firstName = player$firstName,
          familyName = player$familyName,
          nameI = player$nameI,
          playerSlug = player$playerSlug,
          position = player$position,
          comment = player$comment,
          jerseyNum = player$jerseyNum
        ),
        player_stats
      )
    }
  }

  list(
    team = .nba_stats_bind_rows_list(team_rows),
    player = .nba_stats_bind_rows_list(player_rows)
  )
}

.nba_stats_v3_pbp_to_df <- function(raw) {
  if (is.null(raw) || is.null(raw$game) || is.null(raw$game$actions)) return(NULL)
  actions <- raw$game$actions
  if (length(actions) == 0) return(data.frame())
  df <- dplyr::bind_rows(actions)
  if (!"gameId" %in% names(df) && !is.null(raw$game$gameId)) {
    df$gameId <- raw$game$gameId
  }
  df
}

.nba_stats_parse_boxscore_variant <- function(json, season, game_id = NULL) {
  raw <- .nba_stats_read_json(json)
  key <- .nba_stats_v3_boxscore_key(raw)
  v3 <- if (!is.null(key)) .nba_stats_v3_boxscore_tables(raw, key) else NULL
  if (!is.null(v3)) {
    team_df <- v3$team
    player_df <- v3$player
  } else {
    team_rs <- .nba_stats_find_result_set(raw, "TeamStats")
    player_rs <- .nba_stats_find_result_set(raw, "PlayerStats")
    team_df <- .nba_stats_result_set_to_df(team_rs)
    player_df <- .nba_stats_result_set_to_df(player_rs)
  }

  if (!is.null(game_id) && nrow(team_df) > 0) {
    team_df$GAME_ID <- game_id
  }
  if (!is.null(game_id) && nrow(player_df) > 0) {
    player_df$GAME_ID <- game_id
  }

  team_df <- .nba_stats_standardize_box(team_df, season = season)
  player_df <- .nba_stats_standardize_box(player_df, season = season)

  list(team = team_df, player = player_df)
}

.nba_stats_standardize_box <- function(df, season) {
  df <- as.data.frame(df, stringsAsFactors = FALSE, check.names = FALSE)
  df$league <- rep("nba", nrow(df))
  df$source <- rep("nba_stats", nrow(df))
  df$season <- rep(season, nrow(df))

  if ("GAME_ID" %in% names(df)) {
    df$game_id <- as.character(df$GAME_ID)
  } else if ("gameId" %in% names(df)) {
    df$game_id <- as.character(df$gameId)
  }

  if ("TEAM_ID" %in% names(df)) {
    df$team_id <- as.character(df$TEAM_ID)
  } else if ("teamId" %in% names(df)) {
    df$team_id <- as.character(df$teamId)
  }

  if ("PLAYER_ID" %in% names(df)) {
    df$player_id <- as.character(df$PLAYER_ID)
  } else if ("personId" %in% names(df)) {
    df$player_id <- as.character(df$personId)
  } else if ("playerId" %in% names(df)) {
    df$player_id <- as.character(df$playerId)
  }

  df
}

.nba_stats_standardize_pbp <- function(df, season) {
  df <- as.data.frame(df, stringsAsFactors = FALSE, check.names = FALSE)
  df$league <- rep("nba", nrow(df))
  df$source <- rep("nba_stats", nrow(df))
  df$season <- rep(season, nrow(df))

  if ("GAME_ID" %in% names(df)) {
    df$game_id <- as.character(df$GAME_ID)
  } else if ("gameId" %in% names(df)) {
    df$game_id <- as.character(df$gameId)
  }

  if ("EVENTNUM" %in% names(df)) {
    df$event_num <- suppressWarnings(as.integer(df$EVENTNUM))
  } else if ("EVENT_NUM" %in% names(df)) {
    df$event_num <- suppressWarnings(as.integer(df$EVENT_NUM))
  } else if ("eventNum" %in% names(df)) {
    df$event_num <- suppressWarnings(as.integer(df$eventNum))
  } else if ("actionNumber" %in% names(df)) {
    df$event_num <- suppressWarnings(as.integer(df$actionNumber))
  }

  df
}

.nba_stats_standardize_shots <- function(df, season) {
  df <- as.data.frame(df, stringsAsFactors = FALSE, check.names = FALSE)
  df$league <- rep("nba", nrow(df))
  df$source <- rep("nba_stats", nrow(df))
  df$season <- rep(season, nrow(df))

  if ("GAME_ID" %in% names(df)) df$game_id <- as.character(df$GAME_ID)
  if ("GAME_EVENT_ID" %in% names(df)) df$event_num <- suppressWarnings(as.integer(df$GAME_EVENT_ID))
  if (!"event_num" %in% names(df)) df$event_num <- rep(NA_integer_, nrow(df))

  if (!"shot_id" %in% names(df)) {
    df$shot_id <- if (nrow(df) == 0) character(0) else sprintf("%s_%s", df$game_id, seq_len(nrow(df)))
  }

  df
}

.nba_stats_bind_or_empty <- function(items, empty_df) {
  items <- items[!vapply(items, is.null, logical(1))]
  if (length(items) == 0) return(empty_df)
  out <- bind_rows(items)
  if (nrow(out) == 0 && nrow(empty_df) == 0) {
    missing_cols <- setdiff(names(empty_df), names(out))
    for (col in missing_cols) out[[col]] <- empty_df[[col]]
  }
  out
}

.nba_stats_empty_games_index <- function() {
  data.frame(
    game_id = character(),
    game_date = as.Date(character()),
    season = character(),
    home_team_id = character(),
    away_team_id = character(),
    home_team = character(),
    away_team = character(),
    stringsAsFactors = FALSE
  )
}

.nba_stats_game_id_from_path <- function(path) {
  base <- basename(path)
  id <- sub("^.*_", "", base)
  id <- sub("\\.json$", "", id)
  id
}

.nba_stats_empty_games <- function() {
  data.frame(
    league = character(),
    source = character(),
    season = character(),
    game_id = character(),
    game_date = as.Date(character()),
    home_team = character(),
    away_team = character(),
    home_team_id = character(),
    away_team_id = character(),
    stringsAsFactors = FALSE
  )
}

.nba_stats_empty_team_box <- function() {
  data.frame(
    league = character(),
    source = character(),
    season = character(),
    game_id = character(),
    team_id = character(),
    stringsAsFactors = FALSE
  )
}

.nba_stats_empty_player_box <- function() {
  data.frame(
    league = character(),
    source = character(),
    season = character(),
    game_id = character(),
    player_id = character(),
    stringsAsFactors = FALSE
  )
}

.nba_stats_empty_pbp <- function() {
  data.frame(
    league = character(),
    source = character(),
    season = character(),
    game_id = character(),
    event_num = integer(),
    stringsAsFactors = FALSE
  )
}

.nba_stats_empty_shots <- function() {
  data.frame(
    league = character(),
    source = character(),
    season = character(),
    game_id = character(),
    event_num = integer(),
    shot_id = character(),
    stringsAsFactors = FALSE
  )
}

.nba_stats_empty_manifest <- function() {
  data.frame(
    league = character(),
    season = character(),
    game_id = character(),
    downloaded_at = character(),
    source_url = character(),
    status_code = integer(),
    bytes = numeric(),
    has_boxscore_traditional = logical(),
    has_boxscore_advanced = logical(),
    has_boxscore_fourfactors = logical(),
    has_boxscore_usage = logical(),
    has_pbp = logical(),
    has_shots = logical(),
    stringsAsFactors = FALSE
  )
}

.nba_stats_build_manifest <- function(game_dirs, season, shotchart_path) {
  if (length(game_dirs) == 0) return(.nba_stats_empty_manifest())

  has_shots <- file.exists(shotchart_path)
  rows <- vector("list", length(game_dirs))
  for (i in seq_along(game_dirs)) {
    dir_path <- game_dirs[[i]]
    game_id <- .nba_stats_game_id_from_path(dir_path)
    files <- c(
      boxscore_traditional = file.path(dir_path, "boxscore_traditional.json"),
      boxscore_advanced = file.path(dir_path, "boxscore_advanced.json"),
      boxscore_fourfactors = file.path(dir_path, "boxscore_fourfactors.json"),
      boxscore_usage = file.path(dir_path, "boxscore_usage.json"),
      playbyplay = file.path(dir_path, "playbyplay.json")
    )
    exists_flags <- vapply(files, file.exists, logical(1))
    bytes <- sum(vapply(files[exists_flags], function(p) file.info(p)$size, numeric(1)), na.rm = TRUE)

    rows[[i]] <- data.frame(
      league = "nba",
      season = season,
      game_id = as.character(game_id),
      downloaded_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
      source_url = dir_path,
      status_code = if (any(exists_flags)) 200L else NA_integer_,
      bytes = bytes,
      has_boxscore_traditional = exists_flags[["boxscore_traditional"]],
      has_boxscore_advanced = exists_flags[["boxscore_advanced"]],
      has_boxscore_fourfactors = exists_flags[["boxscore_fourfactors"]],
      has_boxscore_usage = exists_flags[["boxscore_usage"]],
      has_pbp = exists_flags[["playbyplay"]],
      has_shots = has_shots,
      stringsAsFactors = FALSE
    )
  }

  dplyr::bind_rows(rows)
}

.nba_stats_parse_games_from_index <- function(raw_dir, season) {
  index_path <- file.path(raw_dir, paste0("index_", season, ".json"))
  if (!file.exists(index_path)) return(.nba_stats_empty_games())
  raw <- jsonlite::fromJSON(index_path, simplifyVector = FALSE)
  nba_stats_parse_game_index(raw, season = season)
}

.nba_stats_build_games_from_finder <- function(df, season) {
  if (!"GAME_ID" %in% names(df)) return(.nba_stats_empty_games())

  df$GAME_ID <- as.character(df$GAME_ID)
  if ("TEAM_ID" %in% names(df)) df$TEAM_ID <- as.character(df$TEAM_ID)
  if ("TEAM_ABBREVIATION" %in% names(df)) df$TEAM_ABBREVIATION <- as.character(df$TEAM_ABBREVIATION)

  game_ids <- unique(df$GAME_ID)
  rows <- vector("list", length(game_ids))

  for (i in seq_along(game_ids)) {
    gid <- game_ids[[i]]
    sub <- df[df$GAME_ID == gid, , drop = FALSE]

    matchup <- if ("MATCHUP" %in% names(sub)) sub$MATCHUP else rep(NA_character_, nrow(sub))
    home_row <- sub[grepl("vs\\.", matchup), , drop = FALSE]
    away_row <- sub[grepl("@", matchup), , drop = FALSE]

    if (nrow(home_row) == 0 && nrow(sub) >= 1) home_row <- sub[1, , drop = FALSE]
    if (nrow(away_row) == 0 && nrow(sub) >= 2) away_row <- sub[2, , drop = FALSE]

    rows[[i]] <- data.frame(
      league = "nba",
      source = "nba_stats",
      season = season,
      game_id = gid,
      game_date = if ("GAME_DATE" %in% names(sub)) sub$GAME_DATE[[1]] else as.Date(NA),
      home_team = if ("TEAM_ABBREVIATION" %in% names(home_row)) home_row$TEAM_ABBREVIATION[[1]] else NA_character_,
      away_team = if ("TEAM_ABBREVIATION" %in% names(away_row)) away_row$TEAM_ABBREVIATION[[1]] else NA_character_,
      home_team_id = if ("TEAM_ID" %in% names(home_row)) home_row$TEAM_ID[[1]] else NA_character_,
      away_team_id = if ("TEAM_ID" %in% names(away_row)) away_row$TEAM_ID[[1]] else NA_character_,
      stringsAsFactors = FALSE
    )
  }

  bind_rows(rows)
}

# ----------------------
# Shot chart helpers
# ----------------------

.nba_shotchart_date_ranges <- function(season, chunk) {
  chunk <- if (is.null(chunk)) "none" else tolower(as.character(chunk))
  if (chunk %in% c("none", "false", "0")) return(list())
  if (chunk != "month") return(list())

  years <- .nba_stats_season_years(season)
  if (any(is.na(years))) return(list())

  start_date <- as.Date(sprintf("%d-10-01", years[1]))
  end_date <- as.Date(sprintf("%d-06-30", years[2]))
  if (start_date > end_date) return(list())

  starts <- seq.Date(start_date, end_date, by = "month")
  next_starts <- c(starts[-1], end_date + 1)
  ends <- pmin(end_date, next_starts - 1)

  ranges <- lapply(seq_along(starts), function(i) {
    list(date_from = starts[[i]], date_to = ends[[i]])
  })
  ranges
}

.nba_shotchart_merge_raw <- function(base_raw, new_raw) {
  if (is.null(base_raw)) return(new_raw)

  base_rs <- base_raw$resultSets
  new_rs <- new_raw$resultSets

  base_idx <- .nba_result_set_index(base_rs, "Shot_Chart_Detail")
  new_idx <- .nba_result_set_index(new_rs, "Shot_Chart_Detail")

  if (is.na(base_idx) || is.na(new_idx)) return(base_raw)

  base_rows <- base_rs[[base_idx]]$rowSet
  new_rows <- new_rs[[new_idx]]$rowSet

  if (is.null(base_rows)) base_rows <- list()
  if (is.null(new_rows)) new_rows <- list()

  base_rs[[base_idx]]$rowSet <- c(base_rows, new_rows)
  base_raw$resultSets <- base_rs

  base_raw
}

.nba_result_set_index <- function(result_sets, target) {
  if (is.null(result_sets)) return(NA_integer_)
  names_vec <- vapply(result_sets, function(x) {
    if (is.null(x$name)) NA_character_ else as.character(x$name)
  }, character(1))
  idx <- which(vapply(names_vec, function(x) .nba_result_set_match(x, target), logical(1)))
  if (length(idx) == 0) return(NA_integer_)
  idx[[1]]
}

.nba_result_set_match <- function(name, target) {
  if (is.na(name) || is.na(target)) return(FALSE)
  norm <- function(x) tolower(gsub("[^a-z0-9]", "", as.character(x)))
  norm(name) == norm(target)
}

.nba_shotchart_empty_raw <- function() {
  list(
    resultSets = list(
      list(
        name = "Shot_Chart_Detail",
        headers = .nba_shotchart_default_headers(),
        rowSet = list()
      )
    )
  )
}

.nba_shotchart_default_headers <- function() {
  c(
    "GRID_TYPE",
    "GAME_ID",
    "GAME_EVENT_ID",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_NAME",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "EVENT_TYPE",
    "ACTION_TYPE",
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "SHOT_ZONE_RANGE",
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "SHOT_ATTEMPTED_FLAG",
    "SHOT_MADE_FLAG",
    "GAME_DATE",
    "HTM",
    "VTM"
  )
}

.nba_shotchart_empty_table <- function(season, season_type, headers = NULL) {
  if (is.null(headers) || length(headers) == 0) headers <- .nba_shotchart_default_headers()
  df <- as.data.frame(matrix(nrow = 0, ncol = length(headers)), stringsAsFactors = FALSE)
  names(df) <- headers
  df <- .nba_shotchart_coerce_types(df)
  df$season <- rep(season, nrow(df))
  df$season_type <- rep(season_type, nrow(df))
  dplyr::as_tibble(df)
}

.nba_shotchart_coerce_types <- function(df) {
  list_cols <- names(df)[vapply(df, is.list, logical(1))]
  if (length(list_cols) > 0) {
    for (col in list_cols) {
      df[[col]] <- vapply(df[[col]], function(v) {
        if (is.null(v) || length(v) == 0) return(NA_character_)
        if (is.list(v)) v <- v[[1]]
        if (is.null(v) || length(v) == 0) return(NA_character_)
        as.character(v)[1]
      }, character(1))
    }
  }

  char_cols <- c("GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_NAME", "SHOT_TYPE",
                 "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "GAME_DATE", "HTM", "VTM",
                 "EVENT_TYPE", "ACTION_TYPE", "GRID_TYPE")
  int_cols <- c("GAME_EVENT_ID", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING",
                "SHOT_ATTEMPTED_FLAG", "SHOT_MADE_FLAG", "SHOT_DISTANCE")
  num_cols <- c("LOC_X", "LOC_Y")

  for (col in intersect(names(df), char_cols)) df[[col]] <- as.character(df[[col]])
  for (col in intersect(names(df), int_cols)) df[[col]] <- as.integer(df[[col]])
  for (col in intersect(names(df), num_cols)) df[[col]] <- as.numeric(df[[col]])

  if ("GAME_DATE" %in% names(df)) {
    df$GAME_DATE <- .nba_safe_date(df$GAME_DATE)
  }

  df
}

.nba_safe_date <- function(x) {
  x <- as.character(x)
  out <- rep(as.Date(NA), length(x))
  if (length(x) == 0) return(out)

  idx <- grepl("^\\d{8}$", x)
  if (any(idx)) {
    out[idx] <- suppressWarnings(as.Date(x[idx], format = "%Y%m%d"))
  }

  idx2 <- !idx & grepl("^\\d{4}-\\d{2}-\\d{2}$", x)
  if (any(idx2)) {
    out[idx2] <- suppressWarnings(as.Date(x[idx2], format = "%Y-%m-%d"))
  }

  idx3 <- !(idx | idx2) & grepl("^\\d{2}/\\d{2}/\\d{4}$", x)
  if (any(idx3)) {
    out[idx3] <- suppressWarnings(as.Date(x[idx3], format = "%m/%d/%Y"))
  }

  out
}
