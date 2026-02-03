#' Collect raw data for a league + season
#'
#' @param league League identifier (currently only "nba" is supported).
#' @param season Season identifier.
#' @param source Source identifier (nba only: espn, nba_stats, or all).
#' @param raw_dir Directory for raw data.
#' @param force Force re-download even if cached.
#' @param progress Show a progress bar while collecting.
#' @param quiet Suppress progress messages.
#' @param shotchart_proxy Optional proxy string or list for NBA Stats requests.
#' @param workers Number of parallel workers for per-game collection (NBA Stats only).
#' @param rate_sleep Random sleep range (seconds) between endpoint calls.
#' @param max_retries Max retries for 403/429/timeouts (NBA Stats only).
#' @param season_type Optional season type (e.g., "regular", "playoffs").
#' @param ... Extra arguments passed to the league source.
#' @return A list with games, raw_paths, league_dir, and index_path.
#' @export

collect_raw <- function(league,
                        season,
                        source = NULL,
                        raw_dir = "data/raw",
                        force = FALSE,
                        progress = TRUE,
                        quiet = FALSE,
                        shotchart_proxy = NULL,
                        workers = 1L,
                        rate_sleep = c(0.5, 1.2),
                        max_retries = 2L,
                        season_type = NULL,
                        ...) {
  sources <- .resolve_sources(league, source)
  if (length(sources) > 1) {
    out <- lapply(sources, function(src_name) {
      collect_raw(
        league = league,
        season = season,
        source = src_name,
        raw_dir = raw_dir,
        force = force,
        progress = progress,
        quiet = quiet,
        shotchart_proxy = shotchart_proxy,
        workers = workers,
        rate_sleep = rate_sleep,
        max_retries = max_retries,
        ...
      )
    })
    names(out) <- sources
    return(out)
  }

  source <- sources
  src <- get_source(league, source = source)
  if (!is.function(src$collect_game_index) || !is.function(src$collect_game_raw)) {
    stop("Source for league ", src$league, " must define collect_game_index() and collect_game_raw()", call. = FALSE)
  }

  league <- tolower(as.character(league))
  season <- as.character(season)
  season_type_norm <- season_type
  if (!is.null(season_type_norm) && league == "nba") {
    stype <- tolower(as.character(season_type_norm))
    if (source == "nba_stats") {
      if (stype %in% c("playoffs", "postseason")) season_type_norm <- "Playoffs"
      if (stype %in% c("regular", "regular season")) season_type_norm <- "Regular Season"
    } else if (source == "espn") {
      if (stype == "playoffs") season_type_norm <- "postseason"
      if (stype %in% c("regular season", "regular")) season_type_norm <- "regular"
    }
  }

  league_dir <- .raw_league_dir(raw_dir, league, season, source, season_type = season_type)
  if (!dir.exists(league_dir)) {
    dir.create(league_dir, recursive = TRUE, showWarnings = FALSE)
  }

  index_out <- if (is.null(season_type_norm)) {
    src$collect_game_index(season, raw_dir = league_dir, force = force, proxy = shotchart_proxy, ...)
  } else {
    src$collect_game_index(season, raw_dir = league_dir, force = force, proxy = shotchart_proxy, season_type = season_type_norm, ...)
  }
  index_path <- NULL
  raw_paths <- NULL

  if (is.list(index_out) && !is.data.frame(index_out)) {
    if (!is.null(index_out$games)) {
      games <- index_out$games
    } else {
      stop("collect_game_index() must return games data", call. = FALSE)
    }
    if (!is.null(index_out$index_path)) index_path <- index_out$index_path
    if (!is.null(index_out$raw_paths)) raw_paths <- index_out$raw_paths
  } else {
    games <- index_out
  }

  if (!is.data.frame(games) || is.null(games$game_id)) {
    stop("collect_game_index() must return a data frame with game_id", call. = FALSE)
  }

  if (is.null(raw_paths)) {
    total <- nrow(games)
    raw_paths <- character(total)

    use_parallel <- is.numeric(workers) && workers > 1 &&
      tolower(as.character(league)) == "nba" &&
      tolower(as.character(source)) == "nba_stats"

    if (workers > 2) workers <- 2L
    if (isTRUE(use_parallel) &&
        requireNamespace("testthat", quietly = TRUE) &&
        isTRUE(testthat::is_testing())) {
      # Avoid multisession side effects in tests (mocked counters live in parent process).
      use_parallel <- FALSE
    }

    pb <- NULL
    if (isTRUE(progress) && total > 0) {
      pb <- utils::txtProgressBar(min = 0, max = total, style = 3)
      on.exit(close(pb), add = TRUE)
    }

    collect_one <- function(i) {
      game_id <- games$game_id[[i]]
      game_date <- if ("game_date" %in% names(games)) games$game_date[[i]] else NULL
      tryCatch(
        src$collect_game_raw(
          game_id,
          season = season,
          raw_dir = league_dir,
          force = force,
          game_date = game_date,
          proxy = shotchart_proxy,
          rate_sleep = rate_sleep,
          max_retries = max_retries,
          ...
        ),
        error = function(e) {
          .log_collect_failure(league_dir, league, source, season, game_id, conditionMessage(e))
          NA_character_
        }
      )
    }

    if (use_parallel && requireNamespace("future", quietly = TRUE) && requireNamespace("future.apply", quietly = TRUE)) {
      plan_old <- future::plan()
      on.exit(future::plan(plan_old), add = TRUE)
      future::plan(future::multisession, workers = workers)

      chunk_size <- max(1L, workers * 5L)
      idx <- seq_len(total)
      start <- 1L
      out_paths <- character(total)

      while (start <= total) {
        end <- min(total, start + chunk_size - 1L)
        chunk_idx <- idx[start:end]
        chunk_paths <- future.apply::future_lapply(chunk_idx, collect_one)
        out_paths[chunk_idx] <- unlist(chunk_paths, use.names = FALSE)

        if (.nba_stats_detect_rate_limit(league_dir, games$game_id[chunk_idx])) {
          use_parallel <- FALSE
          workers <- 1L
          if (!isTRUE(quiet)) {
            message("Detected repeated 403/429 errors; switching to workers = 1 for remaining games.")
          }
          break
        }

        start <- end + 1L
      }

      if (!use_parallel && start <= total) {
        for (i in idx[start:total]) {
          out_paths[[i]] <- collect_one(i)
          if (!is.null(pb)) {
            utils::setTxtProgressBar(pb, i)
          } else if (!isTRUE(quiet)) {
            message(sprintf("[%s/%s] collected %s", i, total, games$game_id[[i]]))
          }
        }
      }

      raw_paths <- out_paths
    } else {
      for (i in seq_len(total)) {
        raw_paths[[i]] <- collect_one(i)

        if (!is.null(pb)) {
          utils::setTxtProgressBar(pb, i)
        } else if (!isTRUE(quiet)) {
          message(sprintf("[%s/%s] collected %s", i, total, games$game_id[[i]]))
        }
      }
    }
  }

  list(
    games = games,
    raw_paths = raw_paths,
    league_dir = league_dir,
    index_path = index_path
  )
}

.log_collect_failure <- function(league_dir, league, source, season, game_id, error_message) {
  dir.create(league_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(league_dir, "collect_failures.csv")
  entry <- data.frame(
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    league = as.character(league),
    source = as.character(source),
    season = as.character(season),
    game_id = as.character(game_id),
    error_message = as.character(error_message),
    stringsAsFactors = FALSE
  )
  utils::write.table(
    entry,
    path,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(path),
    append = file.exists(path)
  )
  invisible(path)
}

.nba_stats_detect_rate_limit <- function(league_dir, game_ids) {
  dirs <- list.dirs(league_dir, full.names = TRUE, recursive = FALSE)
  dirs <- dirs[basename(dirs) != "nba_shotchart"]
  hit <- 0L

  for (gid in game_ids) {
    target <- dirs[grepl(paste0(gid, "$"), dirs)]
    if (length(target) == 0) next
    files <- list.files(target[1], pattern = "\\.json$", full.names = TRUE)
    for (path in files) {
      raw <- tryCatch(jsonlite::fromJSON(path, simplifyVector = FALSE), error = function(e) NULL)
      if (is.null(raw) || is.null(raw$status_code)) next
      if (raw$status_code %in% c(403, 429)) {
        hit <- hit + 1L
        if (hit >= 2L) return(TRUE)
      }
    }
  }
  FALSE
}
