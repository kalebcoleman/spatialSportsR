#' Parse raw data for a league + season into standardized tables
#'
#' @param league League identifier (currently only "nba" is supported).
#' @param season Season identifier.
#' @param source Source identifier (nba only: espn, nba_stats, or all).
#' @param raw_dir Directory for raw data.
#' @param progress Show a progress bar while parsing.
#' @param quiet Suppress progress messages.
#' @param season_type Optional season type (e.g., "regular", "playoffs").
#' @param ... Extra arguments passed to the league source.
#' @return A list of data frames. Always includes games, events, teams, and manifest
#'   (empty if not returned by the source). Additional tables returned by the league
#'   parse_game_raw() are included dynamically.
#'
#' @importFrom dplyr bind_rows
#' @export

parse_raw <- function(league,
                      season,
                      source = NULL,
                      raw_dir = "data/raw",
                      progress = TRUE,
                      quiet = FALSE,
                      season_type = NULL,
                      ...) {
  sources <- .resolve_sources(league, source)
  if (length(sources) > 1) {
    out <- lapply(sources, function(src_name) {
      parse_raw(
        league = league,
        season = season,
        source = src_name,
        raw_dir = raw_dir,
        progress = progress,
        quiet = quiet,
        ...
      )
    })
    names(out) <- sources
    return(out)
  }

  source <- sources
  src <- get_source(league, source = source)
  if (!is.function(src$parse_game_raw)) {
    if (!is.function(src$parse_raw_dir)) {
      stop("Source for league ", src$league, " must define parse_game_raw() or parse_raw_dir()", call. = FALSE)
    }
  }

  league <- tolower(as.character(league))
  season <- as.character(season)
  league_dir <- .raw_league_dir(raw_dir, league, season, source, season_type = season_type)

  if (!dir.exists(league_dir)) {
    stop("Raw directory not found: ", league_dir, call. = FALSE)
  }

  if (is.function(src$parse_raw_dir)) {
    return(src$parse_raw_dir(season = season, raw_dir = league_dir, season_type = season_type, ...))
  }

  if (is.function(src$list_raw_paths)) {
    raw_paths <- src$list_raw_paths(season = season, raw_dir = league_dir, ...)
  } else {
    pattern <- if (!is.null(src$raw_pattern)) src$raw_pattern else "^game_.*\\.json$"
    raw_paths <- list.files(league_dir, pattern = pattern, full.names = TRUE)
  }
  total <- length(raw_paths)
  parsed <- vector("list", total)

  if (isTRUE(progress)) {
    pb <- utils::txtProgressBar(min = 0, max = total, style = 3)
    on.exit(close(pb), add = TRUE)
  }

  for (i in seq_len(total)) {
    parsed[[i]] <- src$parse_game_raw(raw_paths[[i]], season_type = season_type, ...)

    if (isTRUE(progress)) {
      utils::setTxtProgressBar(pb, i)
    } else if (!isTRUE(quiet)) {
      message(sprintf("[%s/%s] parsed %s", i, total, basename(raw_paths[[i]])))
    }
  }

  bind_or_empty <- function(items, name) {
    chunk <- lapply(items, function(x) x[[name]])
    chunk <- chunk[!vapply(chunk, is.null, logical(1))]
    if (length(chunk) == 0) return(data.frame())
    bind_rows(chunk)
  }

  core_names <- c("games", "events", "teams", "manifest")
  all_names <- unique(c(core_names, unlist(lapply(parsed, names))))
  out <- setNames(vector("list", length(all_names)), all_names)

  for (nm in all_names) {
    out[[nm]] <- bind_or_empty(parsed, nm)
  }

  if (league == "nba" && source == "espn") {
    if (!is.null(out$games) && is.data.frame(out$games) && nrow(out$games) > 0) {
      key_cols <- c("league", "source", "game_id")
      if (all(key_cols %in% names(out$games))) {
        out$games <- out$games[!duplicated(out$games[, key_cols]), , drop = FALSE]
      }
    }
  }

  out
}
