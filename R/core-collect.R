#' Collect raw data for a league + season
#'
#' @param league League identifier (nba/nhl/nfl/mlb).
#' @param season Season identifier.
#' @param raw_dir Directory for raw data.
#' @param force Force re-download even if cached.
#' @param progress Show a progress bar while collecting.
#' @param quiet Suppress progress messages.
#' @param ... Extra arguments passed to the league source.
#' @return A list with games, raw_paths, league_dir, and index_path.
#' @export

collect_raw <- function(league, season, raw_dir = "data/raw", force = FALSE, progress = TRUE, quiet = FALSE, ...) {
  src <- get_source(league)
  if (!is.function(src$collect_game_index) || !is.function(src$collect_game_raw)) {
    stop("Source for league ", src$league, " must define collect_game_index() and collect_game_raw()", call. = FALSE)
  }

  league <- tolower(as.character(league))
  season <- as.character(season)

  league_dir <- file.path(raw_dir, league, season)
  if (!dir.exists(league_dir)) {
    dir.create(league_dir, recursive = TRUE, showWarnings = FALSE)
  }

  index_out <- src$collect_game_index(season, raw_dir = league_dir, force = force, ...)
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

    if (isTRUE(progress) && total > 0) {
      pb <- utils::txtProgressBar(min = 0, max = total, style = 3)
      on.exit(close(pb), add = TRUE)
    }

    for (i in seq_len(total)) {
      game_id <- games$game_id[[i]]
      game_date <- if ("game_date" %in% names(games)) games$game_date[[i]] else NULL
      raw_paths[[i]] <- src$collect_game_raw(
        game_id,
        season = season,
        raw_dir = league_dir,
        force = force,
        game_date = game_date,
        ...
      )

      if (isTRUE(progress) && total > 0) {
        utils::setTxtProgressBar(pb, i)
      } else if (!isTRUE(quiet)) {
        message(sprintf("[%s/%s] collected %s", i, total, game_id))
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
