#' Parse raw data for a league + season into standardized tables
#'
#' @param league League identifier (nba/nhl/nfl/mlb).
#' @param season Season identifier.
#' @param raw_dir Directory for raw data.
#' @param progress Show a progress bar while parsing.
#' @param quiet Suppress progress messages.
#' @param ... Extra arguments passed to the league source.
#' @return A list of data frames. Always includes games, events, teams, and manifest
#'   (empty if not returned by the source). Additional tables returned by the league
#'   parse_game_raw() are included dynamically.
#'
#' @importFrom dplyr bind_rows
#' @export

parse_raw <- function(league, season, raw_dir = "data/raw", progress = TRUE, quiet = FALSE, ...) {
  src <- get_source(league)
  if (!is.function(src$parse_game_raw)) {
    stop("Source for league ", src$league, " must define parse_game_raw()", call. = FALSE)
  }

  league <- tolower(as.character(league))
  season <- as.character(season)
  league_dir <- file.path(raw_dir, league, season)

  if (!dir.exists(league_dir)) {
    stop("Raw directory not found: ", league_dir, call. = FALSE)
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
    parsed[[i]] <- src$parse_game_raw(raw_paths[[i]], ...)

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

  out
}
