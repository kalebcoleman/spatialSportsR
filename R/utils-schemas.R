#' Source registry and schema access helpers

#' Get a league source implementation
#'
#' @param league League identifier (nba/nhl/nfl/mlb).
#' @param source Source identifier (nba only: espn, nba_stats, or all).
#' @return Source list for the requested league/source.
#' @export
get_source <- function(league, source = NULL) {
  league <- tolower(as.character(league))
  source <- .normalize_source(league, source)

  fn <- switch(
    league,
    nba = function() source_nba(source),
    nhl = source_nhl,
    nfl = source_nfl,
    mlb = source_mlb,
    NULL
  )

  if (is.null(fn)) {
    stop("Unsupported league: ", league, call. = FALSE)
  }

  src <- fn()
  if (!is.list(src) || is.null(src$league)) {
    stop("Source for league ", league, " must return a list with $league", call. = FALSE)
  }

  src
}

#' Get schema paths for a league/source
#'
#' @param league League identifier (nba/nhl/nfl/mlb).
#' @param source Source identifier (nba only: espn, nba_stats).
#' @return Named list of schema paths (or schema descriptors).
#' @export
schemas <- function(league, source = NULL) {
  src <- get_source(league, source = source)
  if (!is.function(src$schemas)) {
    stop("Source for league ", src$league, " does not define schemas()", call. = FALSE)
  }
  src$schemas()
}

.normalize_source <- function(league, source = NULL, allow_all = FALSE) {
  league <- tolower(as.character(league))
  if (is.null(source) || is.na(source) || !nzchar(as.character(source))) {
    return(if (league == "nba") "espn" else "default")
  }

  source <- tolower(as.character(source))
  if (league == "nba") {
    if (allow_all && source == "all") return("all")
    if (source %in% c("espn", "nba_stats")) return(source)
    stop("Unsupported NBA source: ", source, call. = FALSE)
  }

  if (source %in% c("default", "all")) return("default")
  stop("Unsupported source for league ", league, ": ", source, call. = FALSE)
}

.resolve_sources <- function(league, source = NULL) {
  source <- .normalize_source(league, source, allow_all = TRUE)
  if (league == "nba" && source == "all") return(nba_sources())
  source
}
