#' Source registry and schema access helpers

#' Get a league source implementation
#'
#' @param league League identifier (nba/nhl/nfl/mlb).
#' @return Source list for the requested league.
#' @export
get_source <- function(league) {
  league <- tolower(as.character(league))
  fn <- switch(
    league,
    nba = source_nba,
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

#' Get schema paths for a league
#'
#' @param league League identifier (nba/nhl/nfl/mlb).
#' @return Named list of schema paths.
#' @export
schemas <- function(league) {
  src <- get_source(league)
  if (!is.function(src$schemas)) {
    stop("Source for league ", league, " does not define schemas()", call. = FALSE)
  }
  src$schemas()
}
