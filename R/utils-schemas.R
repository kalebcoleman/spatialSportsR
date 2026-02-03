#' Source registry and schema access helpers

#' Get a league source implementation
#'
#' @param league League identifier (must be 'nba').
#' @param source Source identifier ('espn' or 'nba_stats').
#' @return Source list for the requested league/source.
#' @export
get_source <- function(league, source = NULL) {
  league <- tolower(as.character(league))
  if (league != "nba") {
    stop("Unsupported league: ", league, ". Only 'nba' is supported.", call. = FALSE)
  }

  source <- .normalize_nba_source(source)
  src <- source_nba(source)

  if (!is.list(src) || is.null(src$league)) {
    stop("Source for league ", league, " must return a list with $league", call. = FALSE)
  }

  src
}

.normalize_nba_source <- function(source = NULL, allow_all = FALSE) {
  if (is.null(source) || is.na(source) || !nzchar(as.character(source))) {
    return("espn")
  }

  source <- tolower(as.character(source))
  if (allow_all && source == "all") {
    return("all")
  }
  if (source %in% c("espn", "nba_stats")) {
    return(source)
  }
  stop("Unsupported NBA source: ", source, call. = FALSE)
}

.resolve_sources <- function(league, source = NULL) {
  league <- tolower(as.character(league))
  if (league != "nba") {
    stop("Unsupported league: ", league, ". Only 'nba' is supported.", call. = FALSE)
  }
  source <- .normalize_nba_source(source, allow_all = TRUE)
  if (source == "all") {
    return(nba_sources())
  }
  source
}
