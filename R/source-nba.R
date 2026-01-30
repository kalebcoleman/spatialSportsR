#' NBA source registry
#'
#' @description
#' Returns NBA source implementations for ESPN and NBA Stats.
#'
#' @keywords internal

source_nba <- function(source = c("espn", "nba_stats")) {
  source <- match.arg(source)

  if (source == "espn") return(source_nba_espn())
  if (source == "nba_stats") return(source_nba_stats())

  stop("Unsupported NBA source: ", source, call. = FALSE)
}

nba_sources <- function() {
  c("espn", "nba_stats")
}
