#' Cache helpers

#' Build a cache path for raw data
#'
#' @param raw_dir Root raw directory.
#' @param league League identifier.
#' @param season Season identifier.
#' @param filename File name.
#' @return A cache file path.
#' @export
cache_path <- function(raw_dir, league, season, filename) {
  file.path(raw_dir, tolower(as.character(league)), as.character(season), filename)
}

#' Check if a cache path exists
#'
#' @param path File path.
#' @return Logical indicating existence.
#' @export
cache_exists <- function(path) {
  is.character(path) && nzchar(path) && file.exists(path)
}
