#' Cache helpers

#' Build a cache path for raw data
#'
#' @param raw_dir Root raw directory.
#' @param league League identifier.
#' @param season Season identifier.
#' @param filename File name.
#' @param source Optional source (nba only).
#' @param season_type Optional season type (e.g., "regular", "playoffs").
#' @return A cache file path.
#' @export
cache_path <- function(raw_dir, league, season, filename, source = NULL, season_type = NULL) {
  file.path(.raw_league_dir(raw_dir, league, season, source, season_type = season_type), filename)
}

#' Check if a cache path exists
#'
#' @param path File path.
#' @return Logical indicating existence.
#' @export
cache_exists <- function(path) {
  is.character(path) && nzchar(path) && file.exists(path)
}

.normalize_season_type_dir <- function(season_type = NULL) {
  if (is.null(season_type) || !nzchar(as.character(season_type))) return(NULL)
  stype <- tolower(as.character(season_type))
  if (stype %in% c("regular", "regular season", "reg")) return(NULL)
  if (stype %in% c("postseason", "playoffs", "playoff")) return("playoffs")
  stype
}

.raw_league_dir <- function(raw_dir, league, season, source = NULL, season_type = NULL) {
  league <- tolower(as.character(league))
  season <- as.character(season)
  if (league == "nba" && exists(".nba_stats_season_string", mode = "function")) {
    season <- .nba_stats_season_string(season)
  }
  if (league == "nba" && !is.null(source) && nzchar(as.character(source))) {
    base <- file.path(raw_dir, league, season, tolower(as.character(source)))
  } else {
    base <- file.path(raw_dir, league, season)
  }
  stype_dir <- .normalize_season_type_dir(season_type)
  if (!is.null(stype_dir)) {
    return(file.path(base, stype_dir))
  }
  base
}
