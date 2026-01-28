#' Report what raw data is present for a league + season
#'
#' @param league League identifier (nba/nhl/nfl/mlb).
#' @param season Season identifier.
#' @param raw_dir Directory for raw data.
#' @return A list with present game ids, index_path, and league_dir.
#' @export

manifest <- function(league, season, raw_dir = "data/raw") {
  league <- tolower(as.character(league))
  season <- as.character(season)
  league_dir <- file.path(raw_dir, league, season)

  if (!dir.exists(league_dir)) {
    return(list(present = character(0), missing = character(0), league_dir = league_dir))
  }

  raw_files <- list.files(league_dir, pattern = "^game_.*\\.json$", full.names = FALSE)
  present <- sub("^game_", "", sub("\\.json$", "", raw_files))

  index_path <- file.path(league_dir, paste0("index_", season, ".json"))
  list(
    present = present,
    index_path = index_path,
    league_dir = league_dir
  )
}
