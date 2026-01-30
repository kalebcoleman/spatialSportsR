#' Report what raw data is present for a league + season
#'
#' @param league League identifier (nba/nhl/nfl/mlb).
#' @param season Season identifier.
#' @param source Source identifier (nba only: espn, nba_stats, or all).
#' @param raw_dir Directory for raw data.
#' @param season_type Optional season type (e.g., "regular", "playoffs").
#' @return A list with present game ids, index_path, and league_dir.
#' @export

manifest <- function(league, season, source = NULL, raw_dir = "data/raw", season_type = NULL) {
  league <- tolower(as.character(league))
  season <- as.character(season)
  sources <- .resolve_sources(league, source)

  if (length(sources) > 1) {
    out <- lapply(sources, function(src_name) {
      manifest(league = league, season = season, source = src_name, raw_dir = raw_dir, season_type = season_type)
    })
    names(out) <- sources
    return(out)
  }

  source <- sources
  league_dir <- .raw_league_dir(raw_dir, league, season, source, season_type = season_type)

  if (!dir.exists(league_dir)) {
    return(list(present = character(0), missing = character(0), league_dir = league_dir))
  }

  if (league == "nba" && source == "espn") {
    raw_files <- list.files(league_dir, pattern = "^summary_.*\\.json$", full.names = FALSE)
    present <- .manifest_extract_game_ids(raw_files)
    index_path <- file.path(league_dir, paste0("nba_index_", season, "_regular.json"))
  } else if (league == "nba" && source == "nba_stats") {
    game_dirs <- list.dirs(league_dir, full.names = FALSE, recursive = FALSE)
    game_dirs <- setdiff(game_dirs, basename(league_dir))
    if (length(game_dirs) > 0) {
      present <- game_dirs[file.exists(file.path(league_dir, game_dirs, "boxscore_traditional.json"))]
      present <- vapply(present, .nba_stats_game_id_from_path, character(1))
    } else {
      raw_files <- list.files(league_dir, pattern = "^boxscoretraditionalv2_.*\\.json$", full.names = FALSE)
      present <- .manifest_extract_game_ids(raw_files)
    }
    season_str <- if (exists(".nba_stats_season_string", mode = "function")) .nba_stats_season_string(season) else season
    index_path <- file.path(league_dir, paste0("index_", season_str, ".json"))
  } else {
    raw_files <- list.files(league_dir, pattern = "^game_.*\\.json$", full.names = FALSE)
    present <- sub("^game_", "", sub("\\.json$", "", raw_files))
    index_path <- file.path(league_dir, paste0("index_", season, ".json"))
  }

  list(
    present = present,
    index_path = index_path,
    league_dir = league_dir
  )
}

.manifest_extract_game_ids <- function(files) {
  base <- sub("\\.json$", "", basename(files))
  vapply(strsplit(base, "_"), function(x) {
    if (length(x) == 0) return(NA_character_)
    x[[length(x)]]
  }, character(1))
}
