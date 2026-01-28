#' NFL source helpers (internal)
#'
#' @description
#' Internal ESPN NFL source implementation returned by `get_source("nfl")`.
#'
#' @keywords internal
#' @importFrom dplyr as_tibble bind_rows
#' @importFrom jsonlite fromJSON

source_nfl <- function() {
  list(
    league = "nfl",
    collect_game_index = function(season, ...) {
      stop("collect_game_index() not implemented for NFL", call. = FALSE)
    },
    collect_game_raw = function(game_id, ...) {
      stop("collect_game_raw() not implemented for NFL", call. = FALSE)
    },
    parse_game_raw = function(path, ...) {
      stop("parse_game_raw() not implemented for NFL", call. = FALSE)
    },
    normalize_xy = function(x, y) {
      data.frame(
        x_unit = x / 100,
        y_unit = y / 53.3333333333
      )
    },
    schemas = function() {
      list(
        events = system.file("extdata/schemas/nfl_events.json", package = "spatialSportsR"),
        games = system.file("extdata/schemas/common_games.json", package = "spatialSportsR"),
        teams = system.file("extdata/schemas/common_teams.json", package = "spatialSportsR")
      )
    }
  )
}
