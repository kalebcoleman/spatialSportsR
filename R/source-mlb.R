#' MLB source helpers (internal)
#'
#' @description
#' Internal ESPN MLB source implementation returned by `get_source("mlb")`.
#'
#' @keywords internal
#' @importFrom dplyr as_tibble bind_rows
#' @importFrom jsonlite fromJSON

source_mlb <- function() {
  list(
    league = "mlb",
    collect_game_index = function(season, ...) {
      stop("collect_game_index() not implemented for MLB", call. = FALSE)
    },
    collect_game_raw = function(game_id, ...) {
      stop("collect_game_raw() not implemented for MLB", call. = FALSE)
    },
    parse_game_raw = function(path, ...) {
      stop("parse_game_raw() not implemented for MLB", call. = FALSE)
    },
    normalize_xy = function(x, y) {
      data.frame(
        x_unit = x / 250,
        y_unit = y / 250
      )
    },
    schemas = function() {
      list(
        events = system.file("extdata/schemas/mlb_events.json", package = "spatialSportsR"),
        games = system.file("extdata/schemas/common_games.json", package = "spatialSportsR"),
        teams = system.file("extdata/schemas/common_teams.json", package = "spatialSportsR")
      )
    }
  )
}
