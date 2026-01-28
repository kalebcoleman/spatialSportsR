#' NHL source helpers (internal)
#'
#' @description
#' Internal ESPN NHL source implementation returned by `get_source("nhl")`.
#'
#' @keywords internal
#' @importFrom dplyr as_tibble bind_rows
#' @importFrom jsonlite fromJSON

source_nhl <- function() {
  list(
    league = "nhl",
    collect_game_index = function(season, ...) {
      stop("collect_game_index() not implemented for NHL", call. = FALSE)
    },
    collect_game_raw = function(game_id, ...) {
      stop("collect_game_raw() not implemented for NHL", call. = FALSE)
    },
    parse_game_raw = function(path, ...) {
      stop("parse_game_raw() not implemented for NHL", call. = FALSE)
    },
    normalize_xy = function(x, y) {
      data.frame(
        x_unit = (x + 100) / 200,
        y_unit = (y + 42.5) / 85
      )
    },
    schemas = function() {
      list(
        events = system.file("extdata/schemas/nhl_spatial_events.json", package = "spatialSportsR"),
        games = system.file("extdata/schemas/common_games.json", package = "spatialSportsR"),
        teams = system.file("extdata/schemas/common_teams.json", package = "spatialSportsR")
      )
    }
  )
}
