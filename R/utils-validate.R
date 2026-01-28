#' Validation helpers

#' Assert required columns exist
#'
#' @param df Data frame to validate.
#' @param required Character vector of required columns.
#' @param table_name Name of the table for error messages.
#' @return Invisibly TRUE on success.
assert_required_cols <- function(df, required, table_name) {
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) {
    stop(
      table_name, " missing columns: ", paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

#' Validate standardized tables against the contract
#'
#' @param tables List of tables (games/events required).
#' @param league League identifier.
#' @return Invisibly TRUE on success.
#' @export
validate_tables <- function(tables, league) {
  if (!is.list(tables)) stop("tables must be a list", call. = FALSE)

  games <- tables$games
  events <- tables$events
  teams <- tables$teams
  manifest <- tables$manifest

  if (is.null(games) || !is.data.frame(games)) {
    stop("tables$games must be a data frame", call. = FALSE)
  }
  if (is.null(events) || !is.data.frame(events)) {
    stop("tables$events must be a data frame", call. = FALSE)
  }

  required_games <- c("league", "season", "game_id", "game_date", "home_team", "away_team")
  required_events <- c("league", "season", "game_id", "event_id", "event_type", "x_unit", "y_unit", "label", "meta")
  required_teams <- c("league", "team_id", "team_abbrev", "team_name")
  required_manifest <- c("league", "season", "game_id", "downloaded_at", "source_url", "status_code", "bytes")

  assert_required_cols(games, required_games, "games")
  assert_required_cols(events, required_events, "events")

  if (!is.null(teams)) {
    if (!is.data.frame(teams)) stop("tables$teams must be a data frame", call. = FALSE)
    assert_required_cols(teams, required_teams, "teams")
  }

  if (!is.null(manifest)) {
    if (!is.data.frame(manifest)) stop("tables$manifest must be a data frame", call. = FALSE)
    assert_required_cols(manifest, required_manifest, "manifest")
  }

  dup_games <- duplicated(games[, c("league", "game_id")])
  if (any(dup_games)) {
    stop("games has duplicate (league, game_id): ", sum(dup_games), " rows", call. = FALSE)
  }

  dup_events <- duplicated(events[, c("league", "game_id", "event_id")])
  if (any(dup_events)) {
    stop("events has duplicate (league, game_id, event_id): ", sum(dup_events), " rows", call. = FALSE)
  }

  bad_event_id <- is.na(events$event_id) | events$event_id == ""
  if (any(bad_event_id)) {
    stop("events$event_id has missing values: ", sum(bad_event_id), " rows", call. = FALSE)
  }

  bad_game_id <- is.na(events$game_id) | events$game_id == ""
  if (any(bad_game_id)) {
    stop("events$game_id has missing values: ", sum(bad_game_id), " rows", call. = FALSE)
  }

  if ("x_unit" %in% names(events)) {
    bad_x <- !is.na(events$x_unit) & (events$x_unit < 0 | events$x_unit > 1)
    if (any(bad_x)) {
      stop("events$x_unit out of [0,1] bounds: ", sum(bad_x), " rows", call. = FALSE)
    }
  }

  if ("y_unit" %in% names(events)) {
    bad_y <- !is.na(events$y_unit) & (events$y_unit < 0 | events$y_unit > 1)
    if (any(bad_y)) {
      stop("events$y_unit out of [0,1] bounds: ", sum(bad_y), " rows", call. = FALSE)
    }
  }

  events_key <- paste(events$league, events$game_id)
  games_key <- paste(games$league, games$game_id)
  missing_fk <- setdiff(unique(events_key), unique(games_key))
  if (length(missing_fk) > 0) {
    stop("events reference missing games: ", length(missing_fk), " keys", call. = FALSE)
  }

  invisible(TRUE)
}
