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
#' @param tables List of tables (games/events required for core contract).
#' @param league League identifier.
#' @param source Source identifier (nba only: espn, nba_stats).
#' @return Invisibly TRUE on success.
#' @export
validate_tables <- function(tables, league, source = NULL) {
  if (!is.list(tables)) stop("tables must be a list", call. = FALSE)

  league <- tolower(as.character(league))
  source <- .normalize_source(league, source)

  if (league == "nba" && source == "nba_stats") {
    return(.validate_nba_stats_tables(tables))
  }

  .validate_contract_tables(tables, require_source = (league == "nba"))
}

.validate_contract_tables <- function(tables, require_source = FALSE) {
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

  if (isTRUE(require_source)) {
    required_games <- c(required_games, "source")
    required_events <- c(required_events, "source")
    required_teams <- c(required_teams, "source")
    required_manifest <- c(required_manifest, "source")
  }

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

  game_key_cols <- if (isTRUE(require_source)) c("league", "source", "game_id") else c("league", "game_id")
  event_key_cols <- if (isTRUE(require_source)) c("league", "source", "game_id", "event_id") else c("league", "game_id", "event_id")

  dup_games <- duplicated(games[, game_key_cols])
  if (any(dup_games)) {
    stop("games has duplicate keys: ", sum(dup_games), " rows", call. = FALSE)
  }

  dup_events <- duplicated(events[, event_key_cols])
  if (any(dup_events)) {
    stop("events has duplicate keys: ", sum(dup_events), " rows", call. = FALSE)
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

  events_key <- do.call(paste, c(events[game_key_cols]))
  games_key <- do.call(paste, c(games[game_key_cols]))
  missing_fk <- setdiff(unique(events_key), unique(games_key))
  if (length(missing_fk) > 0) {
    stop("events reference missing games: ", length(missing_fk), " keys", call. = FALSE)
  }

  invisible(TRUE)
}

.validate_nba_stats_tables <- function(tables) {
  games <- tables$games
  team_box_traditional <- tables$team_box_traditional
  player_box_traditional <- tables$player_box_traditional
  team_box_advanced <- tables$team_box_advanced
  player_box_advanced <- tables$player_box_advanced
  team_box_fourfactors <- tables$team_box_fourfactors
  player_box_fourfactors <- tables$player_box_fourfactors
  player_box_usage <- tables$player_box_usage
  pbp <- tables$pbp
  shots <- tables$shots

  required_games <- c("league", "source", "season", "game_id")
  required_team_box <- c("league", "source", "season", "game_id", "team_id")
  required_player_box <- c("league", "source", "season", "game_id", "player_id")
  required_pbp <- c("league", "source", "season", "game_id", "event_num")
  required_shots <- c("league", "source", "season", "game_id")

  if (is.null(games) || !is.data.frame(games)) stop("tables$games must be a data frame", call. = FALSE)
  assert_required_cols(games, required_games, "games")

  .validate_box_table(team_box_traditional, required_team_box, "team_box_traditional")
  .validate_box_table(player_box_traditional, required_player_box, "player_box_traditional")
  .validate_box_table(team_box_advanced, required_team_box, "team_box_advanced")
  .validate_box_table(player_box_advanced, required_player_box, "player_box_advanced")
  .validate_box_table(team_box_fourfactors, required_team_box, "team_box_fourfactors")
  .validate_box_table(player_box_fourfactors, required_player_box, "player_box_fourfactors")
  .validate_box_table(player_box_usage, required_player_box, "player_box_usage")

  if (is.null(pbp) || !is.data.frame(pbp)) stop("tables$pbp must be a data frame", call. = FALSE)
  assert_required_cols(pbp, required_pbp, "pbp")

  if (is.null(shots) || !is.data.frame(shots)) stop("tables$shots must be a data frame", call. = FALSE)
  assert_required_cols(shots, required_shots, "shots")
  if (!"event_num" %in% names(shots) && !"shot_id" %in% names(shots)) {
    stop("shots must include event_num or shot_id", call. = FALSE)
  }

  dup_games <- duplicated(games[, required_games])
  if (any(dup_games)) stop("games has duplicate keys: ", sum(dup_games), " rows", call. = FALSE)

  .validate_pk_unique(team_box_traditional, required_team_box, "team_box_traditional")
  .validate_pk_unique(player_box_traditional, required_player_box, "player_box_traditional")
  .validate_pk_unique(team_box_advanced, required_team_box, "team_box_advanced")
  .validate_pk_unique(player_box_advanced, required_player_box, "player_box_advanced")
  .validate_pk_unique(team_box_fourfactors, required_team_box, "team_box_fourfactors")
  .validate_pk_unique(player_box_fourfactors, required_player_box, "player_box_fourfactors")
  .validate_pk_unique(player_box_usage, required_player_box, "player_box_usage")

  dup_pbp <- duplicated(pbp[, required_pbp])
  if (any(dup_pbp)) stop("pbp has duplicate keys: ", sum(dup_pbp), " rows", call. = FALSE)

  use_event_num <- ("event_num" %in% names(shots) && !all(is.na(shots$event_num)))
  shot_key_cols <- if (isTRUE(use_event_num)) {
    c(required_shots, "event_num")
  } else {
    if (!"shot_id" %in% names(shots)) {
      stop("shots missing shot_id", call. = FALSE)
    }
    c(required_shots, "shot_id")
  }

  if (isTRUE(use_event_num)) {
    bad_shot_event <- is.na(shots$event_num)
    if (any(bad_shot_event)) stop("shots$event_num has missing values: ", sum(bad_shot_event), " rows", call. = FALSE)
  } else {
    bad_shot_id <- is.na(shots$shot_id) | shots$shot_id == ""
    if (any(bad_shot_id)) stop("shots$shot_id has missing values: ", sum(bad_shot_id), " rows", call. = FALSE)
  }

  dup_shots <- duplicated(shots[, shot_key_cols])
  if (any(dup_shots)) stop("shots has duplicate keys: ", sum(dup_shots), " rows", call. = FALSE)

  bad_event <- is.na(pbp$event_num)
  if (any(bad_event)) stop("pbp$event_num has missing values: ", sum(bad_event), " rows", call. = FALSE)

  games_key <- paste(games$league, games$source, games$game_id)
  pbp_key <- paste(pbp$league, pbp$source, pbp$game_id)
  shots_key <- paste(shots$league, shots$source, shots$game_id)

  missing_pbp <- setdiff(unique(pbp_key), unique(games_key))
  if (length(missing_pbp) > 0) stop("pbp reference missing games: ", length(missing_pbp), " keys", call. = FALSE)

  missing_shots <- setdiff(unique(shots_key), unique(games_key))
  if (length(missing_shots) > 0) stop("shots reference missing games: ", length(missing_shots), " keys", call. = FALSE)

  invisible(TRUE)
}

.validate_box_table <- function(df, required_cols, table_name) {
  if (is.null(df) || !is.data.frame(df)) {
    stop("tables$", table_name, " must be a data frame", call. = FALSE)
  }
  assert_required_cols(df, required_cols, table_name)
}

.validate_pk_unique <- function(df, key_cols, table_name) {
  if (nrow(df) == 0) return(invisible(TRUE))
  dup <- duplicated(df[, key_cols])
  if (any(dup)) {
    stop(table_name, " has duplicate keys: ", sum(dup), " rows", call. = FALSE)
  }
  invisible(TRUE)
}
