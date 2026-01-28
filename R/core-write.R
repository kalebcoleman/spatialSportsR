#' Write standardized tables to disk
#'
#' @param tables List of tables.
#' @param format Output format.
#' @param out_dir Output directory for rds/csv.
#' @param db_path SQLite path for sqlite output.
#' @param mode Write mode for sqlite.
#' @param skip_existing Skip writing rds/csv files if they already exist.
#' @return Invisibly TRUE on success.
#' @export

write_tables <- function(tables, format = c("rds", "csv", "sqlite"), out_dir = "data/processed", db_path = NULL, mode = c("append", "upsert"), skip_existing = FALSE) {
  if (!is.list(tables)) stop("tables must be a list", call. = FALSE)

  format <- match.arg(format)
  mode <- match.arg(mode)

  if (format %in% c("rds", "csv")) {
    if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  }

  if (format == "rds") {
    for (nm in names(tables)) {
      tbl <- tables[[nm]]
      if (is.null(tbl)) next
      if (!is.data.frame(tbl)) next
      path <- file.path(out_dir, paste0(nm, ".rds"))
      if (isTRUE(skip_existing) && file.exists(path)) next
      saveRDS(tbl, path)
    }
  }

  if (format == "csv") {
    for (nm in names(tables)) {
      tbl <- tables[[nm]]
      if (is.null(tbl)) next
      if (!is.data.frame(tbl)) next
      path <- file.path(out_dir, paste0(nm, ".csv"))
      if (isTRUE(skip_existing) && file.exists(path)) next
      write.csv(tbl, path, row.names = FALSE)
    }
  }

  if (format == "sqlite") {
    if (is.null(db_path) || db_path == "") {
      stop("db_path is required for sqlite output", call. = FALSE)
    }
    if (!requireNamespace("DBI", quietly = TRUE) || !requireNamespace("RSQLite", quietly = TRUE)) {
      stop("Install DBI and RSQLite to write sqlite", call. = FALSE)
    }

    con <- DBI::dbConnect(RSQLite::SQLite(), db_path)
    on.exit(DBI::dbDisconnect(con), add = TRUE)

    for (nm in names(tables)) {
      tbl <- tables[[nm]]
      if (is.null(tbl)) next
      if (!is.data.frame(tbl)) next
      DBI::dbWriteTable(con, nm, tbl, append = (mode == "append"), overwrite = (mode != "append"))
    }
  }

  invisible(TRUE)
}

#' Collect, parse, validate, and write data for multiple seasons
#'
#' @param league League identifier (nba/nhl/nfl/mlb).
#' @param seasons Vector of seasons (e.g., 2006:2025).
#' @param raw_dir Directory for raw data.
#' @param out_dir Output directory for rds/csv.
#' @param format Output format for write_tables().
#' @param db_path SQLite path for sqlite output.
#' @param mode Write mode for sqlite.
#' @param skip_existing Skip writing rds/csv files if they already exist.
#' @param force Force re-download even if cached.
#' @param progress Show a progress bar while collecting/parsing.
#' @param quiet Suppress progress messages.
#' @param validate Validate tables after parsing.
#' @param keep_tables Keep parsed tables in the return list.
#' @param ... Extra arguments passed to collect_raw() and parse_raw().
#' @return A list of season results with row counts and any errors.
#' @export
collect_parse_write <- function(league,
                                seasons,
                                raw_dir = "data/raw",
                                out_dir = "data/processed",
                                format = c("rds", "csv", "sqlite"),
                                db_path = NULL,
                                mode = c("append", "upsert"),
                                force = FALSE,
                                progress = TRUE,
                                quiet = FALSE,
                                validate = TRUE,
                                keep_tables = FALSE,
                                skip_existing = FALSE,
                                ...) {
  format <- match.arg(format)
  mode <- match.arg(mode)
  seasons <- as.integer(seasons)
  results <- vector("list", length(seasons))
  names(results) <- as.character(seasons)

  for (i in seq_along(seasons)) {
    season <- seasons[[i]]
    res <- list(season = season, error = NULL)
    tryCatch({
      collect_raw(
        league = league,
        season = season,
        raw_dir = raw_dir,
        force = force,
        progress = progress,
        quiet = quiet,
        ...
      )

      tables <- parse_raw(
        league = league,
        season = season,
        raw_dir = raw_dir,
        progress = progress,
        quiet = quiet,
        ...
      )

      if (isTRUE(validate)) {
        validate_tables(tables, league)
      }

      season_out_dir <- out_dir
      if (format %in% c("rds", "csv")) {
        season_out_dir <- file.path(out_dir, league, as.character(season))
      }

      write_tables(
        tables = tables,
        format = format,
        out_dir = season_out_dir,
        db_path = db_path,
        mode = mode,
        skip_existing = skip_existing
      )

      res$rows <- vapply(tables, function(x) if (is.data.frame(x)) nrow(x) else NA_integer_, integer(1))
      if (isTRUE(keep_tables)) {
        res$tables <- tables
      }
    }, error = function(e) {
      res$error <- conditionMessage(e)
    })

    results[[i]] <- res
  }

  results
}
