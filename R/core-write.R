#' Write standardized tables to disk
#'
#' @param tables List of tables.
#' @param format Output format.
#' @param out_dir Output directory for rds/csv.
#' @param db_path SQLite path for sqlite output.
#' @param mode Write mode for sqlite. Use `"append"` to add rows or `"overwrite"` to
#'   replace tables. `"upsert"` is deprecated (alias for `"overwrite"`).
#' @param skip_existing Skip writing rds/csv files if they already exist.
#' @param bundle Write bundled RDS after per-table writes.
#' @param bundle_name Bundle file name without extension.
#' @param table_prefix Optional prefix for sqlite table names. Use "source" to prefix with source name.
#' @return Invisibly TRUE on success.
#' @export

write_tables <- function(tables,
                         format = c("rds", "csv", "sqlite"),
                         out_dir = "data/parsed",
                         db_path = NULL,
                         mode = c("append", "overwrite", "upsert"),
                         skip_existing = TRUE,
                         bundle = FALSE,
                         bundle_name = NULL,
                         table_prefix = NULL) {
  if (!is.list(tables)) stop("tables must be a list", call. = FALSE)

  format <- match.arg(format)
  mode <- match.arg(mode)
  if (mode == "upsert") {
    warning(
      "write_tables(mode = 'upsert') is deprecated and currently behaves like mode = 'overwrite'. Use mode = 'overwrite' instead.",
      call. = FALSE
    )
    mode <- "overwrite"
  }

  if (format %in% c("rds", "csv")) {
    if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  }

  if (format == "rds") {
    table_names <- names(tables)[vapply(tables, is.data.frame, logical(1))]
    for (nm in names(tables)) {
      tbl <- tables[[nm]]
      if (is.null(tbl)) next
      if (!is.data.frame(tbl)) next
      path <- file.path(out_dir, paste0(nm, ".rds"))
      if (isTRUE(skip_existing) && file.exists(path)) next
      saveRDS(tbl, path)
    }
    if (isTRUE(bundle)) {
      write_bundle_rds(
        table_names = unique(table_names),
        out_dir = out_dir,
        bundle_name = bundle_name,
        skip_existing = skip_existing
      )
    }
  }

  if (format == "csv") {
    for (nm in names(tables)) {
      tbl <- tables[[nm]]
      if (is.null(tbl)) next
      if (!is.data.frame(tbl)) next
      path <- file.path(out_dir, paste0(nm, ".csv"))
      if (isTRUE(skip_existing) && file.exists(path)) next
      utils::write.csv(tbl, path, row.names = FALSE)
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

    DBI::dbWithTransaction(con, {
      for (nm in names(tables)) {
        tbl <- tables[[nm]]
        if (is.null(tbl)) next
        if (!is.data.frame(tbl)) next

        # Drop duplicate columns (case-insensitive) to avoid dbWriteTable failures.
        if (any(duplicated(tolower(names(tbl))))) {
          keep_idx <- !duplicated(tolower(names(tbl)))
          tbl <- tbl[, keep_idx, drop = FALSE]
        }

        # Normalize date/time columns to ISO strings for SQLite readability.
        for (col in names(tbl)) {
          if (inherits(tbl[[col]], "Date")) {
            tbl[[col]] <- format(tbl[[col]], "%Y-%m-%d")
          } else if (inherits(tbl[[col]], c("POSIXct", "POSIXt"))) {
            tbl[[col]] <- format(tbl[[col]], "%Y-%m-%d %H:%M:%S")
          }
        }

        table_name <- if (!is.null(table_prefix) && nzchar(table_prefix)) {
          paste(table_prefix, nm, sep = "_")
        } else {
          nm
        }

        if (mode == "append" && DBI::dbExistsTable(con, table_name)) {
          existing_cols <- DBI::dbListFields(con, table_name)
          existing_lower <- tolower(existing_cols)
          incoming_lower <- tolower(names(tbl))
          new_cols <- names(tbl)[!(incoming_lower %in% existing_lower)]
          missing_cols <- existing_cols[!(existing_lower %in% incoming_lower)]

          if (length(new_cols) > 0) {
            for (col in new_cols) {
              DBI::dbExecute(
                con,
                sprintf(
                  "ALTER TABLE %s ADD COLUMN %s",
                  DBI::dbQuoteIdentifier(con, table_name),
                  DBI::dbQuoteIdentifier(con, col)
                )
              )
            }
          }

          if (length(missing_cols) > 0) {
            for (col in missing_cols) {
              tbl[[col]] <- NA
            }
          }

          tbl <- tbl[, c(existing_cols, setdiff(names(tbl), existing_cols)), drop = FALSE]
        }

        DBI::dbWriteTable(
          con,
          table_name,
          tbl,
          append = (mode == "append"),
          overwrite = (mode == "overwrite")
        )
      }
    })
  }

  invisible(TRUE)
}

#' Write a bundled RDS file from per-table RDS outputs
#'
#' @param table_names Character vector of table names (without extensions).
#' @param out_dir Directory containing per-table `.rds` files.
#' @param bundle_name Bundle file name without extension.
#' @param skip_existing If `TRUE` and the bundle exists, return it without rewriting.
#' @return The bundle path (invisibly).
#' @export
write_bundle_rds <- function(table_names, out_dir, bundle_name, skip_existing = FALSE) {
  if (is.null(bundle_name) || !nzchar(bundle_name)) {
    stop("bundle_name is required when bundle = TRUE", call. = FALSE)
  }

  bundle_path <- file.path(out_dir, paste0(bundle_name, ".rds"))
  if (isTRUE(skip_existing) && file.exists(bundle_path)) return(invisible(bundle_path))

  tables <- lapply(table_names, function(nm) {
    path <- file.path(out_dir, paste0(nm, ".rds"))
    if (!file.exists(path)) return(NULL)
    readRDS(path)
  })
  names(tables) <- table_names
  tables <- tables[!vapply(tables, is.null, logical(1))]

  saveRDS(tables, bundle_path)
  invisible(bundle_path)
}

#' Write SQLite DB from parsed RDS files
#'
#' @param root_dir Root parsed directory (e.g., data/parsed).
#' @param league League identifier (nba/nhl/nfl/mlb).
#' @param sources Optional character vector of sources to include.
#' @param seasons Optional vector of season folder names to include.
#' @param db_path SQLite output path.
#' @param mode Write mode for sqlite. Use `"append"` to add rows or `"overwrite"` to
#'   replace tables. `"upsert"` is deprecated (alias for `"overwrite"`).
#' @param bundle Prefer bundled RDS if present.
#' @param bundle_name Optional bundle name override (without extension).
#' @param season_type Optional season type subdirectory filter.
#' @param table_prefix Optional prefix for sqlite table names.
#' @param debug Emit progress messages.
#' @return A list of results per source/season with row counts.
#' @export
write_sqlite_from_rds <- function(root_dir = "data/parsed",
                                  league,
                                  sources = NULL,
                                  seasons = NULL,
                                  db_path,
                                  mode = c("append", "overwrite", "upsert"),
                                  bundle = TRUE,
                                  bundle_name = NULL,
                                  season_type = NULL,
                                  table_prefix = NULL,
                                  debug = FALSE) {
  mode <- match.arg(mode)
  if (mode == "upsert") {
    warning(
      "write_sqlite_from_rds(mode = 'upsert') is deprecated and currently behaves like mode = 'overwrite'. Use mode = 'overwrite' instead.",
      call. = FALSE
    )
    mode <- "overwrite"
  }
  bundle <- isTRUE(bundle)
  debug <- isTRUE(debug)

  if (is.null(db_path) || !nzchar(db_path)) {
    stop("db_path is required", call. = FALSE)
  }
  if (!requireNamespace("DBI", quietly = TRUE) || !requireNamespace("RSQLite", quietly = TRUE)) {
    stop("Install DBI and RSQLite to write sqlite", call. = FALSE)
  }

  league <- tolower(as.character(league))
  base_dir <- file.path(root_dir, league)
  if (!dir.exists(base_dir)) stop("Parsed league dir not found: ", base_dir, call. = FALSE)

  source_dirs <- list.dirs(base_dir, full.names = TRUE, recursive = FALSE)
  if (length(source_dirs) == 0) stop("No sources found in: ", base_dir, call. = FALSE)
  source_names <- basename(source_dirs)
  if (!is.null(sources)) {
    sources <- tolower(as.character(sources))
    source_dirs <- source_dirs[source_names %in% sources]
    source_names <- basename(source_dirs)
  }
  if (length(source_dirs) == 0) stop("No matching sources found", call. = FALSE)

  season_filter <- if (is.null(seasons)) NULL else as.character(seasons)
  results <- list()

  con <- DBI::dbConnect(RSQLite::SQLite(), db_path)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  for (i in seq_along(source_dirs)) {
    src_dir <- source_dirs[[i]]
    src_name <- source_names[[i]]

    season_dirs <- list.dirs(src_dir, full.names = TRUE, recursive = FALSE)
    if (!is.null(season_filter)) {
      season_dirs <- season_dirs[basename(season_dirs) %in% season_filter]
    }
    if (length(season_dirs) == 0) next

    for (season_dir in season_dirs) {
      season_label <- basename(season_dir)
      stype_dir <- .normalize_season_type_dir(season_type)
      data_dir <- if (!is.null(stype_dir)) file.path(season_dir, stype_dir) else season_dir
      if (!dir.exists(data_dir)) next
      if (isTRUE(debug)) {
        message("write_sqlite_from_rds: source=", src_name, " season=", season_label)
      }

      tables <- NULL
      table_paths <- NULL
      if (isTRUE(bundle)) {
        bundle_guess <- bundle_name
        if (is.null(bundle_guess) || !nzchar(bundle_guess)) {
          if (league == "nba") {
            bundle_guess <- if (src_name == "espn") "espn_all" else "nba_stats_all"
          } else {
            bundle_guess <- paste0(league, "_all")
          }
        }
        bundle_path <- file.path(data_dir, paste0(bundle_guess, ".rds"))
        if (file.exists(bundle_path)) {
          tables <- readRDS(bundle_path)
          if (!is.list(tables) || length(tables) == 0) {
            tables <- NULL
          } else {
            non_df <- names(tables)[!vapply(tables, is.data.frame, logical(1))]
            if (length(non_df) > 0) {
              message(
                "write_sqlite_from_rds: bundle has non-data.frame tables; falling back to per-table RDS. bundle=",
                bundle_path,
                " non_df=",
                paste(non_df, collapse = ",")
              )
              tables <- NULL
            } else {
              table_paths <- stats::setNames(rep(bundle_path, length(tables)), names(tables))
              # If bundle has empty tables but per-table RDS has data, override the empty table.
              empty_tbls <- names(tables)[vapply(tables, function(x) is.data.frame(x) && nrow(x) == 0, logical(1))]
              if (length(empty_tbls) > 0) {
                for (nm in empty_tbls) {
                  rds_path <- file.path(data_dir, paste0(nm, ".rds"))
                  if (!file.exists(rds_path)) next
                  rds_tbl <- readRDS(rds_path)
                  if (is.data.frame(rds_tbl) && nrow(rds_tbl) > 0) {
                    tables[[nm]] <- rds_tbl
                    table_paths[[nm]] <- rds_path
                    message(
                      "write_sqlite_from_rds: overriding empty bundled table=",
                      nm,
                      " bundle=",
                      bundle_path,
                      " path=",
                      rds_path,
                      " rows=",
                      nrow(rds_tbl)
                    )
                  }
                }
              }
            }
          }
        }
      }

      if (is.null(tables)) {
        rds_paths <- list.files(data_dir, pattern = "\\.rds$", full.names = TRUE)
        if (length(rds_paths) == 0) next
        rds_paths <- rds_paths[!grepl("_all\\.rds$", rds_paths)]
        tables <- lapply(rds_paths, readRDS)
        names(tables) <- sub("\\.rds$", "", basename(rds_paths))
        table_paths <- stats::setNames(rds_paths, names(tables))
      }

      if (!is.list(tables) || length(tables) == 0) next

      DBI::dbWithTransaction(con, {
        for (nm in names(tables)) {
          tbl <- tables[[nm]]
          if (is.null(tbl) || !is.data.frame(tbl)) {
            src_path <- if (!is.null(table_paths) && nm %in% names(table_paths)) table_paths[[nm]] else NA_character_
            message("write_sqlite_from_rds: skipped non-data.frame table=", nm, " path=", src_path, " rows=0")
            next
          }
          if (any(duplicated(tolower(names(tbl))))) {
            keep_idx <- !duplicated(tolower(names(tbl)))
            if (isTRUE(debug)) {
              drop_cols <- names(tbl)[!keep_idx]
              message("write_sqlite_from_rds: dropping duplicate columns (case-insensitive): ", paste(drop_cols, collapse = ","))
            }
            tbl <- tbl[, keep_idx, drop = FALSE]
          }
          # Normalize date/time columns to ISO strings for SQLite readability.
          for (col in names(tbl)) {
            if (inherits(tbl[[col]], "Date")) {
              tbl[[col]] <- format(tbl[[col]], "%Y-%m-%d")
            } else if (inherits(tbl[[col]], c("POSIXct", "POSIXt"))) {
              tbl[[col]] <- format(tbl[[col]], "%Y-%m-%d %H:%M:%S")
            }
          }
          prefix <- NULL
          if (!is.null(table_prefix) && nzchar(table_prefix)) {
            prefix <- if (identical(table_prefix, "source")) src_name else table_prefix
          } else if (league == "nba") {
            prefix <- if (src_name == "nba_stats") "nba_stats" else paste("nba", src_name, sep = "_")
          }
          table_name <- if (!is.null(prefix) && nzchar(prefix)) paste(prefix, nm, sep = "_") else nm
          if (mode == "append" && DBI::dbExistsTable(con, table_name)) {
            existing_cols <- DBI::dbListFields(con, table_name)
            existing_lower <- tolower(existing_cols)
            incoming_lower <- tolower(names(tbl))
            new_cols <- names(tbl)[!(incoming_lower %in% existing_lower)]
            missing_cols <- existing_cols[!(existing_lower %in% incoming_lower)]

            if (length(new_cols) > 0) {
              for (col in new_cols) {
                DBI::dbExecute(
                  con,
                  sprintf("ALTER TABLE %s ADD COLUMN %s", DBI::dbQuoteIdentifier(con, table_name), DBI::dbQuoteIdentifier(con, col))
                )
              }
            }

            if (length(missing_cols) > 0) {
              for (col in missing_cols) {
                tbl[[col]] <- NA
              }
            }

            tbl <- tbl[, c(existing_cols, setdiff(names(tbl), existing_cols)), drop = FALSE]
          }

          DBI::dbWriteTable(con, table_name, tbl, append = (mode == "append"), overwrite = (mode == "overwrite"))
          if (isTRUE(debug)) {
            src_path <- if (!is.null(table_paths) && nm %in% names(table_paths)) table_paths[[nm]] else NA_character_
            message("write_sqlite_from_rds: wrote table=", table_name, " rows=", nrow(tbl), " path=", src_path)
          }
        }
      })

      key <- paste(src_name, season_label, sep = "/")
      results[[key]] <- list(
        source = src_name,
        season = season_label,
        rows = vapply(tables, function(x) if (is.data.frame(x)) nrow(x) else NA_integer_, integer(1))
      )
    }
  }

  results
}

#' Collect, parse, validate, and write data for multiple seasons
#'
#' @param league League identifier (currently only "nba" is supported).
#' @param source Source identifier (nba only: espn, nba_stats, or all).
#' @param seasons Vector of seasons (e.g., 2006:2025).
#' @param raw_dir Directory for raw data.
#' @param out_dir Output directory for rds/csv.
#' @param format Output format for write_tables().
#' @param db_path SQLite path for sqlite output.
#' @param mode Write mode for sqlite. Use `"append"` to add rows or `"overwrite"` to
#'   replace tables. `"upsert"` is deprecated (alias for `"overwrite"`).
#' @param skip_existing Skip writing rds/csv files if they already exist.
#' @param force Force re-download even if cached.
#' @param progress Show a progress bar while collecting/parsing.
#' @param quiet Suppress progress messages.
#' @param validate Validate tables after parsing.
#' @param keep_tables Keep parsed tables in the return list.
#' @param bundle Write bundled RDS after per-table writes.
#' @param debug Emit debug logging for each season/source.
#' @param log_path Optional CSV path to append run metadata.
#' @param season_type Optional season type (e.g., "regular", "playoffs").
#' @param ... Extra arguments passed to collect_raw() and parse_raw().
#' @return A list of season results with row counts and any errors.
#' @export
collect_parse_write <- function(league,
                                seasons,
                                source = NULL,
                                raw_dir = "data/raw",
                                out_dir = "data/parsed",
                                format = c("rds", "csv", "sqlite"),
                                db_path = NULL,
                                mode = c("append", "overwrite", "upsert"),
                                force = FALSE,
                                progress = TRUE,
                                quiet = FALSE,
                                validate = TRUE,
                                keep_tables = FALSE,
                                skip_existing = TRUE,
                                bundle = FALSE,
                                debug = FALSE,
                                log_path = NULL,
                                season_type = NULL,
                                ...) {
  format <- match.arg(format)
  mode <- match.arg(mode)
  if (mode == "upsert") {
    warning(
      "collect_parse_write(mode = 'upsert') is deprecated and currently behaves like mode = 'overwrite'. Use mode = 'overwrite' instead.",
      call. = FALSE
    )
    mode <- "overwrite"
  }
  validate <- isTRUE(validate)
  keep_tables <- isTRUE(keep_tables)
  skip_existing <- isTRUE(skip_existing)
  bundle <- isTRUE(bundle)
  progress <- isTRUE(progress)
  quiet <- isTRUE(quiet)
  force <- isTRUE(force)
  seasons <- as.integer(seasons)
  results <- vector("list", length(seasons))
  names(results) <- as.character(seasons)

  sources <- .resolve_sources(league, source)
  if (is.null(sources)) sources <- character()
  sources <- as.character(sources)
  if (length(sources) == 0) {
    stop("No sources resolved for league = ", league, " source = ", source, call. = FALSE)
  }

  dots <- list(...)
  parse_dots <- dots
  # Strip collect-only args from parse_raw/parse_raw_dir.
  parse_dots[c("workers", "rate_sleep", "max_retries", "shotchart_proxy", "force")] <- NULL
  if (!is.null(season_type)) {
    dots$season_type <- season_type
    parse_dots$season_type <- season_type
  }

  run_one_source <- function(src_name, season, season_label) {
    tryCatch({
      do.call(collect_raw, c(list(
        league = league,
        source = src_name,
        season = season,
        raw_dir = raw_dir,
        force = force,
        progress = progress,
        quiet = quiet
      ), dots))
      if (isTRUE(debug)) {
        message("collect_parse_write: collected raw for source=", src_name, " season=", season)
      }

      tables <- do.call(parse_raw, c(list(
        league = league,
        source = src_name,
        season = season,
        raw_dir = raw_dir,
        progress = progress,
        quiet = quiet
      ), parse_dots))

      if (isTRUE(debug)) {
        message(
          "collect_parse_write: keep_tables=",
          keep_tables,
          " keep_tables_type=",
          typeof(keep_tables),
          " tables_is_list=",
          is.list(tables),
          " tables_len=",
          length(tables)
        )
        message(
          "collect_parse_write: parsed tables=",
          length(tables),
          " names=",
          if (length(tables) > 0) paste(names(tables), collapse = ",") else "NONE"
        )
      }

      if (isTRUE(validate)) {
        validate_tables(tables, league, source = src_name)
      }

      season_out_dir <- out_dir
      bundle_name <- NULL
      table_prefix <- NULL
      if (format %in% c("rds", "csv")) {
        season_out_dir <- if (league == "nba") {
          file.path(out_dir, league, src_name, season_label)
        } else {
          file.path(out_dir, league, as.character(season))
        }
        stype_dir <- .normalize_season_type_dir(season_type)
        if (!is.null(stype_dir)) {
          season_out_dir <- file.path(season_out_dir, stype_dir)
        }

        if (isTRUE(bundle)) {
          bundle_name <- if (league == "nba") {
            if (src_name == "espn") "espn_all" else "nba_stats_all"
          } else {
            paste0(league, "_all")
          }
        }
      }

      if (format == "sqlite" && league == "nba") {
        table_prefix <- if (src_name == "nba_stats") "nba_stats" else paste("nba", src_name, sep = "_")
      }

      write_tables(
        tables = tables,
        format = format,
        out_dir = season_out_dir,
        db_path = db_path,
        mode = mode,
        skip_existing = skip_existing,
        bundle = bundle,
        bundle_name = bundle_name,
        table_prefix = table_prefix
      )
      if (isTRUE(debug)) {
        message("collect_parse_write: wrote tables to ", season_out_dir)
      }

      out <- list(
        source = src_name,
        rows = vapply(tables, function(x) if (is.data.frame(x)) nrow(x) else NA_integer_, integer(1)),
        tables = if (isTRUE(keep_tables)) tables else NULL,
        error = NULL
      )
      if (isTRUE(debug)) {
        message(
          "collect_parse_write: out$tables_len=",
          length(out$tables),
          " out$tables_is_null=",
          is.null(out$tables)
        )
        message("collect_parse_write: source=", src_name, " season=", season, " error=NULL")
      }
      out
    }, error = function(e) {
      err_txt <- conditionMessage(e)
      if (isTRUE(debug)) {
        message("collect_parse_write: source=", src_name, " season=", season, " error=", err_txt)
      }
      list(source = src_name, rows = NULL, tables = NULL, error = err_txt)
    })
  }

  for (i in seq_along(seasons)) {
    season <- seasons[[i]]
    season_label <- if (league == "nba" && exists(".nba_stats_season_string", mode = "function")) {
      .nba_stats_season_string(season)
    } else {
      as.character(season)
    }
    res <- list(season = season, error = NULL)
    source_results <- list()
    for (src_name in sources) {
      source_results[[src_name]] <- run_one_source(src_name, season, season_label)
    }
    if (length(source_results) > 0) {
      source_errors <- vapply(source_results, function(x) isTRUE(nzchar(x$error)), logical(1))
      if (any(source_errors)) {
        res$error <- source_results[[which(source_errors)[1]]]$error
      }
    }

    if (length(source_results) > 0) {
      res$sources <- source_results
    }
    if (length(sources) == 1 && length(source_results) > 0) {
      src_name <- sources[[1]]
      res$rows <- source_results[[src_name]]$rows
      if (isTRUE(keep_tables)) {
        res$tables <- source_results[[src_name]]$tables
      }
    }

    if (!is.null(log_path) && nzchar(log_path)) {
      errs <- vapply(source_results, function(x) if (is.null(x$error)) "" else x$error, character(1))
      bad <- nzchar(errs)
      if (any(bad)) {
        log_row <- data.frame(
          timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
          league = as.character(league),
          season = as.character(season),
          source = names(errs)[bad],
          error = unname(errs[bad]),
          stringsAsFactors = FALSE
        )
        dir.create(dirname(log_path), recursive = TRUE, showWarnings = FALSE)
        utils::write.table(
          log_row,
          log_path,
          sep = ",",
          row.names = FALSE,
          col.names = !file.exists(log_path),
          append = file.exists(log_path)
        )
      }
    }

    results[[i]] <- res
  }

  results
}
