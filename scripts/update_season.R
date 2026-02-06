#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

parse_args <- function(args) {
  out <- list()
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (startsWith(arg, "--")) {
      if (grepl("=", arg, fixed = TRUE)) {
        parts <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
        key <- parts[[1]]
        val <- paste(parts[-1], collapse = "=")
      } else {
        key <- sub("^--", "", arg)
        val <- "true"
        if (i < length(args) && !startsWith(args[[i + 1L]], "--")) {
          val <- args[[i + 1L]]
          i <- i + 1L
        }
      }
      out[[key]] <- val
    }
    i <- i + 1L
  }
  out
}

get_arg <- function(opts, key, default = NULL) {
  if (!is.null(opts[[key]])) opts[[key]] else default
}

to_bool <- function(value, default = FALSE) {
  if (is.null(value)) return(default)
  if (is.logical(value)) return(value)
  txt <- tolower(trimws(as.character(value)))
  if (!nzchar(txt)) return(default)
  txt %in% c("1", "true", "t", "yes", "y", "on")
}

split_csv <- function(value) {
  if (is.null(value)) return(character())
  parts <- unlist(strsplit(as.character(value), ","))
  parts <- trimws(parts)
  parts[nzchar(parts)]
}

normalize_season_type <- function(value) {
  if (is.null(value) || !nzchar(as.character(value))) return("regular")
  stype <- tolower(trimws(as.character(value)))
  if (stype %in% c("regular", "regular season", "reg")) return("regular")
  if (stype %in% c("postseason", "playoffs", "playoff")) return("playoffs")
  stype
}

normalize_rate_sleep <- function(value, default = c(0.5, 1.2)) {
  if (is.null(value) || !nzchar(as.character(value))) return(default)
  parts <- suppressWarnings(as.numeric(unlist(strsplit(as.character(value), "[, ]+"))))
  parts <- parts[!is.na(parts)]
  if (length(parts) >= 2) return(parts[1:2])
  default
}

normalize_path <- function(repo_root, path) {
  if (is.null(path) || !nzchar(as.character(path))) return(path)
  if (grepl("^/", path)) return(path)
  file.path(repo_root, path)
}

repo_root <- normalizePath(getwd())
renv_path <- file.path(repo_root, "renv", "activate.R")
if (file.exists(renv_path)) {
  source(renv_path)
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))

season_input <- get_arg(opts, "season", "2026")
season_type_arg <- get_arg(opts, "season-type", "regular")
sources <- tolower(split_csv(get_arg(opts, "sources", "espn,nba_stats")))
backfill_days <- suppressWarnings(as.integer(get_arg(opts, "backfill-days", "3")))
db_path <- normalize_path(repo_root, get_arg(opts, "db-path", "data/parsed/nba.sqlite"))
raw_dir <- normalize_path(repo_root, get_arg(opts, "raw-dir", "data/raw"))
out_dir <- normalize_path(repo_root, get_arg(opts, "out-dir", "data/parsed"))
workers <- suppressWarnings(as.integer(get_arg(opts, "workers", "1")))
rate_sleep <- normalize_rate_sleep(get_arg(opts, "rate-sleep", "0.5,1.2"))
force_shotchart <- to_bool(get_arg(opts, "force-shotchart", "true"), default = TRUE)
force_espn <- to_bool(get_arg(opts, "force-espn", "true"), default = TRUE)
force_nba_stats_index <- to_bool(get_arg(opts, "force-nba-stats-index", "true"), default = TRUE)

if (is.na(backfill_days) || backfill_days < 0) backfill_days <- 0L
if (is.na(workers) || workers < 1L) workers <- 1L

required_pkgs <- c("DBI", "RSQLite", "jsonlite", "httr")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Install required packages: ", paste(missing_pkgs, collapse = ", "), call. = FALSE)
}

if (requireNamespace("devtools", quietly = TRUE)) {
  devtools::load_all(repo_root, quiet = TRUE)
} else {
  r_dir <- file.path(repo_root, "R")
  r_files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)
  for (path in r_files) {
    sys.source(path, envir = .GlobalEnv)
  }
}

season_label <- .nba_stats_season_string(season_input)
season_years <- .nba_stats_season_years(season_label)
season_end <- season_years[2]
if (is.na(season_end)) {
  season_end <- suppressWarnings(as.integer(season_input))
}
if (is.na(season_end)) {
  stop("Unable to parse season from input: ", season_input, call. = FALSE)
}

season_type_norm <- normalize_season_type(season_type_arg)
espn_season_type <- if (season_type_norm == "playoffs") "postseason" else "regular"
nba_stats_season_type <- if (season_type_norm == "playoffs") "Playoffs" else "Regular Season"

log_path <- file.path(repo_root, "logs", "update_season_runs.csv")
dir.create(dirname(log_path), recursive = TRUE, showWarnings = FALSE)

log_run <- function(source, rows, error = NULL) {
  entry <- data.frame(
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    source = as.character(source),
    season = as.character(season_label),
    rows = as.integer(rows),
    error = if (is.null(error)) "" else as.character(error),
    stringsAsFactors = FALSE
  )
  utils::write.table(
    entry,
    log_path,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(log_path),
    append = file.exists(log_path),
    quote = TRUE
  )
}

log_info <- function(...) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  message(sprintf("[%s] %s", ts, paste0(..., collapse = "")))
}

normalize_datetime_cols <- function(tbl) {
  for (col in names(tbl)) {
    if (inherits(tbl[[col]], "Date")) {
      tbl[[col]] <- format(tbl[[col]], "%Y-%m-%d")
    } else if (inherits(tbl[[col]], c("POSIXct", "POSIXt"))) {
      tbl[[col]] <- format(tbl[[col]], "%Y-%m-%d %H:%M:%S")
    }
  }
  tbl
}

replace_season_sqlite <- function(tables, db_path, table_prefix, season_values, season_type_norm) {
  con <- DBI::dbConnect(RSQLite::SQLite(), db_path)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  DBI::dbWithTransaction(con, {
    for (nm in names(tables)) {
      tbl <- tables[[nm]]
      if (is.null(tbl) || !is.data.frame(tbl)) next

      if (any(duplicated(tolower(names(tbl))))) {
        keep_idx <- !duplicated(tolower(names(tbl)))
        tbl <- tbl[, keep_idx, drop = FALSE]
      }

      tbl <- normalize_datetime_cols(tbl)

      table_name <- if (!is.null(table_prefix) && nzchar(table_prefix)) {
        paste(table_prefix, nm, sep = "_")
      } else {
        nm
      }

      if (!DBI::dbExistsTable(con, table_name)) {
        DBI::dbWriteTable(con, table_name, tbl, overwrite = TRUE)
        next
      }

      existing_cols <- DBI::dbListFields(con, table_name)
      has_season <- "season" %in% existing_cols
      has_season_type <- "season_type" %in% existing_cols

      if (has_season || has_season_type) {
        where <- character()
        params <- list()
        if (has_season) {
          if (length(season_values) > 1) {
            placeholders <- paste(rep("?", length(season_values)), collapse = ", ")
            where <- c(where, paste0("season IN (", placeholders, ")"))
            params <- c(params, as.list(season_values))
          } else {
            where <- c(where, "season = ?")
            params <- c(params, season_values[[1]])
          }
        }
        if (has_season_type && !is.null(season_type_norm)) {
          where <- c(where, "season_type = ?")
          params <- c(params, season_type_norm)
        }
        delete_sql <- paste0(
          "DELETE FROM ",
          DBI::dbQuoteIdentifier(con, table_name),
          " WHERE ",
          paste(where, collapse = " AND ")
        )
        DBI::dbExecute(con, delete_sql, params = params)
      }

      incoming_cols <- names(tbl)
      new_cols <- setdiff(incoming_cols, existing_cols)
      if (length(new_cols) > 0) {
        for (col in new_cols) {
          add_sql <- paste0(
            "ALTER TABLE ",
            DBI::dbQuoteIdentifier(con, table_name),
            " ADD COLUMN ",
            DBI::dbQuoteIdentifier(con, col)
          )
          DBI::dbExecute(con, add_sql)
        }
      }

      missing_cols <- setdiff(existing_cols, incoming_cols)
      if (length(missing_cols) > 0) {
        for (col in missing_cols) {
          tbl[[col]] <- NA
        }
      }

      tbl <- tbl[, c(existing_cols, setdiff(names(tbl), existing_cols)), drop = FALSE]
      DBI::dbWriteTable(con, table_name, tbl, append = TRUE)
    }
  })
}

get_espn_max_date <- function(db_path, season_values, season_type_norm) {
  if (!file.exists(db_path)) return(NA)
  con <- DBI::dbConnect(RSQLite::SQLite(), db_path)
  on.exit(DBI::dbDisconnect(con), add = TRUE)
  if (!DBI::dbExistsTable(con, "espn_games")) return(NA)
  placeholders <- paste(rep("?", length(season_values)), collapse = ", ")
  query <- paste0(
    "SELECT MAX(game_date) AS max_date FROM espn_games WHERE season IN (",
    placeholders,
    ") AND season_type = ?"
  )
  params <- c(as.list(season_values), season_type_norm)
  res <- tryCatch(DBI::dbGetQuery(con, query, params = params), error = function(e) NULL)
  if (is.null(res) || nrow(res) == 0) return(NA)
  max_date <- res$max_date[[1]]
  if (is.null(max_date) || is.na(max_date) || !nzchar(as.character(max_date))) return(NA)
  max_date <- sub("T.*$", "", as.character(max_date))
  as.Date(max_date)
}

normalize_espn_tables <- function(tables, season_label) {
  for (nm in names(tables)) {
    tbl <- tables[[nm]]
    if (is.null(tbl) || !is.data.frame(tbl)) next
    if ("season" %in% names(tbl)) {
      tbl$season <- rep(season_label, nrow(tbl))
    }
    tables[[nm]] <- tbl
  }
  tables
}

log_info("Update season: ", season_label, " (", season_type_norm, ")")
log_info("Sources: ", paste(sources, collapse = ", "))
log_info("DB: ", db_path)
log_info("Raw dir: ", raw_dir)
log_info("Parsed dir: ", out_dir)

if ("espn" %in% sources) {
  log_info("ESPN update starting...")
  tryCatch({
    today <- Sys.Date()
    max_date <- get_espn_max_date(db_path, list(season_label, season_end), season_type_norm)
    if (!is.na(max_date) && max_date > today) {
      log_info("ESPN max game_date is in the future (", format(max_date), "); clamping to today (", format(today), ").")
      max_date <- today
    }
    if (is.na(max_date)) {
      date_from <- today - backfill_days
    } else {
      date_from <- max_date - backfill_days
    }
    date_to <- today
    if (date_from > date_to) {
      date_from <- date_to - backfill_days
      if (date_from > date_to) date_from <- date_to
    }

    log_info("ESPN backfill window: ", format(date_from), " to ", format(date_to))
    espn_collect <- collect_raw(
      league = "nba",
      source = "espn",
      season = season_end,
      raw_dir = raw_dir,
      force = force_espn,
      progress = FALSE,
      quiet = TRUE,
      season_type = espn_season_type,
      date_from = date_from,
      date_to = date_to
    )

    games_count <- if (is.list(espn_collect) && !is.null(espn_collect$games)) nrow(espn_collect$games) else NA_integer_
    raw_count <- if (is.list(espn_collect) && !is.null(espn_collect$raw_paths)) sum(!is.na(espn_collect$raw_paths)) else NA_integer_
    log_info("ESPN collect complete: games=", games_count, " raw_files=", raw_count)

    tables_espn <- parse_raw(
      league = "nba",
      source = "espn",
      season = season_end,
      raw_dir = raw_dir,
      progress = FALSE,
      quiet = TRUE,
      season_type = espn_season_type
    )
    tables_espn <- normalize_espn_tables(tables_espn, season_label)
    log_info(
      "ESPN parse complete: games=",
      if (!is.null(tables_espn$games)) nrow(tables_espn$games) else 0L,
      " events=",
      if (!is.null(tables_espn$events)) nrow(tables_espn$events) else 0L,
      " teams=",
      if (!is.null(tables_espn$teams)) nrow(tables_espn$teams) else 0L
    )

    espn_out_dir <- file.path(out_dir, "nba", "espn", season_label)
    write_tables(
      tables = tables_espn,
      format = "rds",
      out_dir = espn_out_dir,
      skip_existing = FALSE,
      bundle = TRUE,
      bundle_name = "espn_all"
    )

    replace_season_sqlite(
      tables = tables_espn,
      db_path = db_path,
      table_prefix = "espn",
      season_values = list(season_label, season_end),
      season_type_norm = season_type_norm
    )

    total_rows <- sum(vapply(tables_espn, function(x) if (is.data.frame(x)) nrow(x) else 0L, integer(1)))
    log_run("espn", total_rows, error = NULL)
    log_info("ESPN update complete.")
  }, error = function(e) {
    log_run("espn", 0L, error = conditionMessage(e))
    stop(e)
  })
}

if ("nba_stats" %in% sources) {
  log_info("NBA Stats update starting...")
  tryCatch({
    if (isTRUE(force_nba_stats_index)) {
      log_info("Refreshing NBA Stats index...")
      nba_index <- nba_stats_collect_game_index(
        season = season_end,
        raw_dir = .raw_league_dir(
          raw_dir = raw_dir,
          league = "nba",
          season = season_end,
          source = "nba_stats",
          season_type = season_type_norm
        ),
        force = TRUE,
        season_type = nba_stats_season_type,
        save_index = TRUE
      )
      if (is.list(nba_index) && !is.null(nba_index$games)) {
        log_info("NBA Stats index games: ", nrow(nba_index$games))
      }
    }

    nba_collect <- collect_raw(
      league = "nba",
      source = "nba_stats",
      season = season_end,
      raw_dir = raw_dir,
      force = FALSE,
      progress = FALSE,
      quiet = TRUE,
      season_type = nba_stats_season_type,
      workers = workers,
      rate_sleep = rate_sleep
    )
    nba_games_count <- if (is.list(nba_collect) && !is.null(nba_collect$games)) nrow(nba_collect$games) else NA_integer_
    nba_raw_count <- if (is.list(nba_collect) && !is.null(nba_collect$raw_paths)) sum(!is.na(nba_collect$raw_paths)) else NA_integer_
    log_info("NBA Stats collect complete: games=", nba_games_count, " raw_files=", nba_raw_count)

    if (isTRUE(force_shotchart)) {
      nba_stats_raw_dir <- .raw_league_dir(
        raw_dir = raw_dir,
        league = "nba",
        season = season_end,
        source = "nba_stats",
        season_type = season_type_norm
      )
      log_info("Refreshing NBA Stats shotchart...")
      collect_nba_shotchart(
        season = season_label,
        season_type = nba_stats_season_type,
        raw_dir = nba_stats_raw_dir,
        force = TRUE,
        chunk = "month"
      )
    }

    tables_nba <- parse_raw(
      league = "nba",
      source = "nba_stats",
      season = season_end,
      raw_dir = raw_dir,
      progress = FALSE,
      quiet = TRUE,
      season_type = nba_stats_season_type
    )
    log_info(
      "NBA Stats parse complete: games=",
      if (!is.null(tables_nba$games)) nrow(tables_nba$games) else 0L,
      " shots=",
      if (!is.null(tables_nba$shots)) nrow(tables_nba$shots) else 0L,
      " pbp=",
      if (!is.null(tables_nba$pbp)) nrow(tables_nba$pbp) else 0L
    )

    nba_out_dir <- file.path(out_dir, "nba", "nba_stats", season_label)
    write_tables(
      tables = tables_nba,
      format = "rds",
      out_dir = nba_out_dir,
      skip_existing = FALSE,
      bundle = TRUE,
      bundle_name = "nba_stats_all"
    )

    replace_season_sqlite(
      tables = tables_nba,
      db_path = db_path,
      table_prefix = "nba_stats",
      season_values = list(season_label),
      season_type_norm = season_type_norm
    )

    total_rows <- sum(vapply(tables_nba, function(x) if (is.data.frame(x)) nrow(x) else 0L, integer(1)))
    log_run("nba_stats", total_rows, error = NULL)
    log_info("NBA Stats update complete.")
  }, error = function(e) {
    log_run("nba_stats", 0L, error = conditionMessage(e))
    stop(e)
  })
}

log_info("Update finished.")
