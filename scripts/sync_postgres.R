#!/usr/bin/env Rscript

# Sync SQLite database to PostgreSQL.
#
# Usage:
#   Rscript scripts/sync_postgres.R --season=2026 --season-type=regular
#   Rscript scripts/sync_postgres.R --full
#   Rscript scripts/sync_postgres.R --full --chunk-size=100000
#
# Environment variables (used as defaults):
#   POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
#   SPATIALSPORTSR_DB_PATH  (SQLite path)

options(stringsAsFactors = FALSE)

# ---- Arg parsing (reuse pattern from update_season.R) ----

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

# ---- Setup ----

repo_root <- normalizePath(getwd())
renv_path <- file.path(repo_root, "renv", "activate.R")
if (file.exists(renv_path)) source(renv_path)

# Load .env file if present (so credentials are set once, not every run).
env_file <- file.path(repo_root, ".env")
if (file.exists(env_file)) {
  env_lines <- readLines(env_file, warn = FALSE)
  for (line in env_lines) {
    line <- trimws(line)
    if (!nzchar(line) || startsWith(line, "#")) next
    if (!grepl("=", line, fixed = TRUE)) next
    key <- trimws(sub("=.*", "", line))
    val <- trimws(sub("^[^=]+=", "", line))
    # Strip surrounding quotes
    val <- gsub("^[\"']|[\"']$", "", val)
    if (nzchar(key) && !nzchar(Sys.getenv(key, unset = ""))) {
      do.call(Sys.setenv, stats::setNames(list(val), key))
    }
  }
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))

full_sync    <- to_bool(get_arg(opts, "full", "false"), default = FALSE)
season_arg   <- get_arg(opts, "season", "2026")
stype_arg    <- get_arg(opts, "season-type", NULL)
sqlite_path  <- get_arg(opts, "sqlite-path",
                         Sys.getenv("SPATIALSPORTSR_DB_PATH",
                                    unset = file.path(repo_root, "data", "parsed", "nba.sqlite")))
pg_host      <- get_arg(opts, "pg-host", NULL)
pg_port      <- get_arg(opts, "pg-port", NULL)
pg_db        <- get_arg(opts, "pg-db", NULL)
pg_user      <- get_arg(opts, "pg-user", NULL)
pg_password  <- get_arg(opts, "pg-password", NULL)
tables_arg   <- split_csv(get_arg(opts, "tables", NULL))
chunk_size   <- suppressWarnings(as.integer(get_arg(opts, "chunk-size", "50000")))
if (is.na(chunk_size) || chunk_size < 1L) chunk_size <- 50000L

# ---- Load package ----

if (requireNamespace("devtools", quietly = TRUE)) {
  devtools::load_all(repo_root, quiet = TRUE)
} else {
  r_dir <- file.path(repo_root, "R")
  r_files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)
  for (path in r_files) {
    sys.source(path, envir = .GlobalEnv)
  }
}

# ---- Logging ----

log_path <- file.path(repo_root, "logs", "sync_postgres_runs.csv")
dir.create(dirname(log_path), recursive = TRUE, showWarnings = FALSE)

log_info <- function(...) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  message(sprintf("[%s] %s", ts, paste0(..., collapse = "")))
}

log_run <- function(tables_synced, total_rows, elapsed, error = NULL) {
  entry <- data.frame(
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    mode = if (isTRUE(full_sync)) "full" else "incremental",
    season = if (isTRUE(full_sync)) "all" else as.character(season_arg),
    tables_synced = as.integer(tables_synced),
    total_rows = as.integer(total_rows),
    elapsed_secs = round(elapsed, 1),
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

# ---- Connect ----

log_info("SQLite → Postgres sync starting...")
log_info("SQLite path: ", sqlite_path)
log_info("Mode: ", if (isTRUE(full_sync)) "full backfill" else paste0("incremental (season input=", season_arg, ")"))

pg_args <- list()
if (!is.null(pg_host)) pg_args$host <- pg_host
if (!is.null(pg_port)) pg_args$port <- pg_port
if (!is.null(pg_db)) pg_args$dbname <- pg_db
if (!is.null(pg_user)) pg_args$user <- pg_user
if (!is.null(pg_password)) pg_args$password <- pg_password

pg_con <- do.call(pg_connect, pg_args)
on.exit(DBI::dbDisconnect(pg_con), add = TRUE)

pg_display_host <- if (!is.null(pg_args$host)) pg_args$host else Sys.getenv("POSTGRES_HOST")
log_info("Connected to Postgres: ", pg_display_host)

# ---- Sync ----

t_start <- proc.time()[["elapsed"]]

sync_args <- list(
  sqlite_path = sqlite_path,
  pg_con      = pg_con,
  chunk_size  = chunk_size,
  verbose     = TRUE
)

if (length(tables_arg) > 0) {
  sync_args$tables <- tables_arg
}

if (!isTRUE(full_sync)) {
  # Normalize season to "YYYY-YY" format used in all tables.
  # Accept both "2025-26" and "2026" as input.
  season_label <- if (exists(".nba_stats_season_string", mode = "function")) {
    .nba_stats_season_string(season_arg)
  } else {
    as.character(season_arg)
  }
  sync_args$seasons <- season_label
  log_info("Season filter: ", season_label)
  if (!is.null(stype_arg)) {
    sync_args$season_type <- stype_arg
  }
}

result <- tryCatch({
  do.call(sync_sqlite_to_postgres, sync_args)
}, error = function(e) {
  log_info("FATAL: ", conditionMessage(e))
  log_run(0L, 0L, proc.time()[["elapsed"]] - t_start, error = conditionMessage(e))
  stop(e)
})

elapsed <- proc.time()[["elapsed"]] - t_start
total_rows <- sum(result$rows_synced, na.rm = TRUE)
n_errors <- sum(nzchar(result$error), na.rm = TRUE)

log_run(nrow(result), total_rows, elapsed,
        error = if (n_errors > 0) paste(n_errors, "table errors") else NULL)

log_info("Sync finished in ", round(elapsed, 1), "s: ",
         total_rows, " rows across ", nrow(result), " tables",
         if (n_errors > 0) paste0(" (", n_errors, " errors)") else "")
