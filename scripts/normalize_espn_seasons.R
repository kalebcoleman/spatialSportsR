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
root_dir <- normalize_path(repo_root, get_arg(opts, "root-dir", "data/parsed"))
db_path <- normalize_path(repo_root, get_arg(opts, "db-path", "data/parsed/nba.sqlite"))
apply_db <- to_bool(get_arg(opts, "apply-db", "true"), default = TRUE)
apply_rds <- to_bool(get_arg(opts, "apply-rds", "true"), default = TRUE)

required_pkgs <- c("DBI", "RSQLite", "jsonlite")
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

normalize_season_value <- function(x) {
  if (is.na(x)) return(NA_character_)
  x <- as.character(x)
  if (!nzchar(x)) return(x)
  if (exists(".nba_stats_season_string", mode = "function")) {
    return(.nba_stats_season_string(x))
  }
  if (grepl("^\\d{4}$", x)) {
    end_year <- suppressWarnings(as.integer(x))
    if (is.na(end_year)) return(x)
    start_year <- end_year - 1L
    return(sprintf("%d-%02d", start_year, end_year %% 100))
  }
  x
}

normalize_df_season <- function(df) {
  if (!is.data.frame(df) || !"season" %in% names(df)) return(df)
  df$season <- vapply(df$season, normalize_season_value, character(1))
  df
}

if (isTRUE(apply_rds)) {
  espn_root <- file.path(root_dir, "nba", "espn")
  if (dir.exists(espn_root)) {
    rds_paths <- list.files(espn_root, pattern = "\\.rds$", full.names = TRUE, recursive = TRUE)
    rds_paths <- rds_paths[!grepl("_all\\.rds$", rds_paths)]
    bundle_paths <- list.files(espn_root, pattern = "_all\\.rds$", full.names = TRUE, recursive = TRUE)
    all_paths <- c(rds_paths, bundle_paths)

    for (path in all_paths) {
      obj <- readRDS(path)
      if (is.data.frame(obj)) {
        obj <- normalize_df_season(obj)
        saveRDS(obj, path)
      } else if (is.list(obj)) {
        names_obj <- names(obj)
        obj <- lapply(obj, function(x) {
          if (is.data.frame(x)) normalize_df_season(x) else x
        })
        if (!is.null(names_obj)) names(obj) <- names_obj
        saveRDS(obj, path)
      }
    }
  }
}

if (isTRUE(apply_db)) {
  if (!file.exists(db_path)) {
    stop("SQLite DB not found: ", db_path, call. = FALSE)
  }
  con <- DBI::dbConnect(RSQLite::SQLite(), db_path)
  on.exit(DBI::dbDisconnect(con), add = TRUE)
  tables <- DBI::dbGetQuery(
    con,
    "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'espn_%' ORDER BY name"
  )
  if (nrow(tables) > 0) {
    for (tbl in tables$name) {
      cols <- DBI::dbListFields(con, tbl)
      if (!"season" %in% cols) next
      seasons <- DBI::dbGetQuery(con, paste0("SELECT DISTINCT season FROM ", DBI::dbQuoteIdentifier(con, tbl)))
      if (!"season" %in% names(seasons) || nrow(seasons) == 0) next
      for (val in seasons$season) {
        normalized <- normalize_season_value(val)
        if (is.na(normalized) || identical(as.character(val), normalized)) next
        DBI::dbExecute(
          con,
          paste0("UPDATE ", DBI::dbQuoteIdentifier(con, tbl), " SET season = ? WHERE season = ?"),
          params = list(normalized, val)
        )
      }
    }
  }
}

message("ESPN season normalization complete.")
