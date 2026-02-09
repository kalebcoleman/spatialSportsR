# ---- Internal: SQLite declared type → Postgres type mapping ----

.sqlite_type_to_pg <- function(sqlite_type) {
  if (is.null(sqlite_type) || !nzchar(trimws(sqlite_type))) return("TEXT")
  st <- toupper(trimws(sqlite_type))
  if (grepl("^INT", st))    return("BIGINT")
  if (st == "REAL")          return("DOUBLE PRECISION")
  if (st == "TEXT")          return("TEXT")
  if (st == "BLOB")          return("BYTEA")
  if (st == "NUMERIC")       return("NUMERIC")
  "TEXT"
}

# ---- Internal: R column class → Postgres type mapping ----
# Preferred over .sqlite_type_to_pg because SQLite's declared types
# can be wrong (e.g. column declared INTEGER but storing text).

.r_class_to_pg <- function(r_class) {
  switch(r_class,
    integer   = "BIGINT",
    numeric   = "DOUBLE PRECISION",
    character = "TEXT",
    logical   = "BIGINT",
    Date      = "DATE",
    POSIXct   = "TIMESTAMPTZ",
    POSIXlt   = "TIMESTAMPTZ",
    raw       = "BYTEA",
    "TEXT"
  )
}

# ---- Internal: Default conflict keys per table ----

.default_conflict_keys <- function() {
  list(
    espn_games                        = "game_id",
    espn_events                       = c("game_id", "event_id"),
    espn_manifest                     = "game_id",
    espn_player_box                   = c("game_id", "player_id"),
    espn_team_box                     = c("game_id", "team_id"),
    espn_teams                        = c("team_id", "season_type"),
    nba_stats_games                   = "game_id",
    nba_stats_manifest                = "game_id",
    nba_stats_pbp                     = c("GAME_ID", "event_num"),
    nba_stats_shots                   = c("GAME_ID", "event_num"),
    nba_stats_player_box_traditional  = c("GAME_ID", "player_id"),
    nba_stats_player_box_advanced     = c("GAME_ID", "player_id"),
    nba_stats_player_box_fourfactors  = c("GAME_ID", "player_id"),
    nba_stats_player_box_usage        = c("GAME_ID", "player_id"),
    nba_stats_team_box_traditional    = c("GAME_ID", "team_id"),
    nba_stats_team_box_advanced       = c("GAME_ID", "team_id"),
    nba_stats_team_box_fourfactors    = c("GAME_ID", "team_id")
  )
}

# ---- Internal: Default table list ----

.default_sync_tables <- function() {
  names(.default_conflict_keys())
}

# ---- Internal: Check and fix Postgres column type mismatches ----

.pg_fix_column_types <- function(pg_con, table_name, col_info, sample_df) {
  if (is.null(sample_df) || nrow(sample_df) == 0) return(invisible(NULL))

  # Get actual Postgres column types
  pg_cols <- DBI::dbGetQuery(pg_con, paste0(
    "SELECT column_name, data_type FROM information_schema.columns ",
    "WHERE table_name = ", DBI::dbQuoteLiteral(pg_con, table_name),
    " AND column_name != '_synced_at'"
  ))

  for (i in seq_len(nrow(pg_cols))) {
    pg_col_name <- pg_cols$column_name[i]
    pg_type <- pg_cols$data_type[i]

    if (!(pg_col_name %in% names(sample_df))) next

    # Determine expected type from sample data
    expected_type <- .r_class_to_pg(class(sample_df[[pg_col_name]])[1])

    # Map Postgres type aliases to canonical forms for comparison
    pg_type_canonical <- switch(pg_type,
      boolean = "BOOLEAN",
      integer = "BIGINT",
      smallint = "BIGINT",
      bigint = "BIGINT",
      "double precision" = "DOUBLE PRECISION",
      numeric = "NUMERIC",
      text = "TEXT",
      character = "TEXT",
      bytea = "BYTEA",
      timestamp = "TIMESTAMPTZ",
      "timestamp without time zone" = "TIMESTAMPTZ",
      "timestamp with time zone" = "TIMESTAMPTZ",
      date = "DATE",
      pg_type
    )

    if (pg_type_canonical != expected_type) {
      # Type mismatch detected — alter the column
      tryCatch({
        alter_sql <- paste0(
          "ALTER TABLE ", DBI::dbQuoteIdentifier(pg_con, table_name),
          " ALTER COLUMN ", DBI::dbQuoteIdentifier(pg_con, pg_col_name),
          " TYPE ", expected_type, " USING ",
          DBI::dbQuoteIdentifier(pg_con, pg_col_name), "::", expected_type
        )
        DBI::dbExecute(pg_con, alter_sql, immediate = TRUE)
      }, error = function(e) {
        # Silent; the mismatch may be intentional or unfixable
        NULL
      })
    }
  }

  invisible(NULL)
}

# ---- Internal: Ensure Postgres table exists with correct schema ----

.pg_ensure_table <- function(pg_con, table_name, col_info, conflict_keys,
                             sample_df = NULL) {
  col_defs <- vapply(seq_len(nrow(col_info)), function(i) {
    col_name <- col_info$name[i]
    # Prefer R class from a sample read (accurate) over PRAGMA declared type
    # because SQLite's declared types can be wrong (e.g. INTEGER storing text).
    pg_type <- if (!is.null(sample_df) && col_name %in% names(sample_df)) {
      .r_class_to_pg(class(sample_df[[col_name]])[1])
    } else {
      .sqlite_type_to_pg(col_info$type[i])
    }
    paste(DBI::dbQuoteIdentifier(pg_con, col_name), pg_type)
  }, character(1))

  synced_col <- paste(DBI::dbQuoteIdentifier(pg_con, "_synced_at"),
                      "TIMESTAMPTZ DEFAULT now()")
  col_defs <- c(col_defs, synced_col)

  create_sql <- paste0(
    "CREATE TABLE IF NOT EXISTS ",
    DBI::dbQuoteIdentifier(pg_con, table_name),
    " (\n  ",
    paste(col_defs, collapse = ",\n  "),
    "\n)"
  )
  DBI::dbExecute(pg_con, create_sql, immediate = TRUE)

  # Fix any existing column type mismatches before inserting data
  .pg_fix_column_types(pg_con, table_name, col_info, sample_df)

  constraint_name <- paste0(table_name, "_upsert_key")
  key_cols <- paste(
    vapply(conflict_keys, function(k) DBI::dbQuoteIdentifier(pg_con, k), character(1)),
    collapse = ", "
  )

  # Check if constraint already exists before attempting ALTER TABLE.
  constraint_check <- DBI::dbGetQuery(pg_con, paste0(
    "SELECT 1 FROM information_schema.table_constraints ",
    "WHERE constraint_name = ", DBI::dbQuoteLiteral(pg_con, constraint_name),
    " AND table_name = ", DBI::dbQuoteLiteral(pg_con, table_name)
  ))
  if (nrow(constraint_check) == 0) {
    alter_sql <- paste0(
      "ALTER TABLE ", DBI::dbQuoteIdentifier(pg_con, table_name),
      " ADD CONSTRAINT ", DBI::dbQuoteIdentifier(pg_con, constraint_name),
      " UNIQUE (", key_cols, ")"
    )
    DBI::dbExecute(pg_con, alter_sql, immediate = TRUE)
  }

  existing_cols <- DBI::dbListFields(pg_con, table_name)
  existing_lower <- tolower(existing_cols)

  for (i in seq_len(nrow(col_info))) {
    col_name <- col_info$name[i]
    if (tolower(col_name) %in% existing_lower) next
    pg_type <- if (!is.null(sample_df) && col_name %in% names(sample_df)) {
      .r_class_to_pg(class(sample_df[[col_name]])[1])
    } else {
      .sqlite_type_to_pg(col_info$type[i])
    }
    add_sql <- paste0(
      "ALTER TABLE ", DBI::dbQuoteIdentifier(pg_con, table_name),
      " ADD COLUMN IF NOT EXISTS ",
      DBI::dbQuoteIdentifier(pg_con, col_name), " ", pg_type
    )
    DBI::dbExecute(pg_con, add_sql, immediate = TRUE)
  }

  invisible(TRUE)
}

# ---- Internal: Upsert from staging table ----

.pg_upsert_from_staging <- function(pg_con, target_table, staging_table,
                                    all_cols, conflict_keys) {
  quoted_target <- DBI::dbQuoteIdentifier(pg_con, target_table)
  quoted_staging <- DBI::dbQuoteIdentifier(pg_con, staging_table)
  quoted_cols <- vapply(all_cols, function(c) DBI::dbQuoteIdentifier(pg_con, c), character(1))
  quoted_keys <- vapply(conflict_keys, function(k) DBI::dbQuoteIdentifier(pg_con, k), character(1))

  update_cols <- setdiff(all_cols, conflict_keys)
  set_clause <- if (length(update_cols) > 0) {
    quoted_update <- vapply(update_cols, function(c) DBI::dbQuoteIdentifier(pg_con, c), character(1))
    sets <- paste0(quoted_update, " = EXCLUDED.", quoted_update)
    sets <- c(sets, "_synced_at = now()")
    paste(sets, collapse = ", ")
  } else {
    "_synced_at = now()"
  }

  sql <- paste0(
    "INSERT INTO ", quoted_target, " (",
    paste(quoted_cols, collapse = ", "),
    ")\nSELECT ",
    paste(quoted_cols, collapse = ", "),
    "\nFROM ", quoted_staging,
    "\nON CONFLICT (", paste(quoted_keys, collapse = ", "), ") DO UPDATE SET ",
    set_clause
  )
  DBI::dbExecute(pg_con, sql, immediate = TRUE)
}

# ---- Exported: Connect to Postgres ----

#' Connect to a PostgreSQL database
#'
#' @param host Hostname (default: env var `POSTGRES_HOST`).
#' @param port Port (default: env var `POSTGRES_PORT` or 5432).
#' @param dbname Database name (default: env var `POSTGRES_DB`).
#' @param user Username (default: env var `POSTGRES_USER`).
#' @param password Password (default: env var `POSTGRES_PASSWORD`).
#' @param sslmode SSL mode (default: `"require"`).
#' @return A DBI connection object.
#' @export
pg_connect <- function(host = NULL,
                       port = NULL,
                       dbname = NULL,
                       user = NULL,
                       password = NULL,
                       sslmode = "require") {
  if (!requireNamespace("RPostgres", quietly = TRUE)) {
    stop("Install the RPostgres package to use Postgres sync", call. = FALSE)
  }

  if (is.null(host))     host     <- Sys.getenv("POSTGRES_HOST", unset = "")
  if (is.null(port))     port     <- Sys.getenv("POSTGRES_PORT", unset = "5432")
  if (is.null(dbname))   dbname   <- Sys.getenv("POSTGRES_DB", unset = "")
  if (is.null(user))     user     <- Sys.getenv("POSTGRES_USER", unset = "")
  if (is.null(password)) password <- Sys.getenv("POSTGRES_PASSWORD", unset = "")

  if (!nzchar(host)) stop("Postgres host is required (set POSTGRES_HOST)", call. = FALSE)
  if (!nzchar(dbname)) stop("Postgres dbname is required (set POSTGRES_DB)", call. = FALSE)
  if (!nzchar(user)) stop("Postgres user is required (set POSTGRES_USER)", call. = FALSE)

  DBI::dbConnect(
    RPostgres::Postgres(),
    host     = host,
    port     = as.integer(port),
    dbname   = dbname,
    user     = user,
    password = password,
    sslmode  = sslmode
  )
}

# ---- Exported: Sync SQLite → Postgres ----

#' Sync tables from SQLite to PostgreSQL via chunked UPSERT
#'
#' Reads tables from a SQLite database and upserts them into PostgreSQL.
#' Creates tables in Postgres if they don't exist, adds new columns if the
#' schema has evolved, and never drops tables or existing indexes.
#'
#' @param sqlite_path Path to the SQLite database file.
#' @param pg_con A DBI connection to PostgreSQL (or `NULL` to auto-connect via
#'   environment variables).
#' @param tables Character vector of table names to sync. Defaults to the 17
#'   core pipeline tables.
#' @param conflict_keys Named list mapping table names to character vectors of
#'   conflict (unique key) columns. Defaults to `.default_conflict_keys()`.
#' @param seasons Optional character/integer vector of season values to filter
#'   on. `NULL` means sync all rows (full backfill).
#' @param season_type Optional season type filter (e.g., `"regular"`,
#'   `"Regular Season"`).
#' @param chunk_size Number of rows to read/write per batch (default 50000).
#' @param verbose Print progress messages (default `TRUE`).
#' @return A data frame of sync results (table, rows_synced, elapsed_secs,
#'   error).
#' @export
sync_sqlite_to_postgres <- function(sqlite_path = NULL,
                                    pg_con = NULL,
                                    tables = NULL,
                                    conflict_keys = NULL,
                                    seasons = NULL,
                                    season_type = NULL,
                                    chunk_size = 50000L,
                                    verbose = TRUE) {
  if (!requireNamespace("DBI", quietly = TRUE) ||
      !requireNamespace("RSQLite", quietly = TRUE)) {
    stop("Install DBI and RSQLite packages", call. = FALSE)
  }

  if (is.null(sqlite_path)) {
    sqlite_path <- Sys.getenv("SPATIALSPORTSR_DB_PATH", unset = "data/parsed/nba.sqlite")
  }
  if (!file.exists(sqlite_path)) {
    stop("SQLite database not found: ", sqlite_path, call. = FALSE)
  }

  own_pg <- FALSE
  if (is.null(pg_con)) {
    pg_con <- pg_connect()
    own_pg <- TRUE
  }
  on.exit(if (isTRUE(own_pg)) DBI::dbDisconnect(pg_con), add = TRUE)

  sl_con <- DBI::dbConnect(RSQLite::SQLite(), sqlite_path)
  on.exit(DBI::dbDisconnect(sl_con), add = TRUE)

  if (is.null(tables)) tables <- .default_sync_tables()
  if (is.null(conflict_keys)) conflict_keys <- .default_conflict_keys()
  chunk_size <- as.integer(chunk_size)

  sl_tables <- DBI::dbListTables(sl_con)
  tables <- intersect(tables, sl_tables)
  if (length(tables) == 0) {
    if (isTRUE(verbose)) message("No matching tables found in SQLite.")
    return(invisible(data.frame(
      table = character(), rows_synced = integer(),
      elapsed_secs = numeric(), error = character(),
      stringsAsFactors = FALSE
    )))
  }

  results <- vector("list", length(tables))

  for (idx in seq_along(tables)) {
    tbl_name <- tables[idx]
    t_start <- proc.time()[["elapsed"]]
    total_synced <- 0L
    err_msg <- ""

    tryCatch({
      col_info <- DBI::dbGetQuery(sl_con, paste0("PRAGMA table_info(", DBI::dbQuoteIdentifier(sl_con, tbl_name), ")"))

      keys <- conflict_keys[[tbl_name]]
      if (is.null(keys) || length(keys) == 0) {
        if (isTRUE(verbose)) message("  Skipping ", tbl_name, ": no conflict keys defined")
        err_msg <- "no conflict keys"
      } else {

        missing_keys <- setdiff(keys, col_info$name)
        if (length(missing_keys) > 0) {
          if (isTRUE(verbose)) {
            message("  Skipping ", tbl_name, ": conflict key columns not in table: ",
                    paste(missing_keys, collapse = ", "))
          }
          err_msg <- paste("missing key columns:", paste(missing_keys, collapse = ", "))
        } else {

          if (isTRUE(verbose)) message("Syncing ", tbl_name, "...")

          # Sample a row to get accurate R types for Postgres column mapping.
          # SQLite PRAGMA types can be wrong (e.g. INTEGER column storing text).
          sample_df <- DBI::dbGetQuery(
            sl_con,
            paste0("SELECT * FROM ",
                   DBI::dbQuoteIdentifier(sl_con, tbl_name), " LIMIT 1")
          )

          .pg_ensure_table(pg_con, tbl_name, col_info, keys,
                            sample_df = sample_df)

          all_cols <- col_info$name

          where_parts <- character()
          params <- list()
          sl_cols <- col_info$name
          has_season <- "season" %in% sl_cols
          has_season_type <- "season_type" %in% sl_cols

          if (!is.null(seasons) && has_season) {
            placeholders <- paste(rep("?", length(seasons)), collapse = ", ")
            where_parts <- c(where_parts, paste0("season IN (", placeholders, ")"))
            params <- c(params, as.list(as.character(seasons)))
          }
          if (!is.null(season_type) && has_season_type) {
            where_parts <- c(where_parts, "season_type = ?")
            params <- c(params, as.character(season_type))
          }

          where_clause <- if (length(where_parts) > 0) {
            paste(" WHERE", paste(where_parts, collapse = " AND "))
          } else {
            ""
          }

          count_sql <- paste0(
            "SELECT COUNT(*) AS n FROM ",
            DBI::dbQuoteIdentifier(sl_con, tbl_name),
            where_clause
          )
          total_rows <- if (length(params) > 0) {
            DBI::dbGetQuery(sl_con, count_sql, params = params)$n
          } else {
            DBI::dbGetQuery(sl_con, count_sql)$n
          }

          if (total_rows == 0) {
            if (isTRUE(verbose)) message("  ", tbl_name, ": 0 rows to sync")
          } else {

            if (isTRUE(verbose)) message("  ", tbl_name, ": ", total_rows, " rows to sync")

            n_chunks <- ceiling(total_rows / chunk_size)
            quoted_sl_cols <- paste(
              vapply(all_cols, function(c) DBI::dbQuoteIdentifier(sl_con, c), character(1)),
              collapse = ", "
            )

            for (chunk_i in seq_len(n_chunks)) {
              offset <- (chunk_i - 1L) * chunk_size
              select_sql <- paste0(
                "SELECT ", quoted_sl_cols,
                " FROM ", DBI::dbQuoteIdentifier(sl_con, tbl_name),
                where_clause,
                " LIMIT ", chunk_size, " OFFSET ", offset
              )
              chunk_df <- if (length(params) > 0) {
                DBI::dbGetQuery(sl_con, select_sql, params = params)
              } else {
                DBI::dbGetQuery(sl_con, select_sql)
              }
              if (nrow(chunk_df) == 0) next

              # Deduplicate by conflict keys within the chunk. Postgres raises
              # "ON CONFLICT DO UPDATE cannot affect row a second time" if two
              # rows in one INSERT share the same key. Keep last occurrence
              # (most recent data wins).
              dup_idx <- duplicated(chunk_df[, keys, drop = FALSE], fromLast = TRUE)
              if (any(dup_idx)) {
                chunk_df <- chunk_df[!dup_idx, , drop = FALSE]
              }

              # Convert logical columns to integer. SQLite has no real
              # BOOLEAN type — R sometimes reads 0/1 integers as logical
              # from small samples, which creates type mismatches.
              logical_cols <- vapply(chunk_df, is.logical, logical(1))
              if (any(logical_cols)) {
                for (lc in names(which(logical_cols))) {
                  chunk_df[[lc]] <- as.integer(chunk_df[[lc]])
                }
              }

              staging_name <- paste0("_staging_", tbl_name)

              DBI::dbWithTransaction(pg_con, {
                if (DBI::dbExistsTable(pg_con, staging_name)) {
                  DBI::dbRemoveTable(pg_con, staging_name)
                }
                DBI::dbWriteTable(pg_con, staging_name, chunk_df,
                                  temporary = TRUE, overwrite = TRUE)
                .pg_upsert_from_staging(pg_con, tbl_name, staging_name,
                                        all_cols, keys)
                DBI::dbRemoveTable(pg_con, staging_name)
              })

              total_synced <- total_synced + nrow(chunk_df)
              if (isTRUE(verbose) && n_chunks > 1) {
                message("  ", tbl_name, ": chunk ", chunk_i, "/", n_chunks,
                        " (", total_synced, "/", total_rows, " rows)")
              }
            }

            if (isTRUE(verbose)) {
              elapsed <- round(proc.time()[["elapsed"]] - t_start, 1)
              message("  ", tbl_name, ": done (", total_synced, " rows, ", elapsed, "s)")
            }
          }
        }
      }
    }, error = function(e) {
      err_msg <<- conditionMessage(e)
      if (isTRUE(verbose)) message("  ERROR syncing ", tbl_name, ": ", err_msg)
    })

    elapsed <- round(proc.time()[["elapsed"]] - t_start, 1)
    results[[idx]] <- data.frame(
      table = tbl_name,
      rows_synced = total_synced,
      elapsed_secs = elapsed,
      error = err_msg,
      stringsAsFactors = FALSE
    )
  }

  out <- do.call(rbind, results[!vapply(results, is.null, logical(1))])
  if (is.null(out)) {
    out <- data.frame(
      table = character(), rows_synced = integer(),
      elapsed_secs = numeric(), error = character(),
      stringsAsFactors = FALSE
    )
  }

  if (isTRUE(verbose)) {
    total <- sum(out$rows_synced, na.rm = TRUE)
    errors <- sum(nzchar(out$error), na.rm = TRUE)
    message("Sync complete: ", total, " rows across ", nrow(out), " tables",
            if (errors > 0) paste0(" (", errors, " errors)") else "")
  }

  invisible(out)
}

# ---- Exported: Convenience wrapper for current season ----

#' Sync the current season from SQLite to PostgreSQL
#'
#' A convenience wrapper around [sync_sqlite_to_postgres()] that filters to a
#' single season.
#'
#' @inheritParams sync_sqlite_to_postgres
#' @param season Season identifier (e.g., `"2025-26"` or `2026`).
#' @param season_type Season type filter (e.g., `"regular"`).
#' @return A data frame of sync results (invisibly).
#' @export
sync_current_season <- function(sqlite_path = NULL,
                                pg_con = NULL,
                                season,
                                season_type = NULL,
                                tables = NULL,
                                conflict_keys = NULL,
                                chunk_size = 50000L,
                                verbose = TRUE) {
  sync_sqlite_to_postgres(
    sqlite_path   = sqlite_path,
    pg_con        = pg_con,
    tables        = tables,
    conflict_keys = conflict_keys,
    seasons       = as.character(season),
    season_type   = season_type,
    chunk_size    = chunk_size,
    verbose       = verbose
  )
}
