#' HTTP helpers (rate limiting + basic caching hooks)

#' Download a file with simple caching
#'
#' @param url Source URL.
#' @param dest Destination path.
#' @param force Force download even if cached.
#' @param sleep Seconds to sleep for rate limiting.
#' @param ... Extra arguments (reserved).
#' @return The destination path.
#' @export
#' Download a file with simple caching
#'
#' @param url Source URL.
#' @param dest Destination path.
#' @param force Force download even if cached.
#' @param sleep Seconds to sleep for rate limiting.
#' @param ... Extra arguments (reserved).
#' @return The destination path.
#' @export
http_get <- function(url, dest, force = FALSE, sleep = 0.25, ...) {
  if (!force && file.exists(dest)) {
    return(dest)
  }

  if (!dir.exists(dirname(dest))) {
    dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  }

  Sys.sleep(sleep)

  if (requireNamespace("curl", quietly = TRUE)) {
    curl::curl_download(url, destfile = dest, quiet = TRUE)
  } else if (requireNamespace("httr2", quietly = TRUE)) {
    req <- httr2::request(url)
    httr2::req_perform(req, path = dest)
  } else {
    stop("Install curl or httr2 to enable downloads", call. = FALSE)
  }

  dest
}

# =========================
# NBA (ESPN) HTTP HELPERS
# =========================
# All ESPN NBA request/response helpers live below. Keep any NBA-only
# HTTP utilities in this section so league-specific helpers stay grouped.
# When adding new NBA endpoints, place them in this block.

check_status <- function(res) {
  if (!requireNamespace("httr", quietly = TRUE)) {
    stop("Install httr to use ESPN NBA helpers", call. = FALSE)
  }
  status <- httr::status_code(res)
  if (is.na(status) || status < 200 || status >= 300) {
    stop(sprintf("ESPN request failed with status %s", status), call. = FALSE)
  }
}

.espn_nba_request <- function(url, times = 5, pause_base = 1, pause_cap = 60) {
  if (!requireNamespace("httr", quietly = TRUE)) {
    stop("Install httr to use ESPN NBA helpers", call. = FALSE)
  }
  httr::RETRY(
    "GET",
    url,
    times = times,
    pause_base = pause_base,
    pause_cap = pause_cap,
    httr::add_headers(
      `User-Agent` = "Mozilla/5.0 (compatible; spatialSportsR)",
      Referer = "https://www.espn.com"
    )
  )
}

.espn_nba_get <- function(url, times = 5, pause_base = 1, pause_cap = 60) {
  res <- tryCatch(
    .espn_nba_request(url, times = times, pause_base = pause_base, pause_cap = pause_cap),
    error = function(e) NULL
  )
  if (!is.null(res)) {
    check_status(res)
    return(httr::content(res, as = "text", encoding = "UTF-8"))
  }

  if (requireNamespace("curl", quietly = TRUE)) {
    h <- curl::new_handle()
    curl::handle_setheaders(h, "user-agent" = "Mozilla/5.0 (compatible; spatialSportsR)", "referer" = "https://www.espn.com")
    resp <- tryCatch(curl::curl_fetch_memory(url, handle = h), error = function(e) NULL)
    if (!is.null(resp) && resp$status_code >= 200 && resp$status_code < 300) {
      return(rawToChar(resp$content))
    }
  }

  NULL
}

.espn_nba_summary_response_raw <- function(game_id, times = 5, pause_base = 1, pause_cap = 60) {
  summary_url <- "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?"
  full_url <- paste0(summary_url, "event=", game_id)
  .espn_nba_request(full_url, times = times, pause_base = pause_base, pause_cap = pause_cap)
}

.espn_nba_scoreboard_response <- function(dates) {
  base_urls <- c(
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "https://site.web.api.espn.com/apis/v2/sports/basketball/nba/scoreboard"
  )
  query <- paste0("limit=1000&dates=", dates)
  for (base_url in base_urls) {
    scoreboard_url <- paste0(base_url, "?", query)
    raw_text <- .espn_nba_get(scoreboard_url)
    if (!is.null(raw_text)) return(raw_text)
  }
  NULL
}

# =========================
# NBA Stats HTTP HELPERS
# =========================
# All NBA Stats request/response helpers live below. Keep any NBA Stats
# HTTP utilities in this section so league-specific helpers stay grouped.

#' NBA Stats endpoint helper
#'
#' @param endpoint Endpoint name (e.g., "shotchartdetail").
#' @return Full NBA Stats API URL.
#'
#' @keywords internal
nba_endpoint <- function(endpoint) {
  paste0("https://stats.nba.com/stats/", endpoint)
}

#' Request NBA Stats endpoint (rvest + headers + optional proxy)
#'
#' @param url Request URL.
#' @param params Query params list.
#' @param proxy Optional proxy string or list for httr::use_proxy.
#' @param origin Origin header.
#' @param referer Referer header.
#' @param ... Extra args passed to rvest::session (e.g., httr::timeout()).
#' @return Parsed JSON list.
#'
#' @keywords internal
request_with_proxy <- function(url,
                               params = list(),
                               proxy = NULL,
                               origin = "https://stats.nba.com",
                               referer = "https://www.nba.com/",
                               ...) {
  headers <- c(
    `Host` = "stats.nba.com",
    `User-Agent` = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0",
    `Accept` = "application/json, text/plain, */*",
    `Accept-Language` = "en-US,en;q=0.5",
    `Accept-Encoding` = "gzip, deflate, br",
    `x-nba-stats-origin` = "stats",
    `x-nba-stats-token` = "true",
    `Connection` = "keep-alive",
    `Referer` = referer,
    `Origin` = origin,
    `Pragma` = "no-cache",
    `Cache-Control` = "no-cache"
  )

  proxy_cfg <- NULL
  if (!is.null(proxy)) {
    if (is.list(proxy)) {
      proxy_cfg <- do.call(httr::use_proxy, proxy)
    } else if (is.character(proxy) && length(proxy) == 1) {
      proxy_cfg <- httr::use_proxy(proxy)
    }
  }

  if (length(params) >= 1) {
    url <- httr::modify_url(url, query = params)
  }

  res <- rvest::session(
    url = url,
    httr::add_headers(.headers = headers),
    httr::timeout(60),
    if (!is.null(proxy_cfg)) proxy_cfg,
    ...
  )

  httr::content(res$response, as = "text", encoding = "UTF-8") |>
    jsonlite::fromJSON(simplifyVector = FALSE)
}

#' NBA Stats API GET helper
#'
#' @param url NBA Stats API endpoint.
#' @param query Query list.
#' @param path Optional path to save raw JSON.
#' @param force Force re-download even if cached.
#' @param times Retry attempts.
#' @param pause_base Base sleep between retries.
#' @param pause_cap Maximum sleep between retries.
#' @param proxy Optional proxy string or list for requests.
#' @return A list with status_code, raw, error, and path.
#'
#' @keywords internal
nba_api_get <- function(url,
                        query,
                        path = NULL,
                        force = FALSE,
                        times = 5,
                        pause_base = 1,
                        pause_cap = 60,
                        proxy = NULL) {
  if (!is.null(path) && !isTRUE(force) && file.exists(path)) {
    raw <- tryCatch(jsonlite::fromJSON(path, simplifyVector = FALSE), error = function(e) NULL)
    return(list(status_code = 200L, raw = raw, error = NULL, path = path))
  }

  if (is.null(proxy)) {
    proxy <- getOption("spatialSportsR.nba_proxy", NULL)
  }

  headers <- list(
    Host = "stats.nba.com",
    `User-Agent` = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0",
    Accept = "application/json, text/plain, */*",
    `Accept-Language` = "en-US,en;q=0.5",
    `Accept-Encoding` = "gzip, deflate, br",
    `x-nba-stats-origin` = "stats",
    `x-nba-stats-token` = "true",
    Connection = "keep-alive",
    Origin = "http://stats.nba.com",
    Referer = "https://www.nba.com/",
    Pragma = "no-cache",
    `Cache-Control` = "no-cache"
  )

  if (requireNamespace("rvest", quietly = TRUE) && requireNamespace("httr", quietly = TRUE)) {
    proxy_cfg <- NULL
    if (!is.null(proxy)) {
      if (is.list(proxy)) {
        proxy_cfg <- do.call(httr::use_proxy, proxy)
      } else if (is.character(proxy) && length(proxy) == 1) {
        proxy_cfg <- httr::use_proxy(proxy)
      }
    }

    raw <- NULL
    raw_text <- NULL
    status_code <- NA_integer_
    error_message <- NULL

    header_vec <- unlist(headers, use.names = TRUE)
    for (i in seq_len(times)) {
      full_url <- url
      if (length(query) >= 1) {
        full_url <- httr::modify_url(full_url, query = query)
      }
      res <- tryCatch(
        rvest::session(
          url = full_url,
          httr::add_headers(.headers = header_vec),
          httr::timeout(60),
          if (!is.null(proxy_cfg)) proxy_cfg
        ),
        error = function(e) e
      )

      if (inherits(res, "error")) {
        error_message <- conditionMessage(res)
      } else {
        status_code <- httr::status_code(res$response)
        if (!is.na(status_code) && status_code >= 200 && status_code < 300) {
          raw_text <- httr::content(res$response, as = "text", encoding = "UTF-8")
          raw <- tryCatch(jsonlite::fromJSON(raw_text, simplifyVector = FALSE), error = function(e) NULL)
          error_message <- NULL
          if (!is.null(raw)) break
        } else {
          error_message <- httr::http_status(res$response)$message
        }
      }

      Sys.sleep(min(pause_cap, pause_base * i))
    }

    if (is.null(raw)) {
      return(list(status_code = status_code, raw = NULL, raw_text = raw_text, error = error_message, path = NULL))
    }

    if (!is.null(path)) {
      jsonlite::write_json(raw, path, auto_unbox = TRUE, pretty = TRUE, null = "null")
    }

    return(list(status_code = 200L, raw = raw, raw_text = raw_text, error = NULL, path = path))
  }

  if (requireNamespace("httr", quietly = TRUE)) {
    req_cfg <- NULL
    if (!is.null(proxy)) {
      if (is.list(proxy)) {
        req_cfg <- do.call(httr::use_proxy, proxy)
      } else if (is.character(proxy) && length(proxy) == 1) {
        req_cfg <- httr::use_proxy(proxy)
      }
    }
    res <- httr::RETRY(
      "GET",
      url,
      query = query,
      times = times,
      pause_base = pause_base,
      pause_cap = pause_cap,
      httr::add_headers(.headers = headers),
      if (!is.null(req_cfg)) req_cfg
    )
    status_code <- httr::status_code(res)
    if (is.na(status_code) || status_code < 200 || status_code >= 300) {
      return(list(status_code = status_code, raw = NULL, error = httr::http_status(res)$message, path = NULL))
    }
    raw_text <- httr::content(res, as = "text", encoding = "UTF-8")
    raw <- jsonlite::fromJSON(raw_text, simplifyVector = FALSE)
  } else if (requireNamespace("httr2", quietly = TRUE)) {
    req <- httr2::request(url)
    req <- do.call(httr2::req_headers, c(list(req), headers))
    req <- do.call(httr2::req_url_query, c(list(req), query))
    if (!is.null(proxy)) {
      if (is.list(proxy)) {
        req <- do.call(httr2::req_proxy, c(list(req), proxy))
      } else if (is.character(proxy) && length(proxy) == 1) {
        req <- httr2::req_proxy(req, url = proxy)
      }
    }
    raw <- NULL
    raw_text <- NULL
    status_code <- NA_integer_
    error_message <- NULL

    for (i in seq_len(times)) {
      resp <- tryCatch(httr2::req_perform(req), error = function(e) e)
      if (inherits(resp, "error")) {
        error_message <- conditionMessage(resp)
      } else {
        status_code <- httr2::resp_status(resp)
        if (!is.na(status_code) && status_code >= 200 && status_code < 300) {
          raw_text <- httr2::resp_body_string(resp)
          raw <- jsonlite::fromJSON(raw_text, simplifyVector = FALSE)
          error_message <- NULL
          break
        }
        error_message <- httr2::resp_status_desc(resp)
      }
      Sys.sleep(min(pause_cap, pause_base * i))
    }

    if (is.null(raw)) {
      return(list(status_code = status_code, raw = NULL, error = error_message, path = NULL))
    }
  } else if (requireNamespace("curl", quietly = TRUE)) {
    query_string <- paste0(names(query), "=", vapply(query, function(x) utils::URLencode(as.character(x)), character(1)))
    full_url <- paste0(url, "?", paste(query_string, collapse = "&"))
    h <- curl::new_handle()
    curl::handle_setheaders(h, .list = headers)
    if (!is.null(proxy)) {
      if (is.list(proxy) && !is.null(proxy$url)) {
        curl::handle_setopt(h, proxy = proxy$url)
      } else if (is.character(proxy) && length(proxy) == 1) {
        curl::handle_setopt(h, proxy = proxy)
      }
    }

    raw_text <- NULL
    status_code <- NA_integer_
    for (i in seq_len(times)) {
      resp <- tryCatch(curl::curl_fetch_memory(full_url, handle = h), error = function(e) NULL)
      if (!is.null(resp)) {
        status_code <- resp$status_code
        if (!is.na(status_code) && status_code >= 200 && status_code < 300) {
          raw_text <- rawToChar(resp$content)
          break
        }
      }
      Sys.sleep(min(pause_cap, pause_base * i))
    }

    if (is.null(raw_text)) {
      return(list(status_code = status_code, raw = NULL, error = "NBA Stats request failed", path = NULL))
    }

    raw <- jsonlite::fromJSON(raw_text, simplifyVector = FALSE)
  } else {
    stop("Install httr, httr2, or curl to use nba_api_get()", call. = FALSE)
  }

  if (!is.null(path)) {
    jsonlite::write_json(raw, path, auto_unbox = TRUE, pretty = TRUE, null = "null")
  }

  list(status_code = 200L, raw = raw, raw_text = raw_text, error = NULL, path = path)
}

nba_api_get_with_backoff <- function(url,
                                     query,
                                     path = NULL,
                                     force = FALSE,
                                     times = 5,
                                     pause_base = 1,
                                     pause_cap = 60,
                                     proxy = NULL,
                                     backoff_base = 1,
                                     backoff_cap = 30,
                                     jitter = 0.25,
                                     verbose = FALSE,
                                     label = NULL) {
  out <- list(status_code = NA_integer_, raw = NULL, raw_text = NULL, error = NULL, path = NULL)

  for (i in seq_len(times)) {
    res <- tryCatch(
      suppressWarnings(nba_api_get(
        url = url,
        query = query,
        path = path,
        force = force,
        times = 1,
        pause_base = pause_base,
        pause_cap = pause_cap,
        proxy = proxy
      )),
      error = function(e) list(status_code = NA_integer_, raw = NULL, raw_text = NULL, error = conditionMessage(e), path = NULL)
    )

    out <- res
    if (is.null(res$error) && !is.na(res$status_code) && res$status_code >= 200 && res$status_code < 300) {
      return(out)
    }

    if (isTRUE(verbose)) {
      tag <- if (!is.null(label) && nzchar(label)) paste0(" [", label, "]") else ""
      msg <- if (!is.null(res$error)) res$error else paste0("status ", res$status_code)
      message(sprintf("NBA Stats request failed%s (attempt %s/%s): %s", tag, i, times, msg))
    }

    sleep_time <- min(backoff_cap, backoff_base * (2 ^ (i - 1)))
    if (!is.null(jitter) && jitter > 0) {
      sleep_time <- sleep_time + stats::runif(1, 0, jitter)
    }
    Sys.sleep(sleep_time)
  }

  out
}


save_raw_json <- function(raw, endpoint, key, dir = "data/raw") {
  if (is.null(raw)) {
    return(invisible(NULL))
  }
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  safe_key <- gsub("[^A-Za-z0-9_-]+", "_", as.character(key))
  path <- file.path(dir, paste0(endpoint, "_", safe_key, ".json"))
  jsonlite::write_json(raw, path, auto_unbox = TRUE, pretty = TRUE, null = "null")
  invisible(path)
}

.espn_nba_clean_date_key <- function(date_value) {
  if (is.null(date_value)) {
    return(NULL)
  }
  if (inherits(date_value, "Date")) {
    return(format(date_value, "%Y%m%d"))
  }
  date_str <- as.character(date_value)
  date_str <- sub("Z$", "", date_str)
  date_str <- substr(date_str, 1, 10)
  date_str <- gsub("-", "", date_str)
  if (!nzchar(date_str)) {
    return(NULL)
  }
  date_str
}

.espn_nba_summary_key <- function(season = NULL, game_date = NULL, game_id = NULL) {
  parts <- c(as.character(season), as.character(game_date), as.character(game_id))
  parts <- parts[!is.na(parts) & nzchar(parts)]
  if (length(parts) == 0) {
    return(as.character(game_id))
  }
  paste(parts, collapse = "_")
}

.espn_nba_extract_summary_season <- function(raw_summary) {
  season <- raw_summary$season$year
  if (is.null(season)) {
    season <- raw_summary$header$season$year
  }
  if (is.list(season)) {
    season <- season[[1]]
  } else if (length(season) > 1) {
    season <- season[1]
  }
  if (is.null(season) || is.na(season)) {
    return(NULL)
  }
  as.character(season)
}

.espn_nba_extract_summary_date <- function(raw_summary) {
  date_value <- raw_summary$header$competitions$date
  if (is.null(date_value)) {
    return(NULL)
  }
  if (is.list(date_value)) {
    date_value <- date_value[[1]]
  } else if (length(date_value) > 1) {
    date_value <- date_value[1]
  }
  date_value
}

log_failed_scrape <- function(game_id,
                              game_date,
                              status_code,
                              error_message,
                              dir = "data/raw",
                              filename = "scrape_failures.csv") {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(dir, filename)
  entry <- data.frame(
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    game_id = as.integer(game_id),
    game_date = as.character(game_date),
    status_code = as.integer(status_code),
    error_message = as.character(error_message),
    stringsAsFactors = FALSE
  )

  utils::write.table(
    entry,
    path,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(path),
    append = file.exists(path)
  )
  invisible(path)
}

espn_nba_summary_raw_safe <- function(game_id,
                                      save_raw = FALSE,
                                      raw_dir = "data/raw",
                                      times = 5,
                                      pause_base = 1,
                                      pause_cap = 60,
                                      file_path = NULL) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Install jsonlite to use ESPN NBA helpers", call. = FALSE)
  }

  res <- .espn_nba_summary_response_raw(
    game_id,
    times = times,
    pause_base = pause_base,
    pause_cap = pause_cap
  )
  status_code <- httr::status_code(res)
  if (is.na(status_code) || status_code < 200 || status_code >= 300) {
    message <- httr::http_status(res)$message
    return(list(status_code = status_code, raw = NULL, error = message, file_path = NULL))
  }

  raw_summary <- tryCatch(
    jsonlite::fromJSON(httr::content(res, as = "text", encoding = "UTF-8"), simplifyVector = FALSE),
    error = function(e) e
  )
  if (inherits(raw_summary, "error")) {
    return(list(status_code = status_code, raw = NULL, error = conditionMessage(raw_summary), file_path = NULL))
  }

  saved_path <- NULL
  if (isTRUE(save_raw)) {
    if (!is.null(file_path)) {
      jsonlite::write_json(raw_summary, file_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
      saved_path <- file_path
    } else {
      season <- .espn_nba_extract_summary_season(raw_summary)
      game_date <- .espn_nba_clean_date_key(.espn_nba_extract_summary_date(raw_summary))
      key <- .espn_nba_summary_key(season = season, game_date = game_date, game_id = game_id)
      saved_path <- save_raw_json(raw_summary, "summary", key, dir = raw_dir)
    }
  }

  list(status_code = status_code, raw = raw_summary, error = NULL, file_path = saved_path)
}


espn_nba_scoreboard <- function(dates, save_raw = FALSE, raw_dir = "data/raw") {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Install jsonlite to use ESPN NBA helpers", call. = FALSE)
  }
  dates <- as.character(dates)
  raw_text <- tryCatch(.espn_nba_scoreboard_response(dates), error = function(e) NULL)
  if (is.null(raw_text)) return(NULL)

  raw_sched <- tryCatch(
    jsonlite::fromJSON(raw_text, simplifyVector = FALSE),
    error = function(e) NULL
  )
  if (is.null(raw_sched)) return(NULL)

  if (isTRUE(save_raw)) {
    save_raw_json(raw_sched, "scoreboard", dates, dir = raw_dir)
  }

  events <- raw_sched[["events"]]
  if (is.null(events) || length(events) == 0) {
    return(data.frame(
      game_id = character(),
      game_date = as.Date(character()),
      home_team_id = character(),
      away_team_id = character(),
      home_team = character(),
      away_team = character(),
      stringsAsFactors = FALSE
    ))
  }

  rows <- lapply(events, .espn_nba_parse_scoreboard_event)
  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (length(rows) == 0) {
    return(data.frame(
      game_id = character(),
      game_date = as.Date(character()),
      season = integer(),
      season_type = integer(),
      home_team_id = character(),
      away_team_id = character(),
      home_team = character(),
      away_team = character(),
      stringsAsFactors = FALSE
    ))
  }
  do.call(rbind, rows)
}

.espn_nba_parse_scoreboard_event <- function(event) {
  competition <- event$competitions[[1]]
  if (is.null(competition)) return(NULL)
  date_str <- competition$date
  game_date <- as.Date(NA)
  if (!is.null(date_str) && length(date_str) > 0) {
    date_clean <- sub("Z$", "", date_str)
    game_date_time <- suppressWarnings(as.POSIXct(date_clean, tz = "America/New_York"))
    if (is.na(game_date_time)) {
      game_date_time <- suppressWarnings(as.POSIXct(date_clean, format = "%Y-%m-%dT%H:%M", tz = "America/New_York"))
    }
    if (!is.na(game_date_time)) {
      game_date <- as.Date(substr(game_date_time, 1, 10))
    }
  }

  competitors <- competition$competitors
  if (is.null(competitors) || length(competitors) < 2) return(NULL)
  home_idx <- which(vapply(competitors, function(x) x$homeAway, "") == "home")
  away_idx <- which(vapply(competitors, function(x) x$homeAway, "") == "away")
  if (length(home_idx) == 0) home_idx <- 1
  if (length(away_idx) == 0) away_idx <- 2

  home <- competitors[[home_idx[1]]]
  away <- competitors[[away_idx[1]]]

  safe_chr <- function(x) {
    if (is.null(x) || length(x) == 0) return(NA_character_)
    as.character(x)[1]
  }

  data.frame(
    game_id = safe_chr(competition$id),
    game_date = game_date,
    season = as.integer(safe_chr(event$season$year)),
    season_type = as.integer(safe_chr(event$season$type)),
    home_team_id = safe_chr(home$team$id),
    away_team_id = safe_chr(away$team$id),
    home_team = safe_chr(home$team$abbreviation),
    away_team = safe_chr(away$team$abbreviation),
    stringsAsFactors = FALSE
  )
}
