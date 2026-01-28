make_tables <- function() {
  games <- data.frame(
    league = "nba",
    season = "2024",
    game_id = "001",
    game_date = as.Date("2024-10-01"),
    home_team = "AAA",
    away_team = "BBB",
    stringsAsFactors = FALSE
  )

  events <- data.frame(
    league = "nba",
    season = "2024",
    game_id = "001",
    event_id = "e1",
    event_type = "shot",
    x_unit = 0.5,
    y_unit = 0.4,
    label = 1,
    meta = "{}",
    stringsAsFactors = FALSE
  )

  list(games = games, events = events)
}

test_that("validate_tables() fails on missing columns", {
  tables <- make_tables()
  tables$events$label <- NULL

  expect_error(
    validate_tables(tables, "nba"),
    "events missing columns"
  )
})

test_that("validate_tables() fails on bounds", {
  tables <- make_tables()
  tables$events$x_unit <- 1.5

  expect_error(
    validate_tables(tables, "nba"),
    "x_unit out of \\[0,1\\]"
  )
})

test_that("validate_tables() fails on duplicate keys", {
  tables <- make_tables()
  tables$games <- rbind(tables$games, tables$games)

  expect_error(
    validate_tables(tables, "nba"),
    "games has duplicate"
  )
})

test_that("validate_tables() fails on FK mismatch", {
  tables <- make_tables()
  tables$events$game_id <- "999"

  expect_error(
    validate_tables(tables, "nba"),
    "events reference missing games"
  )
})
