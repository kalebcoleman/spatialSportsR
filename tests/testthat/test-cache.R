testthat::test_that("cache_path() builds season/source paths and cache_exists() detects files", {
  tmp_dir <- tempdir()

  path <- cache_path(
    raw_dir = tmp_dir,
    league = "nba",
    season = 2026,
    source = "espn",
    filename = "example.json"
  )

  testthat::expect_true(grepl("nba", path, fixed = TRUE))
  testthat::expect_true(grepl("espn", path, fixed = TRUE))
  testthat::expect_true(grepl("2025-26", path, fixed = TRUE))
  testthat::expect_false(cache_exists(path))

  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines("{}", path)
  testthat::expect_true(cache_exists(path))
})
