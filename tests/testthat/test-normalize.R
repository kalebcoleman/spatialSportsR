testthat::test_that(".nba_stats_season_string() normalizes season labels", {
  testthat::expect_equal(spatialSportsR:::.nba_stats_season_string(2026), "2025-26")
  testthat::expect_equal(spatialSportsR:::.nba_stats_season_string("2024-2025"), "2024-25")
  testthat::expect_equal(spatialSportsR:::.nba_stats_season_string("2024-25"), "2024-25")
})

testthat::test_that(".normalize_season_type_dir() maps playoffs to subdir", {
  testthat::expect_null(spatialSportsR:::.normalize_season_type_dir("regular"))
  testthat::expect_equal(spatialSportsR:::.normalize_season_type_dir("playoffs"), "playoffs")
  testthat::expect_equal(spatialSportsR:::.normalize_season_type_dir("postseason"), "playoffs")
})
