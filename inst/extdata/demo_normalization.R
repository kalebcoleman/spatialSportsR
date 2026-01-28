# Demo: raw vs normalized coordinates across leagues

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Install ggplot2 to run the demo.", call. = FALSE)
}

raw_nba <- data.frame(
  league = "nba",
  x_raw = c(5, 25, 47, 88),
  y_raw = c(3, 10, 25, 45)
)
raw_nhl <- data.frame(
  league = "nhl",
  x_raw = c(-90, -30, 0, 80),
  y_raw = c(-40, -10, 5, 35)
)
raw_nfl <- data.frame(
  league = "nfl",
  x_raw = c(5, 35, 65, 95),
  y_raw = c(5, 15, 30, 50)
)

normalize_rows <- function(df) {
  norm <- normalize_xy(df$league[1], df$x_raw, df$y_raw)
  data.frame(
    league = df$league,
    x_raw = df$x_raw,
    y_raw = df$y_raw,
    x_unit = norm$x_unit,
    y_unit = norm$y_unit
  )
}

nba_norm <- normalize_rows(raw_nba)
nhl_norm <- normalize_rows(raw_nhl)
nfl_norm <- normalize_rows(raw_nfl)

plot_raw <- function(df) {
  plot_events(
    data.frame(x_unit = df$x_raw, y_unit = df$y_raw),
    league = paste0(df$league[1], " (raw)")
  )
}

plot_norm <- function(df) {
  plot_events(
    data.frame(x_unit = df$x_unit, y_unit = df$y_unit),
    league = paste0(df$league[1], " (normalized)")
  )
}

plots <- list(
  plot_raw(nba_norm), plot_norm(nba_norm),
  plot_raw(nhl_norm), plot_norm(nhl_norm),
  plot_raw(nfl_norm), plot_norm(nfl_norm)
)

if (requireNamespace("patchwork", quietly = TRUE)) {
  print(plots[[1]] + plots[[2]] + plots[[3]] + plots[[4]] + plots[[5]] + plots[[6]])
} else if (requireNamespace("gridExtra", quietly = TRUE)) {
  gridExtra::grid.arrange(grobs = plots, ncol = 2)
} else {
  message("Install patchwork or gridExtra for side-by-side layout.")
  for (p in plots) print(p)
}
