#' Plot helpers

#' Plot a league field/court outline
#'
#' @param league League identifier.
#' @param ... Additional arguments.
#' @return A ggplot object.
#' @export
plot_field <- function(league, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Install ggplot2 to use plotting functions", call. = FALSE)
  }
  ggplot2::ggplot() + ggplot2::labs(title = paste("Field:", league))
}

#' Plot standardized events
#'
#' @param events Events data frame with x_unit/y_unit.
#' @param league League identifier.
#' @param ... Additional arguments.
#' @return A ggplot object.
#' @export
plot_events <- function(events, league, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Install ggplot2 to use plotting functions", call. = FALSE)
  }
  if (!is.data.frame(events)) stop("events must be a data frame", call. = FALSE)

  ggplot2::ggplot(events, ggplot2::aes(x = .data$x_unit, y = .data$y_unit)) +
    ggplot2::geom_point(alpha = 0.3) +
    ggplot2::labs(title = paste("Events:", league))
}

utils::globalVariables(".data")
