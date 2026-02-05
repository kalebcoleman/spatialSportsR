"""
Shared court drawing utilities for NBA shot chart visualizations.

This module consolidates court drawing functions used across the analysis scripts
to avoid code duplication and ensure consistent court representations.
"""

import math
from matplotlib.patches import Arc, Circle, Rectangle
import matplotlib.pyplot as plt


def draw_half_court(ax=None, color='black', lw=2, outer_lines=False):
    """
    Draw an NBA half court on a given Matplotlib axes.
    
    Uses NBA coordinate system where LOC_X and LOC_Y are in 1/10th of a foot.
    Hoop is at origin (0, 0), baseline at Y ≈ -47.5.
    
    Args:
        ax: Matplotlib axes (uses current axes if None)
        color: Line color for court markings
        lw: Line width
        outer_lines: Whether to draw outer boundary lines
        
    Returns:
        The matplotlib axes with court drawn
    """
    if ax is None:
        ax = plt.gca()

    # Hoop and backboard
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    
    # Key/paint area
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)
    
    # Free throw arcs
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180, 
                         linewidth=lw, color=color, fill=False)
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=360, 
                            linewidth=lw, color=color, linestyle='--')
    
    # Restricted area
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)

    # Three-point line
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)

    # Center court arcs (for reference)
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0, 
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0, 
                           linewidth=lw, color=color)

    court_elements = [
        hoop, backboard, outer_box, inner_box, top_free_throw,
        bottom_free_throw, restricted, corner_three_a,
        corner_three_b, three_arc, center_outer_arc, center_inner_arc
    ]

    if outer_lines:
        outer = Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False)
        court_elements.append(outer)

    for element in court_elements:
        ax.add_patch(element)

    return ax


def draw_full_court(
    ax=None,
    color="white",
    lw=2,
    bg="#0b0f1a",
    show_axis=False,
    draw_corner_threes=True,
    zorder=1,
):
    """
    Draw a full NBA court with customizable colors (for heatmaps/density plots).
    
    Args:
        ax: Matplotlib axes (uses current axes if None)
        color: Line color for court markings
        lw: Line width
        bg: Background color of the court
        show_axis: Whether to show axis labels
        draw_corner_threes: Whether to draw corner three-point lines
        zorder: Z-order for court elements
        
    Returns:
        The matplotlib axes with court drawn
    """
    if ax is None:
        ax = plt.gca()

    ax.set_facecolor(bg)
    baseline_y = -52
    halfcourt_y = 418
    corner_x = 220
    arc_radius = 237.5
    corner_y = math.sqrt(arc_radius**2 - corner_x**2)

    def add_half_court(y_flip=False):
        def y(val):
            return 2 * halfcourt_y - val if y_flip else val

        def add_rect(x, y0, w, h):
            y0f = y(y0)
            y1f = y(y0 + h)
            y_min = min(y0f, y1f)
            height = abs(y1f - y0f)
            ax.add_patch(
                Rectangle((x, y_min), w, height, fill=False, linewidth=lw, color=color)
            )

        # Hoop
        ax.add_patch(
            Circle((0, y(0)), 7.5, fill=False, linewidth=lw, color=color, zorder=zorder)
        )
        # Backboard
        ax.add_patch(
            Rectangle((-30, y(-12.5)), 60, 0, linewidth=lw, color=color, zorder=zorder)
        )

        # Key boxes
        add_rect(-80, baseline_y, 160, 190)
        add_rect(-60, baseline_y, 120, 190)

        if not y_flip:
            # Free throw arcs
            ax.add_patch(
                Arc((0, y(142.5)), 120, 120, theta1=0, theta2=180,
                    linewidth=lw, color=color, zorder=zorder)
            )
            ax.add_patch(
                Arc((0, y(142.5)), 120, 120, theta1=180, theta2=360,
                    linewidth=lw, color=color, linestyle="dashed", zorder=zorder)
            )
            # Restricted area
            ax.add_patch(
                Arc((0, y(0)), 80, 80, theta1=0, theta2=180,
                    linewidth=lw, color=color, zorder=zorder)
            )
            # Corner threes
            if draw_corner_threes:
                ax.plot([-corner_x, -corner_x], [y(baseline_y), y(corner_y)],
                        linewidth=lw, color=color, zorder=zorder)
                ax.plot([corner_x, corner_x], [y(baseline_y), y(corner_y)],
                        linewidth=lw, color=color, zorder=zorder)
            # Three-point arc
            ax.add_patch(
                Arc((0, y(0)), 2 * arc_radius, 2 * arc_radius, theta1=22, theta2=158,
                    linewidth=lw, color=color, zorder=zorder)
            )
        else:
            # Mirrored court elements for full court
            ax.add_patch(
                Arc((0, y(142.5)), 120, 120, theta1=180, theta2=360,
                    linewidth=lw, color=color, zorder=zorder)
            )
            ax.add_patch(
                Arc((0, y(142.5)), 120, 120, theta1=0, theta2=180,
                    linewidth=lw, color=color, linestyle="dashed", zorder=zorder)
            )
            ax.add_patch(
                Arc((0, y(0)), 80, 80, theta1=180, theta2=360,
                    linewidth=lw, color=color, zorder=zorder)
            )
            if draw_corner_threes:
                ax.plot([-corner_x, -corner_x], [y(baseline_y), y(corner_y)],
                        linewidth=lw, color=color, zorder=zorder)
                ax.plot([corner_x, corner_x], [y(baseline_y), y(corner_y)],
                        linewidth=lw, color=color, zorder=zorder)
            ax.add_patch(
                Arc((0, y(0)), 2 * arc_radius, 2 * arc_radius, theta1=202, theta2=338,
                    linewidth=lw, color=color, zorder=zorder)
            )

    add_half_court(y_flip=False)
    add_half_court(y_flip=True)
    
    # Halfcourt line and circle
    ax.plot([-250, 250], [halfcourt_y, halfcourt_y], 
            color=color, linewidth=lw, zorder=zorder)
    ax.add_patch(
        Circle((0, halfcourt_y), 60, fill=False, linewidth=lw, color=color, zorder=zorder)
    )
    # Court boundary
    ax.add_patch(
        Rectangle((-250, baseline_y), 500, 940, fill=False, 
                  linewidth=lw, color=color, zorder=zorder)
    )
    
    ax.set_xlim(-250, 250)
    ax.set_ylim(-52, 888)
    ax.set_aspect("equal")
    
    if show_axis:
        ax.tick_params(colors=color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(color)
        ax.set_xlabel("Court X (NBA units; 1 unit = 0.1 ft)", color=color)
        ax.set_ylabel("Court Y (NBA units; 1 unit = 0.1 ft)", color=color)
        ax.set_xticks(range(-250, 251, 50))
        ax.set_yticks(range(-50, 901, 100))
    else:
        ax.axis("off")
        
    return ax


def setup_shot_chart_axes(ax, xlim=(-250, 250), ylim=(-47.5, 422.5)):
    """
    Configure axes for a standard half-court shot chart.
    
    Args:
        ax: Matplotlib axes
        xlim: X-axis limits (default: full court width)
        ylim: Y-axis limits (default: half court)
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_aspect('equal')
