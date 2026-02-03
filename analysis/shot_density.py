import math
import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Arc, Circle, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "parsed" / "nba.sqlite"
DB_PATH = Path(os.getenv("SPATIALSPORTSR_DB_PATH", str(DEFAULT_DB_PATH))).expanduser()


def normalize_season_type(season_type):
    if season_type is None:
        return None
    st = str(season_type).strip().lower()
    if st in {"playoffs", "playoff", "postseason"}:
        return "playoffs"
    if st in {"regular", "regular season", "reg"}:
        return "regular"
    return st


def available_season_types(con, season):
    q = """
    SELECT DISTINCT season_type
    FROM nba_stats_shots
    WHERE season = ?
    ORDER BY season_type
    """
    return pd.read_sql_query(q, con, params=[season])["season_type"].tolist()


def draw_court(
    ax=None,
    color="white",
    lw=2,
    bg="#0b0f1a",
    show_axis=False,
    full_court=False,
    draw_corner_threes=True,
    zorder=1,
):
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

        ax.add_patch(
            Circle((0, y(0)), 7.5, fill=False, linewidth=lw, color=color, zorder=zorder)
        )
        ax.add_patch(
            Rectangle((-30, y(-12.5)), 60, 0, linewidth=lw, color=color, zorder=zorder)
        )

        add_rect(-80, baseline_y, 160, 190)
        add_rect(-60, baseline_y, 120, 190)

        if not y_flip:
            ax.add_patch(
                Arc(
                    (0, y(142.5)),
                    120,
                    120,
                    theta1=0,
                    theta2=180,
                    linewidth=lw,
                    color=color,
                    zorder=zorder,
                )
            )
            ax.add_patch(
                Arc(
                    (0, y(142.5)),
                    120,
                    120,
                    theta1=180,
                    theta2=360,
                    linewidth=lw,
                    color=color,
                    linestyle="dashed",
                    zorder=zorder,
                )
            )
            ax.add_patch(
                Arc(
                    (0, y(0)),
                    80,
                    80,
                    theta1=0,
                    theta2=180,
                    linewidth=lw,
                    color=color,
                    zorder=zorder,
                )
            )
            if draw_corner_threes:
                ax.plot(
                    [-corner_x, -corner_x],
                    [y(baseline_y), y(corner_y)],
                    linewidth=lw,
                    color=color,
                    zorder=zorder,
                )
                ax.plot(
                    [corner_x, corner_x],
                    [y(baseline_y), y(corner_y)],
                    linewidth=lw,
                    color=color,
                    zorder=zorder,
                )
            ax.add_patch(
                Arc(
                    (0, y(0)),
                    2 * arc_radius,
                    2 * arc_radius,
                    theta1=22,
                    theta2=158,
                    linewidth=lw,
                    color=color,
                    zorder=zorder,
                )
            )
        else:
            ax.add_patch(
                Arc(
                    (0, y(142.5)),
                    120,
                    120,
                    theta1=180,
                    theta2=360,
                    linewidth=lw,
                    color=color,
                    zorder=zorder,
                )
            )
            ax.add_patch(
                Arc(
                    (0, y(142.5)),
                    120,
                    120,
                    theta1=0,
                    theta2=180,
                    linewidth=lw,
                    color=color,
                    linestyle="dashed",
                    zorder=zorder,
                )
            )
            ax.add_patch(
                Arc(
                    (0, y(0)),
                    80,
                    80,
                    theta1=180,
                    theta2=360,
                    linewidth=lw,
                    color=color,
                    zorder=zorder,
                )
            )
            if draw_corner_threes:
                ax.plot(
                    [-corner_x, -corner_x],
                    [y(baseline_y), y(corner_y)],
                    linewidth=lw,
                    color=color,
                    zorder=zorder,
                )
                ax.plot(
                    [corner_x, corner_x],
                    [y(baseline_y), y(corner_y)],
                    linewidth=lw,
                    color=color,
                    zorder=zorder,
                )
            ax.add_patch(
                Arc(
                    (0, y(0)),
                    2 * arc_radius,
                    2 * arc_radius,
                    theta1=202,
                    theta2=338,
                    linewidth=lw,
                    color=color,
                    zorder=zorder,
                )
            )

    add_half_court(y_flip=False)
    if full_court:
        add_half_court(y_flip=True)
        ax.plot(
            [-250, 250],
            [halfcourt_y, halfcourt_y],
            color=color,
            linewidth=lw,
            zorder=zorder,
        )
        ax.add_patch(
            Circle(
                (0, halfcourt_y),
                60,
                fill=False,
                linewidth=lw,
                color=color,
                zorder=zorder,
            )
        )
        ax.add_patch(
            Rectangle(
                (-250, baseline_y),
                500,
                940,
                fill=False,
                linewidth=lw,
                color=color,
                zorder=zorder,
            )
        )
        ax.set_xlim(-250, 250)
        ax.set_ylim(-52, 888)
    else:
        ax.set_xlim(-250, 250)
        ax.set_ylim(-52, 418)
    ax.set_aspect("equal")
    if show_axis:
        ax.tick_params(colors=color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(color)
        ax.set_xlabel("Court X (NBA units; 1 unit = 0.1 ft)", color=color)
        ax.set_ylabel("Court Y (NBA units; 1 unit = 0.1 ft)", color=color)
        ax.set_xticks(range(-250, 251, 50))
        if full_court:
            ax.set_yticks(range(-50, 901, 100))
        else:
            ax.set_yticks(range(-50, 451, 50))
        for label in ax.get_yticklabels():
            label.set_rotation(0)
            label.set_horizontalalignment("right")
    else:
        ax.axis("off")
    return ax


def load_shots(season, season_type="playoffs"):
    season_type = normalize_season_type(season_type)
    con = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        """
        SELECT LOC_X, LOC_Y, SHOT_MADE_FLAG, SHOT_TYPE, SHOT_ZONE_BASIC
        FROM nba_stats_shots
        WHERE season = ?
          AND season_type = ?
          AND SHOT_ATTEMPTED_FLAG = 1
        """,
        con,
        params=[season, season_type],
    )
    if df.empty:
        types = available_season_types(con, season)
        print(
            f"no rows for season={season} season_type={season_type}; available season_type={types}"
        )
    con.close()
    return df


def summarize_shots(df):
    total_fga = len(df)
    total_fgm = int(df["SHOT_MADE_FLAG"].sum())
    total_fg = total_fgm / total_fga if total_fga else 0

    is_three = df["SHOT_TYPE"].fillna("").str.contains("3PT", case=False, na=False)
    threes = df[is_three]
    twos = df[~is_three]

    three_fga = len(threes)
    three_fgm = int(threes["SHOT_MADE_FLAG"].sum())
    three_fg = three_fgm / three_fga if three_fga else 0

    two_fga = len(twos)
    two_fgm = int(twos["SHOT_MADE_FLAG"].sum())
    two_fg = two_fgm / two_fga if two_fga else 0

    zone = (
        df.groupby("SHOT_ZONE_BASIC")["SHOT_MADE_FLAG"]
        .agg(fga="size", fgm="sum")
        .reset_index()
    )
    zone["fg_pct"] = zone["fgm"] / zone["fga"]
    zone = zone.sort_values("fga", ascending=False)

    summary = {
        "total_fga": total_fga,
        "total_fgm": total_fgm,
        "total_fg": total_fg,
        "two_fga": two_fga,
        "two_fgm": two_fgm,
        "two_fg": two_fg,
        "three_fga": three_fga,
        "three_fgm": three_fgm,
        "three_fg": three_fg,
        "zone_table": zone,
    }
    return summary



def generate_shot_chart(
    season,
    season_type,
    bg_color,
    court_bg_color,
    court_line_color,
    font_color,
    stat_label_color,
    output_filename,
    title_text,
    subtitle_text,
    author_name_text,
    cmap_hexbin="YlOrRd",
    stat_value_cmap_index=0.6,
    show_plot=True
):
    df = load_shots(season, season_type)
    print(f"rows: {len(df)} for {season} {season_type}")

    summary = summarize_shots(df)
    print(f"FG%: {round(summary['total_fg'] * 100, 1)}%")
    print(f"2PT FG%: {round(summary['two_fg'] * 100, 1)}%")
    print(f"3PT FG%: {round(summary['three_fg'] * 100, 1)}%")
    print("\nTop zones by attempts:")
    print(summary["zone_table"].head(6).to_string(index=False))

    fig = plt.figure(figsize=(14, 16))
    fig.patch.set_facecolor(bg_color)

    gs_main = fig.add_gridspec(4, 1, height_ratios=[0.5, 0.35, 0.2, 8.4], hspace=0.04)
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.05, right=0.95)

    ax_name = fig.add_subplot(gs_main[0, 0])
    ax_name.set_facecolor(bg_color)
    ax_name.axis("off")
    ax_name.text(
        0.5,
        0.7,
        title_text,
        ha="center",
        va="top",
        color=font_color,
        fontsize=28,
        fontweight="bold",
    )

    ax_desc = fig.add_subplot(gs_main[1, 0])
    ax_desc.set_facecolor(bg_color)
    ax_desc.axis("off")
    ax_desc.text(
        0.5,
        0.85,
        subtitle_text,
        ha="center",
        va="top",
        color=font_color,
        fontsize=15,
        fontweight="bold",
    )
    ax_desc.text(
        0.5,
        0.28,
        author_name_text,
        ha="center",
        va="top",
        color=font_color,
        fontsize=15,
        style="italic",
    )

    ax_cbar = fig.add_subplot(gs_main[2, 0])
    ax_cbar.set_facecolor(bg_color)
    ax_cbar.axis("off")

    gs_court = gs_main[3, 0].subgridspec(1, 3, width_ratios=[1.0, 2.2, 1.0], wspace=0.0)
    ax_left = fig.add_subplot(gs_court[0, 0])
    ax_court = fig.add_subplot(gs_court[0, 1])
    ax_right = fig.add_subplot(gs_court[0, 2])
    for ax in (ax_left, ax_right):
        ax.set_facecolor(bg_color)
        ax.axis("off")

    draw_court(
        ax_court,
        color=court_line_color,
        lw=2,
        bg=court_bg_color,
        show_axis=True,
        full_court=True,
        draw_corner_threes=True,
        zorder=3,
    )

    hb = ax_court.hexbin(
        df["LOC_X"],
        df["LOC_Y"],
        gridsize=80,
        mincnt=1,
        bins="log",
        zorder=2,
        cmap=cmap_hexbin,
    )

    ax_cbar.axis("off")
    court_pos = ax_court.get_position()
    legend_width = 0.14
    fig_w, fig_h = fig.get_size_inches()
    dx = -30 / (fig.dpi * fig_w)
    dy = -350 / (fig.dpi * fig_h)
    log_bar_y_offset = 0.0
    legend_left = max(0.04, court_pos.x0 - 0.12 - legend_width / 2 + dx)
    legend_bottom = court_pos.y0 + 0.30 + dy + log_bar_y_offset
    legend_ax = fig.add_axes([legend_left, legend_bottom, legend_width, 0.02])
    legend_ax.set_facecolor(bg_color)
    cbar = plt.colorbar(hb, cax=legend_ax, orientation="horizontal")
    cbar.set_label("Shot Count (log10)", color=font_color, fontsize=10, labelpad=6)
    cbar.set_ticks([])
    cbar.ax.tick_params(length=0)
    cbar.outline.set_visible(False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    zone = summary["zone_table"].copy()
    zone["share_pct"] = zone["fga"] / zone["fga"].sum()

    left_start_y = 0.82
    left_y_spacing = 0.14
    label_to_value_gap = 0.035

    stat_color = plt.get_cmap(cmap_hexbin)(stat_value_cmap_index)
    label_fontsize = 15
    value_fontsize = 24

    left_stats = [
        ("Field Goals Attempted", str(summary["total_fga"])),
        ("Field Goal %", f"{summary['total_fg'] * 100:.1f}%"),
        ("2PT Field Goal %", f"{summary['two_fg'] * 100:.1f}%"),
        ("3PT Field Goal %", f"{summary['three_fg'] * 100:.1f}%"),
    ]
    for i, (label, value) in enumerate(left_stats):
        y = left_start_y - i * left_y_spacing
        ax_left.text(
            0.5,
            y,
            label,
            ha="center",
            va="bottom",
            color=font_color,
            fontsize=label_fontsize,
            fontweight="normal",
            transform=ax_left.transAxes,
        )
        ax_left.text(
            0.5,
            y - label_to_value_gap,
            value,
            ha="center",
            va="top",
            color=stat_color,
            fontsize=value_fontsize,
            fontweight="bold",
            transform=ax_left.transAxes,
        )

    zone_value_gap = 0.035
    right_start_y = left_start_y
    right_y_spacing = 0.14
    col_offset = 0.22

    for i, (_, row) in enumerate(zone.head(4).iterrows()):
        y_pos = right_start_y - i * right_y_spacing
        zone_name = row["SHOT_ZONE_BASIC"]

        if len(zone_name) > 15:
            if "Above the Break" in zone_name:
                zone_name = "Above Break 3"
            elif "In The Paint" in zone_name:
                zone_name = "Paint (Non-RA)"

        ax_right.text(
            0.5,
            y_pos,
            zone_name,
            ha="center",
            va="bottom",
            color=font_color,
            fontsize=label_fontsize,
            fontweight="normal",
            transform=ax_right.transAxes,
        )

        ax_right.text(
            0.5 - col_offset,
            y_pos - zone_value_gap,
            f"{row['share_pct'] * 100:.1f}%",
            ha="center",
            va="top",
            color=stat_color,
            fontsize=value_fontsize,
            fontweight="bold",
            transform=ax_right.transAxes,
        )

        ax_right.text(
            0.5 + col_offset,
            y_pos - zone_value_gap,
            f"{row['fg_pct'] * 100:.1f}%",
            ha="center",
            va="top",
            color=stat_color,
            fontsize=value_fontsize,
            fontweight="bold",
            transform=ax_right.transAxes,
        )

    ax_right_pos = ax_right.get_position()
    label_y_offset = 0.0
    label_y = (legend_bottom - ax_right_pos.y0) / ax_right_pos.height + label_y_offset
    label_y = max(0.02, min(0.98, label_y))
    ax_right.text(
        0.5 - col_offset,
        label_y,
        "Shot Share",
        ha="center",
        va="center",
        color=stat_label_color,
        fontsize=label_fontsize,
        style="italic",
        alpha=1.0,
        transform=ax_right.transAxes,
    )
    ax_right.text(
        0.5 + col_offset,
        label_y,
        "Field Goal %",
        ha="center",
        va="center",
        color=stat_label_color,
        fontsize=label_fontsize,
        style="italic",
        alpha=1.0,
        transform=ax_right.transAxes,
    )

    # Use the provided output_filename
    plt.savefig(
        output_filename,
        dpi=300,
        bbox_inches="tight",
        facecolor=bg_color,
    )
    print(f"\nSaved improved visualization to: {output_filename}")
    if show_plot:
        plt.show()
    plt.close(fig) # Close the figure to free memory

if __name__ == "__main__":
    current_season = "2025-26"
    current_season_type = "regular"

    # Dark Theme Plot
    generate_shot_chart(
        season=current_season,
        season_type=current_season_type,
        bg_color="#23272B",
        court_bg_color="#30343A",
        court_line_color="white",
        font_color="white",
        stat_label_color="white",
        output_filename="shot_density_hexbin_2025-26_dark.png",
        title_text="Shot Density Heatmap Analysis",
        subtitle_text="2025-2026 NBA Regular Season",
        author_name_text="Kaleb Coleman",
        cmap_hexbin="YlOrRd",
        stat_value_cmap_index=0.6,
        show_plot=False # Don't show this plot immediately
    )

    # Light Theme Plot
    generate_shot_chart(
        season=current_season,
        season_type=current_season_type,
        bg_color="white",
        court_bg_color="#D0D0D0", # Darkened light grey
        court_line_color="black",
        font_color="black",
        stat_label_color="gray", # Slightly lighter than black for labels
        output_filename="shot_density_hexbin_2025-26_light.png",
        title_text="Shot Density Heatmap Analysis",
        subtitle_text="2025-2026 NBA Regular Season",
        author_name_text="Kaleb Coleman",
        cmap_hexbin="YlOrRd", # Changed back to YlOrRd
        stat_value_cmap_index=0.6,
        show_plot=True # Show the last plot generated
    )

