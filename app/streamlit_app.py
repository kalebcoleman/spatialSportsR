import os
import sqlite3
from pathlib import Path

import math

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.patches import Arc, Circle, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "parsed" / "nba.sqlite"
APP_BG = "#0B0F1A"
PANEL_BG = "#121826"
COURT_BG = "#23272B"
COURT_LINE = "#FFFFFF"
TEXT_COLOR = "#E6E8EE"
MUTED_TEXT = "#98A1B3"
ACCENT = "#F5C84C"
PLOT_WIDTH = 620


def resolve_db_path(sidebar_value: str) -> Path:
    env_path = os.getenv("SPATIALSPORTSR_DB_PATH")
    candidate = sidebar_value or env_path or str(DEFAULT_DB_PATH)
    return Path(candidate).expanduser()


@st.cache_data(show_spinner=False)
def get_seasons(db_path: str) -> list[str]:
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(
            "SELECT DISTINCT season FROM nba_stats_shots ORDER BY season",
            con,
        )
    return df["season"].dropna().tolist()


@st.cache_data(show_spinner=False)
def get_season_types(db_path: str, season: str) -> list[str]:
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT season_type
            FROM nba_stats_shots
            WHERE season = ?
            ORDER BY season_type
            """,
            con,
            params=[season],
        )
    return df["season_type"].dropna().tolist()


@st.cache_data(show_spinner=False)
def get_teams(db_path: str, season: str, season_type: str) -> list[str]:
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT TEAM_NAME
            FROM nba_stats_shots
            WHERE season = ?
              AND season_type = ?
            ORDER BY TEAM_NAME
            """,
            con,
            params=[season, season_type],
        )
    return df["TEAM_NAME"].dropna().tolist()


@st.cache_data(show_spinner=False)
def get_players(
    db_path: str,
    season: str,
    season_type: str,
    team_name: str | None,
) -> list[str]:
    sql = """
        SELECT DISTINCT PLAYER_NAME
        FROM nba_stats_shots
        WHERE season = ?
          AND season_type = ?
    """
    params = [season, season_type]
    if team_name:
        sql += " AND TEAM_NAME = ?"
        params.append(team_name)
    sql += " ORDER BY PLAYER_NAME"
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(sql, con, params=params)
    return df["PLAYER_NAME"].dropna().tolist()


def load_shots(
    db_path: str,
    season: str,
    season_type: str,
    team_name: str | None,
    player_name: str | None,
) -> pd.DataFrame:
    sql = """
        SELECT LOC_X, LOC_Y, SHOT_MADE_FLAG, SHOT_TYPE, SHOT_ZONE_BASIC
        FROM nba_stats_shots
        WHERE season = ?
          AND season_type = ?
          AND SHOT_ATTEMPTED_FLAG = 1
    """
    params = [season, season_type]
    if team_name:
        sql += " AND TEAM_NAME = ?"
        params.append(team_name)
    if player_name:
        sql += " AND PLAYER_NAME = ?"
        params.append(player_name)
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(sql, con, params=params)
    return df


@st.cache_data(show_spinner=False)
def get_shot_points(
    db_path: str,
    season: str,
    season_type: str,
    team_name: str | None,
    player_name: str | None,
) -> pd.DataFrame:
    sql = """
        SELECT
          shot_id,
          LOC_X,
          LOC_Y,
          SHOT_MADE_FLAG
        FROM nba_stats_shots
        WHERE season = ?
          AND season_type = ?
          AND SHOT_ATTEMPTED_FLAG = 1
    """
    params = [season, season_type]
    if team_name:
        sql += " AND TEAM_NAME = ?"
        params.append(team_name)
    if player_name:
        sql += " AND PLAYER_NAME = ?"
        params.append(player_name)
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(sql, con, params=params)
    return df


@st.cache_data(show_spinner=False)
def get_shot_detail(db_path: str, shot_id: str) -> dict | None:
    if not shot_id:
        return None
    sql = """
        SELECT
          shot_id,
          GAME_DATE,
          GAME_ID,
          HTM,
          VTM,
          PLAYER_NAME,
          TEAM_NAME,
          SHOT_TYPE,
          SHOT_DISTANCE,
          SHOT_ZONE_BASIC,
          SHOT_ZONE_AREA,
          SHOT_ZONE_RANGE,
          PERIOD,
          MINUTES_REMAINING,
          SECONDS_REMAINING,
          SHOT_MADE_FLAG,
          LOC_X,
          LOC_Y
        FROM nba_stats_shots
        WHERE shot_id = ?
        LIMIT 1
    """
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(sql, con, params=[shot_id])
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def draw_court(
    ax=None,
    color="white",
    lw=2,
    bg="#0b0f1a",
    show_axis=False,
    full_court=False,
    draw_corner_threes=True,
    zorder=1,
    pad_x=0,
    pad_y=0,
):
    if ax is None:
        ax = plt.gca()

    ax.set_facecolor(bg)
    baseline_y = -52
    halfcourt_y = 418
    corner_x = 220
    arc_radius = 237.5
    corner_y = (arc_radius**2 - corner_x**2) ** 0.5

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
        ax.add_patch(
            Rectangle(
                (-250 - pad_x, baseline_y),
                500 + (2 * pad_x),
                (halfcourt_y - baseline_y) + pad_y,
                fill=False,
                linewidth=lw,
                color=color,
                zorder=zorder,
            )
        )
        ax.plot(
            [-250 - pad_x, 250 + pad_x],
            [halfcourt_y, halfcourt_y],
            color=color,
            linewidth=lw,
            zorder=zorder,
        )
        ax.add_patch(
            Arc(
                (0, halfcourt_y),
                120,
                120,
                theta1=180,
                theta2=360,
                linewidth=lw,
                color=color,
                zorder=zorder,
            )
        )
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


def compute_summary(df: pd.DataFrame) -> dict:
    total_fga = len(df)
    total_fgm = int(df["SHOT_MADE_FLAG"].sum()) if total_fga else 0
    total_fg = total_fgm / total_fga if total_fga else 0

    is_three = df["SHOT_TYPE"].fillna("").str.contains("3PT", case=False, na=False)
    threes = df[is_three]
    twos = df[~is_three]

    three_fga = len(threes)
    three_fgm = int(threes["SHOT_MADE_FLAG"].sum()) if three_fga else 0
    three_fg = three_fgm / three_fga if three_fga else 0

    two_fga = len(twos)
    two_fgm = int(twos["SHOT_MADE_FLAG"].sum()) if two_fga else 0
    two_fg = two_fgm / two_fga if two_fga else 0

    zone = (
        df.groupby("SHOT_ZONE_BASIC")["SHOT_MADE_FLAG"]
        .agg(fga="size", fgm="sum")
        .reset_index()
    )
    zone["share_pct"] = zone["fga"] / zone["fga"].sum()
    zone["fg_pct"] = zone["fgm"] / zone["fga"]
    zone = zone.sort_values("fga", ascending=False).head(5)

    return {
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


def plot_shots(
    df: pd.DataFrame,
    chart_type: str,
    court_view: str,
) -> plt.Figure:
    full_court = court_view == "Full court"
    figsize = (5.6, 7.0) if full_court else (5.6, 5.0)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(APP_BG)
    if full_court:
        xpad = 35
        ypad = 35
    else:
        xpad = 25
        ypad = 0
    draw_court(
        ax=ax,
        color=COURT_LINE,
        lw=2,
        bg=COURT_BG,
        show_axis=False,
        full_court=full_court,
        draw_corner_threes=True,
        zorder=2,
        pad_x=xpad,
        pad_y=ypad,
    )
    if full_court:
        ax.set_xlim(-250 - xpad, 250 + xpad)
        ax.set_ylim(-52 - ypad, 888 + ypad)
    else:
        ax.set_xlim(-250 - xpad, 250 + xpad)
        ax.set_ylim(-52 - ypad, 418 + ypad)
    ax.margins(0)

    if chart_type == "Hexbin density":
        hb = ax.hexbin(
            df["LOC_X"],
            df["LOC_Y"],
            gridsize=80,
            mincnt=1,
            bins="log",
            cmap="YlOrRd",
            zorder=1,
        )
        cbar = fig.colorbar(hb, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Shot Count (log10)", color=TEXT_COLOR)
        cbar.ax.tick_params(colors=TEXT_COLOR, length=0)
        cbar.outline.set_edgecolor(TEXT_COLOR)
    else:
        made = df[df["SHOT_MADE_FLAG"] == 1]
        missed = df[df["SHOT_MADE_FLAG"] == 0]
        ax.scatter(
            missed["LOC_X"],
            missed["LOC_Y"],
            c="#FF6B6B",
            marker="x",
            s=12,
            alpha=0.35,
            zorder=1,
        )
        ax.scatter(
            made["LOC_X"],
            made["LOC_Y"],
            c=ACCENT,
            marker="o",
            s=16,
            alpha=0.45,
            zorder=1,
        )

    return fig


def arc_points(cx, cy, r, theta1, theta2, n=120):
    angles = [theta1 + (theta2 - theta1) * i / (n - 1) for i in range(n)]
    x = [cx + r * math.cos(math.radians(a)) for a in angles]
    y = [cy + r * math.sin(math.radians(a)) for a in angles]
    return x, y


def add_court_traces(fig, full_court: bool, line_color: str = "#FFFFFF"):
    baseline_y = -52
    halfcourt_y = 418
    corner_x = 220
    arc_radius = 237.5
    corner_y = (arc_radius**2 - corner_x**2) ** 0.5

    def add_half(y_flip=False):
        def y(val):
            return 2 * halfcourt_y - val if y_flip else val

        fig.add_trace(
            go.Scatter(
                x=[-80, 80, 80, -80, -80],
                y=[y(baseline_y), y(baseline_y), y(baseline_y + 190), y(baseline_y + 190), y(baseline_y)],
                mode="lines",
                line=dict(color=line_color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[-60, 60, 60, -60, -60],
                y=[y(baseline_y), y(baseline_y), y(baseline_y + 190), y(baseline_y + 190), y(baseline_y)],
                mode="lines",
                line=dict(color=line_color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        hoop_x, hoop_y = arc_points(0, y(0), 7.5, 0, 360, n=90)
        fig.add_trace(
            go.Scatter(
                x=hoop_x,
                y=hoop_y,
                mode="lines",
                line=dict(color=line_color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[-30, 30],
                y=[y(-12.5), y(-12.5)],
                mode="lines",
                line=dict(color=line_color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        ft_solid = arc_points(0, y(142.5), 60, 0, 180, n=90)
        ft_dashed = arc_points(0, y(142.5), 60, 180, 360, n=90)
        if not y_flip:
            fig.add_trace(
                go.Scatter(
                    x=ft_solid[0],
                    y=ft_solid[1],
                    mode="lines",
                    line=dict(color=line_color, width=2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ft_dashed[0],
                    y=ft_dashed[1],
                    mode="lines",
                    line=dict(color=line_color, width=2, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=ft_dashed[0],
                    y=ft_dashed[1],
                    mode="lines",
                    line=dict(color=line_color, width=2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ft_solid[0],
                    y=ft_solid[1],
                    mode="lines",
                    line=dict(color=line_color, width=2, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        ra = arc_points(0, y(0), 40, 0, 180, n=90)
        ra_alt = arc_points(0, y(0), 40, 180, 360, n=90)
        use_arc = ra_alt if y_flip else ra
        fig.add_trace(
            go.Scatter(
                x=use_arc[0],
                y=use_arc[1],
                mode="lines",
                line=dict(color=line_color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[-corner_x, -corner_x],
                y=[y(baseline_y), y(corner_y)],
                mode="lines",
                line=dict(color=line_color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[corner_x, corner_x],
                y=[y(baseline_y), y(corner_y)],
                mode="lines",
                line=dict(color=line_color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        arc_theta1, arc_theta2 = (202, 338) if y_flip else (22, 158)
        three_arc = arc_points(0, y(0), arc_radius, arc_theta1, arc_theta2, n=150)
        fig.add_trace(
            go.Scatter(
                x=three_arc[0],
                y=three_arc[1],
                mode="lines",
                line=dict(color=line_color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    add_half(y_flip=False)
    if full_court:
        add_half(y_flip=True)
        fig.add_trace(
            go.Scatter(
                x=[-250, 250],
                y=[halfcourt_y, halfcourt_y],
                mode="lines",
                line=dict(color=line_color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        center = arc_points(0, halfcourt_y, 60, 0, 360, n=120)
        fig.add_trace(
            go.Scatter(
                x=center[0],
                y=center[1],
                mode="lines",
                line=dict(color=line_color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[-250, 250, 250, -250, -250],
                y=[baseline_y, baseline_y, 888, 888, baseline_y],
                mode="lines",
                line=dict(color=line_color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[-250, 250, 250, -250, -250],
                y=[baseline_y, baseline_y, halfcourt_y, halfcourt_y, baseline_y],
                mode="lines",
                line=dict(color=line_color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )


def sample_shots_with_far(shots: pd.DataFrame, max_points: int, far_threshold: int) -> pd.DataFrame:
    shots = shots.copy()
    shots["LOC_X"] = pd.to_numeric(shots["LOC_X"], errors="coerce")
    shots["LOC_Y"] = pd.to_numeric(shots["LOC_Y"], errors="coerce")
    shots = shots.dropna(subset=["LOC_X", "LOC_Y", "SHOT_MADE_FLAG"])
    shots["shot_distance_ft"] = ((shots["LOC_X"] ** 2 + shots["LOC_Y"] ** 2) ** 0.5) / 10.0
    if len(shots) <= max_points:
        return shots
    far = shots[shots["shot_distance_ft"] >= far_threshold].copy()
    near = shots[shots["shot_distance_ft"] < far_threshold].copy()
    if len(far) >= max_points:
        return far.sort_values("shot_distance_ft", ascending=False).head(max_points)
    remaining = max_points - len(far)
    if len(near) > remaining:
        near = near.sample(n=remaining, random_state=42)
    return pd.concat([far, near], ignore_index=True)


def plot_scatter_plotly(
    shots: pd.DataFrame,
    court_view: str,
) -> go.Figure:
    shots = shots.copy()
    shots["result"] = shots["SHOT_MADE_FLAG"].map({1: "Make", 0: "Miss"}).fillna("Miss")
    fig = px.scatter(
        shots,
        x="LOC_X",
        y="LOC_Y",
        color="result",
        color_discrete_map={"Make": ACCENT, "Miss": "#FF6B6B"},
        category_orders={"result": ["Miss", "Make"]},
        custom_data=["shot_id"],
        opacity=0.9,
        width=PLOT_WIDTH,
        height=520 if court_view == "Full court" else 460,
    )
    fig.update_traces(
        marker=dict(size=9),
        selected=dict(marker=dict(opacity=0.98, size=9)),
        unselected=dict(marker=dict(opacity=0.98)),
        selectedpoints=[],
        selector=dict(mode="markers"),
    )
    add_court_traces(fig, full_court=court_view == "Full court", line_color=COURT_LINE)
    fig.update_xaxes(range=[-250, 250], showgrid=False, zeroline=False, showticklabels=False)
    y_range = [-52, 888] if court_view == "Full court" else [-52, 418]
    fig.update_yaxes(
        range=y_range,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        autosize=False,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        plot_bgcolor=COURT_BG,
        paper_bgcolor=APP_BG,
        clickmode="event+select",
    )
    return fig


def render_selected_shot(selected_shot: dict | None) -> None:
    st.subheader("Selected Shot")
    if not selected_shot:
        st.markdown(
            f"""
            <div style="width: {PLOT_WIDTH}px; max-width: 100%; margin: 0;">
              <div style="padding: 12px; border-radius: 12px; background: {PANEL_BG};
                   border: 1px solid rgba(255,255,255,0.08); color: {MUTED_TEXT};">
                Click a shot on the chart to see details.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    period = selected_shot.get("PERIOD")
    minutes = selected_shot.get("MINUTES_REMAINING")
    seconds = selected_shot.get("SECONDS_REMAINING")
    if minutes is None or seconds is None or pd.isna(minutes) or pd.isna(seconds):
        clock = "N/A"
    else:
        clock = f"{int(minutes):02d}:{int(seconds):02d}"
    if period is None or pd.isna(period):
        period_text = "N/A"
    else:
        period_text = str(int(period))
    is_make = selected_shot.get("SHOT_MADE_FLAG") == 1
    result = "Make" if is_make else "Miss"
    result_color = ACCENT if is_make else "#FF6B6B"
    shot_distance = selected_shot.get("SHOT_DISTANCE")
    if shot_distance is None or pd.isna(shot_distance):
        distance_text = "N/A"
    else:
        distance_text = f"{int(round(float(shot_distance)))} ft"
    zone_basic = selected_shot.get("SHOT_ZONE_BASIC") or "N/A"
    zone_area = selected_shot.get("SHOT_ZONE_AREA") or "N/A"
    zone_range = selected_shot.get("SHOT_ZONE_RANGE") or "N/A"
    game_date = selected_shot.get("GAME_DATE") or "N/A"
    htm = selected_shot.get("HTM") or ""
    vtm = selected_shot.get("VTM") or ""
    matchup = f"{htm} vs {vtm}".strip()

    st.markdown(
        f"""
        <div style="width: {PLOT_WIDTH}px; max-width: 100%; margin: 0;">
          <div style="padding: 12px; border-radius: 12px; background: {PANEL_BG};
               border: 1px solid rgba(255,255,255,0.08);">
            <div style="font-weight: 600; margin-bottom: 8px;">{selected_shot.get("PLAYER_NAME", "Unknown")}</div>
            <div style="color: {MUTED_TEXT}; font-size: 0.9rem; margin-bottom: 6px;">
              {selected_shot.get("TEAM_NAME", "Unknown Team")}
            </div>
            <div style="margin-bottom: 6px;">{game_date} · {matchup}</div>
            <div style="margin-bottom: 6px;">Period {period_text} · {clock}</div>
            <div style="margin-bottom: 6px;">{selected_shot.get("SHOT_TYPE", "Shot")} · {distance_text}</div>
            <div style="margin-bottom: 6px;">{zone_basic} / {zone_area} / {zone_range}</div>
            <div style="font-weight: 600; color: {result_color};">{result}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="NBA Shot Density Dashboard",
        layout="wide",
    )
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            background: {APP_BG};
            color: {TEXT_COLOR};
            font-family: "Space Grotesk", "IBM Plex Sans", "SF Pro Display", "Segoe UI", sans-serif;
        }}
        .stApp {{
            background: {APP_BG};
        }}
        [data-testid="stSidebar"] {{
            background: {PANEL_BG};
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {{
            color: {TEXT_COLOR};
        }}
        h1, h2, h3, h4 {{
            color: {TEXT_COLOR};
            letter-spacing: 0.02em;
        }}
        p, span, label {{
            color: {TEXT_COLOR};
        }}
        .stMetric {{
            background: {PANEL_BG};
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px;
        }}
        .stMetric div {{
            overflow: visible !important;
            text-overflow: unset !important;
            white-space: normal !important;
        }}
        .stMetric label {{
            color: {MUTED_TEXT};
            font-size: 0.85rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("NBA Shot Map Dashboard")

    with st.sidebar:
        st.header("Filters")
        default_db = os.getenv("SPATIALSPORTSR_DB_PATH", str(DEFAULT_DB_PATH))
        db_input = st.text_input("SQLite DB path", value=default_db)
        db_path = resolve_db_path(db_input)
        if not db_path.exists():
            st.error(f"DB not found: {db_path}")
            st.stop()

        seasons = get_seasons(str(db_path))
        if not seasons:
            st.error("No seasons found in the database.")
            st.stop()

        season = st.selectbox("Season", seasons, index=len(seasons) - 1)
        season_types = get_season_types(str(db_path), season)
        default_season_type = (
            season_types.index("regular")
            if "regular" in season_types
            else 0
        )
        season_type = st.selectbox(
            "Season Type",
            season_types,
            index=default_season_type,
        )

        teams = get_teams(str(db_path), season, season_type)
        team_options = ["All Teams"] + teams
        team_choice = st.selectbox("Team", team_options)
        selected_team = None if team_choice == "All Teams" else team_choice

        players = get_players(
            str(db_path),
            season,
            season_type,
            selected_team,
        )
        player_options = ["All Players"] + players
        player_choice = st.selectbox("Player", player_options)
        selected_player = None if player_choice == "All Players" else player_choice

        court_view = st.radio(
            "Court View",
            ["Full court", "Half court"],
            index=1,
        )
        max_points = st.slider(
            "Max points",
            min_value=5000,
            max_value=100000,
            value=40000,
            step=5000,
        )
        far_threshold = st.slider(
            "Always include shots beyond (ft)",
            min_value=30,
            max_value=60,
            value=40,
            step=5,
        )

    subtitle_parts = [season, season_type]
    if selected_team:
        subtitle_parts.append(selected_team)
    if selected_player:
        subtitle_parts.append(selected_player)
    st.caption(" | ".join(subtitle_parts))

    shots = load_shots(
        str(db_path),
        season,
        season_type,
        selected_team,
        selected_player,
    )
    filter_signature = (
        season,
        season_type,
        selected_team or "ALL_TEAMS",
        selected_player or "ALL_PLAYERS",
    )
    if st.session_state.get("filter_signature") != filter_signature:
        st.session_state["filter_signature"] = filter_signature
        st.session_state.pop("selected_shot", None)

    shots_points = get_shot_points(
        str(db_path),
        season,
        season_type,
        selected_team,
        selected_player,
    )
    raw_points_count = len(shots_points)
    shots_points = sample_shots_with_far(
        shots_points,
        max_points=max_points or 40000,
        far_threshold=far_threshold or 40,
    )
    omitted_points = max(0, raw_points_count - len(shots_points))
    if omitted_points > 0:
        st.caption(
            f"Showing {len(shots_points):,} points. Omitted {omitted_points:,} for speed "
            f"(all shots ≥ {far_threshold} ft included)."
        )

    if shots.empty:
        available_types = get_season_types(str(db_path), season)
        st.warning(
            f"No shots found for {season} {season_type}. "
            f"Available season types: {', '.join(available_types)}"
        )
        st.stop()

    summary = compute_summary(shots)

    col_plot, col_side = st.columns([2.2, 1])

    with col_plot:
        fig = plot_scatter_plotly(shots_points, court_view)
        selection = st.plotly_chart(
            fig,
            use_container_width=False,
            config={"displayModeBar": False},
            on_select="rerun",
            selection_mode="points",
        )
        selected_points = None
        if isinstance(selection, dict):
            selected_points = selection.get("points", []) or selection.get("selection", {}).get("points", [])
        else:
            sel = getattr(selection, "selection", None)
            if isinstance(sel, dict):
                selected_points = sel.get("points", [])

        if selected_points is not None and len(selected_points) == 0:
            st.session_state.pop("selected_shot", None)
        elif selected_points:
            shot_id = selected_points[0]["customdata"][0]
            detail = get_shot_detail(str(db_path), str(shot_id))
            if detail:
                st.session_state["selected_shot"] = detail

        render_selected_shot(st.session_state.get("selected_shot"))

    with col_side:
        st.markdown(
            '<div style="margin-top:-18px; margin-bottom: 8px; font-size: 1.35rem; font-weight: 700;">Summary</div>',
            unsafe_allow_html=True,
        )
        metrics_cols = st.columns(2)
        metrics_cols[0].metric("Field Goal Attempts", f"{summary['total_fga']:,}")
        metrics_cols[1].metric("Field Goal %", f"{summary['total_fg'] * 100:.1f}%")
        metrics_cols = st.columns(2)
        metrics_cols[0].metric("2PT Field Goal %", f"{summary['two_fg'] * 100:.1f}%")
        metrics_cols[1].metric("3PT Field Goal %", f"{summary['three_fg'] * 100:.1f}%")

        st.subheader("Top Shot Zones")
        zone_table = summary["zone_table"].copy()
        zone_table["share_pct"] = zone_table["share_pct"] * 100
        zone_table["fg_pct"] = zone_table["fg_pct"] * 100
        zone_table = zone_table.sort_values("share_pct")

        fig_bar, ax_bar = plt.subplots(figsize=(4, 3.4))
        fig_bar.patch.set_facecolor(PANEL_BG)
        ax_bar.set_facecolor(PANEL_BG)
        ax_bar.barh(
            zone_table["SHOT_ZONE_BASIC"],
            zone_table["share_pct"],
            color=ACCENT,
            alpha=0.85,
        )
        ax_bar.set_xlabel("Shot Share (%)", color=MUTED_TEXT, fontsize=9)
        ax_bar.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax_bar.grid(axis="x", color="white", alpha=0.08, linewidth=1)
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        max_share = max(zone_table["share_pct"].max(), 1)
        ax_bar.set_xlim(0, max_share * 1.25)
        for idx, share in enumerate(zone_table["share_pct"]):
            ax_bar.text(
                share + max_share * 0.03,
                idx,
                f"{share:.1f}%",
                va="center",
                ha="left",
                color=TEXT_COLOR,
                fontsize=9,
            )
        st.pyplot(fig_bar, use_container_width=True)

        made_count = int((shots["SHOT_MADE_FLAG"] == 1).sum())
        miss_count = int((shots["SHOT_MADE_FLAG"] == 0).sum())
        st.markdown(
            f"""
            <div style="margin-top: 8px; margin-bottom: 4px; font-weight: 600;">Legend</div>
            <div style="display: grid; grid-template-columns: 16px auto auto; gap: 8px 10px; align-items: center;">
              <span style="width: 10px; height: 10px; border-radius: 50%; background: {ACCENT}; display: inline-block;"></span>
              <span>Make</span>
              <span style="color: {MUTED_TEXT}; text-align: right;">{made_count:,}</span>
              <span style="width: 10px; height: 10px; border-radius: 2px; background: #FF6B6B; display: inline-block;"></span>
              <span>Miss</span>
              <span style="color: {MUTED_TEXT}; text-align: right;">{miss_count:,}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )




if __name__ == "__main__":
    main()
