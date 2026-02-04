import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.patches import Arc, Circle, Rectangle

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "parsed" / "nba.sqlite"
APP_BG = "#0B0F1A"
PANEL_BG = "#121826"
COURT_BG = "#23272B"
COURT_LINE = "#FFFFFF"
TEXT_COLOR = "#E6E8EE"
MUTED_TEXT = "#98A1B3"
ACCENT = "#F5C84C"


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
    figsize = (7, 9) if full_court else (7, 6)
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
        cbar = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
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

    st.title("NBA Shot Density Dashboard")

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

        chart_type = st.radio(
            "Chart Type",
            ["Hexbin density", "Make/Miss scatter"],
            index=0,
        )
        court_view = st.radio(
            "Court View",
            ["Full court", "Half court"],
            index=1,
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
        fig = plot_shots(shots, chart_type, court_view)
        st.pyplot(fig, use_container_width=True)

    with col_side:
        st.subheader("Summary")
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

        if chart_type == "Make/Miss scatter":
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
