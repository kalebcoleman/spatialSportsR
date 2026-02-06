"""
Advanced NBA Analytics Module.

Three integrated analytics products using existing xFG model:
1. Shot Quality (Residual Analysis)
2. Shot Difficulty Index (SDI)
3. Player Shot Archetype Clustering
4. Player Metric Correlation Heatmap

Uses pre-computed xP_prob from expected_points_analysis.py
"""

import os
import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Add analysis directory to path to allow imports if run from root
ANALYSIS_DIR = Path(__file__).parent
sys.path.append(str(ANALYSIS_DIR))

# Import from utils
from utils.court_utils import draw_half_court, setup_shot_chart_axes


# Configuration
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "parsed" / "nba.sqlite"
DB_PATH = Path(os.getenv("SPATIALSPORTSR_DB_PATH", str(DEFAULT_DB_PATH))).expanduser()
DATA_DIR = ANALYSIS_DIR / "data"
FIGURES_DIR = ANALYSIS_DIR / "figures"
MIN_SHOTS = 200  # Minimum shots for player analysis
DISTANCE_BINS = [0, 3, 10, 16, 24, 30, 40]
MIN_PLAYERS_PER_COMPONENT = 8
MIN_CLUSTER_SIZE = 4

# Position/role configuration (easy to tweak)
POSITION_CONFIG = {
    "guard_assist_quantile": 0.50,   # used if guard_assist_threshold is None
    "guard_assist_threshold": None,  # set numeric (e.g., 25.0) to override quantile
    "use_assist_for_guard_split": False,
    "guard_distance_threshold": 14.0,
    "guard_3pt_threshold": 0.33,
    "pf_rim_threshold": 0.35,
    "pf_distance_threshold": 12.0,
}

ROLE_CONFIG = {
    "fallback_big_rim": 0.45,
    "fallback_big_distance": 8.0,
    "fallback_guard_3pt": 0.40,
    "fallback_guard_distance": 15.0,
    # Override guards that look like frontcourt (stretch bigs)
    "frontcourt_rim_override": 0.35,
    "frontcourt_hook_override": 0.03,
    "frontcourt_dunk_override": 0.03,
    "big_override_rim": 0.50,
    "big_override_distance": 8.0,
}

USE_POSITIONS = False  # set True to use derived positions for role grouping/plots

CLUSTER_FEATURES = [
    "pct_rim",
    "pct_midrange",
    "pct_3pt",
    "pct_corner_3",
    "distance_entropy",
    "distance_std",
    "avg_sdi",
    "pullup_rate",
    "usage_pct",
    "attempts_per_game",
]

# Feature weighting after standardization (1.0 = neutral)
CLUSTER_WEIGHTS = {
    "avg_sdi": 1.4,
    "distance_entropy": 1.2,
    "pullup_rate": 1.1,
}

POSITION_SHAPES = {
    "PG": "o",
    "SG": "s",
    "SF": "^",
    "PF": "D",
    "C": "P",
    "Unknown": "X",
}

POSITION_COLORS = {
    "PG": "#1f77b4",
    "SG": "#ff7f0e",
    "SF": "#2ca02c",
    "PF": "#d62728",
    "C": "#9467bd",
    "Unknown": "#7f7f7f",
}


# =============================================================================
# PHASE 1: Shot Quality (Residual Analysis)
# =============================================================================

def _parse_minutes(value):
    """Parse minutes string like '32:15' into float minutes."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        parts = text.split(":")
        try:
            mins = int(parts[0])
            secs = int(parts[1]) if len(parts) > 1 else 0
            return mins + (secs / 60.0)
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def load_usage_from_sqlite(season, season_type):
    """Load minutes-weighted usage% from NBA Stats usage table."""
    if not DB_PATH.exists():
        print(f"Warning: SQLite DB not found at {DB_PATH}; usage% will be missing.")
        return pd.DataFrame(columns=["player_id", "usage_pct"])

    season_type = str(season_type).strip().lower()
    query = """
    SELECT player_id, minutes, usagePercentage
    FROM nba_stats_player_box_usage
    WHERE season = ? AND season_type = ? AND usagePercentage IS NOT NULL
    """

    with sqlite3.connect(str(DB_PATH)) as con:
        df = pd.read_sql_query(query, con, params=[season, season_type])

    if df.empty:
        print("Warning: No usage% rows found; usage% will be missing.")
        return pd.DataFrame(columns=["player_id", "usage_pct"])

    df["usagePercentage"] = pd.to_numeric(df["usagePercentage"], errors="coerce")
    df["minutes_num"] = df["minutes"].apply(_parse_minutes)
    df = df[df["usagePercentage"].notna()].copy()

    if df.empty:
        return pd.DataFrame(columns=["player_id", "usage_pct"])

    def weighted_usage(group):
        minutes = group["minutes_num"].sum()
        if minutes > 0:
            return (group["usagePercentage"] * group["minutes_num"]).sum() / minutes
        return group["usagePercentage"].mean()

    usage = df.groupby("player_id").apply(weighted_usage).reset_index(name="usage_pct")
    return usage


def load_position_from_sqlite(season, season_type):
    """Load minutes-weighted position (G/F/C) from NBA Stats boxscore table."""
    if not DB_PATH.exists():
        print(f"Warning: SQLite DB not found at {DB_PATH}; positions will be missing.")
        return pd.DataFrame(columns=["player_id", "position_raw", "position_source"])

    season_type = str(season_type).strip().lower()
    query = """
    SELECT player_id, position, minutes
    FROM nba_stats_player_box_traditional
    WHERE season = ? AND season_type = ? AND position IS NOT NULL
    """

    with sqlite3.connect(str(DB_PATH)) as con:
        df = pd.read_sql_query(query, con, params=[season, season_type])

    if df.empty:
        print("Warning: No position rows found; positions will be missing.")
        return pd.DataFrame(columns=["player_id", "position_raw", "position_source"])

    df["position"] = df["position"].astype(str).str.strip().str.upper()
    df = df[df["position"].ne("")]
    df["minutes_num"] = df["minutes"].apply(_parse_minutes)
    if df.empty:
        return pd.DataFrame(columns=["player_id", "position_raw", "position_source"])

    pos_minutes = (
        df.groupby(["player_id", "position"])["minutes_num"]
        .sum()
        .reset_index()
        .sort_values(["player_id", "minutes_num"], ascending=[True, False])
    )
    pos_pick = pos_minutes.drop_duplicates("player_id")
    pos_pick = pos_pick.rename(columns={"position": "position_raw"})
    pos_pick["position_source"] = "nba_stats_player_box_traditional"
    return pos_pick[["player_id", "position_raw", "position_source"]]


def load_assist_pct_from_sqlite(season, season_type):
    """Load minutes-weighted assist% from NBA Stats advanced boxscore table."""
    if not DB_PATH.exists():
        print(f"Warning: SQLite DB not found at {DB_PATH}; assist% will be missing.")
        return pd.DataFrame(columns=["player_id", "assist_pct"])

    season_type = str(season_type).strip().lower()
    query = """
    SELECT player_id, assistPercentage, minutes
    FROM nba_stats_player_box_advanced
    WHERE season = ? AND season_type = ? AND assistPercentage IS NOT NULL
    """

    with sqlite3.connect(str(DB_PATH)) as con:
        df = pd.read_sql_query(query, con, params=[season, season_type])

    if df.empty:
        print("Warning: No assist% rows found; assist% will be missing.")
        return pd.DataFrame(columns=["player_id", "assist_pct"])

    df["assistPercentage"] = pd.to_numeric(df["assistPercentage"], errors="coerce")
    df["minutes_num"] = df["minutes"].apply(_parse_minutes)
    df = df[df["assistPercentage"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["player_id", "assist_pct"])

    def weighted_assist(group):
        minutes = group["minutes_num"].sum()
        if minutes > 0:
            return (group["assistPercentage"] * group["minutes_num"]).sum() / minutes
        return group["assistPercentage"].mean()

    assist = df.groupby("player_id").apply(weighted_assist).reset_index(name="assist_pct")
    return assist


def derive_positions(features, position_df=None, assist_df=None, config=None):
    """
    Derive PG/SG/SF/PF/C using position + assist% + shot diet heuristics.
    """
    if config is None:
        config = POSITION_CONFIG

    features = features.copy()
    if position_df is not None and not position_df.empty:
        position_df = position_df.copy()
        position_df["player_id"] = position_df["player_id"].astype(str)
        features["PLAYER_ID"] = features["PLAYER_ID"].astype(str)
        features = features.merge(position_df, left_on="PLAYER_ID", right_on="player_id", how="left")
        features = features.drop(columns=["player_id"], errors="ignore")
    else:
        features["position_raw"] = np.nan
        features["position_source"] = np.nan

    if assist_df is not None and not assist_df.empty:
        assist_df = assist_df.copy()
        assist_df["player_id"] = assist_df["player_id"].astype(str)
        features["PLAYER_ID"] = features["PLAYER_ID"].astype(str)
        features = features.merge(assist_df, left_on="PLAYER_ID", right_on="player_id", how="left")
        features = features.drop(columns=["player_id"], errors="ignore")
    else:
        features["assist_pct"] = np.nan

    use_assist = bool(config.get("use_assist_for_guard_split", False))
    assist_threshold = config.get("guard_assist_threshold")
    if use_assist and assist_threshold is None:
        guard_pool = features.loc[features["position_raw"] == "G", "assist_pct"]
        if guard_pool.notna().any():
            assist_threshold = guard_pool.dropna().quantile(config.get("guard_assist_quantile", 0.5))
        else:
            all_pool = features["assist_pct"]
            if all_pool.notna().any():
                assist_threshold = all_pool.dropna().quantile(config.get("guard_assist_quantile", 0.5))
            else:
                assist_threshold = np.nan

    pct_rim = features.get("pct_rim", pd.Series([0] * len(features)))
    pct_3pt = features.get("pct_3pt", pd.Series([0] * len(features)))
    avg_distance = features.get("avg_distance", pd.Series([0] * len(features)))

    pf_rim_threshold = config.get("pf_rim_threshold", 0.35)
    pf_distance_threshold = config.get("pf_distance_threshold", 12.0)

    derived = pd.Series(index=features.index, data="Unknown")
    raw = features.get("position_raw")

    if raw is not None:
        is_c = raw == "C"
        is_g = raw == "G"
        is_f = raw == "F"

        derived.loc[is_c] = "C"

        if use_assist and assist_threshold is not None and not np.isnan(assist_threshold):
            derived.loc[is_g & (features["assist_pct"] >= assist_threshold)] = "PG"
            derived.loc[is_g & (features["assist_pct"] < assist_threshold)] = "SG"
        else:
            guard_dist = config.get("guard_distance_threshold", 14.0)
            guard_3pt = config.get("guard_3pt_threshold", 0.33)
            sg_mask = (pct_3pt >= guard_3pt) | (avg_distance >= guard_dist)
            derived.loc[is_g & sg_mask] = "SG"
            derived.loc[is_g & ~sg_mask] = "PG"

        pf_mask = (pct_rim >= pf_rim_threshold) | (avg_distance <= pf_distance_threshold)
        derived.loc[is_f & pf_mask] = "PF"
        derived.loc[is_f & ~pf_mask] = "SF"

    # Fallback when no raw position
    missing_raw = raw.isna() if raw is not None else pd.Series([True] * len(features))
    if missing_raw.any():
        guard_fallback = (avg_distance >= ROLE_CONFIG["fallback_guard_distance"]) | (
            pct_3pt >= ROLE_CONFIG["fallback_guard_3pt"]
        )
        big_fallback = (pct_rim >= ROLE_CONFIG["fallback_big_rim"]) & (
            avg_distance <= ROLE_CONFIG["fallback_big_distance"]
        )

        if use_assist and assist_threshold is not None and not np.isnan(assist_threshold):
            derived.loc[missing_raw & guard_fallback & (features["assist_pct"] >= assist_threshold)] = "PG"
            derived.loc[missing_raw & guard_fallback & (features["assist_pct"] < assist_threshold)] = "SG"
        else:
            guard_dist = config.get("guard_distance_threshold", 14.0)
            guard_3pt = config.get("guard_3pt_threshold", 0.33)
            sg_mask = guard_fallback & ((pct_3pt >= guard_3pt) | (avg_distance >= guard_dist))
            derived.loc[missing_raw & sg_mask] = "SG"
            derived.loc[missing_raw & guard_fallback & ~sg_mask] = "PG"

        derived.loc[missing_raw & big_fallback & ~guard_fallback] = "C"

        pf_mask = (pct_rim >= pf_rim_threshold) | (avg_distance <= pf_distance_threshold)
        derived.loc[missing_raw & ~guard_fallback & ~big_fallback & pf_mask] = "PF"
        derived.loc[missing_raw & ~guard_fallback & ~big_fallback & ~pf_mask] = "SF"

    features["position_derived"] = derived
    return features

def compute_residuals(df):
    """
    Compute shot-level residuals (actual - expected).
    
    Positive residual = player made a shot they were expected to miss
    Negative residual = player missed a shot they were expected to make
    """
    df = df.copy()
    df['residual'] = df['SHOT_MADE_FLAG'] - df['xP_prob']
    return df


def aggregate_player_residuals(df, min_shots=MIN_SHOTS):
    """
    Aggregate residuals by player.
    
    Returns DataFrame with:
    - avg_xFG: average expected FG%
    - actual_fg_pct: actual FG%
    - residual_fg_pct: actual - expected (overperformance)
    - volume: total shots
    """
    player_stats = df.groupby('PLAYER_NAME').agg({
        'xP_prob': 'mean',
        'SHOT_MADE_FLAG': ['mean', 'sum', 'count'],
        'residual': 'mean'
    }).reset_index()
    
    player_stats.columns = ['player', 'avg_xFG', 'actual_fg_pct', 'makes', 'attempts', 'avg_residual']
    player_stats = player_stats[player_stats['attempts'] >= min_shots]
    player_stats['residual_fg_pct'] = player_stats['actual_fg_pct'] - player_stats['avg_xFG']
    
    return player_stats.sort_values('residual_fg_pct', ascending=False)


def aggregate_team_residuals(df):
    """Aggregate residuals by team (using GAME_ID prefix for team)."""
    # Extract team from player's most common game
    # For now, use zone-level aggregation since we don't have clean team column
    zone_stats = df.groupby('SHOT_ZONE_BASIC').agg({
        'xP_prob': 'mean',
        'SHOT_MADE_FLAG': 'mean',
        'residual': ['mean', 'count']
    }).reset_index()
    
    zone_stats.columns = ['zone', 'avg_xFG', 'actual_fg_pct', 'avg_residual', 'attempts']
    zone_stats['residual_fg_pct'] = zone_stats['actual_fg_pct'] - zone_stats['avg_xFG']
    
    return zone_stats.sort_values('attempts', ascending=False)


def plot_residual_heatmap(df, player_name, output_path):
    """
    Create court heatmap showing where a player over/underperforms.
    
    Red = underperforming (missing expected makes)
    Green = overperforming (making expected misses)
    """
    player_df = df[df['PLAYER_NAME'] == player_name].copy()
    
    if len(player_df) < 50:
        print(f"Not enough shots for {player_name}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 11))
    draw_half_court(ax, outer_lines=True)
    
    # Color by residual
    scatter = ax.scatter(
        player_df['LOC_X'], 
        player_df['LOC_Y'],
        c=player_df['residual'],
        cmap='RdYlGn',
        vmin=-0.5, vmax=0.5,
        s=50, alpha=0.7,
        edgecolors='black', linewidths=0.5
    )
    
    setup_shot_chart_axes(ax)
    ax.set_title(f'{player_name} - Shot Residuals (Green=Overperform, Red=Underperform)', fontsize=14)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Residual (Actual - Expected)', fontsize=12)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# PHASE 2: Shot Difficulty Index (SDI)
# =============================================================================

def compute_sdi(df):
    """
    Compute Shot Difficulty Index per shot.
    
    SDI = weighted combination of:
    - Distance (30%): farther = harder
    - Shot clock pressure (20%): less time = harder
    - Shot type difficulty (20%): pull-up > catch-and-shoot
    - Zone difficulty (15%): mid-range harder than paint
    - Angle difficulty (15%): extreme angles harder
    
    Higher SDI = more difficult shot
    """
    df = df.copy()
    
    # 1. Distance component (0-1 scale, max at 35 feet)
    df['sdi_distance'] = (df['shot_distance_feet'].clip(0, 35) / 35.0)
    
    # 2. Shot clock pressure (less time = harder)
    # seconds_in_period: 720 = start of period, 0 = end
    # We invert: closer to 0 = more pressure
    df['sdi_clock'] = 1 - (df['seconds_in_period'].clip(0, 720) / 720.0)
    
    # 3. Shot type difficulty
    # Pull-up/fadeaway harder than catch-and-shoot
    action = df['ACTION_TYPE'].str.lower().fillna('')
    df['sdi_shot_type'] = 0.3  # baseline
    df.loc[action.str.contains('pullup|step back|fadeaway|turnaround'), 'sdi_shot_type'] = 0.8
    df.loc[action.str.contains('driving|running'), 'sdi_shot_type'] = 0.6
    df.loc[action.str.contains('dunk'), 'sdi_shot_type'] = 0.1  # dunks are easy
    df.loc[action.str.contains('layup') & ~action.str.contains('driving'), 'sdi_shot_type'] = 0.2
    
    # 4. Zone difficulty
    zone_difficulty = {
        'Restricted Area': 0.1,
        'In The Paint (Non-RA)': 0.4,
        'Mid-Range': 0.7,
        'Left Corner 3': 0.5,
        'Right Corner 3': 0.5,
        'Above the Break 3': 0.6,
        'Backcourt': 0.9
    }
    df['sdi_zone'] = df['SHOT_ZONE_BASIC'].map(zone_difficulty).fillna(0.5)
    
    # 5. Angle difficulty (extreme angles harder)
    df['sdi_angle'] = np.abs(df['shot_angle']) / (np.pi / 2)  # 0-1 scale
    
    # Weighted combination
    df['SDI'] = (
        0.30 * df['sdi_distance'] +
        0.20 * df['sdi_clock'] +
        0.20 * df['sdi_shot_type'] +
        0.15 * df['sdi_zone'] +
        0.15 * df['sdi_angle']
    )
    
    return df


def aggregate_player_sdi(df, min_shots=MIN_SHOTS):
    """Aggregate SDI by player."""
    player_sdi = df.groupby('PLAYER_NAME').agg({
        'SDI': 'mean',
        'xP_prob': 'mean',
        'SHOT_MADE_FLAG': ['mean', 'count']
    }).reset_index()
    
    player_sdi.columns = ['player', 'avg_sdi', 'avg_xFG', 'actual_fg_pct', 'attempts']
    player_sdi = player_sdi[player_sdi['attempts'] >= min_shots]
    
    return player_sdi.sort_values('avg_sdi', ascending=False)


def plot_sdi_vs_xfg(player_sdi, output_path):
    """
    Scatter plot: Avg SDI vs Actual FG% per player.
    
    Quadrants:
    - Top-right: difficult shots, high conversion (elite shot-makers)
    - Bottom-right: difficult shots, low conversion (volume shooters)
    - Top-left: easy shots, high conversion (efficient role players)
    - Bottom-left: easy shots, low conversion (inefficient)
    
    Color shows FG% residual (actual - expected) to highlight overperformers.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Compute residual for color coding
    player_sdi = player_sdi.copy()
    player_sdi['residual'] = player_sdi['actual_fg_pct'] - player_sdi['avg_xFG']
    
    scatter = ax.scatter(
        player_sdi['avg_sdi'],
        player_sdi['actual_fg_pct'],  # Y-axis = Actual FG%
        s=player_sdi['attempts'] / 5,  # size by volume
        alpha=0.6,
        c=player_sdi['residual'],  # Color = Residual (actual - expected)
        cmap='RdYlGn',
        vmin=-0.10, vmax=0.10,  # Symmetric colorscale around 0
        edgecolors='black', linewidths=0.5
    )
    
    # Add quadrant lines
    ax.axhline(player_sdi['actual_fg_pct'].median(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(player_sdi['avg_sdi'].median(), color='gray', linestyle='--', alpha=0.5)
    
    def _normalize_name(name):
        if not isinstance(name, str):
            return ""
        text = unicodedata.normalize("NFKD", name)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.lower().replace(".", "").replace(",", "").strip()

    highlight_names = {
        "giannis antetokounmpo",
        "shai gilgeous-alexander",
        "nikola jokic",
        "nikola jokić",
    }
    highlight_names = {_normalize_name(n) for n in highlight_names}

    for _, row in player_sdi.iterrows():
        if _normalize_name(row["player"]) in highlight_names:
            ax.annotate(
                row["player"],
                (row["avg_sdi"], row["actual_fg_pct"]),
                fontsize=9,
                alpha=0.95,
            )
    
    ax.set_xlabel('Average Shot Difficulty Index (SDI)', fontsize=12)
    ax.set_ylabel('Actual Field Goal %', fontsize=12)
    ax.set_title('Shot Difficulty vs Actual Efficiency\n(Size = Volume, Color = FG% Residual)', fontsize=14)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('FG% Residual (Actual - Expected)', fontsize=10)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_player_correlation(features, output_path, season_label=None):
    """
    Correlation heatmap across player-level metrics.

    Includes usage/volume, shot selection mix, and efficiency signals.
    """
    corr_cols = [
        "avg_distance",
        "avg_xFG",
        "avg_sdi",
        "actual_fg_pct",
        "pullup_rate",
        "attempts_per_game",
        "usage_pct",
        "pct_restricted_area",
        "pct_mid-range",
        "pct_above_the_break_3",
        "pct_left_corner_3",
        "pct_right_corner_3",
    ]

    corr_df = features[[c for c in corr_cols if c in features.columns]].copy()
    if "avg_xFG" in corr_df.columns and "actual_fg_pct" in corr_df.columns:
        corr_df["residual_fg_pct"] = corr_df["actual_fg_pct"] - corr_df["avg_xFG"]

    if corr_df.empty:
        print("Skipping correlation plot: no numeric columns available.")
        return

    corr_df = corr_df.dropna(axis=1, how="all")
    corr_df = corr_df.loc[:, corr_df.nunique(dropna=True) > 1]
    if corr_df.shape[1] < 2:
        print("Skipping correlation plot: not enough signal columns.")
        return

    corr = corr_df.corr()

    label_map = {
        "avg_distance": "Avg Distance (ft)",
        "avg_xFG": "Avg xFG",
        "avg_sdi": "Avg SDI",
        "actual_fg_pct": "Actual FG%",
        "residual_fg_pct": "FG% Residual",
        "pullup_rate": "Pull-up Rate",
        "attempts_per_game": "Attempts/Game",
        "usage_pct": "Usage%",
        "pct_restricted_area": "Restricted Area",
        "pct_mid-range": "Mid-Range",
        "pct_above_the_break_3": "Above Break 3",
        "pct_left_corner_3": "Left Corner 3",
        "pct_right_corner_3": "Right Corner 3",
    }
    labels = [label_map.get(col, col.replace("_", " ").title()) for col in corr.columns]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.tick_params(axis="both", length=0)

    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.values[i, j]
            if np.isnan(value):
                continue
            color = "white" if abs(value) >= 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    title = "Player Metric Correlations"
    if season_label:
        title = f"{title} ({season_label})"
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# PHASE 3: Player Shot Archetype Clustering
# =============================================================================

def build_player_features(df, usage_df=None, min_shots=MIN_SHOTS):
    """
    Build player-level feature matrix for clustering.
    
    Features:
    - % of shots by zone (6 zones)
    - Avg shot distance
    - Pull-up rate
    - Avg xFG
    - Avg SDI
    - Usage% (minutes-weighted when available)
    - Attempts per game (usage proxy)
    """
    if "PLAYER_ID" not in df.columns:
        raise ValueError("shots data missing PLAYER_ID; re-run expected_points_analysis.py")

    # Filter players with enough shots
    player_counts = df.groupby('PLAYER_ID').size()
    valid_players = player_counts[player_counts >= min_shots].index
    df = df[df['PLAYER_ID'].isin(valid_players)].copy()

    # Volume / usage proxy
    volume = df.groupby('PLAYER_ID').agg(
        total_attempts=('SHOT_MADE_FLAG', 'size'),
        games_played=('GAME_ID', 'nunique')
    ).reset_index()
    volume['attempts_per_game'] = volume['total_attempts'] / volume['games_played'].replace(0, np.nan)
    volume['attempts_per_game'] = volume['attempts_per_game'].fillna(0)

    # Zone percentages (pivot)
    zone_pcts = df.groupby(['PLAYER_ID', 'SHOT_ZONE_BASIC']).size().unstack(fill_value=0)
    zone_pcts = zone_pcts.div(zone_pcts.sum(axis=1), axis=0)
    zone_pcts.columns = [f'pct_{col.replace(" ", "_").lower()}' for col in zone_pcts.columns]

    # Ensure expected zone columns exist (fill with 0 if missing)
    expected_zones = [
        "restricted_area",
        "in_the_paint_(non-ra)",
        "mid-range",
        "above_the_break_3",
        "left_corner_3",
        "right_corner_3",
        "backcourt",
    ]
    for zone in expected_zones:
        col = f"pct_{zone}"
        if col not in zone_pcts.columns:
            zone_pcts[col] = 0.0

    # Distance distribution features
    def _distance_entropy(distances):
        values = distances.dropna().to_numpy()
        if values.size == 0:
            return np.nan
        counts, _ = np.histogram(values, bins=DISTANCE_BINS)
        total = counts.sum()
        if total == 0:
            return np.nan
        probs = counts / total
        probs = probs[probs > 0]
        return -(probs * np.log(probs)).sum()

    def _iqr(series):
        return series.quantile(0.75) - series.quantile(0.25)

    distance_stats = df.groupby('PLAYER_ID')['shot_distance_feet'].agg(
        distance_std='std',
        distance_iqr=_iqr
    ).reset_index()
    distance_entropy = df.groupby('PLAYER_ID')['shot_distance_feet'].apply(
        _distance_entropy
    ).reset_index(name='distance_entropy')

    # Other player-level stats
    agg_map = {
        'player': ('PLAYER_NAME', 'first'),
        'avg_distance': ('shot_distance_feet', 'mean'),
        'avg_xFG': ('xP_prob', 'mean'),
        'avg_sdi': ('SDI', 'mean'),
        'actual_fg_pct': ('SHOT_MADE_FLAG', 'mean'),
        'pullup_rate': ('is_jump_shot', 'mean')
    }
    shot_type_cols = {
        'is_layup': 'layup_rate',
        'is_dunk': 'dunk_rate',
        'is_hook': 'hook_rate',
        'is_floater': 'floater_rate'
    }
    for col, out_col in shot_type_cols.items():
        if col in df.columns:
            agg_map[out_col] = (col, 'mean')

    player_stats = df.groupby('PLAYER_ID').agg(**agg_map).reset_index()

    # Merge
    features = player_stats.merge(zone_pcts.reset_index(), on='PLAYER_ID', how='left')
    features = features.merge(volume, on='PLAYER_ID', how='left')
    features = features.merge(distance_stats, on='PLAYER_ID', how='left')
    features = features.merge(distance_entropy, on='PLAYER_ID', how='left')

    # Shot diet aggregates
    pct_restricted = features.get('pct_restricted_area', 0)
    pct_paint = features.get('pct_in_the_paint_(non-ra)', 0)
    features['pct_rim'] = pct_restricted + pct_paint
    features['pct_midrange'] = features.get('pct_mid-range', 0)
    features['pct_3pt'] = (
        features.get('pct_above_the_break_3', 0)
        + features.get('pct_left_corner_3', 0)
        + features.get('pct_right_corner_3', 0)
    )
    features['pct_corner_3'] = (
        features.get('pct_left_corner_3', 0) + features.get('pct_right_corner_3', 0)
    )

    if usage_df is not None and not usage_df.empty:
        usage_df = usage_df.copy()
        usage_df['player_id'] = usage_df['player_id'].astype(str)
        features['PLAYER_ID'] = features['PLAYER_ID'].astype(str)
        features = features.merge(usage_df, left_on='PLAYER_ID', right_on='player_id', how='left')
        features = features.drop(columns=['player_id'], errors='ignore')

    return features


def assign_roles(features):
    """
    Assign coarse role groups using shot diet (or positions if enabled).
    """
    features = features.copy()
    role = pd.Series(index=features.index, data=pd.NA, dtype="object")

    if USE_POSITIONS and "position_derived" in features.columns:
        role_map = {
            "PG": "Guard",
            "SG": "Guard",
            "SF": "Wing",
            "PF": "Wing",
            "C": "Big"
        }
        role = features["position_derived"].map(role_map)

    pct_rim = features.get('pct_rim', pd.Series([0] * len(features)))
    pct_3pt = features.get('pct_3pt', pd.Series([0] * len(features)))
    avg_distance = features.get('avg_distance', pd.Series([0] * len(features)))
    hook_rate = features.get('hook_rate', pd.Series([0] * len(features)))
    dunk_rate = features.get('dunk_rate', pd.Series([0] * len(features)))

    fallback_mask = role.isna()
    if fallback_mask.any():
        big_mask = (pct_rim >= ROLE_CONFIG["fallback_big_rim"]) & (
            avg_distance <= ROLE_CONFIG["fallback_big_distance"]
        )
        guard_mask = (pct_3pt >= ROLE_CONFIG["fallback_guard_3pt"]) | (
            avg_distance >= ROLE_CONFIG["fallback_guard_distance"]
        )
        role.loc[fallback_mask & big_mask] = "Big"
        role.loc[fallback_mask & guard_mask & ~big_mask] = "Guard"
        role.loc[fallback_mask & ~big_mask & ~guard_mask] = "Wing"

    # Hybrid override to ensure true centers are treated as Big
    big_override = (pct_rim >= ROLE_CONFIG["big_override_rim"]) & (
        avg_distance <= ROLE_CONFIG["big_override_distance"]
    )
    role.loc[big_override] = "Big"

    # Frontcourt override to avoid stretch bigs being labeled as guards
    frontcourt_override = (
        (pct_rim >= ROLE_CONFIG["frontcourt_rim_override"]) |
        (hook_rate >= ROLE_CONFIG["frontcourt_hook_override"]) |
        (dunk_rate >= ROLE_CONFIG["frontcourt_dunk_override"])
    )
    guard_override = (role == "Guard") & frontcourt_override & ~big_override
    role.loc[guard_override] = "Wing"

    features['role'] = role.fillna("Wing")
    return features


def cluster_players(features, min_k=2, max_k=6):
    """
    Role-aware GMM clustering on player features.
    
    Returns features DataFrame with cluster labels added.
    cluster_confidence is in-sample posterior assignment probability from GMM
    (useful for relative certainty within this fit, but not calibrated uncertainty).
    """
    if "role" not in features.columns:
        features = assign_roles(features)

    # Select features for clustering
    if CLUSTER_FEATURES:
        feature_cols = [c for c in CLUSTER_FEATURES if c in features.columns]
    else:
        exclude_cols = {
            'player',
            'PLAYER_ID',
            'total_attempts',
            'games_played',
            'role'
        }
        feature_cols = [
            c for c in features.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(features[c])
        ]
    X = features[feature_cols].fillna(0).values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply feature weights after standardization
    if CLUSTER_WEIGHTS:
        weights = np.ones(len(feature_cols))
        for i, col in enumerate(feature_cols):
            weights[i] = CLUSTER_WEIGHTS.get(col, 1.0)
        X_scaled = X_scaled * weights
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    features['pca_1'] = X_pca[:, 0]
    features['pca_2'] = X_pca[:, 1]

    # Role-aware GMM clustering with BIC selection
    features['cluster'] = -1
    features['cluster_id'] = None
    features['cluster_confidence'] = np.nan
    gmm_models = {}
    bic_rows = []

    selected_meta = {}
    roles = features['role'].fillna("Unknown").unique()
    for role in roles:
        role_mask = features['role'] == role
        X_role = X_scaled[role_mask]
        n_role = X_role.shape[0]

        if n_role < 2:
            features.loc[role_mask, 'cluster'] = 0
            features.loc[role_mask, 'cluster_id'] = f"{role}-0"
            features.loc[role_mask, 'cluster_confidence'] = 1.0
            bic_rows.append({
                "role": role,
                "k": 1,
                "bic": np.nan,
                "n_players": n_role,
                "min_cluster_size": n_role,
                "at_boundary": True,
                "selection_reason": "single_player_role",
            })
            selected_meta[role] = {
                "k": 1,
                "selection_reason": "single_player_role",
            }
            continue

        # Guardrail: ensure enough players per component to avoid over-fragmented roles.
        role_max_k = min(max_k, n_role // MIN_PLAYERS_PER_COMPONENT)
        if role_max_k < 2:
            features.loc[role_mask, 'cluster'] = 0
            features.loc[role_mask, 'cluster_id'] = f"{role}-0"
            features.loc[role_mask, 'cluster_confidence'] = 1.0
            bic_rows.append({
                "role": role,
                "k": 1,
                "bic": np.nan,
                "n_players": n_role,
                "min_cluster_size": n_role,
                "at_boundary": True,
                "selection_reason": "insufficient_players_for_k_ge_2",
            })
            selected_meta[role] = {
                "k": 1,
                "selection_reason": "insufficient_players_for_k_ge_2",
            }
            continue

        candidate_ks = range(min_k, role_max_k + 1)
        candidate_results = []

        for k in candidate_ks:
            gmm = GaussianMixture(
                n_components=k,
                n_init=5,
                reg_covar=1e-6,
                random_state=42
            )
            gmm.fit(X_role)
            bic = gmm.bic(X_role)
            labels = gmm.predict(X_role)
            counts = np.bincount(labels, minlength=k)
            min_cluster_size = int(counts.min()) if counts.size else 0
            at_boundary = (k == min(candidate_ks)) or (k == max(candidate_ks))
            bic_rows.append({
                "role": role,
                "k": k,
                "bic": bic,
                "n_players": n_role,
                "min_cluster_size": min_cluster_size,
                "at_boundary": at_boundary,
                "selection_reason": "candidate",
            })
            candidate_results.append({
                "k": k,
                "model": gmm,
                "bic": bic,
                "labels": labels,
                "min_cluster_size": min_cluster_size,
            })

        valid_results = [r for r in candidate_results if r["min_cluster_size"] >= MIN_CLUSTER_SIZE]
        if valid_results:
            best_result = min(valid_results, key=lambda r: r["bic"])
            selection_reason = "lowest_bic_with_size_constraint"
        elif candidate_results:
            # Constrained fallback: pick the smallest feasible k when all candidates
            # violate minimum cluster size.
            best_result = min(candidate_results, key=lambda r: r["k"])
            selection_reason = "constrained_fallback_smallest_k"
        else:
            best_result = None
            selection_reason = "no_candidates"

        if best_result is None:
            features.loc[role_mask, 'cluster'] = 0
            features.loc[role_mask, 'cluster_id'] = f"{role}-0"
            features.loc[role_mask, 'cluster_confidence'] = 1.0
            continue

        labels = best_result["labels"]
        probs = best_result["model"].predict_proba(X_role)
        conf = probs.max(axis=1)

        features.loc[role_mask, 'cluster'] = labels
        features.loc[role_mask, 'cluster_id'] = [
            f"{role}-{label}" for label in labels
        ]
        features.loc[role_mask, 'cluster_confidence'] = conf

        gmm_models[role] = best_result["model"]
        selected_meta[role] = {
            "k": best_result["k"],
            "selection_reason": selection_reason,
        }
    
    bic_df = pd.DataFrame(bic_rows)
    if not bic_df.empty:
        bic_df["selected"] = False
        for role, meta in selected_meta.items():
            k_selected = meta["k"]
            mask = (bic_df["role"] == role) & (bic_df["k"] == k_selected)
            bic_df.loc[mask, "selected"] = True
            bic_df.loc[mask, "selection_reason"] = meta["selection_reason"]

    return features, gmm_models, scaler, bic_df


def plot_gmm_bic(bic_df, output_path):
    """
    Plot BIC vs k for each role to show model selection.
    """
    if bic_df is None or bic_df.empty:
        print("Skipping GMM BIC plot: no data available.")
        return

    roles = sorted(bic_df["role"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(roles)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for role, color in zip(roles, colors):
        role_df = bic_df[bic_df["role"] == role].sort_values("k")
        ax.plot(
            role_df["k"],
            role_df["bic"],
            marker="o",
            label=f"{role} (n={int(role_df['n_players'].iloc[0])})",
            color=color
        )
        selected = role_df[role_df["selected"]]
        if not selected.empty:
            ax.scatter(
                selected["k"],
                selected["bic"],
                s=120,
                color=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=5
            )

    ax.set_title("GMM Model Selection by Role (BIC)", fontsize=13)
    ax.set_xlabel("Number of Components (k)")
    ax.set_ylabel("BIC (lower is better)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def label_clusters(features):
    """
    Assign interpretable spatial labels to clusters based on shot distribution.
    
    DISCLAIMER: These archetypes describe shot location and difficulty patterns only.
    They do NOT imply offensive skill, versatility, or overall player quality.
    """
    cluster_labels = {}

    def _quantile(series, q):
        if series is None or series.empty or series.isna().all():
            return None
        return series.dropna().quantile(q)

    usage_series = None
    if "usage_pct" in features.columns and not features["usage_pct"].isna().all():
        usage_series = features["usage_pct"]
    elif "attempts_per_game" in features.columns:
        usage_series = features["attempts_per_game"]

    usage_low = _quantile(usage_series, 0.30)
    usage_high = _quantile(usage_series, 0.70)

    rim_high = _quantile(features.get("pct_rim"), 0.70)
    three_high = _quantile(features.get("pct_3pt"), 0.70)
    mid_high = _quantile(features.get("pct_midrange"), 0.70)
    sdi_median = _quantile(features.get("avg_sdi"), 0.50)

    for cluster_id in features['cluster_id'].dropna().unique():
        cluster_data = features[features['cluster_id'] == cluster_id]
        if cluster_data.empty:
            continue

        role = cluster_data['role'].mode().iat[0] if 'role' in cluster_data.columns else "Unknown"

        pct_rim = cluster_data.get('pct_rim', pd.Series([0])).mean()
        pct_3pt = cluster_data.get('pct_3pt', pd.Series([0])).mean()
        pct_midrange = cluster_data.get('pct_midrange', pd.Series([0])).mean()
        avg_sdi = cluster_data.get('avg_sdi', pd.Series([0])).mean()

        usage_val = None
        if "usage_pct" in cluster_data.columns and not cluster_data["usage_pct"].isna().all():
            usage_val = cluster_data["usage_pct"].mean()
        elif "attempts_per_game" in cluster_data.columns:
            usage_val = cluster_data["attempts_per_game"].mean()

        # Spatial-descriptive labels use season-relative quantiles (q70),
        # so labels stay robust as league shot diets shift season-to-season.
        if rim_high is not None and pct_rim > rim_high:
            base_label = "Paint-Dominant"
        elif three_high is not None and pct_3pt > three_high:
            base_label = "Perimeter-Focused"
        elif mid_high is not None and pct_midrange > mid_high:
            base_label = "Mid-Range Specialist"
        else:
            # Mixed Shot Profile with SDI-based split
            if sdi_median is not None and avg_sdi > sdi_median:
                base_label = "Shot Creator (Mixed)"
            else:
                base_label = "Role Player (Mixed)"

        usage_suffix = ""
        if usage_val is not None and usage_low is not None and usage_high is not None:
            if usage_val <= usage_low:
                usage_suffix = " (Low Usage)"
            elif usage_val >= usage_high:
                usage_suffix = " (High Usage)"

        label = f"{role} – {base_label}{usage_suffix}"
        cluster_labels[cluster_id] = label

    features['archetype'] = features['cluster_id'].map(cluster_labels)
    
    return features, cluster_labels


def plot_cluster_scatter(features, output_path):
    """2D PCA scatter plot of player clusters."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    archetypes = features['archetype'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(archetypes)))
    rep_map = select_representative_players(features)
    
    for archetype, color in zip(archetypes, colors):
        mask = features['archetype'] == archetype
        ax.scatter(
            features.loc[mask, 'pca_1'],
            features.loc[mask, 'pca_2'],
            label=archetype,
            s=90,
            alpha=0.7,
            c=[color],
            edgecolors='black',
            linewidths=0.5,
            marker="o"
        )
    
    # Label representative players per archetype
    for archetype in archetypes:
        rep_names = rep_map.get(archetype, [])
        if not rep_names:
            continue
        arch_rows = features[
            (features['archetype'] == archetype) & (features['player'].isin(rep_names))
        ]
        for _, row in arch_rows.iterrows():
            ax.annotate(row['player'], (row['pca_1'], row['pca_2']), fontsize=8)
    
    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_title('Player Shot Archetypes (Role-Aware GMM)', fontsize=14)

    # Archetype legend (colors) — keep a single legend to reduce clutter
    archetype_handles = []
    for archetype, color in zip(archetypes, colors):
        archetype_handles.append(
            plt.Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=archetype,
                markerfacecolor=color,
                markeredgecolor='black',
                markersize=9
            )
        )
    ax.legend(
        handles=archetype_handles,
        title='Archetype',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=True,
        fontsize=8,
        title_fontsize=9
    )
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_position_scatter(features, output_path):
    """2D PCA scatter plot colored by derived position."""
    fig, ax = plt.subplots(figsize=(14, 10))

    positions = features.get('position_derived', pd.Series(["Unknown"] * len(features))).fillna("Unknown")
    position_order = ["PG", "SG", "SF", "PF", "C", "Unknown"]

    for pos in position_order:
        mask = positions == pos
        if not mask.any():
            continue
        ax.scatter(
            features.loc[mask, 'pca_1'],
            features.loc[mask, 'pca_2'],
            label=pos,
            s=90,
            alpha=0.7,
            c=POSITION_COLORS.get(pos, "#7f7f7f"),
            edgecolors='black',
            linewidths=0.5,
            marker="o"
        )

    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_title('Player Positions (Derived)', fontsize=14)

    ax.legend(
        title='Position',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=True,
        fontsize=8,
        title_fontsize=9
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_cluster_summary(features, cluster_labels):
    """Create summary table of cluster characteristics."""
    summary_cols = [
        'avg_distance',
        'avg_xFG',
        'avg_sdi',
        'actual_fg_pct',
        'pullup_rate',
        'pct_rim',
        'pct_midrange',
        'pct_3pt',
        'pct_corner_3',
        'distance_entropy',
        'usage_pct',
        'attempts_per_game'
    ]

    available_cols = [c for c in summary_cols if c in features.columns]
    group_cols = [c for c in ['role', 'archetype'] if c in features.columns]

    summary = features.groupby(group_cols)[available_cols].mean().round(3)
    summary['count'] = features.groupby(group_cols).size()
    summary = summary.reset_index()

    return summary


def select_representative_players(features):
    """
    Pick 2-4 representative players per archetype based on volume.
    """
    if features.empty or "archetype" not in features.columns:
        return {}

    volume_col = None
    if "total_attempts" in features.columns:
        volume_col = "total_attempts"
    elif "attempts_per_game" in features.columns:
        volume_col = "attempts_per_game"

    rep_map = {}
    for archetype, group in features.groupby("archetype"):
        n = len(group)
        if n <= 6:
            k = 2
        elif n <= 20:
            k = 3
        else:
            k = 4

        if volume_col:
            group = group.copy()
            group[volume_col] = pd.to_numeric(group[volume_col], errors="coerce").fillna(0)
            group = group.sort_values(volume_col, ascending=False)

        rep_map[archetype] = group["player"].head(k).tolist()

    return rep_map


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    
    # Load enriched shot data
    print("Loading enriched shot data...")
    current_season = "2025-26"
    current_season_type = "regular"
    shots_path = DATA_DIR / f"shots_with_xp_{current_season}.parquet"
    if not shots_path.exists():
        raise FileNotFoundError(
            f"Missing {shots_path}. Run expected_points_analysis.py first."
        )
    df = pd.read_parquet(shots_path)
    print(f"Loaded {len(df):,} shots")

    usage_df = load_usage_from_sqlite(current_season, current_season_type)
    position_df = None
    assist_df = None
    if USE_POSITIONS:
        position_df = load_position_from_sqlite(current_season, current_season_type)
        assist_df = load_assist_pct_from_sqlite(current_season, current_season_type)

    required_cols = {
        'PLAYER_NAME',
        'SHOT_MADE_FLAG',
        'SHOT_ZONE_BASIC',
        'ACTION_TYPE',
        'xP_prob',
        'shot_distance_feet',
        'seconds_in_period',
        'shot_angle',
        'is_jump_shot',
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"shots_with_xp file missing required columns: {', '.join(sorted(missing))}"
        )
    
    # =========================================================================
    # PHASE 1: Residual Analysis
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: SHOT QUALITY (RESIDUAL ANALYSIS)")
    print("="*60)
    
    df = compute_residuals(df)
    
    # Player residuals
    player_residuals = aggregate_player_residuals(df)
    
    print("\nTOP 15 OVERPERFORMERS (Positive Residuals):")
    print(player_residuals.head(15)[['player', 'avg_xFG', 'actual_fg_pct', 'residual_fg_pct', 'attempts']].to_string(index=False))
    
    print("\nBOTTOM 15 UNDERPERFORMERS (Negative Residuals):")
    print(player_residuals.tail(15)[['player', 'avg_xFG', 'actual_fg_pct', 'residual_fg_pct', 'attempts']].to_string(index=False))
    
    # Zone residuals
    zone_residuals = aggregate_team_residuals(df)
    print("\nRESIDUALS BY ZONE:")
    print(zone_residuals.to_string(index=False))
    
    # Heatmap for top overperformer
    top_player = player_residuals.iloc[0]['player']
    plot_residual_heatmap(df, top_player, FIGURES_DIR / f"{top_player.replace(' ', '_')}_residual_heatmap.png")
    
    # Save residuals
    player_residuals.to_csv(DATA_DIR / "player_residuals.csv", index=False)
    print(f"Saved: {DATA_DIR / 'player_residuals.csv'}")
    
    # =========================================================================
    # PHASE 2: Shot Difficulty Index
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: SHOT DIFFICULTY INDEX (SDI)")
    print("="*60)
    
    df = compute_sdi(df)
    
    player_sdi = aggregate_player_sdi(df)
    
    print("\nTOP 15 PLAYERS BY SHOT DIFFICULTY:")
    print(player_sdi.head(15)[['player', 'avg_sdi', 'avg_xFG', 'actual_fg_pct', 'attempts']].to_string(index=False))
    
    # SDI vs xFG plot
    plot_sdi_vs_xfg(player_sdi, FIGURES_DIR / "sdi_vs_xfg_scatter.png")
    
    # Identify interesting quadrants
    median_sdi = player_sdi['avg_sdi'].median()
    median_xfg = player_sdi['avg_xFG'].median()
    
    elite = player_sdi[(player_sdi['avg_sdi'] > median_sdi) & (player_sdi['actual_fg_pct'] > player_sdi['avg_xFG'])]
    print(f"\nELITE SHOT MAKERS (High SDI + Positive Residual):")
    print(elite.head(10)[['player', 'avg_sdi', 'actual_fg_pct', 'avg_xFG']].to_string(index=False))
    
    # =========================================================================
    # PHASE 3: Player Clustering
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 3: PLAYER SHOT ARCHETYPE CLUSTERING")
    print("="*60)
    print("\n⚠️  DISCLAIMER: These archetypes describe shot location and")
    print("   difficulty patterns ONLY. They do NOT imply offensive skill,")
    print("   versatility, or overall player quality.\n")
    
    features = build_player_features(df, usage_df=usage_df)
    if USE_POSITIONS:
        features = derive_positions(
            features,
            position_df=position_df,
            assist_df=assist_df,
            config=POSITION_CONFIG
        )
    print(f"Built features for {len(features)} players")
    features = assign_roles(features)

    # Correlation heatmap
    plot_player_correlation(
        features,
        FIGURES_DIR / "player_metric_correlations.png",
        season_label=current_season,
    )
    
    features, gmm_models, scaler, bic_df = cluster_players(features, min_k=2, max_k=6)
    features, cluster_labels = label_clusters(features)
    
    print("\nCLUSTER SUMMARY:")
    summary = create_cluster_summary(features, cluster_labels)
    print(summary.to_string())
    
    print("\nSAMPLE PLAYERS BY ARCHETYPE:")
    rep_map = select_representative_players(features)
    for archetype in features['archetype'].unique():
        players = rep_map.get(archetype, [])
        if players:
            print(f"  {archetype}: {', '.join(players)}")
    
    # Cluster visualization
    plot_cluster_scatter(features, FIGURES_DIR / "player_archetypes_scatter.png")
    if USE_POSITIONS:
        plot_position_scatter(features, FIGURES_DIR / "player_positions_scatter.png")

    # BIC model selection visualization
    if bic_df is not None and not bic_df.empty:
        bic_df.to_csv(DATA_DIR / "gmm_bic_by_role.csv", index=False)
        print(f"Saved: {DATA_DIR / 'gmm_bic_by_role.csv'}")
        plot_gmm_bic(bic_df, FIGURES_DIR / "gmm_bic_by_role.png")
    
    # Save cluster assignments
    output_cols = [
        "player",
        "PLAYER_ID",
        "role",
        "cluster_id",
        "cluster_confidence",
        "archetype",
        "avg_distance",
        "avg_xFG",
        "avg_sdi",
        "actual_fg_pct",
        "pullup_rate",
        "usage_pct",
        "attempts_per_game",
        "total_attempts",
        "games_played",
        "pct_rim",
        "pct_midrange",
        "pct_3pt",
        "pct_corner_3",
        "distance_std",
        "distance_iqr",
        "distance_entropy",
        "layup_rate",
        "dunk_rate",
        "hook_rate",
        "floater_rate"
    ]
    if USE_POSITIONS:
        output_cols = output_cols[:3] + [
            "position_raw",
            "position_derived",
            "position_source",
            "assist_pct"
        ] + output_cols[3:]
    output_cols = [c for c in output_cols if c in features.columns]
    features[output_cols].to_csv(DATA_DIR / "player_clusters.csv", index=False)
    print(f"Saved: {DATA_DIR / 'player_clusters.csv'}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Data saved to: {DATA_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
