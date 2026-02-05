"""
Advanced NBA Analytics Module.

Three integrated analytics products using existing xFG model:
1. Shot Quality (Residual Analysis)
2. Shot Difficulty Index (SDI)
3. Player Shot Archetype Clustering

Uses pre-computed xP_prob from expected_points_analysis.py
"""

import os
import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
    Scatter plot: Avg SDI vs Avg xFG per player.
    
    Quadrants:
    - Top-right: difficult shots, high efficiency (elite scorers)
    - Bottom-left: easy shots, low efficiency (inefficient)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(
        player_sdi['avg_sdi'],
        player_sdi['avg_xFG'],
        s=player_sdi['attempts'] / 5,  # size by volume
        alpha=0.6,
        c=player_sdi['actual_fg_pct'],
        cmap='RdYlGn',
        edgecolors='black', linewidths=0.5
    )
    
    # Add quadrant lines
    ax.axhline(player_sdi['avg_xFG'].median(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(player_sdi['avg_sdi'].median(), color='gray', linestyle='--', alpha=0.5)
    
    # Label interesting players
    top_sdi = player_sdi.nlargest(5, 'avg_sdi')
    for _, row in top_sdi.iterrows():
        ax.annotate(row['player'], (row['avg_sdi'], row['avg_xFG']), fontsize=8)
    
    ax.set_xlabel('Average Shot Difficulty Index (SDI)', fontsize=12)
    ax.set_ylabel('Average Expected FG% (xFG)', fontsize=12)
    ax.set_title('Shot Difficulty vs Expected Efficiency\n(Size = Volume, Color = Actual FG%)', fontsize=14)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Actual FG%', fontsize=10)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
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

    # Other player-level stats
    player_stats = df.groupby('PLAYER_ID').agg(
        player=('PLAYER_NAME', 'first'),
        avg_distance=('shot_distance_feet', 'mean'),
        avg_xFG=('xP_prob', 'mean'),
        avg_sdi=('SDI', 'mean'),
        actual_fg_pct=('SHOT_MADE_FLAG', 'mean'),
        pullup_rate=('is_jump_shot', 'mean')
    ).reset_index()

    # Merge
    features = player_stats.merge(zone_pcts.reset_index(), on='PLAYER_ID', how='left')
    features = features.merge(volume, on='PLAYER_ID', how='left')

    if usage_df is not None and not usage_df.empty:
        usage_df = usage_df.copy()
        usage_df['player_id'] = usage_df['player_id'].astype(str)
        features['PLAYER_ID'] = features['PLAYER_ID'].astype(str)
        features = features.merge(usage_df, left_on='PLAYER_ID', right_on='player_id', how='left')
        features = features.drop(columns=['player_id'], errors='ignore')

    return features


def cluster_players(features, n_clusters=5):
    """
    K-Means clustering on player features.
    
    Returns features DataFrame with cluster labels added.
    """
    # Select numeric features for clustering
    exclude_cols = {
        'player',
        'PLAYER_ID',
        'usage_pct',
        'attempts_per_game',
        'total_attempts',
        'games_played'
    }
    feature_cols = [c for c in features.columns if c not in exclude_cols]
    X = features[feature_cols].fillna(0).values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features['cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    features['pca_1'] = X_pca[:, 0]
    features['pca_2'] = X_pca[:, 1]
    
    return features, kmeans, scaler


def label_clusters(features):
    """
    Assign interpretable basketball labels to clusters based on characteristics.
    """
    cluster_labels = {}

    usage_series = None
    if "usage_pct" in features.columns and not features["usage_pct"].isna().all():
        usage_series = features["usage_pct"]
    elif "attempts_per_game" in features.columns:
        usage_series = features["attempts_per_game"]

    if usage_series is not None:
        usage_series = usage_series.fillna(usage_series.median())
        low_usage = usage_series.quantile(0.25)
        high_usage = usage_series.quantile(0.75)
    else:
        low_usage = None
        high_usage = None

    sdi_cut = features["avg_sdi"].median()
    
    for cluster_id in features['cluster'].unique():
        cluster_data = features[features['cluster'] == cluster_id]
        
        # Determine archetype based on dominant characteristics
        avg_distance = cluster_data['avg_distance'].mean()
        avg_sdi = cluster_data['avg_sdi'].mean()
        
        # Check zone percentages
        restricted_pct = cluster_data.get('pct_restricted_area', pd.Series([0])).mean()
        three_pct = sum([
            cluster_data.get('pct_above_the_break_3', pd.Series([0])).mean(),
            cluster_data.get('pct_left_corner_3', pd.Series([0])).mean(),
            cluster_data.get('pct_right_corner_3', pd.Series([0])).mean()
        ])
        midrange_pct = cluster_data.get('pct_mid-range', pd.Series([0])).mean()
        
        usage_val = None
        if "usage_pct" in cluster_data.columns:
            usage_val = cluster_data["usage_pct"].mean()
        if (usage_val is None or np.isnan(usage_val)) and "attempts_per_game" in cluster_data.columns:
            usage_val = cluster_data["attempts_per_game"].mean()

        # Assign label (neutral, shot-profile focused)
        if restricted_pct > 0.4:
            label = "Rim Heavy / Low Distance"
        elif three_pct > 0.5:
            if usage_val is not None and low_usage is not None and usage_val <= low_usage:
                label = "High 3PT / Low Usage"
            elif avg_sdi >= sdi_cut:
                label = "High 3PT / High SDI"
            else:
                label = "High 3PT / Low SDI"
        elif midrange_pct > 0.2:
            label = "Mid-Range Heavy"
        elif avg_sdi >= sdi_cut:
            label = "High SDI / Non-3"
        else:
            label = "Balanced"
        
        cluster_labels[cluster_id] = label
    
    features['archetype'] = features['cluster'].map(cluster_labels)
    
    return features, cluster_labels


def plot_cluster_scatter(features, output_path):
    """2D PCA scatter plot of player clusters."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    archetypes = features['archetype'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(archetypes)))
    
    for archetype, color in zip(archetypes, colors):
        mask = features['archetype'] == archetype
        ax.scatter(
            features.loc[mask, 'pca_1'],
            features.loc[mask, 'pca_2'],
            label=archetype,
            s=100, alpha=0.7,
            c=[color], edgecolors='black', linewidths=0.5
        )
    
    # Label some players
    for archetype in archetypes:
        arch_players = features[features['archetype'] == archetype].nlargest(2, 'avg_xFG')
        for _, row in arch_players.iterrows():
            ax.annotate(row['player'], (row['pca_1'], row['pca_2']), fontsize=8)
    
    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_title('Player Shot Archetypes (K-Means Clustering)', fontsize=14)
    ax.legend(title='Archetype', loc='best')
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_cluster_summary(features, cluster_labels):
    """Create summary table of cluster characteristics."""
    summary_cols = ['avg_distance', 'avg_xFG', 'avg_sdi', 'actual_fg_pct', 'pullup_rate']
    
    summary = features.groupby('archetype')[summary_cols].mean().round(3)
    summary['count'] = features.groupby('archetype').size()
    
    return summary


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
    print("PHASE 3: PLAYER SHOT ARCHETYPE CLUSTERING)")
    print("="*60)
    
    features = build_player_features(df, usage_df=usage_df)
    print(f"Built features for {len(features)} players")
    
    features, kmeans, scaler = cluster_players(features, n_clusters=5)
    features, cluster_labels = label_clusters(features)
    
    print("\nCLUSTER SUMMARY:")
    summary = create_cluster_summary(features, cluster_labels)
    print(summary.to_string())
    
    print("\nSAMPLE PLAYERS BY ARCHETYPE:")
    for archetype in features['archetype'].unique():
        players = features[features['archetype'] == archetype].nlargest(3, 'actual_fg_pct')['player'].tolist()
        print(f"  {archetype}: {', '.join(players)}")
    
    # Cluster visualization
    plot_cluster_scatter(features, FIGURES_DIR / "player_archetypes_scatter.png")
    
    # Save cluster assignments
    output_cols = [
        "player",
        "PLAYER_ID",
        "archetype",
        "avg_distance",
        "avg_xFG",
        "avg_sdi",
        "actual_fg_pct",
        "pullup_rate",
        "usage_pct",
        "attempts_per_game",
        "total_attempts",
        "games_played"
    ]
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
