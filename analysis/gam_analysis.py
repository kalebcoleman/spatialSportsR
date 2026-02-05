"""
Generalized Additive Model (GAM) Analysis for NBA Shot Probability.

Uses pygam to fit non-linear smooths for:
- Spatial location (tensor product of X, Y)
- Shot distance
- Shot angle
- Game context (period, clock, clutch)
- Shot type (layup, dunk, jump shot)

Generates Partial Dependence Plots (PDPs) to visualize feature effects.
"""

import sys
import pickle
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LogisticGAM, s, te, l
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

# Configuration
ANALYSIS_DIR = Path(__file__).parent
sys.path.append(str(ANALYSIS_DIR))

from utils.court_utils import draw_half_court, setup_shot_chart_axes

DATA_DIR = ANALYSIS_DIR / "data"
FIGURES_DIR = ANALYSIS_DIR / "figures"
MODELS_DIR = ANALYSIS_DIR / "models"
SEED = 42
DISTANCE_CAP_FT = 35  # optional cap if enabled for stability

# Rolling window configuration (matches expected_points_analysis.py)
TARGET_SEASON = '2025-26'
ROLLING_WINDOW = 5

# Import DB path from expected_points module
import os
import sqlite3
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "parsed" / "nba.sqlite"
DB_PATH = Path(os.getenv("SPATIALSPORTSR_DB_PATH", str(DEFAULT_DB_PATH))).expanduser()


def get_rolling_window_seasons(target_season, window_size=5):
    """Generate list of training seasons for rolling window."""
    start_year = int(target_season.split('-')[0])
    training_seasons = []
    for i in range(window_size, 0, -1):
        year = start_year - i
        season = f"{year}-{str(year + 1)[-2:]}"
        training_seasons.append(season)
    return training_seasons


def load_shot_data_from_db(seasons, season_type="regular", cap_distance=False, exclude_backcourt=False):
    """Load shot data directly from SQLite for training."""
    if isinstance(seasons, str):
        seasons = [seasons]
    
    con = sqlite3.connect(str(DB_PATH))
    placeholders = ','.join(['?' for _ in seasons])
    query = f"""
    SELECT
        LOC_X, LOC_Y, SHOT_MADE_FLAG, SHOT_TYPE, ACTION_TYPE,
        SHOT_ZONE_BASIC, SHOT_ZONE_AREA, SHOT_DISTANCE,
        PERIOD, MINUTES_REMAINING, SECONDS_REMAINING, season
    FROM nba_stats_shots
    WHERE season IN ({placeholders}) AND season_type = ? AND SHOT_ATTEMPTED_FLAG = 1
    """
    params = seasons + [season_type]
    df = pd.read_sql_query(query, con, params=params)
    con.close()
    
    # Engineer features inline
    for col in ['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG', 'SHOT_DISTANCE', 
                'PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG', 'ACTION_TYPE'], inplace=True)
    df['shot_distance_feet'] = df['SHOT_DISTANCE'].fillna(
        np.sqrt(df['LOC_X']**2 + df['LOC_Y']**2) / 10
    )
    df['shot_angle'] = np.arctan2(df['LOC_X'], df['LOC_Y'].clip(lower=1))
    df['seconds_in_period'] = df['MINUTES_REMAINING'].fillna(0) * 60 + df['SECONDS_REMAINING'].fillna(0)
    df['is_clutch'] = ((df['PERIOD'] >= 4) & (df['seconds_in_period'] <= 120)).astype(int)
    
    # Optional filters
    if cap_distance:
        df = df[df['shot_distance_feet'].between(0, DISTANCE_CAP_FT)].copy()
    if exclude_backcourt:
        df = df[df['SHOT_ZONE_BASIC'].ne('Backcourt')].copy()
    
    return df


def load_data(cap_distance=True, exclude_backcourt=True):
    """Load latest shot data (target season with enriched xP)."""
    files = list(DATA_DIR.glob("shots_with_xp_*.parquet"))
    if not files:
        raise FileNotFoundError("No shot data found in analysis/data/")
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Loading {latest_file.name}...")
    season = latest_file.stem.replace("shots_with_xp_", "")
    df = pd.read_parquet(latest_file)
    if cap_distance and "shot_distance_feet" in df.columns:
        df = df[df["shot_distance_feet"].between(0, DISTANCE_CAP_FT)].copy()
    if exclude_backcourt and "SHOT_ZONE_BASIC" in df.columns:
        df = df[df["SHOT_ZONE_BASIC"].ne("Backcourt")].copy()
    return df, season


def add_shot_type_features(df):
    """Create mutually exclusive shot-type categories and indicators."""
    df = df.copy()
    action = df.get("ACTION_TYPE", pd.Series("", index=df.index)).str.lower().fillna("")

    is_dunk = action.str.contains("dunk")
    is_layup = action.str.contains("layup|finger roll")
    is_hook = action.str.contains("hook")
    is_floater = action.str.contains("float")
    is_jump = action.str.contains("jump shot|pullup|step back|fadeaway")

    if "SHOT_TYPE" in df.columns:
        is_three = df["SHOT_TYPE"].str.contains("3PT", case=False, na=False)
    elif "shot_value" in df.columns:
        is_three = df["shot_value"].fillna(0).astype(int) == 3
    else:
        is_three = pd.Series(False, index=df.index)

    shot_type_cat = np.select(
        [
            is_dunk,
            is_layup,
            is_hook,
            is_floater,
            is_jump & (~is_three),
            is_jump & is_three,
        ],
        [
            "dunk",
            "layup",
            "hook",
            "floater",
            "2pt_jump",
            "3pt_jump",
        ],
        default="other",
    )

    df["shot_type_cat"] = shot_type_cat
    df["is_dunk"] = (shot_type_cat == "dunk").astype(int)
    df["is_layup"] = (shot_type_cat == "layup").astype(int)
    df["is_hook"] = (shot_type_cat == "hook").astype(int)
    df["is_floater"] = (shot_type_cat == "floater").astype(int)
    df["is_jump_shot_2"] = (shot_type_cat == "2pt_jump").astype(int)
    df["is_jump_shot_3"] = (shot_type_cat == "3pt_jump").astype(int)

    return df


def fit_gam(X_train, y_train):
    """
    Fit LogisticGAM with:
    - te(LOC_X, LOC_Y): Tensor product smooth for spatial surface
    - s(shot_distance_feet): Smooth for distance
    - s(shot_angle): Smooth for angle
    - s(seconds_in_period): Smooth for time
    - s(PERIOD): Smooth for period
    - l(is_dunk, is_layup, is_hook, is_floater, is_jump_shot_2, is_jump_shot_3, is_clutch): Linear effects
    """
    print("Fitting LogisticGAM (this may take a minute)...")
    
    # Features: 0=LOC_X, 1=LOC_Y, 2=distance, 3=angle, 4=seconds, 5=period,
    #           6=is_dunk, 7=is_layup, 8=is_hook, 9=is_floater,
    #           10=is_jump_shot_2, 11=is_jump_shot_3, 12=is_clutch
    gam = LogisticGAM(
        te(0, 1, n_splines=10) +  # Spatial surface
        s(2, n_splines=15) +      # Distance
        s(3, n_splines=10) +      # Angle
        s(4, n_splines=8) +       # Time in period
        s(5, n_splines=5) +       # Period (1-4+OT)
        l(6) +                    # is_dunk
        l(7) +                    # is_layup
        l(8) +                    # is_hook
        l(9) +                    # is_floater
        l(10) +                   # is_jump_shot_2
        l(11) +                   # is_jump_shot_3
        l(12)                     # is_clutch
    )
    
    gam.fit(X_train, y_train)
    return gam


def plot_distance_effect(gam, output_dir, target_df_all=None, obs_max=None):
    """Plot GAM distance effect (log-odds)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    XX = gam.generate_X_grid(term=1)
    pdp = gam.partial_dependence(term=1, X=XX)
    conf = gam.partial_dependence(term=1, X=XX, width=0.95)[1]

    if obs_max is None:
        if target_df_all is not None and "shot_distance_feet" in target_df_all.columns:
            dist_series = pd.to_numeric(target_df_all["shot_distance_feet"], errors="coerce")
            obs_max = np.nanmax(dist_series.values)
        else:
            obs_max = np.nanmax(XX[:, 2])
        if not np.isfinite(obs_max):
            obs_max = np.nanmax(XX[:, 2])
    obs_max = float(np.ceil(obs_max))

    mask = (XX[:, 2] >= 0) & (XX[:, 2] <= obs_max)
    ax.plot(
        XX[mask, 2],
        pdp[mask],
        color="#1f77b4",
        linewidth=2,
        label="GAM smooth (log-odds)",
    )
    ax.fill_between(
        XX[mask, 2],
        conf[mask, 0],
        conf[mask, 1],
        alpha=0.2,
        color="#1f77b4",
        label="95% CI",
    )

    # Add reference lines (no legend)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, label="_nolegend_")
    ax.axvline(23.75, color="orange", linestyle="--", alpha=0.7, label="_nolegend_")
    ax.axvline(4, color="green", linestyle="--", alpha=0.7, label="_nolegend_")

    ax.set_xlabel("Shot Distance (feet)", fontsize=12)
    ax.set_ylabel("Marginal log-odds contribution (additive term; 0 = baseline)", fontsize=11)
    ax.set_title("GAM Distance Effect on Shot Success (Marginal Log-Odds; Relative to Baseline)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, obs_max)

    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "gam_effect_distance.png"
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close()


def plot_angle_effect(gam, output_dir):
    """Plot how shot probability changes with angle."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    XX = gam.generate_X_grid(term=2)
    pdp = gam.partial_dependence(term=2, X=XX)
    conf = gam.partial_dependence(term=2, X=XX, width=.95)[1]
    
    ax.plot(XX[:, 3], pdp, 'b-', linewidth=2, label='GAM smooth')
    ax.fill_between(XX[:, 3], conf[:, 0], conf[:, 1], alpha=0.2, color='blue', label='95% CI')
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='purple', linestyle='--', alpha=0.7, label='Straight on')
    
    ax.set_xlabel('Shot Angle (radians)', fontsize=12)
    ax.set_ylabel('Log Odds Contribution', fontsize=12)
    ax.set_title('Effect of Shot Angle on Make Probability\n(0 = straight on, ±1.57 = baseline)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "gam_effect_angle.png"
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close()


def plot_clock_effect(gam, output_dir):
    """Plot how shot probability changes with game clock."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    XX = gam.generate_X_grid(term=3)
    pdp = gam.partial_dependence(term=3, X=XX)
    conf = gam.partial_dependence(term=3, X=XX, width=.95)[1]
    
    ax.plot(XX[:, 4], pdp, 'b-', linewidth=2, label='GAM smooth')
    ax.fill_between(XX[:, 4], conf[:, 0], conf[:, 1], alpha=0.2, color='blue', label='95% CI')
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(120, color='red', linestyle='--', alpha=0.7, label='Clutch Time')
    
    ax.set_xlabel('Seconds Remaining in Period (start → end)', fontsize=12)
    ax.set_ylabel('Log Odds Contribution', fontsize=12)
    ax.set_title('Effect of Game Clock on Make Probability\n(End-of-period pressure)', fontsize=14)
    ax.legend()
    # Flip so start-of-period is on the left and end-of-period on the right
    ax.set_xlim(720, 0)
    ax.set_xticks(np.arange(0, 721, 120))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "gam_effect_clock.png"
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close()


def plot_period_effect(gam, output_dir):
    """Plot how shot probability changes across periods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    XX = gam.generate_X_grid(term=4)
    pdp = gam.partial_dependence(term=4, X=XX)
    conf = gam.partial_dependence(term=4, X=XX, width=.95)[1]
    
    ax.plot(XX[:, 5], pdp, 'b-', linewidth=2, label='GAM smooth')
    ax.fill_between(XX[:, 5], conf[:, 0], conf[:, 1], alpha=0.2, color='blue', label='95% CI')
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Period (1-4 = regulation, 5+ = OT)', fontsize=12)
    ax.set_ylabel('Log Odds Contribution', fontsize=12)
    ax.set_title('Effect of Game Period on Make Probability\n(Fatigue and pressure effects)', fontsize=14)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(['1st', '2nd', '3rd', '4th', 'OT1', 'OT2+'])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / "gam_effect_period.png"
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close()


def plot_shot_type_effects(df, output_dir):
    """Bar chart of shot type effects from a shot-type-only GAM."""
    fig, ax = plt.subplots(figsize=(10, 6))

    shot_types = ['Dunk', 'Layup', 'Hook', 'Floater', '2PT Jump', '3PT Jump']
    cols = ['is_dunk', 'is_layup', 'is_hook', 'is_floater', 'is_jump_shot_2', 'is_jump_shot_3']
    for col in cols:
        if col not in df.columns:
            df[col] = 0

    X = df[cols].values
    y = df['SHOT_MADE_FLAG'].astype(int).values

    terms = l(0)
    for i in range(1, X.shape[1]):
        terms += l(i)
    gam_types = LogisticGAM(terms).fit(X, y)

    coeffs = []
    for term_idx in range(len(cols)):
        coef_idx = gam_types.terms.get_coef_indices(term_idx)
        coeffs.append(gam_types.coef_[coef_idx][0])
    
    colors = ['green' if c > 0 else 'red' for c in coeffs]
    bars = ax.bar(shot_types, coeffs, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_ylabel('Log Odds Contribution', fontsize=12)
    ax.set_title('Shot Type Effects (Exclusive Categories)\nShot-type-only GAM; baseline = Other', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, coef in zip(bars, coeffs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{coef:+.2f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11)
    
    plt.tight_layout()
    output_path = output_dir / "gam_effect_shot_types.png"
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close()


def plot_spatial_tensor(gam, output_dir, df, min_count=25):
    """Visualizes predicted probability + spatial effect across the court."""
    print("Generating Spatial Probability + Effect Plots...")
    
    x_range = np.linspace(-250, 250, 80)
    y_range = np.linspace(-50, 420, 80)
    XX, YY = np.meshgrid(x_range, y_range)
    
    grid_X = XX.flatten()
    grid_Y = YY.flatten()
    
    N = len(grid_X)
    X_matrix = np.zeros((N, 13))
    X_matrix[:, 0] = grid_X
    X_matrix[:, 1] = grid_Y
    # Calculate actual distance from each point
    X_matrix[:, 2] = np.sqrt(grid_X**2 + grid_Y**2) / 10  # Distance in feet
    # Calculate actual angle
    X_matrix[:, 3] = np.arctan2(grid_X, np.clip(grid_Y, 1, None))  # Angle
    # Use average context to avoid overfitting to a specific shot type
    X_matrix[:, 4] = df["seconds_in_period"].mean()
    X_matrix[:, 5] = df["PERIOD"].mean()
    X_matrix[:, 6] = df["is_dunk"].mean()
    X_matrix[:, 7] = df["is_layup"].mean()
    X_matrix[:, 8] = df["is_hook"].mean()
    X_matrix[:, 9] = df["is_floater"].mean()
    X_matrix[:, 10] = df["is_jump_shot_2"].mean()
    X_matrix[:, 11] = df["is_jump_shot_3"].mean()
    X_matrix[:, 12] = df["is_clutch"].mean()
    
    # Get FULL predicted probability (not just partial dependence)
    probs = gam.predict_proba(X_matrix)
    Z = probs.reshape(XX.shape)

    # Mask low-density regions and out-of-range distances to prevent artifacts
    counts, _, _ = np.histogram2d(
        df["LOC_Y"],
        df["LOC_X"],
        bins=[len(y_range), len(x_range)],
        range=[[y_range.min(), y_range.max()], [x_range.min(), x_range.max()]]
    )
    dist_grid = np.sqrt(XX**2 + YY**2) / 10  # feet
    mask = (counts < min_count) | (dist_grid > DISTANCE_CAP_FT)
    Z[mask] = np.nan
    
    fig, ax = plt.subplots(figsize=(12, 11))
    draw_half_court(ax, outer_lines=True)
    
    # Use 'RdYlGn' so green = high probability, red = low
    vmin = np.nanpercentile(Z, 2)
    vmax = np.nanpercentile(Z, 98)
    mesh = ax.pcolormesh(
        XX, YY, Z, cmap='RdYlGn', shading='auto', alpha=0.85,
        vmin=vmin, vmax=vmax
    )
    
    setup_shot_chart_axes(ax)
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Predicted Make Probability', fontsize=11)
    plt.title("GAM Predicted Shot Probability by Location\n(Green = easier, Red = harder; court coords in 0.1 ft)", fontsize=16)
    
    output_path = output_dir / "gam_spatial_probability.png"
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close()

    # Also plot the spatial tensor effect (log-odds contribution)
    effect = gam.partial_dependence(term=0, X=X_matrix).reshape(XX.shape)
    effect[mask] = np.nan

    fig, ax = plt.subplots(figsize=(12, 11))
    draw_half_court(ax, outer_lines=True)
    lim = np.nanpercentile(np.abs(effect), 98)
    mesh = ax.pcolormesh(
        XX, YY, effect, cmap='coolwarm', shading='auto', alpha=0.85,
        vmin=-lim, vmax=lim
    )
    setup_shot_chart_axes(ax)
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Log Odds (Spatial Effect)', fontsize=11)
    plt.title("Spatial Effect on Shot Probability\n(GAM Tensor Smooth; court coords in 0.1 ft)", fontsize=16)

    output_path = output_dir / "gam_spatial_tensor.png"
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("GAM ANALYSIS (PyGAM) - ROLLING WINDOW")
    print("=" * 60)
    
    # Generate rolling window training seasons
    training_seasons = get_rolling_window_seasons(TARGET_SEASON, window_size=ROLLING_WINDOW)
    print(f"Rolling Window Training: {training_seasons}")
    print(f"Target Season: {TARGET_SEASON}")
    print("=" * 60)
    
    # 1. Load TRAINING data (5 prior seasons)
    print(f"\nLoading training data from {training_seasons[0]} to {training_seasons[-1]}...")
    train_df = load_shot_data_from_db(training_seasons, cap_distance=False, exclude_backcourt=False)
    train_df = add_shot_type_features(train_df)
    print(f"Loaded {len(train_df):,} training shots")
    
    # 2. Load TARGET data (for visualizations)
    print(f"\nLoading target season {TARGET_SEASON} for visualizations...")
    target_df, season = load_data(cap_distance=False, exclude_backcourt=False)
    target_df = add_shot_type_features(target_df)
    print(f"Loaded {len(target_df):,} target shots")

    feature_cols = [
        'LOC_X', 'LOC_Y', 
        'shot_distance_feet', 
        'shot_angle', 
        'seconds_in_period',
        'PERIOD',
        'is_dunk',
        'is_layup',
        'is_hook',
        'is_floater',
        'is_jump_shot_2',
        'is_jump_shot_3',
        'is_clutch'
    ]
    
    # Prepare training data
    train_clean = train_df.dropna(subset=feature_cols + ['SHOT_MADE_FLAG']).copy()
    X_train = train_clean[feature_cols].values
    y_train = train_clean['SHOT_MADE_FLAG'].values
    
    # Prepare target data for evaluation
    target_clean = target_df.dropna(subset=feature_cols + ['SHOT_MADE_FLAG']).copy()
    X_target = target_clean[feature_cols].values
    y_target = target_clean['SHOT_MADE_FLAG'].values
    
    print(f"\nTrain (prior 5 seasons): {len(X_train):,} | Target (out-of-sample): {len(X_target):,}")
    
    # 3. Fit GAM on training data
    gam = fit_gam(X_train, y_train)
    
    # 4. Evaluate on TARGET season (out-of-sample)
    print("\n" + "=" * 60)
    print("EVALUATION (Out-of-Sample)")
    print(f"Training: {training_seasons[0]} to {training_seasons[-1]}")
    print(f"Evaluation: {TARGET_SEASON}")
    print("=" * 60)
    
    accuracy = gam.accuracy(X_target, y_target)
    y_pred_proba = gam.predict_proba(X_target)
    auc_roc = roc_auc_score(y_target, y_pred_proba)
    logloss = log_loss(y_target, y_pred_proba)
    
    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(y_target, y_pred_proba)

    print(f"Accuracy:     {accuracy:.4f}")
    print(f"AUC-ROC:      {auc_roc:.4f}")
    print(f"Log Loss:     {logloss:.4f}")
    print(f"Brier Score:  {brier:.4f}")
    
    # 5. Generate All Visualizations (using target season shot distribution)
    print("\n" + "=" * 40)
    print("GENERATING VISUALIZATIONS")
    print("=" * 40)
    
    plot_distance_effect(gam, FIGURES_DIR, target_df, obs_max=None)
    plot_angle_effect(gam, FIGURES_DIR)
    plot_clock_effect(gam, FIGURES_DIR)
    plot_period_effect(gam, FIGURES_DIR)
    plot_shot_type_effects(target_clean, FIGURES_DIR)
    plot_spatial_tensor(gam, FIGURES_DIR, target_clean)
    
    # 6. Save Model
    model_path = MODELS_DIR / f"gam_model_{season}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(gam, f)
    print(f"\nModel saved to {model_path}")

    metrics_path = DATA_DIR / "model_metrics_gam.csv"
    metrics_df = pd.DataFrame([{
        "model": "GAM (PyGAM)",
        "train_seasons": f"{training_seasons[0]} to {training_seasons[-1]}",
        "eval_season": TARGET_SEASON,
        "season_type": "regular",
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "log_loss": logloss,
        "brier_score": brier,
        "n_train": len(X_train),
        "n_eval": len(X_target),
        "updated_at": datetime.now().isoformat(timespec="seconds")
    }])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Model trained on: {len(X_train):,} shots ({training_seasons[0]} to {training_seasons[-1]})")
    print(f"Evaluated on:     {len(X_target):,} shots ({TARGET_SEASON}) - TRUE OUT-OF-SAMPLE")
    print("=" * 60)
