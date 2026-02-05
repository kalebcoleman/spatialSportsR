"""
Expected Points (xP) Analysis for NBA Shot Prediction.

Uses Logistic Regression to predict shot success probability,
calculates Expected Points (xP) for each shot, and computes 
Points Over Expected (POE).

Features:
- Spatial: LOC_X, LOC_Y, shot distance, shot angle
- Shot type: layup, dunk, jump shot, hook, floater
- Context: period, game clock, clutch indicator
- Zone: SHOT_ZONE_BASIC, SHOT_ZONE_AREA

Accuracy: ~63% (near ceiling without defender tracking data)
"""

import os
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, log_loss, accuracy_score

from court_utils import draw_half_court, setup_shot_chart_axes


# Configuration
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "parsed" / "nba.sqlite"
DB_PATH = Path(os.getenv("SPATIALSPORTSR_DB_PATH", str(DEFAULT_DB_PATH))).expanduser()


def load_shot_data(season, season_type="regular"):
    """Load NBA shot chart data from SQLite database."""
    con = sqlite3.connect(str(DB_PATH))
    
    query = """
    SELECT
        LOC_X, LOC_Y, SHOT_MADE_FLAG, SHOT_TYPE, ACTION_TYPE,
        SHOT_ZONE_BASIC, SHOT_ZONE_AREA, SHOT_DISTANCE,
        PERIOD, MINUTES_REMAINING, SECONDS_REMAINING,
        PLAYER_NAME, GAME_ID
    FROM nba_stats_shots
    WHERE season = ? AND season_type = ? AND SHOT_ATTEMPTED_FLAG = 1
    """
    
    df = pd.read_sql_query(query, con, params=[season, season_type])
    con.close()
    
    if df.empty:
        print(f"Warning: No data found for season={season}, season_type={season_type}")
    return df


def engineer_features(df):
    """Create features for shot prediction."""
    df = df.copy()
    
    # Convert to numeric
    for col in ['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG', 'SHOT_DISTANCE', 
                'PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG', 'ACTION_TYPE'], inplace=True)
    
    # Spatial features
    df['shot_distance_feet'] = df['SHOT_DISTANCE'].fillna(
        np.sqrt(df['LOC_X']**2 + df['LOC_Y']**2) / 10
    )
    df['shot_angle'] = np.arctan2(df['LOC_X'], df['LOC_Y'].clip(lower=1))
    
    # Game clock
    df['seconds_in_period'] = (
        df['MINUTES_REMAINING'].fillna(0) * 60 + 
        df['SECONDS_REMAINING'].fillna(0)
    )
    
    # Clutch (last 2 min of 4th/OT)
    df['is_clutch'] = ((df['PERIOD'] >= 4) & (df['seconds_in_period'] <= 120)).astype(int)
    
    # Shot type indicators
    action = df['ACTION_TYPE'].str.lower().fillna('')
    df['is_layup'] = action.str.contains('layup|finger roll').astype(int)
    df['is_dunk'] = action.str.contains('dunk').astype(int)
    df['is_jump_shot'] = action.str.contains('jump shot|pullup|step back|fadeaway').astype(int)
    df['is_hook'] = action.str.contains('hook').astype(int)
    df['is_floater'] = action.str.contains('float').astype(int)
    
    # Shot value
    df['shot_value'] = df['SHOT_TYPE'].apply(lambda x: 3 if '3PT' in str(x) else 2)
    
    return df


def build_model():
    """Build logistic regression pipeline with preprocessing."""
    numeric_features = [
        'LOC_X', 'LOC_Y', 'shot_distance_feet', 'shot_angle',
        'PERIOD', 'seconds_in_period', 'is_clutch',
        'is_layup', 'is_dunk', 'is_jump_shot', 'is_hook', 'is_floater'
    ]
    categorical_features = ['SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42))
    ])
    
    return pipeline, numeric_features + categorical_features


def generate_visualizations(df, season, output_dir):
    """Generate shot distribution and distance analysis plots."""
    plt.style.use('fivethirtyeight')
    
    # Shot distribution chart
    fig, ax = plt.subplots(figsize=(12, 11))
    draw_half_court(ax, outer_lines=True)
    
    made = df[df['SHOT_MADE_FLAG'] == 1]
    missed = df[df['SHOT_MADE_FLAG'] == 0]
    
    ax.scatter(missed['LOC_X'], missed['LOC_Y'], c='r', marker='x', s=10, alpha=0.3, label='Miss')
    ax.scatter(made['LOC_X'], made['LOC_Y'], c='g', marker='o', s=15, alpha=0.5, label='Make')
    
    setup_shot_chart_axes(ax)
    ax.set_title(f'NBA Shot Distribution: {season} Season', fontsize=18)
    ax.legend(fontsize=12)
    
    plt.savefig(output_dir / f'shot_chart_{season}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'shot_chart_{season}.png'}")
    plt.close()
    
    # Shot distance analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    distance_bins = df.groupby(df['shot_distance_feet'].round())
    distance_analysis = distance_bins['SHOT_MADE_FLAG'].agg(['count', 'mean']).reset_index()
    distance_analysis.columns = ['distance', 'attempts', 'fg_pct']
    distance_analysis = distance_analysis[distance_analysis['distance'] < 38]
    
    ax1.bar(distance_analysis['distance'], distance_analysis['attempts'], width=0.8, alpha=0.8, color='skyblue')
    ax1.set_title(f'NBA Shot Distribution by Distance ({season})', fontsize=16)
    ax1.set_ylabel('Number of Attempts')
    ax1.axvline(x=23.75, linestyle='--', color='grey', label='3-Point Line')
    ax1.legend()
    
    ax2.plot(distance_analysis['distance'], distance_analysis['fg_pct'], marker='o', linestyle='-', color='orangered')
    ax2.set_title(f'NBA Field Goal % by Distance ({season})', fontsize=16)
    ax2.set_xlabel('Shot Distance (feet)')
    ax2.set_ylabel('Field Goal Percentage')
    ax2.axvline(x=23.75, linestyle='--', color='grey')
    
    plt.savefig(output_dir / f'shot_distance_analysis_{season}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'shot_distance_analysis_{season}.png'}")
    plt.close()


if __name__ == "__main__":
    SEASON = '2025-26'
    SEASON_TYPE = 'regular'
    OUTPUT_DIR = Path(__file__).parent / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load and prepare data
    print(f"Loading {SEASON} {SEASON_TYPE} season...")
    shots_df = load_shot_data(SEASON, SEASON_TYPE)
    print(f"Loaded {len(shots_df):,} shots")
    
    shots_df = engineer_features(shots_df)
    print(f"After cleaning: {len(shots_df):,} shots")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(shots_df, SEASON, OUTPUT_DIR)
    
    # Prepare features
    feature_cols = [
        'LOC_X', 'LOC_Y', 'shot_distance_feet', 'shot_angle',
        'PERIOD', 'seconds_in_period', 'is_clutch',
        'is_layup', 'is_dunk', 'is_jump_shot', 'is_hook', 'is_floater',
        'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA'
    ]
    
    X = shots_df[feature_cols].copy()
    y = shots_df['SHOT_MADE_FLAG'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Train model
    print("\nTraining Logistic Regression...")
    model, _ = build_model()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("MODEL RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Log Loss:  {log_loss(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    model_path = OUTPUT_DIR / f'xp_model_{SEASON}.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Generate xP and POE
    print("\nGenerating xP and POE for all shots...")
    shots_df['xP_prob'] = model.predict_proba(shots_df[feature_cols])[:, 1]
    shots_df['xP_value'] = shots_df['xP_prob'] * shots_df['shot_value']
    shots_df['actual_points'] = shots_df['SHOT_MADE_FLAG'] * shots_df['shot_value']
    shots_df['POE'] = shots_df['actual_points'] - shots_df['xP_value']
    
    # Save enriched data
    output_path = OUTPUT_DIR / f'shots_with_xp_{SEASON}.parquet'
    shots_df.to_parquet(output_path)
    print(f"Saved: {output_path}")
    
    # Top shots by POE
    print("\nTop 5 shots with highest POE:")
    top_cols = ['PLAYER_NAME', 'ACTION_TYPE', 'shot_distance_feet', 'xP_prob', 'POE']
    print(shots_df.nlargest(5, 'POE')[top_cols].to_string(index=False))
