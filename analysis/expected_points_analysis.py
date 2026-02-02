
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, log_loss
import joblib

def draw_nba_court(ax=None, color='black', lw=2, outer_lines=False):
    """
    Draws an NBA half court on a given Matplotlib axes.
    """
    if ax is None:
        ax = plt.gca()

    # Create the key
    hoop = Circle((0, 0), radius=0.75, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-3, -0.75), 6, -0.1, linewidth=lw, color=color)
    outer_box = Rectangle((-8, -4.75), 16, 19, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-6, -4.75), 12, 19, linewidth=lw, color=color, fill=False)
    top_free_throw = Arc((0, 14.25), 12, 12, theta1=0, theta2=180, linewidth=lw, color=color, fill=False)
    bottom_free_throw = Arc((0, 14.25), 12, 12, theta1=180, theta2=360, linewidth=lw, color=color, linestyle='--')
    restricted = Arc((0, 0), 8, 8, theta1=0, theta2=180, linewidth=lw, color=color)

    # Three-point line
    corner_three_a = Rectangle((-22, -4.75), 0, 14, linewidth=lw, color=color)
    corner_three_b = Rectangle((22, -4.75), 0, 14, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 47.5, 47.5, theta1=22, theta2=158, linewidth=lw, color=color)

    # Center court
    center_outer_arc = Arc((0, 42.25), 12, 12, theta1=180, theta2=0, linewidth=lw, color=color)
    center_inner_arc = Arc((0, 42.25), 4, 4, theta1=180, theta2=0, linewidth=lw, color=color)

    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        outer_lines = Rectangle((-25, -4.75), 50, 47, linewidth=lw, color=color, fill=False)
        court_elements.append(outer_lines)

    for element in court_elements:
        ax.add_patch(element)

    return ax

DB_PATH = "/Users/itzjuztmya/Kaleb/spatialSportsR/data/parsed/nba.sqlite"

def load_shot_data(season, season_type="regular"):
    """
    Loads NBA shot chart data from the SQLite database for a given season.
    """
    con = sqlite3.connect(DB_PATH)
    
    # Construct the query to get all necessary columns
    query = """
    SELECT
        LOC_X,
        LOC_Y,
        SHOT_MADE_FLAG,
        SHOT_TYPE,
        PLAYER_NAME
    FROM nba_stats_shots
    WHERE season = ? AND season_type = ? AND SHOT_ATTEMPTED_FLAG = 1
    """
    
    df = pd.read_sql_query(query, con, params=[season, season_type])
    con.close()
    
    if df.empty:
        print(f"Warning: No data found for season={season}, season_type={season_type}")
        
    return df

if __name__ == "__main__":
    # Define the season and output paths
    SEASON = '2025-26'
    SEASON_TYPE = 'regular'

    SHOT_CHART_OUTPUT_FILE = f'shot_chart_{SEASON}.png'
    HEATMAP_OUTPUT_FILE = f'shot_efficiency_heatmap_{SEASON}.png'
    DISTANCE_OUTPUT_FILE = f'shot_distance_analysis_{SEASON}.png'
    MODEL_OUTPUT_FILE = f'expected_points_model_{SEASON}.joblib'
    FINAL_DATA_OUTPUT_FILE = f'shots_with_xp_{SEASON}.parquet'

    print(f"Loading data for {SEASON} {SEASON_TYPE} season...")
    shots_df = load_shot_data(SEASON, SEASON_TYPE)

    # Convert columns to numeric, coercing errors, and drop rows with invalid data
    for col in ['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']:
        shots_df[col] = pd.to_numeric(shots_df[col], errors='coerce')
    shots_df.dropna(subset=['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG'], inplace=True)


    print("Data loaded. Generating shot distribution plot...")
    
    # --- Shot Distribution Chart ---
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(12, 11))
    draw_nba_court(ax, outer_lines=True)
    made_shots = shots_df[shots_df['SHOT_MADE_FLAG'] == 1]
    missed_shots = shots_df[shots_df['SHOT_MADE_FLAG'] == 0]
    ax.scatter(missed_shots['LOC_X'], missed_shots['LOC_Y'], c='r', marker='x', s=10, alpha=0.3, label='Miss')
    ax.scatter(made_shots['LOC_X'], made_shots['LOC_Y'], c='g', marker='o', s=15, alpha=0.5, label='Make')
    ax.set_xlim(-250, 250)
    ax.set_ylim(-47.5, 422.5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_title(f'NBA Shot Distribution: {SEASON} Season', fontsize=18)
    ax.legend(fontsize=12)
    plt.savefig(SHOT_CHART_OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {SHOT_CHART_OUTPUT_FILE}")
    plt.close(fig) # Close figure to free up memory

    # --- Shot Efficiency Heatmap ---
    print("Generating shot efficiency heatmap...")
    fig, ax = plt.subplots(figsize=(12, 11))
    x_bins = 50
    y_bins = 50
    x_range = (-250, 250)
    y_range = (-47.5, 422.5)
    total_shots_hist, yedges, xedges = np.histogram2d(shots_df['LOC_Y'], shots_df['LOC_X'], bins=(y_bins, x_bins), range=[y_range, x_range])
    made_shots_hist, _, _ = np.histogram2d(made_shots['LOC_Y'], made_shots['LOC_X'], bins=(y_bins, x_bins), range=[y_range, x_range])
    shot_efficiency = np.divide(made_shots_hist, total_shots_hist, out=np.zeros_like(made_shots_hist), where=total_shots_hist != 0)
    shot_efficiency = np.ma.masked_where(total_shots_hist == 0, shot_efficiency)
    if shot_efficiency.count() > 0:
        vmin, vmax = np.percentile(shot_efficiency.compressed(), [5, 95])
    else:
        vmin, vmax = 0.2, 0.7
    im = ax.imshow(shot_efficiency, extent=[x_range[1], x_range[0], y_range[1], y_range[0]], cmap='viridis', interpolation='bilinear', vmin=vmin, vmax=vmax)
    draw_nba_court(ax, color="white", lw=2, outer_lines=True)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Field Goal %', fontsize=12)
    ax.set_xlim(-250, 250)
    ax.set_ylim(-47.5, 422.5)
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_title(f'NBA Shot Efficiency: {SEASON} Season', fontsize=18)
    plt.savefig(HEATMAP_OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {HEATMAP_OUTPUT_FILE}")
    plt.close(fig) # Close figure

    # --- Shot Distance Analysis ---
    print("Generating shot distance analysis plot...")
    shots_df['SHOT_DISTANCE_FEET'] = np.sqrt(shots_df['LOC_X']**2 + shots_df['LOC_Y']**2) / 10
    distance_bins = shots_df.groupby(shots_df['SHOT_DISTANCE_FEET'].round())
    distance_analysis = distance_bins['SHOT_MADE_FLAG'].agg(['count', 'mean']).reset_index()
    distance_analysis.rename(columns={'count': 'attempts', 'mean': 'fg_percentage'}, inplace=True)
    distance_analysis = distance_analysis[distance_analysis['SHOT_DISTANCE_FEET'] < 38]

    # Create a two-panel plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    
    # Top panel: Shot Attempts
    ax1.bar(distance_analysis['SHOT_DISTANCE_FEET'], distance_analysis['attempts'], 
            width=0.8, alpha=0.8, color='skyblue')
    ax1.set_title(f'NBA Shot Distribution by Distance ({SEASON})', fontsize=16)
    ax1.set_ylabel('Number of Attempts')
    ax1.axvline(x=23.75, linestyle='--', color='grey', label='3-Point Line (Arc)')
    ax1.set_xlim(0, 38)
    ax1.legend()
    
    # Bottom panel: Field Goal Percentage
    ax2.plot(distance_analysis['SHOT_DISTANCE_FEET'], distance_analysis['fg_percentage'], 
             marker='o', linestyle='-', color='orangered')
    ax2.set_title(f'NBA Field Goal % by Distance ({SEASON})', fontsize=16)
    ax2.set_xlabel('Shot Distance (feet)')
    ax2.set_ylabel('Field Goal Percentage')
    ax2.axvline(x=23.75, linestyle='--', color='grey', label='3-Point Line (Arc)')
    ax2.set_xlim(0, 38)
    plt.savefig(DISTANCE_OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Shot distance analysis plot saved to {DISTANCE_OUTPUT_FILE}")
    plt.close(fig) # Close figure

    # --- Build Expected Points (xP) Model ---
    print("Building and training the Expected Points model...")
    features = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE_FEET']
    target = 'SHOT_MADE_FLAG'
    X = shots_df[features]
    y = shots_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xp_model = LogisticRegression(solver='liblinear', random_state=42)
    xp_model.fit(X_train, y_train)
    joblib.dump(xp_model, MODEL_OUTPUT_FILE)
    print(f"Trained model saved to {MODEL_OUTPUT_FILE}")

    # --- Evaluate Model and Generate xP/POE ---
    print("\n--- Model Evaluation ---")
    loaded_model = joblib.load(MODEL_OUTPUT_FILE)
    y_pred = loaded_model.predict(X_test)
    y_pred_proba = loaded_model.predict_proba(X_test)[:, 1]
    print(f"Test Set Accuracy: {loaded_model.score(X_test, y_test):.4f}")
    print(f"Test Set ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Test Set Log Loss: {log_loss(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("--- Generating xP and POE for all shots ---")
    all_shots_X = shots_df[features]
    shots_df['xP_prob'] = loaded_model.predict_proba(all_shots_X)[:, 1]
    shots_df['shot_value'] = shots_df['SHOT_TYPE'].apply(lambda x: 3 if '3PT' in x else 2)
    shots_df['xP_value'] = shots_df['xP_prob'] * shots_df['shot_value']
    shots_df['actual_points'] = shots_df['SHOT_MADE_FLAG'] * shots_df['shot_value']
    shots_df['POE'] = shots_df['actual_points'] - shots_df['xP_value']
    shots_df.to_parquet(FINAL_DATA_OUTPUT_FILE)
    
    print(f"\nFinal dataset with xP and POE saved to {FINAL_DATA_OUTPUT_FILE}")
    print("Top 5 shots with highest POE:")
    print(shots_df.nlargest(5, 'POE')[['PLAYER_NAME', 'SHOT_TYPE', 'SHOT_DISTANCE_FEET', 'actual_points', 'xP_value', 'POE']])
