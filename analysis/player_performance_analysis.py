
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc

def draw_nba_court(ax=None, color='black', lw=2, outer_lines=False):
    """
    Draws an NBA half court on a given Matplotlib axes.
    """
    if ax is None:
        ax = plt.gca()

    hoop = Circle((0, 0), radius=0.75, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-3, -0.75), 6, -0.1, linewidth=lw, color=color)
    outer_box = Rectangle((-8, -4.75), 16, 19, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-6, -4.75), 12, 19, linewidth=lw, color=color, fill=False)
    top_free_throw = Arc((0, 14.25), 12, 12, theta1=0, theta2=180, linewidth=lw, color=color, fill=False)
    bottom_free_throw = Arc((0, 14.25), 12, 12, theta1=180, theta2=360, linewidth=lw, color=color, linestyle='--')
    restricted = Arc((0, 0), 8, 8, theta1=0, theta2=180, linewidth=lw, color=color)
    corner_three_a = Rectangle((-22, -4.75), 0, 14, linewidth=lw, color=color)
    corner_three_b = Rectangle((22, -4.75), 0, 14, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 47.5, 47.5, theta1=22, theta2=158, linewidth=lw, color=color)
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

if __name__ == "__main__":
    # Load the enriched dataset
    DATA_FILE = 'analysis/shots_with_xp.parquet'
    print(f"Loading final dataset from {DATA_FILE}...")
    shots_df = pd.read_parquet(DATA_FILE)

    # --- Player Level Aggregation ---
    print("\n--- Aggregating player performance ---")
    player_summary = shots_df.groupby('PLAYER_NAME').agg(
        total_attempts=('SHOT_ATTEMPTED_FLAG', 'count'),
        total_poe=('POE', 'sum'),
        total_xp=('xP_value', 'sum'),
        total_actual_points=('actual_points', 'sum')
    ).reset_index()

    # Filter for players with a significant number of shots
    min_shots = 250
    player_summary = player_summary[player_summary['total_attempts'] >= min_shots].copy()
    player_summary['poe_per_100_shots'] = (player_summary['total_poe'] / player_summary['total_attempts']) * 100

    # Sort to find top and bottom players
    top_players = player_summary.sort_values('total_poe', ascending=False)
    bottom_players = player_summary.sort_values('total_poe', ascending=True)

    print(f"\n--- Top 15 Players by Total Points Over Expected (min. {min_shots} attempts) ---")
    print(top_players.head(15).to_string(index=False, formatters={'total_poe':'{:.2f}'.format, 'poe_per_100_shots':'{:.2f}'.format}))

    print(f"\n--- Bottom 15 Players by Total Points Over Expected (min. {min_shots} attempts) ---")
    print(bottom_players.head(15).to_string(index=False, formatters={'total_poe':'{:.2f}'.format, 'poe_per_100_shots':'{:.2f}'.format}))

    # --- Visualize Top Player's Shot Chart ---
    if not top_players.empty:
        top_player_name = top_players.iloc[0]['PLAYER_NAME']
        print(f"\n--- Generating POE Shot Chart for: {top_player_name} ---")
        
        player_shots = shots_df[shots_df['PLAYER_NAME'] == top_player_name]
        
        plt.style.use('fivethirtyeight')
        fig, ax = plt.subplots(figsize=(12, 11))

        # Draw court and plot shots, colored by POE
        draw_nba_court(ax, outer_lines=True)
        
        # Separate made and missed shots for the player
        player_made_shots = player_shots[player_shots['SHOT_MADE_FLAG'] == 1]
        player_missed_shots = player_shots[player_shots['SHOT_MADE_FLAG'] == 0]

        # Plot made shots with stars
        scatter_made = ax.scatter(
            player_made_shots['LOC_X'], 
            player_made_shots['LOC_Y'],
            c=player_made_shots['POE'],
            cmap='coolwarm', 
            marker='*',  # Star for made shots
            s=100,       # Larger size for stars
            alpha=0.8,
            label='Made Shot'
        )
        
        # Plot missed shots with circles
        scatter_missed = ax.scatter(
            player_missed_shots['LOC_X'], 
            player_missed_shots['LOC_Y'],
            c=player_missed_shots['POE'],
            cmap='coolwarm', 
            marker='o',  # Circle for missed shots
            s=50,        # Smaller size for circles
            alpha=0.6,
            label='Missed Shot'
        )
        
        # Create a single color bar based on the made shots scatter (or either, as colors are global)
        cbar = fig.colorbar(scatter_made, ax=ax, shrink=0.6)
        cbar.set_label('Points Over Expected (POE)', fontsize=12)

        # Add a legend for the markers
        ax.legend(loc='upper left', fontsize=10)

        ax.set_xlim(-250, 250)
        ax.set_ylim(-47.5, 422.5)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_title(f'{top_player_name} Shot Value Chart (2022-23)', fontsize=18)
        
        output_file = f"analysis/{top_player_name.replace(' ', '_')}_poe_shotchart.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Player shot chart saved to {output_file}")
    else:
        print("\nNo players met the minimum shot criteria for visualization.")
