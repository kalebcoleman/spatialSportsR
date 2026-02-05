"""
Player Value Analysis.

Combines shot performance (POE) with salary data to identify:
- Best value players (high POE, low salary)
- Overpaid players (low POE, high salary)
- Salary efficiency rankings
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Configuration
ANALYSIS_DIR = Path(__file__).parent
DATA_DIR = ANALYSIS_DIR / "data"
FIGURES_DIR = ANALYSIS_DIR / "figures"


def load_data():
    """Load player stats and salary data."""
    stats_path = DATA_DIR / "player_summary.csv"
    salary_path = DATA_DIR / "player_salaries_2024-25.csv"

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing {stats_path}. Run player_performance_analysis.py first."
        )
    if not salary_path.exists():
        raise FileNotFoundError(
            f"Missing {salary_path}. Run salary_collector.py first."
        )

    stats_df = pd.read_csv(stats_path)
    salary_df = pd.read_csv(salary_path)

    # Normalize column names from different sources
    if 'player' in stats_df.columns and 'PLAYER_NAME' not in stats_df.columns:
        stats_df = stats_df.rename(columns={'player': 'PLAYER_NAME'})

    rename_map = {
        'total_POE': 'total_poe',
        'total_actual_points': 'total_actual_points',
        'actual_points': 'total_actual_points',
        'total_attempts': 'total_attempts',
        'attempts': 'total_attempts'
    }
    stats_df = stats_df.rename(columns={k: v for k, v in rename_map.items() if k in stats_df.columns})

    required_cols = {'PLAYER_NAME', 'total_poe', 'total_actual_points', 'total_attempts'}
    missing = required_cols - set(stats_df.columns)
    if missing:
        raise ValueError(
            f"player_summary.csv missing required columns: {', '.join(sorted(missing))}. "
            f"Available: {', '.join(stats_df.columns)}"
        )

    # Ensure FG residual metrics exist
    if 'fg_pct' in stats_df.columns and 'avg_xp_prob' in stats_df.columns:
        if 'fg_residual' not in stats_df.columns:
            stats_df['fg_residual'] = stats_df['fg_pct'] - stats_df['avg_xp_prob']
        if 'fg_residual_pct' not in stats_df.columns:
            stats_df['fg_residual_pct'] = stats_df['fg_residual'] * 100
    elif 'shooting_skill' in stats_df.columns:
        if 'fg_residual' not in stats_df.columns:
            stats_df['fg_residual'] = stats_df['shooting_skill']
        if 'fg_residual_pct' not in stats_df.columns:
            stats_df['fg_residual_pct'] = stats_df['fg_residual'] * 100

    return stats_df, salary_df


def match_and_merge(stats_df, salary_df):
    """Match salary data to player stats using name matching."""
    from difflib import get_close_matches
    
    salary_names = salary_df['player_name'].tolist()
    
    # Create mapping
    matches = []
    for _, row in stats_df.iterrows():
        player_name = row['PLAYER_NAME']
        
        # Exact match
        exact = salary_df[salary_df['player_name'] == player_name]
        if not exact.empty:
            matches.append({
                'PLAYER_NAME': player_name,
                'salary_millions': exact.iloc[0]['salary_millions']
            })
            continue
        
        # Fuzzy match
        close = get_close_matches(player_name, salary_names, n=1, cutoff=0.8)
        if close:
            match = salary_df[salary_df['player_name'] == close[0]].iloc[0]
            matches.append({
                'PLAYER_NAME': player_name,
                'salary_millions': match['salary_millions']
            })
    
    salary_matches = pd.DataFrame(matches)
    
    # Merge
    merged = stats_df.merge(salary_matches, on='PLAYER_NAME', how='left')
    
    return merged


def compute_value_metrics(df):
    """Compute value efficiency metrics."""
    df = df.copy()
    
    # Only compute for players with salary data
    df_with_salary = df[df['salary_millions'].notna()].copy()
    
    # POE per $1M
    df_with_salary['POE_per_million'] = (
        df_with_salary['total_poe'] / df_with_salary['salary_millions'].clip(lower=1)
    )
    
    # Points per $1M
    df_with_salary['points_per_million'] = (
        df_with_salary['total_actual_points'] / df_with_salary['salary_millions'].clip(lower=1)
    )
    
    # Expected cost per POE (how much they're paid per point above expectation)
    # Higher = more expensive per added value
    df_with_salary['cost_per_POE'] = (
        df_with_salary['salary_millions'] / df_with_salary['total_poe'].clip(lower=1)
    )
    
    return df_with_salary


def plot_value_chart(df, output_path):
    """Create scatter plot of POE vs Salary."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color by POE_per_million
    scatter = ax.scatter(
        df['salary_millions'],
        df['total_poe'],
        s=df['total_attempts'] / 5,
        c=df['POE_per_million'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Add diagonal reference line (league average value)
    x_max = df['salary_millions'].max()
    y_max = df['total_poe'].max()
    
    # Label all players
    for _, row in df.iterrows():
        ax.annotate(
            row['PLAYER_NAME'].split()[-1],  # Last name only
            (row['salary_millions'], row['total_poe']),
            fontsize=8,
            alpha=0.8
        )
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Salary ($M)', fontsize=12)
    ax.set_ylabel('Total POE (Points Over Expected)', fontsize=12)
    ax.set_title('Player Value: POE vs Salary\n(Size = Volume, Color = POE per $M)', fontsize=14)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('POE per $1M', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_fg_residual_chart(df, output_path):
    """Create scatter plot of FG% residual vs Salary."""
    fig, ax = plt.subplots(figsize=(14, 10))

    scatter = ax.scatter(
        df['salary_millions'],
        df['fg_residual_pct'],
        s=df['total_attempts'] / 5,
        c=df['fg_residual_pct'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )

    for _, row in df.iterrows():
        ax.annotate(
            row['PLAYER_NAME'].split()[-1],
            (row['salary_millions'], row['fg_residual_pct']),
            fontsize=8,
            alpha=0.8
        )

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Salary ($M)', fontsize=12)
    ax.set_ylabel('FG% Residual (Actual - Expected) in % pts', fontsize=12)
    ax.set_title('Player Value: FG% Residual vs Salary\n(Size = Volume, Color = FG% Residual)', fontsize=14)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('FG% Residual (pct pts)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_value_report(df):
    """Print value analysis report."""
    print("\n" + "=" * 60)
    print("PLAYER VALUE ANALYSIS")
    print("=" * 60)
    
    # Best value (high POE per $M)
    print("\n🏆 BEST VALUE (Highest POE per $1M):")
    best = df.nlargest(10, 'POE_per_million')
    for _, row in best.iterrows():
        print(f"  {row['PLAYER_NAME']:25} {row['POE_per_million']:+6.2f} POE/$M  (${row['salary_millions']:.1f}M, {row['total_poe']:+.0f} POE)")
    
    # Worst value (lowest POE per $M with high salary)
    print("\n💸 LOWEST VALUE (Low POE, High Salary):")
    high_salary = df[df['salary_millions'] >= 20]
    worst = high_salary.nsmallest(10, 'POE_per_million')
    for _, row in worst.iterrows():
        print(f"  {row['PLAYER_NAME']:25} {row['POE_per_million']:+6.2f} POE/$M  (${row['salary_millions']:.1f}M, {row['total_poe']:+.0f} POE)")
    
    # Overperformers on cheap contracts
    print("\n💎 HIDDEN GEMS (High POE, Low Salary):")
    cheap = df[(df['salary_millions'] < 15) & (df['total_poe'] > 50)]
    for _, row in cheap.nlargest(5, 'total_poe').iterrows():
        print(f"  {row['PLAYER_NAME']:25} {row['total_poe']:+6.0f} POE  (${row['salary_millions']:.1f}M)")


if __name__ == "__main__":
    print("=" * 60)
    print("PLAYER VALUE ANALYSIS: POE + SALARY")
    print("=" * 60)

    DATA_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    
    # Load data
    stats_df, salary_df = load_data()
    print(f"Loaded {len(stats_df)} players with performance data")
    print(f"Loaded {len(salary_df)} players with salary data")
    
    # Match and merge
    merged = match_and_merge(stats_df, salary_df)
    print(f"Matched {merged['salary_millions'].notna().sum()} players with salary")
    
    # Compute value metrics
    value_df = compute_value_metrics(merged)
    
    # Create report
    create_value_report(value_df)
    
    # Save
    value_df.to_csv(DATA_DIR / "player_value_with_salary.csv", index=False)
    print(f"\nSaved: {DATA_DIR / 'player_value_with_salary.csv'}")
    
    # Plot
    plot_value_chart(value_df, FIGURES_DIR / "poe_vs_salary_scatter.png")

    if 'fg_residual_pct' in value_df.columns:
        plot_fg_residual_chart(value_df, FIGURES_DIR / "fg_residual_vs_salary_scatter.png")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
