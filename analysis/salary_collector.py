"""
NBA Salary Data Collection.

Fetches player salary data from Basketball-Reference.
Combines with performance metrics for value analysis.
"""

import re
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup


# Configuration
ANALYSIS_DIR = Path(__file__).parent
DATA_DIR = ANALYSIS_DIR / "data"

# Basketball-Reference salary URL
BBR_SALARY_URL = "https://www.basketball-reference.com/contracts/players.html"


def fetch_salaries_bbr():
    """
    Fetch salary data from Basketball-Reference.
    
    Returns DataFrame with all NBA player salaries.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(BBR_SALARY_URL, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find salary table
        table = soup.find('table', id='player-contracts')
        
        if not table:
            print("Could not find salary table on BBR")
            return pd.DataFrame()
        
        rows = []
        for tr in table.find_all('tr')[1:]:  # Skip header
            cells = tr.find_all(['th', 'td'])
            if len(cells) >= 4:
                # Player name in first th/td
                name_cell = cells[1]
                name = name_cell.get_text(strip=True)
                
                # Team in second cell
                team = cells[2].get_text(strip=True) if len(cells) > 2 else ''
                
                # 2025-26 salary in fourth cell (or current season)
                salary_text = cells[3].get_text(strip=True) if len(cells) > 3 else ''
                
                salary = parse_salary(salary_text)
                
                if name and salary and salary > 0:
                    rows.append({
                        'player_name': clean_name(name),
                        'team': team,
                        'salary': salary,
                        'salary_millions': salary / 1_000_000
                    })
        
        df = pd.DataFrame(rows)
        print(f"Fetched {len(df)} player salaries from Basketball-Reference")
        return df
        
    except Exception as e:
        print(f"Error fetching from BBR: {e}")
        return pd.DataFrame()


def parse_salary(salary_str):
    """Parse salary string like '$12,345,678' to integer."""
    if not salary_str:
        return None
    
    # Remove $ and commas
    cleaned = re.sub(r'[,$]', '', str(salary_str))
    
    try:
        return int(cleaned)
    except ValueError:
        return None


def clean_name(name):
    """Clean player name for matching."""
    # Remove Jr., Sr., III, etc.
    name = re.sub(r'\s+(Jr\.?|Sr\.?|II|III|IV)$', '', name)
    return name.strip()


def create_sample_salary_data():
    """
    Create sample salary data for testing (if scraping fails).
    
    Top 50 salaries for 2024-25 season.
    """
    data = [
        ('Stephen Curry', 55761217),
        ('Nikola Jokić', 51415938),
        ('Joel Embiid', 51415938),
        ('Kevin Durant', 51179020),
        ('LeBron James', 50434636),
        ('Giannis Antetokounmpo', 48787676),
        ('Kawhi Leonard', 49350000),
        ('Paul George', 49350000),
        ('Damian Lillard', 48787676),
        ('Jimmy Butler', 48787676),
        ('Bradley Beal', 46741590),
        ('Anthony Davis', 43219440),
        ('Devin Booker', 43000000),
        ('Karl-Anthony Towns', 36016200),
        ('Jayson Tatum', 34848340),
        ('Jaylen Brown', 49500000),
        ('Luka Dončić', 43032015),
        ('Trae Young', 43031940),
        ('Donovan Mitchell', 36840000),
        ('Zion Williamson', 36016200),
        ('Rudy Gobert', 41000000),
        ('Bam Adebayo', 37000000),
        ('Darius Garland', 37090000),
        ('Tyrese Haliburton', 45000000),
        ('Ja Morant', 38175000),
        ('Anthony Edwards', 42200000),
        ('Jalen Brunson', 29727900),
        ('Cade Cunningham', 36000000),
        ('Shai Gilgeous-Alexander', 40064220),
        ('DeMar DeRozan', 28600000),
        ('Jamal Murray', 33833400),
        ('Fred VanVleet', 40806960),
        ('CJ McCollum', 33333333),
        ('Pascal Siakam', 37893408),
        ('Julius Randle', 28915040),
        ('Tobias Harris', 35366667),
        ('Khris Middleton', 31650000),
        ('Mike Conley', 22500000),
        ('Jrue Holiday', 35000000),
        ('Al Horford', 9500000),
        ('De\'Aaron Fox', 34671360),
        ('Scottie Barnes', 25000000),
        ('Paolo Banchero', 12000000),
        ('Victor Wembanyama', 12000000),
        ('Evan Mobley', 10000000),
        ('Desmond Bane', 25000000),
        ('Tyler Herro', 27000000),
        ('Jaren Jackson Jr.', 27000000),
        ('Mikal Bridges', 23000000),
        ('OG Anunoby', 38000000),
    ]
    
    df = pd.DataFrame(data, columns=['player_name', 'salary'])
    df['salary_millions'] = df['salary'] / 1_000_000
    df['team'] = ''  # No team in sample data
    
    return df


def match_salaries_to_players(salary_df, player_df):
    """
    Match salary data to player performance data.
    
    Uses fuzzy name matching since names may differ slightly.
    """
    from difflib import get_close_matches
    
    salary_names = salary_df['player_name'].tolist()
    
    matches = []
    for _, row in player_df.iterrows():
        player_name = row['PLAYER_NAME']
        
        # Try exact match first
        exact = salary_df[salary_df['player_name'] == player_name]
        if not exact.empty:
            matches.append({
                'PLAYER_ID': row['PLAYER_ID'],
                'PLAYER_NAME': player_name,
                'salary': exact.iloc[0]['salary'],
                'salary_millions': exact.iloc[0]['salary_millions']
            })
            continue
        
        # Try fuzzy match
        close = get_close_matches(player_name, salary_names, n=1, cutoff=0.8)
        if close:
            matched_salary = salary_df[salary_df['player_name'] == close[0]].iloc[0]
            matches.append({
                'PLAYER_ID': row['PLAYER_ID'],
                'PLAYER_NAME': player_name,
                'salary': matched_salary['salary'],
                'salary_millions': matched_salary['salary_millions']
            })
    
    return pd.DataFrame(matches)


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("NBA SALARY DATA COLLECTION")
    print("=" * 60)
    
    # Fetch from Basketball-Reference (has all players)
    salary_df = fetch_salaries_bbr()
    
    if salary_df.empty:
        print("Using sample salary data...")
        salary_df = create_sample_salary_data()
    
    # Save
    salary_df.to_csv(DATA_DIR / "player_salaries_2024-25.csv", index=False)
    print(f"Saved: {DATA_DIR / 'player_salaries_2024-25.csv'} ({len(salary_df)} players)")
    
    # Show top salaries
    print("\nTop 15 Salaries:")
    print(salary_df.nlargest(15, 'salary')[['player_name', 'team', 'salary_millions']].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
