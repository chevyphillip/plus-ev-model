"""Script to sync NBA player data from both NBA API and Odds API."""

import logging
from typing import Dict, List, Tuple
import duckdb
import requests
from nba_api.stats.endpoints import commonallplayers
from datetime import datetime, timezone
from difflib import get_close_matches

logger = logging.getLogger(__name__)

ODDS_API_KEY = "bcab2a03da8de48a2a68698a40b78b4c"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

def create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create necessary database tables."""
    # NBA API players table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nba_players (
            player_id INTEGER PRIMARY KEY,
            first_name VARCHAR,
            last_name VARCHAR,
            full_name VARCHAR,
            is_active BOOLEAN,
            team_id INTEGER,
            team_name VARCHAR,
            team_abbreviation VARCHAR,
            last_updated TIMESTAMP
        )
    """)
    
    # Odds API teams table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds_teams (
            team_id VARCHAR PRIMARY KEY,
            full_name VARCHAR,
            last_updated TIMESTAMP
        )
    """)
    
    # Odds API players table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds_players (
            player_id VARCHAR PRIMARY KEY,
            full_name VARCHAR,
            team_id VARCHAR,
            last_updated TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES odds_teams(team_id)
        )
    """)
    
    # Player ID mapping table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_mapping (
            nba_player_id INTEGER,
            odds_player_id VARCHAR,
            match_confidence FLOAT,
            last_updated TIMESTAMP,
            PRIMARY KEY (nba_player_id, odds_player_id),
            FOREIGN KEY (nba_player_id) REFERENCES nba_players(player_id),
            FOREIGN KEY (odds_player_id) REFERENCES odds_players(player_id)
        )
    """)

def get_odds_teams() -> List[Dict]:
    """Get all NBA teams from Odds API."""
    url = f"{ODDS_API_BASE_URL}/sports/basketball_nba/participants"
    response = requests.get(url, params={"apiKey": ODDS_API_KEY})
    response.raise_for_status()
    
    teams = response.json()
    logger.info(f"Found {len(teams)} teams from Odds API")
    return teams

def get_odds_players(team_id: str) -> List[Dict]:
    """Get all players for a team from Odds API."""
    url = f"{ODDS_API_BASE_URL}/sports/basketball_nba/participants/{team_id}/players"
    response = requests.get(url, params={"apiKey": ODDS_API_KEY})
    response.raise_for_status()
    
    players = response.json()
    logger.info(f"Found {len(players)} players for team {team_id}")
    return players

def get_nba_players() -> List[Dict]:
    """Get all active NBA players from NBA API."""
    try:
        # Get all active players
        all_players = commonallplayers.CommonAllPlayers(
            is_only_current_season=1,
            league_id="00"
        ).get_data_frames()[0]
        
        # Convert to list of dicts
        players = []
        for _, row in all_players.iterrows():
            # Use DISPLAY_FIRST_LAST for full name
            name_parts = row['DISPLAY_FIRST_LAST'].split(' ', 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else ''
            
            players.append({
                'player_id': row['PERSON_ID'],
                'first_name': first_name,
                'last_name': last_name,
                'full_name': row['DISPLAY_FIRST_LAST'],
                'is_active': True,
                'team_id': row['TEAM_ID'],
                'team_name': row['TEAM_NAME'],
                'team_abbreviation': row['TEAM_ABBREVIATION'],
                'last_updated': datetime.now(timezone.utc)
            })
        
        logger.info(f"Found {len(players)} active players from NBA API")
        return players
        
    except Exception as e:
        logger.error(f"Error getting NBA player data: {str(e)}")
        raise

def match_players(nba_players: List[Dict], odds_players: List[Dict]) -> List[Tuple[Dict, Dict, float]]:
    """Match players between NBA API and Odds API using fuzzy name matching."""
    matches = []
    
    # Create list of Odds API player names
    odds_names = [p['full_name'] for p in odds_players]
    
    # Find matches for each NBA player
    for nba_player in nba_players:
        # Try to find close matches
        close_matches = get_close_matches(
            nba_player['full_name'],
            odds_names,
            n=1,
            cutoff=0.8
        )
        
        if close_matches:
            # Find the matching Odds player
            odds_player = next(
                p for p in odds_players 
                if p['full_name'] == close_matches[0]
            )
            
            # Calculate match confidence
            confidence = 1.0 if nba_player['full_name'] == odds_player['full_name'] else 0.8
            
            matches.append((nba_player, odds_player, confidence))
    
    logger.info(f"Found {len(matches)} player matches")
    return matches

def sync_players(db_path: str = 'data/nba_stats.duckdb') -> None:
    """Sync player data from both APIs to database."""
    try:
        # Connect to database
        with duckdb.connect(db_path) as conn:
            # Create tables
            create_tables(conn)
            
            # Get NBA API players
            nba_players = get_nba_players()
            
            # Get Odds API teams and players
            odds_teams = get_odds_teams()
            odds_players = []
            for team in odds_teams:
                team_players = get_odds_players(team['id'])
                for player in team_players:
                    player['team_id'] = team['id']
                odds_players.extend(team_players)
            
            # Update teams
            for team in odds_teams:
                conn.execute("""
                    INSERT OR REPLACE INTO odds_teams (
                        team_id,
                        full_name,
                        last_updated
                    ) VALUES (?, ?, ?)
                """, [
                    team['id'],
                    team['full_name'],
                    datetime.now(timezone.utc)
                ])
            
            # Update NBA players
            for player in nba_players:
                conn.execute("""
                    INSERT OR REPLACE INTO nba_players (
                        player_id,
                        first_name,
                        last_name,
                        full_name,
                        is_active,
                        team_id,
                        team_name,
                        team_abbreviation,
                        last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    player['player_id'],
                    player['first_name'],
                    player['last_name'],
                    player['full_name'],
                    player['is_active'],
                    player['team_id'],
                    player['team_name'],
                    player['team_abbreviation'],
                    player['last_updated']
                ])
            
            # Update Odds players
            for player in odds_players:
                conn.execute("""
                    INSERT OR REPLACE INTO odds_players (
                        player_id,
                        full_name,
                        team_id,
                        last_updated
                    ) VALUES (?, ?, ?, ?)
                """, [
                    player['id'],
                    player['full_name'],
                    player['team_id'],
                    datetime.now(timezone.utc)
                ])
            
            # Match players and update mapping
            matches = match_players(nba_players, odds_players)
            for nba_player, odds_player, confidence in matches:
                conn.execute("""
                    INSERT OR REPLACE INTO player_mapping (
                        nba_player_id,
                        odds_player_id,
                        match_confidence,
                        last_updated
                    ) VALUES (?, ?, ?, ?)
                """, [
                    nba_player['player_id'],
                    odds_player['id'],
                    confidence,
                    datetime.now(timezone.utc)
                ])
            
            logger.info("Successfully synced player data")
            
    except Exception as e:
        logger.error(f"Error syncing player data: {str(e)}")
        raise

def get_player_mapping(db_path: str = 'data/nba_stats.duckdb') -> Dict[str, int]:
    """Get mapping of player names to NBA IDs from database.
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary mapping player names to NBA IDs
    """
    try:
        with duckdb.connect(db_path) as conn:
            # Get active players with high confidence matches
            result = conn.execute("""
                SELECT 
                    op.full_name,
                    np.player_id
                FROM odds_players op
                JOIN player_mapping pm ON op.player_id = pm.odds_player_id
                JOIN nba_players np ON pm.nba_player_id = np.player_id
                WHERE np.is_active = true
                AND pm.match_confidence >= 0.8
                ORDER BY op.full_name
            """).fetchall()
            
            # Create mapping
            return {name: id for name, id in result}
            
    except Exception as e:
        logger.error(f"Error getting player mapping: {str(e)}")
        raise

def main() -> None:
    """Sync player data and print mapping."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Sync player data
    sync_players()
    
    # Get and print mapping
    player_ids = get_player_mapping()
    
    print(f"\nFound {len(player_ids)} active players with high confidence matches")
    print("\nSample of player mappings:")
    for name, id in list(player_ids.items())[:10]:
        print(f"{name}: {id}")

if __name__ == "__main__":
    main()
