"""NBA statistics data pipeline for active players with recent games."""

import logging
import os
from datetime import datetime, timedelta
import time
import pandas as pd
from nba_api.stats.static import players  # type: ignore
from nba_api.stats.endpoints import (
    playergamelog,          # type: ignore
    commonplayerinfo,       # type: ignore
    leaguedashplayerstats   # type: ignore
)
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# API settings
DELAY = 3.0  # increased delay between requests
BATCH_SIZE = 20  # further reduced batch size
GAMES_TO_ANALYZE = 20  # number of recent games to analyze
MAX_RETRIES = 3  # number of retries for API requests

def get_active_players() -> List[Dict[str, Any]]:
    """Get list of active NBA players."""
    logger.info("Fetching active players list...")
    return [p for p in players.get_active_players() if p['is_active']]

def calculate_game_stats(games: pd.DataFrame) -> Dict[str, Any]:
    """Calculate detailed stats from game log."""
    if len(games) == 0:
        return {}
        
    # Get last 20 games
    recent_games = games.head(GAMES_TO_ANALYZE)
    
    # Calculate basic stats
    stats = {
        'games_played': len(recent_games),
        'pts_avg': float(recent_games['PTS'].mean()),
        'ast_avg': float(recent_games['AST'].mean()),
        'reb_avg': float(recent_games['REB'].mean()),
        'min_avg': float(recent_games['MIN'].mean()),
        'fga_avg': float(recent_games['FGA'].mean()),
        'fgm_avg': float(recent_games['FGM'].mean()),
        'fg3a_avg': float(recent_games['FG3A'].mean()),
        'fg3m_avg': float(recent_games['FG3M'].mean()),
        'fta_avg': float(recent_games['FTA'].mean()),
        'ftm_avg': float(recent_games['FTM'].mean()),
        'pts_std': float(recent_games['PTS'].std()),
        'ast_std': float(recent_games['AST'].std()),
        'reb_std': float(recent_games['REB'].std()),
        
        # Last 5 games
        'pts_last_5': float(recent_games.head(5)['PTS'].mean()),
        'ast_last_5': float(recent_games.head(5)['AST'].mean()),
        'reb_last_5': float(recent_games.head(5)['REB'].mean()),
        
        # Home/Away splits
        'pts_home': float(recent_games[recent_games['MATCHUP'].str.contains(' vs. ')]['PTS'].mean()),
        'pts_away': float(recent_games[recent_games['MATCHUP'].str.contains(' @ ')]['PTS'].mean()),
        'ast_home': float(recent_games[recent_games['MATCHUP'].str.contains(' vs. ')]['AST'].mean()),
        'ast_away': float(recent_games[recent_games['MATCHUP'].str.contains(' @ ')]['AST'].mean()),
        
        # Last game
        'last_game_date': pd.to_datetime(recent_games.iloc[0]['GAME_DATE']).date(),
        'last_game_pts': int(recent_games.iloc[0]['PTS']),
        'last_game_ast': int(recent_games.iloc[0]['AST']),
        'last_game_reb': int(recent_games.iloc[0]['REB'])
    }
    
    return stats

def make_api_request(request_fn: callable, player_name: str, request_type: str) -> Any:
    """Make API request with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                sleep_time = DELAY * (2 ** attempt)
                logger.debug(f"Retry {attempt + 1}/{MAX_RETRIES} for {player_name} {request_type}")
                time.sleep(sleep_time)
            return request_fn()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            logger.warning(f"Attempt {attempt + 1} failed for {player_name} {request_type}: {str(e)}")
    raise Exception(f"All retries failed for {player_name}")

def get_player_stats(player_id: int, player_name: str) -> Optional[Dict[str, Any]]:
    """Get comprehensive stats for a player."""
    try:
        # Get basic info
        time.sleep(DELAY)
        info = make_api_request(
            lambda: commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0],
            player_name,
            "basic info"
        )
        
        # Get game log
        time.sleep(DELAY)
        games = make_api_request(
            lambda: playergamelog.PlayerGameLog(
                player_id=player_id,
                season='2024-25'
            ).get_data_frames()[0],
            player_name,
            "game log"
        )
        
        # Check if player has played enough recent games
        if len(games) < 5:  # Require at least 5 games
            logger.warning(f"Not enough games for {player_name} ({len(games)} games)")
            return None
            
        # Calculate game stats
        game_stats = calculate_game_stats(games)
        if not game_stats:
            return None
            
        # Combine all stats
        stats = {
            'player_id': player_id,
            'player_name': player_name,
            'position': info['POSITION'].iloc[0],
            'team': info['TEAM_NAME'].iloc[0],
            'height': info['HEIGHT'].iloc[0],
            'weight': info['WEIGHT'].iloc[0],
            'last_updated': datetime.now(),
            **game_stats
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats for {player_name}: {str(e)}")
        return None

def main():
    """Main function to fetch and process player stats."""
    try:
        # Get active players
        active_players = get_active_players()
        total_players = len(active_players)
        logger.info(f"Found {total_players} active players")
        
        # Process players with progress bar
        all_stats = []
        with tqdm(total=total_players, desc="Processing players") as pbar:
            for i in range(0, total_players, BATCH_SIZE):
                batch = active_players[i:i + BATCH_SIZE]
                
                for player in batch:
                    try:
                        stats = get_player_stats(player['id'], player['full_name'])
                        if stats:
                            all_stats.append(stats)
                            logger.debug(f"Successfully processed {player['full_name']}")
                    except Exception as e:
                        logger.error(f"Failed to process {player['full_name']}: {str(e)}")
                    finally:
                        pbar.update(1)
                
                if i + BATCH_SIZE < total_players:
                    logger.debug("Batch complete, waiting before next batch...")
                    time.sleep(DELAY * 3)  # Increased delay between batches
        
        # Convert to DataFrame and sort by scoring average
        df = pd.DataFrame(all_stats)
        df = df.sort_values('pts_avg', ascending=False)
        
        # Save to CSV
        output_dir = 'data/processed'
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'{output_dir}/player_stats_{datetime.now().strftime("%Y%m%d")}.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Saved stats for {len(df)} active players to {output_file}")
        
        # Print summary
        logger.info("\nTop 10 Scorers:")
        print(df[['player_name', 'team', 'pts_avg', 'games_played']].head(10))
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
