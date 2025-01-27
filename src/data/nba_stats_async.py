"""Async NBA statistics data pipeline for active players with recent games."""

import logging
import os
import asyncio
import aiohttp
import time
from datetime import datetime
import pandas as pd
from nba_api.stats.static import players  # type: ignore
from nba_api.stats.endpoints import (
    playergamelog,          # type: ignore
    commonplayerinfo,       # type: ignore
    leaguedashplayerstats   # type: ignore
)
from typing import Dict, Any, List, Optional, Set
from tqdm import tqdm
from aiohttp import ClientSession
from asyncio import Semaphore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# API settings
CONCURRENT_REQUESTS = 3  # reduced concurrent requests
REQUEST_TIMEOUT = 60  # increased timeout
GAMES_TO_ANALYZE = 20  # number of recent games to analyze
MAX_RETRIES = 3  # number of retries for API requests

# NBA API headers to avoid timeouts
HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}

class NBAStatsCollector:
    """Async NBA stats collector."""
    
    def __init__(self):
        self.semaphore = Semaphore(CONCURRENT_REQUESTS)
        self.session: Optional[ClientSession] = None
        
    async def get_player_info(self, player_id: int, player_name: str) -> Optional[Dict[str, Any]]:
        """Get player info asynchronously."""
        async with self.semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    info = commonplayerinfo.CommonPlayerInfo(
                        player_id=player_id,
                        timeout=REQUEST_TIMEOUT,
                        headers=HEADERS
                    ).get_data_frames()[0]
                    
                    return {
                        'position': info['POSITION'].iloc[0],
                        'team': info['TEAM_NAME'].iloc[0],
                        'height': info['HEIGHT'].iloc[0],
                        'weight': info['WEIGHT'].iloc[0]
                    }
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        logger.error(f"Failed to get info for {player_name}: {str(e)}")
                        return None
                    await asyncio.sleep(1 * (2 ** attempt))
    
    async def get_player_games(self, player_id: int, player_name: str) -> Optional[pd.DataFrame]:
        """Get player game log asynchronously."""
        async with self.semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    games = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season='2024-25',
                        timeout=REQUEST_TIMEOUT,
                        headers=HEADERS
                    ).get_data_frames()[0]
                    
                    if len(games) < 5:
                        logger.warning(f"Not enough games for {player_name}")
                        return None
                    
                    return games
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        logger.error(f"Failed to get games for {player_name}: {str(e)}")
                        return None
                    await asyncio.sleep(1 * (2 ** attempt))
    
    def calculate_game_stats(self, games: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed stats from game log."""
        recent_games = games.head(GAMES_TO_ANALYZE)
        
        # Calculate basic stats
        stats = {
            'games_played': len(recent_games),
            'pts_avg': float(recent_games['PTS'].mean()),
            'ast_avg': float(recent_games['AST'].mean()),
            'reb_avg': float(recent_games['REB'].mean()),
            'min_avg': float(recent_games['MIN'].mean()),
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
    
    async def process_player(self, player: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single player asynchronously."""
        try:
            player_id = player['id']
            player_name = player['full_name']
            
            # Get player info and games concurrently
            info_task = asyncio.create_task(self.get_player_info(player_id, player_name))
            games_task = asyncio.create_task(self.get_player_games(player_id, player_name))
            
            info, games = await asyncio.gather(info_task, games_task)
            
            if not info or games is None:
                return None
            
            # Calculate stats
            game_stats = self.calculate_game_stats(games)
            
            # Combine all stats
            return {
                'player_id': player_id,
                'player_name': player_name,
                **info,
                **game_stats,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error processing {player['full_name']}: {str(e)}")
            return None
    
    async def process_all_players(self) -> List[Dict[str, Any]]:
        """Process all active players concurrently."""
        # Get active players
        active_players = [p for p in players.get_active_players() if p['is_active']]
        logger.info(f"Found {len(active_players)} active players")
        
        # Process players with progress bar
        all_stats = []
        with tqdm(total=len(active_players), desc="Processing players") as pbar:
            # Process players in smaller chunks to avoid overwhelming the API
            chunk_size = 20
            for i in range(0, len(active_players), chunk_size):
                chunk = active_players[i:i + chunk_size]
                tasks = []
                for player in chunk:
                    task = asyncio.create_task(self.process_player(player))
                    task.add_done_callback(lambda _: pbar.update(1))
                    tasks.append(task)
                
                # Process chunk and wait before next chunk
                chunk_results = await asyncio.gather(*tasks)
                all_stats.extend([r for r in chunk_results if r is not None])
                
                if i + chunk_size < len(active_players):
                    await asyncio.sleep(2)  # Delay between chunks
        
        return all_stats

async def main():
    """Main async function."""
    try:
        collector = NBAStatsCollector()
        
        # Process all players
        all_stats = await collector.process_all_players()
        
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
    asyncio.run(main())
