"""Script to get NBA player IDs and create a mapping."""

import logging
from typing import Dict
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonallplayers

logger = logging.getLogger(__name__)

def get_active_players() -> Dict[str, int]:
    """Get mapping of active player names to IDs.
    
    Returns:
        Dictionary mapping player names to NBA IDs
    """
    try:
        # Get all active players
        all_players = commonallplayers.CommonAllPlayers(
            is_only_current_season=1,
            league_id="00"
        ).get_data_frames()[0]
        
        # Print column names
        print("\nAvailable columns:")
        print(all_players.columns.tolist())
        
        # Print first few rows
        print("\nFirst few rows:")
        print(all_players.head())
        
        # Create mapping
        player_ids = {}
        for _, row in all_players.iterrows():
            # Use display_first_last for full name
            player_ids[row['DISPLAY_FIRST_LAST']] = row['PERSON_ID']
        
        logger.info(f"Found {len(player_ids)} active players")
        return player_ids
        
    except Exception as e:
        logger.error(f"Error getting player IDs: {str(e)}")
        raise

def main() -> None:
    """Print active player IDs."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get and print player IDs
    player_ids = get_active_players()
    
    print("\nActive Player IDs:")
    print("player_ids = {")
    for name, id in sorted(player_ids.items()):
        print(f"    '{name}': {id},")
    print("}")

if __name__ == "__main__":
    main()
