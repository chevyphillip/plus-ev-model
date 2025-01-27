"""Flask web application for prop betting model."""

import os
import logging
from typing import Dict, Any, List, Optional, cast, Union, TypedDict
from datetime import datetime
from flask import Flask, render_template, jsonify, request, Response
from flask.typing import ResponseReturnValue
from src.data.odds_api_client import (
    OddsAPIClient,
    GameDict,
    PropsDict,
    BookmakerDict,
    MarketDict,
    OutcomeDict
)
from src.models.find_prop_edges import PropEdgeFinder

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Initialize clients
api_key = os.getenv('ODDS_API_KEY')
if not api_key:
    raise ValueError("ODDS_API_KEY environment variable not set")

odds_client = OddsAPIClient(api_key=api_key)

# Initialize edge finder with model training
logger.info("Initializing edge finder and training models...")
edge_finder = PropEdgeFinder(
    api_key=api_key,
    force_retrain=True  # Force retrain models
)

def format_odds(odds: int) -> str:
    """Format American odds for display."""
    if odds > 0:
        return f"+{odds}"
    return str(odds)

def get_props_data(
    game_id: str,
    prop_type: str,
    min_line: float = 0.0
) -> List[Dict[str, Any]]:
    """Get formatted props data.
    
    Args:
        game_id: Game ID to get props for
        prop_type: Type of prop to get
        min_line: Minimum line to include
        
    Returns:
        List of prop dictionaries
    """
    props = odds_client.get_player_props(game_id, [f'player_{prop_type}'])
    if not props or 'bookmakers' not in props:
        return []
        
    # Collect all props
    rows: List[Dict[str, Any]] = []
    for book in props['bookmakers']:
        for market in book['markets']:
            if market['key'] == f'player_{prop_type}':
                for outcome in market['outcomes']:
                    if float(outcome['point']) >= min_line:
                        rows.append({
                            'player': outcome['description'],
                            'line': float(outcome['point']),
                            'odds': format_odds(int(outcome['price'])),
                            'side': outcome['name'],
                            'book': book['title']
                        })
    
    # Sort by player name and line
    rows.sort(key=lambda x: (x['player'], x['line']))
    return rows

@app.route('/')
def index() -> ResponseReturnValue:
    """Render main page."""
    try:
        games = odds_client.get_nba_games()
        return cast(ResponseReturnValue, render_template(
            'index.html',
            games=games,
            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
    except Exception as e:
        logger.error(f"Error loading games: {str(e)}")
        return cast(ResponseReturnValue, render_template('error.html', error=str(e)))

@app.route('/props/<game_id>')
def props(game_id: str) -> ResponseReturnValue:
    """Render props page for a game."""
    try:
        games = odds_client.get_nba_games()
        game = next((g for g in games if g['id'] == game_id), None)
        if not game:
            return cast(ResponseReturnValue, render_template('error.html', error="Game not found"))
            
        return cast(ResponseReturnValue, render_template(
            'props.html',
            game=game,
            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
    except Exception as e:
        logger.error(f"Error loading props: {str(e)}")
        return cast(ResponseReturnValue, render_template('error.html', error=str(e)))

@app.route('/edges')
def edges() -> ResponseReturnValue:
    """Render edges page."""
    try:
        return cast(ResponseReturnValue, render_template(
            'edges.html',
            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
    except Exception as e:
        logger.error(f"Error loading edges page: {str(e)}")
        return cast(ResponseReturnValue, render_template('error.html', error=str(e)))

@app.route('/api/props/<game_id>')
def api_props(game_id: str) -> Response:
    """API endpoint for props data."""
    try:
        prop_type = request.args.get('type', 'points')
        min_line = float(request.args.get('min_line', '0.0'))
        
        props = get_props_data(game_id, prop_type, min_line)
        return cast(Response, jsonify({
            'success': True,
            'props': props
        }))
    except Exception as e:
        logger.error(f"Error getting props: {str(e)}")
        return cast(Response, jsonify({
            'success': False,
            'error': str(e)
        }))

@app.route('/api/edges')
def api_edges() -> Response:
    """API endpoint for prop edges."""
    try:
        min_edge = float(request.args.get('min_edge', '5.0'))
        edges_df = edge_finder.find_edges()
        
        if edges_df.empty:
            return cast(Response, jsonify({
                'success': True,
                'edges': []
            }))
        
        # Filter by minimum edge
        edges_df = edges_df[edges_df['edge'].abs() >= min_edge]
        
        # Convert to list of dictionaries
        edges = edges_df.to_dict('records')
        
        # Format edge data
        formatted_edges = []
        for edge in edges:
            formatted_edges.append({
                'player_name': edge['player'],
                'team': edge.get('game', '').split('@')[0].strip(),
                'prop_type': edge['prop_type'],
                'line': edge['line'],
                'side': edge['bet_type'],
                'edge': abs(edge['edge']),
                'model_prob': edge['model_prob'],
                'market_prob': edge['implied_prob'],
                'odds': edge['odds'],
                'book': edge['book'],
                'kelly': edge.get('kelly', 0.0),
                'ev': edge.get('ev', 0.0)
            })
        
        return cast(Response, jsonify({
            'success': True,
            'edges': formatted_edges
        }))
    except Exception as e:
        logger.error(f"Error finding edges: {str(e)}")
        return cast(Response, jsonify({
            'success': False,
            'error': str(e)
        }))

@app.route('/api/best_odds/<game_id>/<player_name>')
def api_best_odds(game_id: str, player_name: str) -> Response:
    """API endpoint for best odds."""
    try:
        prop_type = request.args.get('type', 'points')
        odds = odds_client.get_best_odds(game_id, player_name, prop_type)
        return cast(Response, jsonify({
            'success': True,
            'odds': odds
        }))
    except Exception as e:
        logger.error(f"Error getting best odds: {str(e)}")
        return cast(Response, jsonify({
            'success': False,
            'error': str(e)
        }))

def main() -> None:
    """Run the Flask application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    app.run(debug=True, port=5000)

if __name__ == '__main__':
    main()
