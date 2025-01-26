import pytest
from src.data.nba_api import NBADataPipeline
from unittest.mock import Mock, patch
import pandas as pd
import duckdb
from datetime import datetime

@pytest.fixture
def mock_nba_api():
    with patch('nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats') as mock:
        mock.return_value.get_data_frames.return_value = [pd.DataFrame({
            'PLAYER_ID': [203999],
            'PLAYER_NAME': ['Test Player'],
            'TEAM_ID': [1],
            'TEAM_ABBREVIATION': ['TST'],
            'AGE': [25.5],
            'GP': [10],
            'MIN': [30.0],
            'FGM': [5.0],
            'FGA': [10.0],
            'FG_PCT': [0.5],
            'FG3M': [2.0],
            'FG3A': [5.0],
            'FG3_PCT': [0.4],
            'FTM': [3.0],
            'FTA': [3.0],
            'FT_PCT': [1.0],
            'OREB': [1.0],
            'DREB': [4.0],
            'REB': [5.0],
            'AST': [5.0],
            'STL': [2.0],
            'BLK': [1.0],
            'TOV': [2.0],
            'PTS': [15.0],
            'PLUS_MINUS': [10.0]
        })]
        yield mock

@pytest.fixture
def test_db():
    conn = duckdb.connect(':memory:')
    yield conn
    conn.close()

def test_schema_initialization(test_db):
    pipeline = NBADataPipeline()
    tables = test_db.execute("SHOW TABLES").fetchall()
    assert any('player_stats' in table for table in tables[0])
    pipeline.close()

def test_data_validation(mock_nba_api):
    pipeline = NBADataPipeline()
    try:
        df = pipeline.fetch_player_stats()
        assert 'last_updated' in df.columns
        assert df.shape == (1, 25)
        assert list(df.columns) == [col.lower() for col in df.columns]
    finally:
        pipeline.close()

def test_db_upsert_operation(mock_nba_api):
    pipeline = NBADataPipeline(db_path=":memory:")
import pytest
from src.data.nba_api import NBADataPipeline
from unittest.mock import Mock, patch
import pandas as pd
import duckdb
from datetime import datetime

@pytest.fixture
def mock_nba_api():
    with patch('nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats') as mock:
        mock.return_value.get_data_frames.return_value = [pd.DataFrame({
            'PLAYER_ID': [203999],
            'PLAYER_NAME': ['Test Player'],
            'TEAM_ID': [1],
            'TEAM_ABBREVIATION': ['TST'],
            'AGE': [25.5],
            'GP': [10],
            'MIN': [30.0],
            'FGM': [5.0],
            'FGA': [10.0],
            'FG_PCT': [0.5],
            'FG3M': [2.0],
            'FG3A': [5.0],
            'FG3_PCT': [0.4],
            'FTM': [3.0],
            'FTA': [3.0],
            'FT_PCT': [1.0],
            'OREB': [1.0],
            'DREB': [4.0],
            'REB': [5.0],
            'AST': [5.0],
            'STL': [2.0],
            'BLK': [1.0],
            'TOV': [2.0],
            'PTS': [15.0],
            'PLUS_MINUS': [10.0]
        })]
        yield mock

@pytest.fixture
def test_db():
    conn = duckdb.connect(':memory:')
    yield conn
    conn.close()

def test_schema_initialization(test_db):
    pipeline = NBADataPipeline()
    tables = test_db.execute("SHOW TABLES").fetchall()
    assert any('player_stats' in table for table in tables[0])
    pipeline.close()

def test_data_validation(mock_nba_api):
    pipeline = NBADataPipeline()
    try:
        df = pipeline.fetch_player_stats()
        assert 'last_updated' in df.columns
        assert df.shape == (1, 25)
        assert list(df.columns) == [col.lower() for col in df.columns]
    finally:
        pipeline.close()

def test_db_upsert_operation(mock_nba_api, test_db):
    pipeline = NBADataPipeline()
    try:
        df = pipeline.fetch_player_stats()
        pipeline.store_stats_in_db(df)
        
        result = test_db.execute("SELECT COUNT(*) FROM player_stats").fetchone()
        assert result[0] == 1
        
        player_data = test_db.execute("SELECT * FROM player_stats").fetchall()
        assert player_data[0][1] == 'Test Player'
        assert isinstance(player_data[0][-1], datetime)
    finally:
        pipeline.close()

def test_invalid_data_handling(mock_nba_api):
    mock_nba_api.side_effect = Exception("API Error")
    pipeline = NBADataPipeline()
    try:
        with pytest.raises(Exception):
            pipeline.fetch_player_stats()
    finally:
        pipeline.close()
