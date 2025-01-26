1. Get all the NBA Teams:

<https://api.the-odds-api.com/v4/sports/basketball_nba/participants?apiKey=bcab2a03da8de48a2a68698a40b78b4c>

Sample Data:

```json
[
{
"full_name": "Atlanta Hawks",
"id": "par_01hqmkq6fceknv7cwebesgrx03"
},
{
"full_name": "Boston Celtics",
"id": "par_01hqmkq6fdf1pvq7jgdd7hdmpf"
}
]
```

2. Get all the NBA Players for each team:

<https://api.the-odds-api.com/v4/sports/basketball_nba/participants/par_01hqmkq6fceknv7cwebesgrx03/players?apiKey=bcab2a03da8de48a2a68698a40b78b4c>

Sample Data:

```json
[
{
"id": "pla_01hrbs4bnpft49rb0gxjbxmvw6",
"full_name": "A.J. Griffin Jr"
},
{
"id": "pla_01hrbs4bhyf8ftwv05ry1mvcax",
"full_name": "Bogdan Bogdanovic"
},
{
"id": "pla_01hrbs4bjyfkbrq084ftzhvc99",
"full_name": "Clint Capela"
}
]
```

# GET NBA PLAY PROPS

## FIRST GET THE EVENTS FOR THE DAY

<https://api.the-odds-api.com/v4/sports/basketball_nba/events?apiKey=bcab2a03da8de48a2a68698a40b78b4c>

```json
{
"id": "240a84b2dddf3a224173e70c23b1d624",
"sport_key": "basketball_nba",
"sport_title": "NBA",
"commence_time": "2025-01-25T20:10:00Z",
"home_team": "Minnesota Timberwolves",
"away_team": "Denver Nuggets",
"bookmakers": [
{
"key": "underdog",
"title": "Underdog",
"markets": [
{
"key": "player_assists",
"last_update": "2025-01-25T02:39:22Z",
"outcomes": [
{
"name": "Over",
"description": "Julius Randle",
"price": -137,
"point": 5.5
},
{
"name": "Under",
"description": "Julius Randle",
"price": -137,
"point": 5.5
},
{
"name": "Over",
"description": "Jamal Murray",
"price": -137,
"point": 5.5
}
]]]
}
```

## TAKE THE ID FROM THE EVENTS TO GET EACH PLAYER PROPS

<https://api.the-odds-api.com/v4/sports/basketball_nba/events/240a84b2dddf3a224173e70c23b1d624/odds?apiKey=bcab2a03da8de48a2a68698a40b78b4c&regions=eu,us,us2,us_dfs,us_ex&markets=player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals,player_first_basket,player_double_double,player_triple_double,player_method_of_first_basket&oddsFormat=american>

## Markets

player_first_basket First Basket Scorer (Yes/No)
player_double_double Double Double (Yes/No)
player_triple_double Triple Double (Yes/No)
player_method_of_first_basket Method of First Basket (Various)
player_threes Threes (Over/Under)
player_blocks Blocks (Over/Under)
player_steals Steals (Over/Under)
player_assists Assists (Over/Under)
player_rebounds Rebounds (Over/Under)
player_points Points (Over/Under)

## Regions

eu
us
us2
us_dfs
us_ex

## Bookmakers

pinnacle (SHARPEST)
betonlineag (SHARP #2)
betmgm
betrivers
betonlineag
betmgm
betrivers
ballybet
espnbet
fliff
underdog
prizepicks
novig
prophetx
