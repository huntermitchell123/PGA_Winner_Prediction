# PGA_Winner_Prediction

API: https://developer.sportradar.com/docs/read/golf/Golf_v3#golf-api-overview

High-level approach:
1. Define variables including which players we're predicting, which tournament is next, and how far back to use training data for.
2. Define helper functions
3. Ensure we have most recent season stats data and player info
4. Get all player historical tournaments data, historical season stats data, and relevant course info data. Write all to disk if necessary.
5. Combine data of all relevant stats from player and course with the total difference to par this person ended up shooting.
6. Clean up data, categorize some columns, scale others
7. Train model
8. Make prediction with current stats from players and info for next tournament



Important Notes:
- I chose to predict the players final scores relative to par, instead of their finishing position. This is because their finishing position is dependent upon
    how others perform in the tournament. I could have also predicted total strokes, but that is dependent upon what par is
- Right now I'm predicting how a player will perform in a tournament. We could instead predict how a player will perform in a given round. This would allow us to use
    tournament weather and conditions on a day-to-day basis better.
- The API only has data aggregated by season. This poses two problems with training/prediction:
    1. During training, the model will be using data after the tournament it's training on. For example, when training on the Masters tournament (which occurs in May),
        it should ideally only be using player data up to May, but will actually be using data from the entire season including after May.
    2. When predicting, we will be using the player data from the current season. For tournaments that are early in the season, this will be minimal data.
        It may be better to use previous season player data for first few tournaments until we have enough data.
- The https://datagolf.com/raw-data-archive API has much more granular strokes gained data. This is paid subscription and requires scratch plus ($30/month)
- I tried incorporating more player-specific features such as past performance on harder courses, windier conditions, and rain. However, I didn't find a good way to 
    meaningfully calculate this, since their scores in these conditions had so many other factors attributing to them, which ended up being very noisy. The code to
    calculate these features is still in the notebook.


Data:
- Tournament Schedule -- {URL}/tournaments/schedule -- https://developer.sportradar.com/docs/read/golf/Golf_v3#tournament-schedule:
        High level tournament data for specific season. Includes course details. Sample data in "temp/tournaments_schedule_2023.csv"

- Tournament Leaderboard -- {URL}/tournaments/{TOURNAMENT_ID}/leaderboard -- https://developer.sportradar.com/docs/read/golf/Golf_v3#tournament-leaderboard:
        Full leaderboard data of specific tournament. Sample data in "temp/zozo_leaderboard_2023.csv"
    
- Tournament Summary -- {URL}/tournaments/{TOURNAMENT_ID}/summary -- https://developer.sportradar.com/docs/read/golf/Golf_v3#tournament-summary:
        High level tournament info. "venue" has par and course length. "rounds" has weather and conditions of each round.
    
- Player Statistics -- {URL}/players/statistics -- https://developer.sportradar.com/docs/read/golf/Golf_v3#player-statistics:
        Season statistics for all golfers. Sample data in "temp/player_stats_2023.csv". IMPORTANT: this only is accurate for v2 of API, NOT v3
    
- Player Profile -- {URL}/players/{player_id}/profile -- https://developer.sportradar.com/docs/read/golf/Golf_v3#player-profile:
        Contains "previous_tournaments" and "statistics" of specific player for all previous seasons. Sample data in "temp/jt_player_previous_tournaments.csv" and 
        "temp/jt_player_statistics.csv"

- Course Difficulty -- Downloaded manually - https://datagolf.com/course-table?sort_cat=sg_difficulty&sort=app_sg&diff=hardest:
        Had to manually change a lot of course names to match what's available in the sportradar API