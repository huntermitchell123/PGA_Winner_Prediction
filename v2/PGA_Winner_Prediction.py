"""
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
"""

import requests
from datetime import datetime
import time
import json
import os
import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


PROJECT_PATH = '/Users/huntermitchell/Documents/Documents/PYTHON_FILES/PGA_Winner_Prediction/v2'

PLAYER_NAMES = ['xander_schauffele','collin_morikawa','keegan_bradley','sungjae_im','rickie_fowler','hideki_matsuyama','sahith_theegala',
                'cameron_davis','eric_cole','adam_schenk','adam_scott','beau_hossler','vincent_norrman','aaron_rai','thomas_detry',
                'adam_svensson']

# Need to research and set these manually
NEXT_TOURNAMENT_CONDITION_INFO = {'avg_temp': 71, 'avg_wind_speed': 9, 'cloudy': 1, 'sunny': 1, 'rainy': 0}

YEARS_BACK = 5

NEW_REQUEST_CURRENT_SEASON_STATS = False
NEW_REQUEST_TOURNAMENT_SCHEDULE = False
NEW_REQUEST_PREVIOUS_TOURNAMENTS = False

SEED = 2015
SIMULATIONS = 5

FEATURES = ['course_difficulty', 'drive_avg', 'drive_acc', 'gir_pct', 'putt_avg','strokes_gained', 'scrambling_pct', 'scoring_avg',
    'strokes_gained_tee_green', 'strokes_gained_total', 'yardage/par', 'avg_temp', 'avg_wind_speed', 'cloudy', 'sunny', 'rainy']
LABELS = ['score']
TEST_SIZE = 0.1

FEATURES_TO_NOT_SCALE = ['cloudy','sunny','rainy']
FEATURES_TO_SCALE = [col for col in FEATURES if col not in FEATURES_TO_NOT_SCALE]

STATISTICS_FEATURES = ['drive_avg','drive_acc','gir_pct','putt_avg','strokes_gained','scrambling_pct','scoring_avg','strokes_gained_tee_green','strokes_gained_total']

API_KEY = os.environ['pga_api_key']

CURRENT_YEAR = datetime.today().year

ALL_PLAYER_STATS_URL = "https://api.sportradar.us/golf/trial/pga/v2/en/{year}/players/statistics.json?api_key={api_key}"
PLAYER_STATS_URL = "https://api.sportradar.us/golf/trial/v3/en/players/{player_id}/profile.json?api_key={api_key}"
TOURNAMENTS_SCHEDULE_URL = "https://api.sportradar.us/golf/trial/pga/v3/en/{year}/tournaments/schedule.json?api_key={api_key}"
TOURNAMENT_SUMMARY_URL = "https://api.sportradar.us/golf/trial/v3/en/{year}/tournaments/{tournament_id}/summary.json?api_key={api_key}"
TOURNAMENT_LEADERBOARD_URL = "https://api.sportradar.us/golf/trial/pga/v3/en/{year}/tournaments/{tournament_id}/leaderboard.json?api_key={api_key}"



def get_player_info(player_name, current_season_stats_df):
    player_info = {}
    first_name = player_name.split('_')[0].title()
    last_name = player_name.split('_')[1].title()
    id = current_season_stats_df[(current_season_stats_df['first_name'] == first_name) & (current_season_stats_df['last_name'] == last_name)]['id'].values[0]
    player_info['first_name'] = first_name
    player_info['last_name'] = last_name
    player_info['id'] = id
    return player_info

def get_tournaments_schedule(url, year, new_request=False):
    print(f'Getting {year} tournaments schedule...')
    file_path = f'{PROJECT_PATH}/tournaments_schedule/{year}_tournaments.csv'
    if new_request:
        response = requests.get(url)
        df = pd.DataFrame(response.json()['tournaments'])
        df.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)
    return df

def get_next_tournament(tournaments_schedule_df):
    return tournaments_schedule_df[tournaments_schedule_df['status']=='scheduled'].iloc[0]

def get_player_previous_tournaments_info(url, player_name, new_request=False):
    file_path = f'{PROJECT_PATH}/players_previous_tournaments/{player_name}.csv'
    if new_request:
        response = requests.get(url)
        df = pd.DataFrame(response.json()['previous_tournaments'])
        df.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)
    return df

def get_all_player_previous_tournaments_df(players):
    print('Getting previous tournaments info for all players...')
    player_previous_tournaments_dfs = []
    for player in players.keys():
        if NEW_REQUEST_PREVIOUS_TOURNAMENTS: time.sleep(1)
        temp_df = get_player_previous_tournaments_info(PLAYER_STATS_URL.format(player_id=players[player]['id'],api_key=API_KEY),
                                                                                player,
                                                                                new_request=NEW_REQUEST_PREVIOUS_TOURNAMENTS)
        temp_df['first_name'] = players[player]['first_name']
        temp_df['last_name'] = players[player]['last_name']
        player_previous_tournaments_dfs.append(temp_df)
    return pd.concat(player_previous_tournaments_dfs)

def get_current_season_stats_df(year, new_request=False):
    print(f'Getting this year season stats for player info lookup...')
    file_path = f'{PROJECT_PATH}/all_player_season_stats/{year}_season_stats.csv'
    if new_request:
        response = requests.get(ALL_PLAYER_STATS_URL.format(year=year, api_key=API_KEY))
        df = pd.DataFrame(response.json()['players'])
        df.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)
    return df

def backfill_all_player_season_stats(years_back = YEARS_BACK):
    print("Backfilling all player season stats...")
    for year in range(CURRENT_YEAR - years_back, CURRENT_YEAR + 1):
        file_path = f'{PROJECT_PATH}/all_player_season_stats/{year}_season_stats.csv'
        if not os.path.isfile(file_path):
            time.sleep(3)
            response = requests.get(ALL_PLAYER_STATS_URL.format(year=year, api_key=API_KEY))
            df = pd.DataFrame(response.json()['players'])
            df.to_csv(file_path, index=False)

def backfill_tournament_summaries(tournament_ids):
    print("Backfilling tournament summaries of all previous tournaments for players we're predicting...")
    for tournament_id in tournament_ids:
        file_path = f'{PROJECT_PATH}/tournament_summaries/{tournament_id}.json'
        if not os.path.isfile(file_path):
            time.sleep(1)
            url = TOURNAMENT_SUMMARY_URL.format(year=CURRENT_YEAR,tournament_id=tournament_id,api_key=API_KEY)
            response = requests.get(url)
            with open(file_path, 'w') as fp:
                json.dump(response.json(), fp)

def generate_previous_tournaments_training_df(player_previous_tournaments_df):
    """
    Stroke play only, no playoffs, only PGA Tour, no TOUR Championship. Then filter to only last YEARS_BACK years
    """
    training_df = player_previous_tournaments_df[
        (player_previous_tournaments_df['event_type']=='stroke') & \
        (~player_previous_tournaments_df['name'].str.endswith('Playoff')) & \
        (player_previous_tournaments_df['name'] != "TOUR Championship")
        ][['id','name','first_name','last_name','seasons','leaderboard','start_date']]\
        .rename(columns={'id':'tournament_id','name':'tournament_name'})

    training_df['full_name'] = training_df['first_name'] + '_' + training_df['last_name']
    training_df['seasons'] = training_df['seasons'].apply(ast.literal_eval)
    training_df['leaderboard'] = training_df['leaderboard'].apply(ast.literal_eval)

    training_df = training_df.explode('seasons')

    training_df = training_df[training_df['seasons'].apply(lambda x: x['tour']['name']) == 'PGA Tour']

    training_df['season_id'] = training_df['seasons'].apply(lambda x: x['id'])
    training_df['score'] = training_df['leaderboard'].apply(lambda x: x['score'])
    training_df['start_date'] = pd.to_datetime(training_df['start_date'], format='%Y-%m-%d')

    training_df = training_df[training_df['start_date'] >= datetime(year=CURRENT_YEAR - YEARS_BACK,month=5,day=1)]
    training_df = training_df.drop(columns=['seasons','leaderboard','first_name','last_name']).reset_index(drop=True)

    return training_df

def get_course_difficulty_df():
    course_difficulty_df = pd.read_csv(f'{PROJECT_PATH}/course_difficulty_stats/dg_course_table.csv')[['course','adj_score_to_par']]
    return course_difficulty_df.rename(columns={'adj_score_to_par':'course_difficulty'})

def generate_season_stats_df(first_names, last_names):
    """
    Combine all season stats from "all_player_season_stats" folder
    """
    season_stats_df_list = []

    for year in range(CURRENT_YEAR-YEARS_BACK, CURRENT_YEAR+1):
        temp_df = pd.read_csv(f'{PROJECT_PATH}/all_player_season_stats/{year}_season_stats.csv')
        temp_df['year'] = year
        season_stats_df_list.append(temp_df)

    season_stats_df = pd.concat(season_stats_df_list)

    season_stats_df = season_stats_df[(season_stats_df['first_name'].isin(first_names)) & (season_stats_df['last_name'].isin(last_names))]
    season_stats_df['statistics'] = season_stats_df['statistics'].apply(ast.literal_eval)

    return season_stats_df

def generate_season_stats_training_df(season_stats_df, statistics_features):
    """
    Extract wanted features from "statistics" object column
    """
    for feature in statistics_features:
        season_stats_df[feature] = season_stats_df['statistics'].apply(lambda x: x[feature])
    
    season_stats_df['full_name'] = season_stats_df['first_name'] + '_' + season_stats_df['last_name']
    return season_stats_df.drop(columns=['id','country','statistics','first_name','last_name']).reset_index(drop=True)

def get_tournament_summary_training_df(tournament_ids):
    """
    Combine relevant course stats from "tournament_summaries" folder
    """
    tournament_summaries_data = []

    for tournament_id in tournament_ids:
        with open(f'{PROJECT_PATH}/tournament_summaries/{tournament_id}.json') as json_file:
            json_data = json.load(json_file)
            venue_data = json_data['venue']
            rounds_data = json_data['rounds']
            tournament_data = [tournament_id,
                            venue_data['name'],
                            venue_data['courses'][0]['yardage'],
                            venue_data['courses'][0]['par'],
                            [round_data['weather']['temp'] for round_data in rounds_data if 'weather' in round_data],
                            [round_data['weather']['condition'] for round_data in rounds_data if 'weather' in round_data],
                            [round_data['weather']['wind']['speed'] for round_data in rounds_data if 'weather' in round_data]
                            ]
            tournament_summaries_data.append(tournament_data)

    return pd.DataFrame(data=tournament_summaries_data, columns=['tournament_id','course','yardage','par','temps','conditions','wind_speeds'])

def combine_training_data(previous_tournaments_training_df, season_stats_training_df, tournament_summary_training_df, course_difficulty_df):
    training_df = previous_tournaments_training_df.merge(tournament_summary_training_df, how='left', on='tournament_id')
    training_df['year'] = training_df['start_date'].apply(lambda x: x.year if x.month >=3 else x.year-1) # if tourney is before March, we use last season stats
    training_df = training_df.merge(season_stats_training_df, how='left', on=['full_name','year'])
    return training_df.merge(course_difficulty_df, how='left', on='course')

def condition_mapping(condition):
    if 'rain' in condition or 'shower' in condition: return 'rainy'
    elif 'cloudy' in condition or condition == 'overcast': return 'cloudy'
    elif condition in ['clear','sunny']: return 'sunny'

def condition_one_hot_encode(conditions, condition_category):
    if len([condition for condition in conditions if condition == condition_category]) >= 2:
        return 1
    else:
        return 0

def engineer_features(training_df):
    # we get rid of rows with less than 3 days value of course conditions
    training_df = training_df[(training_df['temps'].str.len() >= 3) & (training_df['conditions'].str.len() >= 3) & (training_df['wind_speeds'].str.len() >= 3)]
    # we don't have season stats data for some players from 5+ years ago, so drop rows with nulls in those columns
    training_df = training_df.dropna(subset=STATISTICS_FEATURES)

    training_df['yardage/par'] = training_df['yardage']/training_df['par']
    training_df['avg_temp'] = training_df['temps'].apply(lambda x: round(sum(x) / len(x)))
    training_df['avg_wind_speed'] = training_df['wind_speeds'].apply(lambda x: round(sum(x) / len(x)))

    training_df['conditions_mapped'] = training_df['conditions'].apply(lambda conditions: [condition_mapping(condition.lower()) for condition in conditions])

    # fill course_difficulty null values with column mean
    training_df['course_difficulty'] = training_df['course_difficulty'].fillna(0.0)

    # we do 3 one-hot categories: Cloudy, Sunny, and Rain.
    condition_categories = ['cloudy', 'sunny','rainy']

    for condition_category in condition_categories:
        training_df[condition_category] = training_df['conditions_mapped'].apply(lambda conditions: condition_one_hot_encode(conditions, condition_category))

    return training_df.drop(columns=['yardage','par','temps','conditions','wind_speeds','conditions_mapped'])

def plot(x_label, y_label, df):
    plt.figure(figsize=(5,5))

    x = df[x_label].values
    y = df[y_label].values
    plt.scatter(x, y)

    m, b = np.polyfit(x, y, 1)

    plt.plot(x, m*x+b)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

def get_score(model, x_train_scaled, y_train):
  cv_score = cross_val_score(model, x_train_scaled, y_train, scoring = "neg_mean_squared_error", cv = 8)
  rmse = np.sqrt(-cv_score)
  print('Cross-Validation Root Mean Squared Error:', rmse)
  print('Average Root Mean Squared Error:', round(np.mean(rmse), 5))
  print('Standard deviation:', round(rmse.std(), 5))

def get_results(preds, y_test):
  score = np.sqrt(mean_squared_error(preds,y_test.values))
  print(f"Final RMSE: {round(score,5)}")

def get_next_tournament_course_difficulty(course_difficulty_df, course):
    try:
        return course_difficulty_df[course_difficulty_df['course']==course]['course_difficulty'].values[0]
    except IndexError:
        print('Course difficulty not available. Setting to 0.0')
        return 0.0

def generate_pred_df(season_stats_df, next_tournament_course_info, next_tournament_course_difficulty, next_tournament_condition_info):
    df = season_stats_df[season_stats_df['year']==CURRENT_YEAR]
    df['course_difficulty'] = next_tournament_course_difficulty
    df['yardage/par'] = next_tournament_course_info['yardage']/next_tournament_course_info['yardage']
    for i in next_tournament_condition_info.items():
        df[i[0]] = i[1]

    return df.reset_index(drop=True)


def main():
    """
    Predict finishing scores of specified players for next PGA Tour tournament
    """
    # Setup
    current_season_stats_df = get_current_season_stats_df(CURRENT_YEAR, new_request=NEW_REQUEST_CURRENT_SEASON_STATS)
    players = {player: get_player_info(player, current_season_stats_df) for player in PLAYER_NAMES}
    player_first_names = [info['first_name'] for _,info in players.items()]
    player_last_names = [info['last_name'] for _,info in players.items()]
    print('Players Info:',players)
    tournaments_schedule_df = get_tournaments_schedule(url=TOURNAMENTS_SCHEDULE_URL.format(year=CURRENT_YEAR,api_key=API_KEY),
                                                   year=CURRENT_YEAR, new_request=NEW_REQUEST_TOURNAMENT_SCHEDULE)
    next_tournament = get_next_tournament(tournaments_schedule_df)
    print('Next tournament info:',next_tournament)

    # Generate Training Data
    player_previous_tournaments_df = get_all_player_previous_tournaments_df(players)
    backfill_all_player_season_stats(YEARS_BACK)
    previous_tournaments_training_df = generate_previous_tournaments_training_df(player_previous_tournaments_df)
    backfill_tournament_summaries(previous_tournaments_training_df['tournament_id'].unique())
    season_stats_df = generate_season_stats_df(player_first_names, player_last_names)
    print("Available features from season stats data: ",season_stats_df['statistics'].values[0])
    season_stats_training_df = generate_season_stats_training_df(season_stats_df, STATISTICS_FEATURES)
    tournament_summary_training_df = get_tournament_summary_training_df(previous_tournaments_training_df['tournament_id'].unique())
    course_difficulty_df = get_course_difficulty_df()
    course_difficulty_courses = sorted(course_difficulty_df.course.unique())
    training_data_courses = sorted(tournament_summary_training_df.course.unique())
    print('Courses in training data but not in course difficulty data:\n',sorted(set(training_data_courses)-set(course_difficulty_courses)))
    training_df = combine_training_data(previous_tournaments_training_df, season_stats_training_df, tournament_summary_training_df, course_difficulty_df)
    print('Joined data for training:',training_df.head())
    training_df = training_df[['yardage','par','course_difficulty','temps','conditions','wind_speeds','drive_avg','drive_acc','gir_pct','putt_avg',\
                           'strokes_gained','scrambling_pct','scoring_avg','strokes_gained_tee_green','strokes_gained_total','score']]

    # Feature Engineering
    training_df = engineer_features(training_df)
    assert not (training_df['cloudy']+training_df['sunny']+training_df['rainy']).any() == 0, "training data contains one or more rows with no weather condition categories"
    assert not training_df.isnull().values.any(), "training data contains one or more rows with null values"

    # Analysis
    print(training_df.corr()['score'].sort_values(ascending=False))
    print(training_df.describe())

    # Split Data
    features_df = training_df[FEATURES]
    labels_df = training_df[LABELS]
    x_train, x_test, y_train, y_test = train_test_split(features_df, labels_df, test_size=TEST_SIZE)

    # Preprocessing
    scaler = StandardScaler()
    x_train_scaled = pd.concat(
        [pd.DataFrame(scaler.fit_transform(x_train[FEATURES_TO_SCALE]), index=x_train.index, columns=FEATURES_TO_SCALE),
        pd.DataFrame(x_train[FEATURES_TO_NOT_SCALE], index=x_train.index, columns=FEATURES_TO_NOT_SCALE)],
        axis=1)
    x_test_scaled = pd.concat(
        [pd.DataFrame(scaler.transform(x_test[FEATURES_TO_SCALE]), index=x_test.index, columns=FEATURES_TO_SCALE),
        pd.DataFrame(x_test[FEATURES_TO_NOT_SCALE], index=x_test.index, columns=FEATURES_TO_NOT_SCALE)],
        axis=1)

    # Prep final pred df
    next_tournament_course_info = ast.literal_eval(next_tournament['venue'])['courses'][0]
    next_tournament_course_difficulty = get_next_tournament_course_difficulty(course_difficulty_df, next_tournament_course_info['name'])
    pred_df = generate_pred_df(season_stats_training_df, next_tournament_course_info, next_tournament_course_difficulty, NEXT_TOURNAMENT_CONDITION_INFO)
    print('Prediction DataFrame:',pred_df.head())
    final_pred_df = pd.concat(
        [pd.DataFrame(scaler.transform(pred_df[FEATURES_TO_SCALE]), index=pred_df.index, columns=FEATURES_TO_SCALE),
        pd.DataFrame(pred_df[FEATURES_TO_NOT_SCALE], index=pred_df.index, columns=FEATURES_TO_NOT_SCALE)],
        axis=1)

    models = {}
    final_preds_dfs = []
    for i in range(SIMULATIONS):
        # Training
        models[i] = XGBRegressor(random_state=SEED+i, objective='reg:squarederror')
        models[i].fit(x_train_scaled,y_train)
        get_score(models[i], x_train_scaled, y_train)

        # Testing
        preds = np.array(models[i].predict(x_test_scaled))
        print(f'Model Predicted Scores vs Actual Scores for Players Previous Tournaments:')
        for a,b in zip(preds[:30], y_test.values[:30]):
            print(f'Predicted Score: {int(a)}, Actual Score: {int(*b,)}')
        get_results(preds, y_test)
        print('Feature Importances:')
        sorted_idx = np.argsort(models[i].feature_importances_)[::-1]
        for index in sorted_idx:
            print([x_train_scaled.columns[index], models[i].feature_importances_[index]]) 

        # Predict
        final_preds = np.array(models[i].predict(final_pred_df))
        final_preds_dfs.append(pd.DataFrame({'full_name': pred_df['full_name'].values, 'projected_score': final_preds}))

    final_preds_df = pd.concat(final_preds_dfs)
    print(f'Final predictions for next tournament:\n')
    print(final_preds_df.groupby('full_name').mean().sort_values(by='projected_score'))


if __name__=="__main__":
    main()