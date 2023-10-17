
import os
import sys
import requests
import json
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


TOURNAMENTS_URL = "https://api.sportsdata.io/golf/v2/json/Tournaments"

PLAYERS_URL = "https://api.sportsdata.io/golf/v2/json/Players"

LEADERBOARDS_URL = "https://api.sportsdata.io/golf/v2/json/Leaderboard/{}"

SEASON_URL = "https://api.sportsdata.io/golf/v2/json/PlayerSeasonStats/{}"

PROJECT_PATH = '/Users/huntermitchell/Documents/PYTHON_FILES/PGA_Winners_Prediction'

#API_KEY = "917bd281e6f04f44997a173fb9fa4edf" # My key
API_KEY = "2302daee6fed4435aa4f110128af8b40" # Ciana's key

SEED = 15
K_FOLDS = 5


def get_player_id(PLAYER, new_request=False):
    if new_request:
        response = requests.get(PLAYERS_URL,headers={"Ocp-Apim-Subscription-Key": API_KEY})
        players_df = pd.DataFrame(response.json())
        players_df.to_csv(f"{PROJECT_PATH}/players.csv", index=False)
    else:
        players_df = pd.read_csv(f'{PROJECT_PATH}/players.csv')
    try:
        return players_df[(players_df.FirstName == PLAYER.split(' ')[0]) & \
                    (players_df.LastName == PLAYER.split(' ')[1])]['PlayerID'].values[0]
    except Exception as E:
        print(f'Player ID not found: {E}')
        sys.exit()


def get_tournaments_df(new_request=False):
    if new_request:
        response = requests.get(TOURNAMENTS_URL,
                        headers={"Ocp-Apim-Subscription-Key": API_KEY})
        tournaments_df = pd.DataFrame(response.json())
        tournaments_df.to_csv(f'{PROJECT_PATH}/tournaments.csv',index=False)
    else:
        tournaments_df = pd.read_csv(f'{PROJECT_PATH}/tournaments.csv')
    return tournaments_df



def get_next_tournament_info(tournaments_df):
    tournaments_df['EndDate'] = tournaments_df['EndDate'].apply(lambda x: datetime.strptime(str(x)[:10], '%Y-%m-%d'))
    return tournaments_df[tournaments_df.EndDate > datetime.today()].sort_values(by='StartDate')[['TournamentID','Name']].iloc[0]


def get_player_data(leaderboard_df, player_id):
    try:
        player = leaderboard_df[leaderboard_df['PlayerID']==player_id]
        odds_to_win = player['OddsToWin'].values[0]
        if player['IsWithdrawn'].values[0]: return 'WD', odds_to_win
        else:
            score = player['TotalScore'].values[0]
            if math.isnan(score):
                return 'CUT', odds_to_win
            else: return float(leaderboard_df[leaderboard_df['TotalScore']==score].index[0] + 1), odds_to_win
    except Exception as E:
        return None, None




def get_next_tournament_info(tournaments_df):
    tournaments_df['EndDate'] = tournaments_df['EndDate'].apply(lambda x: datetime.strptime(str(x)[:10], '%Y-%m-%d'))
    return tournaments_df[tournaments_df.EndDate > datetime.today()].sort_values(by='StartDate')[['TournamentID','Name']].iloc[0]




def write_leaderboard_data(tournament_ids):
    for tournament in tournament_ids:
        file_path = f'{PROJECT_PATH}/leaderboards/tournament_{tournament}.csv'
        if not os.path.isfile(file_path):
            response = requests.get(LEADERBOARDS_URL.format(tournament),
                                    headers={"Ocp-Apim-Subscription-Key": API_KEY})
            pd.DataFrame(response.json()['Players']).to_csv(f'{PROJECT_PATH}/leaderboards/tournament_{tournament}.csv',
                                                        index=False)

def get_leaderboard_data(tournament_ids, player_id):
    data=[]
    for tournament in tournament_ids:
        try: data.append([tournament,
                     *get_player_data(pd.read_csv(f'{PROJECT_PATH}/leaderboards/tournament_{tournament}.csv'),
                                      player_id)])
        except Exception as E:
            pass
    return data


def get_slope(Y):
    X = list(range(1,len(Y)+1))
    return round(np.polyfit(X,Y,1)[0]*-1,2)


def prep_data(df):
    # remove tournaments they weren't in
    df = df[(~df['Position'].isna()) & (df['Position']!='WD')]

    # convert CUTs to position 75.0
    df['Position'] = df['Position'].apply(lambda x: x if x!='CUT' else 75.0)
    
    df['Position'] = df['Position'].astype('float')

    df['Last10PositionAvg'] = df['Position'].shift(-1).rolling(10).mean().shift(-9)
    df['Last5PositionAvg'] = df['Position'].shift(-1).rolling(5).mean().shift(-4)
    df['Last3PositionAvg'] = df['Position'].shift(-1).rolling(3).mean().shift(-2)
    
    df['Last10PositionSlope'] = df['Position'].shift(-1).rolling(10).apply(get_slope).shift(-9)
    df['Last5PositionSlope'] = df['Position'].shift(-1).rolling(5).apply(get_slope).shift(-4)
    df['Last3PositionSlope'] = df['Position'].shift(-1).rolling(3).apply(get_slope).shift(-2)
    
    # need to fix. Venues can be named differently. And course avg includes current tournament
    #df = df.join(tournaments_df.set_index('TournamentID')['Venue'], how='left', lsuffix='_left', rsuffix='_right')
    #df['CourseAvg'] = df.groupby('Venue')['Position'].transform('mean')
    #df = df.drop(columns='Venue').sort_index(ascending=False)
    
    # cut oldest 10 which don't have all data
    df = df[:-10]

    return df


def plot(x_label, y_label, df):
    plt.figure(figsize=(10,10))

    x = df[x_label].values
    y = df[y_label].values
    plt.scatter(x, y)

    m, b = np.polyfit(x, y, 1)

    plt.plot(x, m*x+b)

    plt.xlabel(x_label)
    plt.ylabel(y_label)


def get_score(model, x_train, y_train):
  cv_score = cross_val_score(model, x_train, y_train, scoring = "neg_mean_squared_error", cv = K_FOLDS)
  rmse = np.sqrt(-cv_score)
  print('Cross-Validation Root Mean Squared Error:', rmse)
  print('Average Root Mean Squared Error:', round(np.mean(rmse), 5))
  print('Standard deviation:', round(rmse.std(), 5))


def get_results(preds, y_test):
  score = np.sqrt(mean_squared_error(preds,y_test.values))
  print(f"Final RMSE: {round(score,5)}")


def main():
    """
    Creates a fitted XGBoost model to predict a players position in next PGA golf tournament
    """
    PLAYER =  sys.argv[1]
    PLAYER_ID = get_player_id(PLAYER, new_request=False)
    print(f'Player {PLAYER} is PLAYER_ID={PLAYER_ID}')
    tournaments_df = get_tournaments_df(new_request=False)
    TOURNAMENT_ID, TOURNAMENT_NAME = get_next_tournament_info(tournaments_df)
    print(f'Predicting {PLAYER} for\nTOURNAMENT_ID: {TOURNAMENT_ID}\nTOURNAMENT_NAME: {TOURNAMENT_NAME}')
    print('Prepping Data...')
    tournament_ids = [TOURNAMENT_ID] + list(tournaments_df[tournaments_df['IsOver']==True]['TournamentID'].unique())
    print(f'Using data from {len(tournament_ids)} tournaments')

    print('Writing tournament data to disk...')
    write_leaderboard_data(tournament_ids)

    print(f'Collecting leaderboard data for {PLAYER}...')
    data = get_leaderboard_data(tournament_ids, PLAYER_ID)

    df_raw = pd.DataFrame(data,columns=['TournamentID','Position','Odds']).set_index('TournamentID')
    print('Preparing data...')
    df = prep_data(df_raw)

    df_next_tournament = df[df.index == TOURNAMENT_ID]
    df = df[1:]

    print('Prepared data:')
    print(df.head())

    features = ['Odds', 'Last10PositionAvg', 'Last5PositionAvg', 'Last3PositionAvg',
            'Last10PositionSlope', 'Last5PositionSlope', 'Last3PositionSlope']
    labels = ['Position']

    features_df = df[features]
    labels_df = df[labels]

    print('Splitting data for training...')
    x_train, x_test, y_train, y_test = train_test_split(features_df, labels_df, test_size=1/K_FOLDS, random_state=SEED)

    print('Fitting model...')
    model = XGBRegressor(random_state=SEED, objective='reg:squarederror')

    model.fit(x_train,y_train)

    get_score(model, x_train, y_train)

    preds = np.array(model.predict(x_test))

    print(f'Model Predicted positions vs Actual Positions for {PLAYER} Previous Tournaments')
    arr = []
    for a,b in zip(preds, y_test.values):
        print(f'Predicted Position: {int(a)}, Actual Position: {int(*b,)}')
        arr.append(abs(int(a)-int(*b)))
    
    get_results(preds, y_test)

    final_pred = np.array(model.predict(df_next_tournament[features]))[0]
    print(f'Final prediction = {final_pred}')



if __name__=="__main__":
    main()
