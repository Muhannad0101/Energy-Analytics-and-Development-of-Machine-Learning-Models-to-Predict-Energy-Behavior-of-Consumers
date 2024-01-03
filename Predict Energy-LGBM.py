import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, glob
import math
import cartopy.crs as ccrs
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import pickle
import polars as pl

path = "/kaggle/input/predict-energy-behavior-of-prosumers"

def preprocess_and_join_datasets(df_data, df_gas, df_client,
                                 df_electricity, df_forecast, 
                                 df_historical, df_location, df_target):
    
    df_data['date'] = pd.to_datetime(df_data['datetime']).dt.date
    df_data['datetime'] = pd.to_datetime(df_data['datetime'])
    
    df_gas = df_gas.rename(columns={'forecast_date': 'date'})
    df_gas['date'] = (pd.to_datetime(df_gas['date']) + pd.Timedelta(days=1)).dt.date

    df_client['date'] = (pd.to_datetime(df_client['date']) + pd.Timedelta(days=2)).dt.date
    
    df_electricity = df_electricity.rename(columns={'forecast_date': 'datetime'})
    df_electricity['datetime'] = pd.to_datetime(df_electricity['datetime'])

    df_forecast = df_forecast.rename(columns={'forecast_datetime': 'datetime'})
    df_forecast['datetime'] = pd.to_datetime(df_forecast['datetime']).dt.tz_localize('UTC').dt.tz_convert('Europe/Bucharest')

    df_forecast = df_forecast.merge(df_location, on=['latitude', 'longitude'], how='left')

    df_historical['datetime'] = pd.to_datetime(df_historical['datetime']) + pd.Timedelta(hours=37)
    df_historical = df_historical.merge(df_location, on=['latitude', 'longitude'], how='left')

    # Joins
    df = df_data.merge(df_gas, on='date', how='left')
    df = df.merge(df_client, on=['county', 'is_business', 'product_type', 'date'], how='left')
    df = df.merge(df_electricity, on='datetime', how='left', suffixes=('_train', '_electricity'))

    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year

    df['sin_dayofyear'] = np.sin(np.pi * df['dayofyear'] / 183)
    df['cos_dayofyear'] = np.cos(np.pi * df['dayofyear'] / 183)
    df['sin_hour'] = np.sin(np.pi * df['hour'] / 12)
    df['cos_hour'] = np.cos(np.pi * df['hour'] / 12)
    
    df['origin_date'] = pd.to_datetime(df['origin_date']).map(pd.Timestamp.toordinal)
    df['origin_date'] = pd.to_datetime(df['origin_date']).map(pd.Timestamp.toordinal)
    df['datetime'] = pd.to_datetime(df['datetime']).map(pd.Timestamp.toordinal)
    df['datetime'] = pd.to_datetime(df['datetime']).map(pd.Timestamp.toordinal)
    df['date'] = pd.to_datetime(df['date']).map(pd.Timestamp.toordinal)
    df['date'] = pd.to_datetime(df['date']).map(pd.Timestamp.toordinal)

    # df.drop(['date', 'datetime', 'hour', 'dayofyear'], axis=1, inplace=True)

    # Merge with target data if needed
    # if df_target is not None:
    #     df = df.merge(df_target, on='some_common_column', how='left')

    return df


train_cols = ['target', 'county', 'is_business', 'product_type', 'is_consumption', 'datetime', 'row_id',]
client_cols = ['product_type', 'county', 'eic_count', 'installed_capacity', 'is_business', 'date']
gas_cols  = ['forecast_date', 'lowest_price_per_mwh', 'highest_price_per_mwh']
electricity_cols = ['forecast_date', 'euros_per_mwh','origin_date']
forecast_cols = ['latitude', 'longitude', 'hours_ahead', 'temperature', 'dewpoint', 'cloudcover_high', 'cloudcover_low', 'cloudcover_mid', 'cloudcover_total', '10_metre_u_wind_component', '10_metre_v_wind_component', 'forecast_datetime', 'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall', 'total_precipitation']
historical_cols= ['datetime', 'temperature', 'dewpoint', 'rain', 'snowfall', 'surface_pressure','cloudcover_total','cloudcover_low','cloudcover_mid','cloudcover_high','windspeed_10m','winddirection_10m','shortwave_radiation','direct_solar_radiation','diffuse_radiation','latitude','longitude']
location_cols = ['longitude', 'latitude', 'county']
target_cols = ['target', 'county', 'is_business', 'product_type', 'is_consumption', 'datetime']

df_train = pl.read_csv(os.path.join(path, "train.csv"), columns=train_cols, try_parse_dates=True)
df_client = pl.read_csv(os.path.join(path, "client.csv"), columns=client_cols, try_parse_dates=True)
df_gas = pl.read_csv(os.path.join(path, "gas_prices.csv"), columns=gas_cols, try_parse_dates=True)
df_electricity = pl.read_csv(os.path.join(path, "electricity_prices.csv"), columns=electricity_cols, try_parse_dates=True)
df_forecast = pl.read_csv(os.path.join(path, "forecast_weather.csv"), columns=forecast_cols, try_parse_dates=True)
df_historical = pl.read_csv(os.path.join(path, "historical_weather.csv"), columns=historical_cols, try_parse_dates=True)
df_location = pl.read_csv(os.path.join(path, "weather_station_to_county_mapping.csv"), columns=location_cols, try_parse_dates=True)
df_target = df_train.select(target_cols)

df_train = df_train.to_pandas()
df_gas = df_gas.to_pandas()
df_client = df_client.to_pandas()
df_electricity = df_electricity.to_pandas()
df_forecast = df_forecast.to_pandas()
df_historical = df_historical.to_pandas()
df_location = df_location.to_pandas()
df_target = df_target.to_pandas()


final_df = preprocess_and_join_datasets(
    df_train, df_gas, df_client, df_electricity, df_forecast, df_historical, df_location, df_target
)


unique_count = []
for x in final_df.columns:
    unique_count.append([x, len(final_df[x].unique()), final_df[x].isnull().sum()])
pd.DataFrame(unique_count, columns=['Column', 'Unique', 'Missing']).set_index('Column')


final_df.dropna(subset=['target'], inplace=True)

final_df['lowest_price_per_mwh'].fillna(final_df['lowest_price_per_mwh'].mean(), inplace=True)
final_df['highest_price_per_mwh'].fillna(final_df['highest_price_per_mwh'].mean(), inplace=True)

eic_count_mode = final_df['eic_count'].mode()[0]
final_df['eic_count'].fillna(eic_count_mode, inplace=True)

final_df['euros_per_mwh'].fillna(final_df['euros_per_mwh'].median(), inplace=True)

final_df['origin_date'].fillna(final_df['origin_date'].mode()[0], inplace=True)

final_df['installed_capacity'].fillna(final_df['installed_capacity'].median(), inplace=True)


save_path = None
load_path = None

best_params = {
    'n_iter': 700,
    'verbose': -1,
    'objective': 'l2',
    'learning_rate': 0.05689066836106983,
    'colsample_bytree': 0.8915976762048253,
    'colsample_bynode': 0.5942203285139224,
    'lambda_l1': 3.6277555139102864,
    'lambda_l2': 1.6591278779517808,
    'min_data_in_leaf': 186,
    'max_depth' : 9,
    'max_bin' : 813,
} # val score is 62.24 for the last month

best_params_solar = {
    'n_iter': 500,
    'verbose'          : -1,
    'objective': 'l2',
    'learning_rate': 0.05689066836106983,
    'colsample_bytree': 0.8915976762048253,
    'colsample_bynode': 0.5942203285139224,
    'lambda_l1': 3.6277555139102864,
    'lambda_l2': 1.6591278779517808,
    'min_data_in_leaf': 186,
    'max_depth': 9,
    'max_bin': 813,
} # val score is 62.24 for the last month

if load_path is not None:
    model = pickle.load(open(load_path, "rb"))
else:
    model = VotingRegressor([
        ('lgb_1', lgb.LGBMRegressor(**best_params, random_state=100)), 
        ('lgb_2', lgb.LGBMRegressor(**best_params, random_state=101)), 
        ('lgb_3', lgb.LGBMRegressor(**best_params, random_state=102)), 
        ('lgb_4', lgb.LGBMRegressor(**best_params, random_state=103)), 
        ('lgb_5', lgb.LGBMRegressor(**best_params, random_state=104)), 
    ])
    
    model_solar = VotingRegressor([
        ('lgb_6', lgb.LGBMRegressor(**best_params_solar, random_state=105)), 
        ('lgb_7', lgb.LGBMRegressor(**best_params_solar, random_state=106)), 
        ('lgb_8', lgb.LGBMRegressor(**best_params_solar, random_state=107)), 
        ('lgb_9', lgb.LGBMRegressor(**best_params_solar, random_state=108)), 
        ('lgb_10', lgb.LGBMRegressor(**best_params_solar, random_state=109)), 
    ])
X = final_df.drop('target', axis=1)
y = final_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.columns)

model.fit(X_train, y_train)
model_solar.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)


if save_path is not None:
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    with open(save_path, "wb") as f:
        pickle.dump(model_solar, f)
import enefit
import polars as pl

env = enefit.make_env()
iter_test = env.iter_test()

schema_train = df_train.schema
schema_client = df_client.schema
schema_gas = df_gas.schema
schema_electricity = df_electricity.schema
schema_forecast = df_forecast.schema
schema_historical = df_historical.schema
schema_target = df_target.schema

for (test, revealed_targets, client, historical_weather,
        forecast_weather, electricity_prices, gas_prices, sample_prediction) in iter_test:
    
    test = test.rename(columns={"prediction_datetime": "datetime"})
    
    df_test  = pl.from_pandas(test[train_cols[1:]], schema_overrides=schema_train)
    df_new_client = pl.from_pandas(client[client_cols], schema_overrides=schema_client)
    df_new_gas = pl.from_pandas(gas_prices[gas_cols], schema_overrides=schema_gas)
    df_new_electricity = pl.from_pandas(electricity_prices[electricity_cols], schema_overrides=schema_electricity)
    df_new_forecast = pl.from_pandas(forecast_weather[forecast_cols], schema_overrides=schema_forecast)
    df_new_historical = pl.from_pandas(historical_weather[historical_cols], schema_overrides=schema_historical)
    df_new_target = pl.from_pandas(revealed_targets[target_cols], schema_overrides=schema_target)
    
    df_forecast = pl.concat([df_forecast, df_new_forecast]).unique()
    df_historical = pl.concat([df_historical, df_new_historical]).unique()
    df_target = pl.concat([df_target, df_new_target]).unique()
    
    df_test = df_test.to_pandas()
    df_new_client = df_new_client.to_pandas()
    df_new_gas = df_new_gas.to_pandas()
    df_new_electricity = df_new_electricity.to_pandas()
    df_new_forecast = df_new_forecast.to_pandas()
    df_new_historical = df_new_historical.to_pandas()
    # df_location = df_location.to_pandas()
    df_new_target = df_new_target.to_pandas()


    X_test = preprocess_and_join_datasets(
        df_test, df_new_client, df_new_gas, df_new_electricity, df_new_forecast, df_new_historical, df_location, df_new_target
    )
    
    test['target'] = model.predict(X_test).clip(0)
    test['target_solar'] = model_solar.predict(X_test).clip(0)
    test.loc[test['is_consumption']==0, "target"] = test.loc[test['is_consumption']==0, "target_solar"]    
    
    sample_prediction["target"] = test['target']
    
    env.predict(sample_prediction)