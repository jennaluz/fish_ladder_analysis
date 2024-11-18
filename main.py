import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Read in Data =================================================================
fish_ladder_data = pd.read_csv('data/fish_ladder_data.csv')
fish_ladder_data = fish_ladder_data.convert_dtypes(convert_floating=True)

# Extract datetime object
datetime_format = ':%Y-%m-%d %H::%M::%S+00::00:'
fish_ladder_data['datetime'] = pd.to_datetime(fish_ladder_data['datetime'],
                                              format=datetime_format)


# Dworshak Water Temp Decision Tree ============================================
features = ['dworshak_discharge', 'dworshak_spill', 'orofino_water_temp',
            'orofino_discharge', 'ahsahka_temp', 'ahsahka_humidity']
outcome = ['dworshak_water_temp']

data = fish_ladder_data[['datetime'] + features + outcome]

time_series_features = ['orofino_water_temp', 'orofino_discharge']

for feature in time_series_features:
    data[feature + '_t-1'] = pd.NA
    data[feature + '_t-2'] = pd.NA
    data[feature + '_t-3'] = pd.NA
    print(data.columns)

    data.loc[1, feature + '_t-1'] = data.loc[0, feature]
    data.loc[2, feature + '_t-1'] = data.loc[1, feature]

    data.loc[2, feature + '_t-2'] = data.loc[0, feature]

    for i in range(3, data.shape[0]):
        data.loc[i, feature + '_t-1'] = data.loc[i - 1, feature]
        data.loc[i, feature + '_t-2'] = data.loc[i - 2, feature]
        data.loc[i, feature + '_t-3'] = data.loc[i - 3, feature]

    print(data[feature + ''])
    print(data[feature + '_t-1'])
    print(data[feature + '_t-2'])
    print(data[feature + '_t-3'])

# Or replace with average?
data = data.dropna(ignore_index=True)
