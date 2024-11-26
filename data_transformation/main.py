import datetime
from enum import Enum
import pandas as pd


# User-Defined Types ===========================================================
# DatetimeError defines the type of error a data instance could have with
# respect to its corresponding `datetime` feature
class DatetimeError(Enum):
    duplicated = 0
    missing = 1


# User-Defined Functions =======================================================
def log_data_errors(dataframe):
    time_interval = datetime.timedelta(hours=1)
    current_time = dataframe.loc[0, 'datetime']
    errorneous_times = {}

    # Check for missing and repeated data
    for index, row in dataframe.iterrows():
        # The `datetime` is as expected
        if row['datetime'] == current_time:
            current_time += time_interval
            continue

        # The `datetime` likely repeated
        elif row['datetime'] < current_time:
            errorneous_times[row['datetime']] = DatetimeError.duplicated
            dataframe.drop(index, inplace=True)

        # There `datetime` instance was skipped
        else:
            # Log all missing `datetime` instances
            while row['datetime'] > current_time:
                errorneous_times[current_time] = DatetimeError.missing
                current_time += time_interval

            current_time += time_interval

    # Save erroneous times to file
    with open('duplicated_data_log_test.txt', 'w') as duplicated_file, \
            open('missing_data_log_test.txt', 'w') as missing_file:
        missing_file.write('datetime\n')
        for date, error in errorneous_times.items():
            if error == DatetimeError.duplicated:
                duplicated_file.write(f'{date} ---------- {error}\n')
            else:
                missing_file.write(f'{date}\n')


    return dataframe


# Read in Data =================================================================
fish_ladder_data = pd.read_csv(
        'fish_ladder_data_original.csv',
        low_memory=False)
fish_ladder_data = fish_ladder_data.convert_dtypes(convert_floating=True)

# Extract datetime object
datetime_format = ':%Y-%m-%d %H::%M::%S+00::00:'
fish_ladder_data['datetime'] = pd.to_datetime(fish_ladder_data['datetime'],
                                              format=datetime_format)

print('Removing duplicated data...')
#fish_ladder_data = log_data_errors(fish_ladder_data)
#fish_ladder_data.to_csv('fish_ladder_data_no_duplicates.csv', index=False)

#fish_ladder_data = pd.read_csv(
#        'fish_ladder_data_no_duplicates.csv',
#        low_memory=False)
#
## Strongly type numeric features
#fish_ladder_data = fish_ladder_data.convert_dtypes(convert_floating=True)
#
## Extract datetime object
#fish_ladder_data['datetime'] = pd.to_datetime(fish_ladder_data['datetime'])

#fish_ladder_data = fish_ladder_data.drop_duplicates('datetime')

# Remove duplicates by "merging" the columns
# The data is grouped by duplicated datetime intervals
# Duplicated rows are merged together, based on the first appearance
fish_ladder_data = fish_ladder_data.groupby('datetime').agg('first').reset_index()
log_data_errors(fish_ladder_data)
print(fish_ladder_data.shape)


# Convert Dworshak water temperature from Fahrenheit to Celsius ================
print('Converting Dworshak water temperature from F to C...')
fish_ladder_data['dworshak_water_temp'] = \
        (5/ 9) * (fish_ladder_data['dworshak_water_temp_f'] - 32)

fish_ladder_data.drop('dworshak_water_temp_f', axis=1, inplace=True)

# Convert all 'object' dtypes to a
fish_ladder_data = fish_ladder_data.convert_dtypes(convert_floating=True)


# Feature Extraction ===========================================================
features = [
        'dworshak_water_temp', 'dworshak_discharge', 'dworshak_spill',
        'asotin_creek_discharge', 'big_canyon_discharge',
        'lapwai_creek_discharge', 'hells_canyon_dam_discharge',
        'orofino_water_temp', 'orofino_discharge', 'peck_water_temp',
        'peck_discharge', 'potlatch_creek_discharge', 'spalding_water_temp',
        'spalding_discharge', 'anatone_water_temp', 'anatone_discharge',
        'ahsahka_temp', 'ahsahka_humidity', 'cherrylane_temp',
        'cherrylane_humidity', 'spalding_temp', 'spalding_humidity',
        'alpowa_temp', 'alpowa_humidity', 'lgr_temp', 'lgr_humidity']
outcome = ['lgr_water_temp']

data = fish_ladder_data[['datetime'] + features + outcome]


# Fill Missing Data ============================================================
# Since not too much data is missing, interpolate the values
print('Filling missing data...')
#missing_data = pd.read_csv('missing_data_log.txt')
#missing_data['datetime'] = pd.to_datetime(missing_data['datetime'])

# Add missing rows with empty feature data
#for _, row in missing_data.iterrows():
#    new_row = pd.DataFrame({'datetime': [row['datetime']]})
#    data = pd.concat([data, new_row], ignore_index=True)
#
## Resort rows
## Log empy values
#with open('empty_values_log.txt', 'w') as file:
#    file.write('===== Empty Values Before Interpolation ======\n')
#    file.write(f'{data.isna().sum()}')
#
#data.sort_values(by=['datetime'], ignore_index=True, inplace=True)

data = data.set_index('datetime').resample('1h').first().reset_index() \
        .reindex(columns=data.columns)

# Interpolate numeric rows
print('Interpolating data...')
interpolated_data = data.select_dtypes(include='number') \
        .interpolate(limit_area='inside')
data = pd.DataFrame(data['datetime']).join(interpolated_data)


# Add Time Series Columns ======================================================
time_series_features = [
        'dworshak_water_temp', 'dworshak_discharge', 'dworshak_spill',
        'asotin_creek_discharge', 'big_canyon_discharge',
        'lapwai_creek_discharge', 'hells_canyon_dam_discharge',
        'orofino_water_temp', 'orofino_discharge', 'peck_water_temp',
        'peck_discharge', 'potlatch_creek_discharge', 'spalding_water_temp',
        'spalding_discharge', 'anatone_water_temp', 'anatone_discharge']

print('Generating time series...')
for feature in time_series_features:
    print(f'\t{feature}')

    for i in range(1, 11):
        new_feature = \
                pd.DataFrame({f'{feature}_t-{i}': data[feature].shift(i)})
        data = pd.concat([data, new_feature], axis=1)

#for feature in time_series_features:
#    print(f'\t{feature}')
#    data.loc[:, feature + '_t-1'] = pd.NA
#    data.loc[:, feature + '_t-2'] = pd.NA
#    data.loc[:, feature + '_t-3'] = pd.NA
#    data.loc[:, feature + '_t-4'] = pd.NA
#    data.loc[:, feature + '_t-5'] = pd.NA
#    data.loc[:, feature + '_t-6'] = pd.NA
#
#    data.loc[1, feature + '_t-1'] = data.loc[0, feature]
#    data.loc[2, feature + '_t-1'] = data.loc[1, feature]
#    data.loc[3, feature + '_t-1'] = data.loc[2, feature]
#    data.loc[4, feature + '_t-1'] = data.loc[3, feature]
#    data.loc[5, feature + '_t-1'] = data.loc[4, feature]
#    data.loc[6, feature + '_t-1'] = data.loc[5, feature]
#
#    data.loc[2, feature + '_t-2'] = data.loc[0, feature]
#    data.loc[3, feature + '_t-2'] = data.loc[1, feature]
#    data.loc[4, feature + '_t-2'] = data.loc[2, feature]
#    data.loc[5, feature + '_t-2'] = data.loc[3, feature]
#    data.loc[6, feature + '_t-2'] = data.loc[4, feature]
#
#    data.loc[3, feature + '_t-3'] = data.loc[0, feature]
#    data.loc[4, feature + '_t-3'] = data.loc[1, feature]
#    data.loc[5, feature + '_t-3'] = data.loc[2, feature]
#    data.loc[6, feature + '_t-3'] = data.loc[3, feature]
#
#    data.loc[4, feature + '_t-4'] = data.loc[0, feature]
#    data.loc[5, feature + '_t-4'] = data.loc[1, feature]
#    data.loc[6, feature + '_t-4'] = data.loc[2, feature]
#
#    data.loc[5, feature + '_t-5'] = data.loc[0, feature]
#    data.loc[6, feature + '_t-5'] = data.loc[1, feature]
#
#    for i in range(6, data.shape[0]):
#        data.loc[i, feature + '_t-1'] = data.loc[i - 1, feature]
#        data.loc[i, feature + '_t-2'] = data.loc[i - 2, feature]
#        data.loc[i, feature + '_t-3'] = data.loc[i - 3, feature]
#        data.loc[i, feature + '_t-4'] = data.loc[i - 4, feature]
#        data.loc[i, feature + '_t-5'] = data.loc[i - 5, feature]


# Extract Data Range ===========================================================
# Find the first index of a non-null value
print('Limiting data date range...')
data_start_idx = data.notna().idxmax()
data_start = pd.Series(index=data.drop('datetime', axis=1).columns,
                       dtype=data['datetime'].dtype)

# Extract the corresponding datetime
# for the first non-null value
for idx, _ in data_start.items():
    data_start[idx] = data.loc[data_start_idx[idx], 'datetime']

# Record the amount of null values
with open('empty_values_log.txt', 'a') as file:
    file.write('\n\n')
    file.write('===== Empty Values After Interpolation ======\n')
    file.write(f'{data.isna().sum()}')

    file.write('\n\n')
    file.write('===== Date of First Non-NA Value ======\n')
    file.write(f'{data_start}')

# Find the last index of the non-null values
data_end_idx = data.sort_index(ascending=False).notna().idxmax()
data_end = pd.Series(index=data.drop('datetime', axis=1).columns,
                     dtype=data['datetime'].dtype)

# Extract the corresponding datetime
# for the last non-null value
for idx, _ in data_end.items():
    data_end[idx] = data.loc[data_end_idx[idx], 'datetime']

# Shrink the data range to only include non-null values
data = data.loc[data_start.max() <= data['datetime'], :]
data = data.loc[data['datetime'] <= data_end.min(), :]

# Record the amount of null values after data shrinkage
with open('empty_values_log.txt', 'a') as file:
    file.write('\n\n')
    file.write('===== Empty Values After Date Range Minimization =====\n')
    file.write(f'{data.isna().sum()}')

data.to_csv('fish_ladder_data_final.csv', index=False)


# Example of Creating Training & Test Sets =====================================
print('Creating training and test sets...')
inputs = data[['datetime'] + features]
targets = data[outcome]

# Generate test size & calcualte the first index of the test set
test_size = 0.2
test_start_idx = data.size - int(data.size * 0.2)

# Split inputs and targets to create training and test sets
training_inputs = inputs[:test_start_idx]
training_targets = targets[:test_start_idx]

test_inputs = inputs[test_start_idx:]
test_targets = targets[test_start_idx:]
