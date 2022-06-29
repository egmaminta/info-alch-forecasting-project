# import dependencies
import pandas
import numpy
import tqdm
import datetime
import holidays
import sklearn.preprocessing


def create_close_change(dataframe: pandas.DataFrame):
    dataframe["Prev Close"] = dataframe.shift(1)["Close"]
    dataframe["Close Change"] = dataframe.apply(
        lambda row: 0 if numpy.isnan(row["Prev Close"]) else row["Close"] - row["Prev Close"],
        axis=1
    )
    return dataframe


def create_features(dataframe: pandas.DataFrame):
    rows = []
    for _, row in tqdm.tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        row_data = dict(
            day_of_week = _.dayofweek,
            day_of_month = _.day,
            week_of_year = _.week,
            month = _.month,
            open = row["Open"],
            high = row["High"],
            low = row["Low"],
            close_change = row["Close Change"],
            close = row["Close"],
        )
        rows.append(row_data)
    
    return pandas.DataFrame(rows, index=dataframe.index)


def generate_cyclical_features(dataframe: pandas.DataFrame, col: str, period: int, start_num=0):
    kwargs = {
        f"sin_{col}": lambda x: numpy.sin(2*numpy.pi*(dataframe[col]-start_num)/period),
        f"cos_{col}": lambda x: numpy.cos(2*numpy.pi*(dataframe[col]-start_num)/period),
    }
    return dataframe.assign(**kwargs).drop(columns=[col])


def add_holiday_features(dataframe: pandas.DataFrame, country="us", **kwargs):

    def is_holiday(date:datetime.date):
        holidays_dict = {
            "us": holidays.US,
            "india": holidays.India,
        }
        _holidays = holidays_dict.get(country.lower())(**kwargs)
        date = date.replace(hour=0)
        return 1 if (date in _holidays) else 0

    # requires dates as index
    return dataframe.assign(is_holiday=dataframe.index.to_series().apply(is_holiday))


def create_sequences(dataframe: pandas.DataFrame, target: str, sequence_length: int):
    sequences = []
    size = len(dataframe.index)
    
    for i in tqdm.tqdm(range(size-sequence_length)):
        sequence = dataframe.iloc[i:i+sequence_length]
        label_position = i + sequence_length
        label = dataframe.iloc[label_position][target]
        sequences.append((sequence, label))
    
    return sequences


def descale(scaler, values):
    _descaler = sklearn.preprocessing.MinMaxScaler()
    _descaler.min_, _descaler.scale_ = scaler.min_[-1], scaler.scale_[-1]

    values_2d = numpy.array(values)[:, numpy.newaxis]
    return _descaler.inverse_transform(values_2d).flatten()