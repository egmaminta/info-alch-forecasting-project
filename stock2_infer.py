# import dependencies
from turtle import forward
import matplotlib.pyplot as plt
import matplotlib
from numpy import dtype
import pandas
import seaborn
import tqdm

import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler

import torch

# custom imports
import utils
import model
import dataset


HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
seaborn.set(style="whitegrid", palette="muted", font_scale=1.2)
seaborn.set_palette(seaborn.color_palette(HAPPY_COLORS_PALETTE))


if __name__ == "__main__":
    seed = 42
    pytorch_lightning.seed_everything(seed)

    # load data
    filepath = "./data/nifty50-stock-market-data/COALINDIA.csv"
    data = pandas.read_csv(filepath)
    data = data.sort_values(by="Date").reset_index(drop=True)
    data["Date"] = pandas.to_datetime(data["Date"], format="%Y-%m-%d")
    data.set_index(keys="Date", inplace=True)
    data = utils.create_features_date_only(dataframe=data)
    data = utils.add_holiday_features(data, country="india", years=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])
    data = utils.generate_cyclical_features(dataframe=data, col="day_of_week", period=7, start_num=0)
    data = utils.generate_cyclical_features(dataframe=data, col="day_of_month", period=30, start_num=1)
    data = utils.generate_cyclical_features(dataframe=data, col="week_of_year", period=52, start_num=0)
    data = utils.generate_cyclical_features(dataframe=data, col="month", period=12, start_num=1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data[["close"]])

    data["close"] = scaler.transform(data[["close"]])

    train_size_ratio = 0.8
    train_size = int(len(data.index) * train_size_ratio)
    train_data, test_data = data[:train_size], data[train_size:]

    start_date, freq = utils.get_datetime_index(dataframe=test_data)
    
    forecast_data = utils.future_predictions(scaler=scaler, test_data=test_data, freq=freq, start=start_date, future_steps=60, country="india", years=[2021, 2022, 2023, 2024])
    #print(forecast_data.tail(20))

    N_FEATURES = forecast_data.shape[1]

    trained_lstm = model.LitStockModel.load_from_checkpoint(
        checkpoint_path="./checkpoints/best_model_lstm_stock2.ckpt",
        n_features = N_FEATURES,
    )
    trained_gru = model.LitStockModel.load_from_checkpoint(
        checkpoint_path="./checkpoints/best_model_gru_stock2.ckpt",
        n_features = N_FEATURES,
    )
    
    trained_lstm.freeze()
    trained_gru.freeze()

    trained_lstm.eval()
    trained_gru.eval()

    sequence_length=21

    keep_index = forecast_data.index
    forecast_data.reset_index(drop=True, inplace=True)

    with torch.inference_mode():
        predictions = []
        labels = []
        size = len(test_data.index)

        for i in tqdm.tqdm(range(size-sequence_length)):
            sequence = test_data.iloc[i:i+sequence_length]
            label_position = i+sequence_length
            label = test_data.iloc[label_position]["close"]
            
            sequence=torch.from_numpy(sequence.values).float()

            pred = trained_lstm(torch.unsqueeze(sequence, dim=0)).data.item()
            #print(pred)
            predictions.append(pred)
            labels.append(label)
            ctr = 0
            if i == (size-sequence_length-1):
                while ctr<60:
                    #print(forecast_data.loc[label_position, "close"])
                    sequence = forecast_data.iloc[i-19:i+2]
                    sequence=torch.from_numpy(sequence.values).float()
                    pred = trained_lstm(torch.unsqueeze(sequence, dim=0)).data.item()
                    label_position += 1
                    forecast_data.loc[label_position, "close"] = pred
                    predictions.append(pred)
                    i += 1
                    ctr += 1
    
    #print(test_data.tail(25))
    
    forecast_data.set_index(keep_index, inplace=True)
    
    #print(forecast_data.tail(30))

    predictions_descaled = utils.descale(scaler=scaler, values=predictions)
    forecast_descaled = utils.descale(scaler=scaler, values=forecast_data["close"])
    labels_descaled = utils.descale(scaler=scaler, values=labels)

    #print(len(predictions_descaled))
    #print(len(forecast_descaled))
    #print(len(labels_descaled))

    #print(predictions_descaled)
    #print(labels_descaled)

    #print(len(predictions_descaled))
    #print(predictions_descaled[580:680])
    seaborn.lineplot(x=range(len(predictions_descaled)), y=predictions_descaled, label="predictions")
    seaborn.lineplot(x=range(len(labels_descaled)), y=labels_descaled, label="labels")
    plt.show()

    """
    plt.plot(predictions_descaled, label="predictions", linestyle="-.")
    #plt.plot(forecast_descaled[sequence_length:], label="future", linestyle=":")
    plt.plot(labels_descaled, label="actual", linestyle="-")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    """