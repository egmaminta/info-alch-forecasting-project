# import dependencies
import matplotlib.pyplot as plt
import matplotlib
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
    data = utils.create_close_change(dataframe=data)
    data = utils.create_features(dataframe=data)
    data = utils.add_holiday_features(data, country="india", years=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])
    data = utils.generate_cyclical_features(dataframe=data, col="day_of_week", period=7, start_num=0)
    data = utils.generate_cyclical_features(dataframe=data, col="day_of_month", period=30, start_num=1)
    data = utils.generate_cyclical_features(dataframe=data, col="week_of_year", period=52, start_num=0)
    data = utils.generate_cyclical_features(dataframe=data, col="month", period=12, start_num=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    print(data.head(5))
    scaler.fit(data[["open", "high", "low", "close_change","close"]])
    data[["open", "high", "low", "close_change","close"]] = scaler.transform(data[["open", "high", "low", "close_change","close"]])
    print(data.head(5))
    train_size_ratio = 0.8
    train_size = int(len(data.index) * 0.80)
    train_data, test_data = data[:train_size], data[train_size:]
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    train_data = pandas.DataFrame(
        train_data.values,
        columns=train_data.columns,
        index=train_data.index
    )
    test_data = pandas.DataFrame(
        test_data.values,
        columns=test_data.columns,
        index=test_data.index
    )
    print(train_data.head(5))
    SEQUENCE_LENGTH = 21
    train_sequences = utils.create_sequences(dataframe=train_data, target="close", sequence_length=SEQUENCE_LENGTH)
    test_sequences = utils.create_sequences(dataframe=test_data, target="close", sequence_length=SEQUENCE_LENGTH)
    print(f"Train sequences data: \n {train_sequences[0][0]} \n shape: {train_sequences[0][0].shape}")
    print(f"Train sequences label: \n {train_sequences[0][1]} \n shape: {train_sequences[0][1].shape}")
    N_EPOCHS = 100
    BATCH_SIZE = 32
    data_module = dataset.LitStockData(train_sequences=train_sequences, test_sequences=test_sequences, batch_size=BATCH_SIZE)
    data_module.setup()
    N_FEATURES = train_data.shape[1]
    N_HIDDEN = 256
    N_LAYERS = 2
    LEARNING_RATE = 0.001
    LitLSTM = model.LitStockModel(
        model="lstm",
        n_features = N_FEATURES,
        n_hidden = N_HIDDEN,
        n_layers = N_LAYERS,
        lr = LEARNING_RATE,
        n_epochs=N_EPOCHS,
    )
    LitGRU = model.LitStockModel(
        model="gru",
        n_features = N_FEATURES,
        n_hidden = N_HIDDEN,
        n_layers = N_LAYERS,
        lr = LEARNING_RATE,
        n_epochs=N_EPOCHS,
    )
    checkpoint_callback_lstm = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best_model_lstm",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )
    checkpoint_callback_gru = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best_model_gru",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )
    early_stopping_callback_lstm = EarlyStopping(
        monitor="val_loss",
        patience=10,
    )
    early_stopping_callback_gru = EarlyStopping(
        monitor="val_loss",
        patience=10,
    )
    logger_lstm = TensorBoardLogger(save_dir="lightning_logs", name="lstm_stock")
    logger_gru = TensorBoardLogger(save_dir="lightning_logs", name="gru_stock")
    trainer_lstm = pytorch_lightning.Trainer(
        accelerator="gpu",
        gpus=1,
        precision=16,
        max_epochs=N_EPOCHS,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback_lstm, early_stopping_callback_lstm],
        logger=logger_lstm,
    )
    trainer_gru = pytorch_lightning.Trainer(
        accelerator="gpu",
        gpus=1,
        precision=16,
        max_epochs=N_EPOCHS,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback_gru, early_stopping_callback_gru],
        logger=logger_gru,
    )
    trainer_lstm.fit(datamodule=data_module, model=LitLSTM)
    trainer_gru.fit(datamodule=data_module, model=LitGRU)

    # inference
    trained_lstm = model.LitStockModel.load_from_checkpoint(
        checkpoint_path="./checkpoints/best_model_lstm.ckpt",
        n_features = N_FEATURES,
    )
    trained_gru = model.LitStockModel.load_from_checkpoint(
        checkpoint_path="./checkpoints/best_model_gru.ckpt",
        n_features = N_FEATURES,
    )

    trained_lstm.freeze()
    trained_gru.freeze()

    trained_lstm.eval()
    trained_gru.eval()

    test_data = dataset.StockData(sequences=test_sequences)
    predictions_lstm = []
    predictions_gru = []
    labels = []
    for item in tqdm.tqdm(test_data):
        sequence = item["sequence"]
        label = item["label"]
        oup_lstm = trained_lstm(torch.unsqueeze(sequence, dim=0))
        oup_gru = trained_gru(torch.unsqueeze(sequence, dim=0))
        predictions_lstm.append(oup_lstm.data.item())
        predictions_gru.append(oup_gru.data.item())
        labels.append(label.data.item())
    
    predictions_lstm_descaled = utils.descale(scaler=scaler, values=predictions_lstm)
    predictions_gru_descaled = utils.descale(scaler=scaler, values=predictions_gru)
    labels_descaled = utils.descale(scaler=scaler, values=labels)
    test_data = data[train_size:]
    test_sequences_data = test_data.iloc[SEQUENCE_LENGTH:]
    dates = matplotlib.dates.date2num(test_sequences_data.index.tolist())
    plt.plot_date(dates, predictions_lstm_descaled, "r-", label="LSTM")
    plt.plot_date(dates, predictions_gru_descaled, "b-", label="GRU")
    plt.plot_date(dates, labels_descaled, "g-", label="Actual")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()