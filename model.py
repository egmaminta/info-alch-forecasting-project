# import dependencies
import torch
import torch.utils.data
import pytorch_lightning


class LSTM_model(torch.nn.Module):

    def __init__(self, n_features: int, n_hidden: int=128, n_layers: int=1, **kwargs):
        super(LSTM_model, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size = n_features,
            hidden_size = n_hidden,
            num_layers = n_layers,
            batch_first = True,
            **kwargs
        )

        self.fc = torch.nn.Linear(n_hidden, 1)
    
    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(x)
        oup = self.fc(h_n[-1])
        return oup


class GRU_model(torch.nn.Module):

    def __init__(self, n_features: int, n_hidden: int=128, n_layers: int=1, **kwargs):
        super(GRU_model, self).__init__()

        self.gru = torch.nn.GRU(
            input_size = n_features,
            hidden_size = n_hidden,
            num_layers = n_layers,
            batch_first = True,
            **kwargs
        )

        self.fc = torch.nn.Linear(n_hidden, 1)
    
    def forward(self, x):
        self.gru.flatten_parameters()
        _, h_n = self.gru(x)
        oup = self.fc(h_n[-1])
        return oup


class LitStockModel(pytorch_lightning.LightningModule):

    def __init__(self, model: str, lr: float=0.001, n_epochs: int=50, **kwargs):
        super(LitStockModel, self).__init__()

        self.save_hyperparameters()

        if model.lower() == "lstm":
            self.model = LSTM_model(**kwargs)
        elif model.lower() == "gru":
            self.model = GRU_model(**kwargs)

        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x, y=None):
        oup = self.model(x)
        return oup
    
    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss = 0
        oup = self(sequences, labels)
        if labels is not None:
            loss = self.criterion(oup, torch.unsqueeze(labels, dim=1))
        return {"loss": loss}
    
    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss = 0
        oup = self(sequences, labels)
        if labels is not None:
            loss = self.criterion(oup, torch.unsqueeze(labels, dim=1))
        return {"test_loss": loss}
    
    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss = 0
        oup = self(sequences, labels)
        if labels is not None:
            loss = self.criterion(oup, torch.unsqueeze(labels, dim=1))
        return {"val_loss": loss}
    
    def training_epoch_end(self, oup):
        avg_loss = torch.stack([x["loss"] for x in oup]).mean()
        self.log("train_loss", avg_loss, on_epoch=True)
    
    def test_epoch_end(self, oup):
        avg_loss = torch.stack([x["test_loss"] for x in oup]).mean()
        self.log("test_loss", avg_loss, on_epoch=True, prog_bar=True)
    
    def validation_epoch_end(self, oup):
        avg_loss = torch.stack([x["val_loss"] for x in oup]).mean()
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams["n_epochs"])
        return [optimizer], [scheduler]