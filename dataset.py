# import dependencies
import torch
import torch.utils.data
import pytorch_lightning


class StockData(torch.utils.data.Dataset):
    
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(
            sequence=torch.from_numpy(sequence.values).float(),
            label=torch.tensor(label).float()
        )


class LitStockData(pytorch_lightning.LightningDataModule):

    def __init__(self, train_sequences, test_sequences, batch_size=32, **kwargs):
        super(LitStockData, self).__init__()

        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = StockData(self.train_sequences)
        self.test_dataset = StockData(self.test_sequences)
    
    def train_dataloader(self, **kwargs):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            **kwargs
        )
    
    def val_dataloader(self, **kwargs):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            **kwargs
        )
    
    def test_dataloader(self, **kwargs):
        return self.val_dataloader(**kwargs)