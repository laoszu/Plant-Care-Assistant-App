from torch.utils.data import DataLoader, random_split
from dataset import PlantNetDataset

class PlantNetDataLoader:
    def __init__(self, data_dir, batch_size=32, **kwargs):
        self.batch_size = batch_size
        
        full_dataset = PlantNetDataset(data_dir)
        
        self.train_dataset, self.val_dataset, self.test_dataset = \
            self._split_dataset(full_dataset)
    
    def get_train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )