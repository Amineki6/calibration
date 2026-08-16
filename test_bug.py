import pandas as pd
from dataset import BatchData
import torch

class DummyDataset:
    def __init__(self):
        # simulate the paths
        self.path = pd.Series(["path0", "path1", "path2"])
        # imagine some filtering happened, so the original dataframe had these paths
        # but the filtered dataframe has index 1, 4, 7...
        # oh wait, we do reset_index(drop=True)!
        self.path = pd.Series(["path0", "path1", "path2"]).reset_index(drop=True)
        
    def __getitem__(self, index):
        return BatchData(
            indices=torch.tensor(index, dtype=torch.long),
            inputs=torch.tensor(0),
            labels=torch.tensor(0),
            drains=torch.tensor(0),
            weights=torch.tensor(0)
        )
    def __len__(self):
        return len(self.path)

ds = DummyDataset()
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

features_dict = {}
for batch in loader:
    for idx_in_batch, original_idx in enumerate(batch.indices):
        original_idx = original_idx.item()
        path_str = str(ds.path.iloc[original_idx])
        features_dict[path_str] = original_idx

print(features_dict)
