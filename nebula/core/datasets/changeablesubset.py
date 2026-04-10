from torch.utils.data import Subset


class ChangeableSubset(Subset):
    def __init__(
        self,
        dataset,
        indices,
    ):
        super().__init__(dataset, indices)
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

    def __len__(self):
        return len(self.indices)
