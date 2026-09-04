from typing import Dict, List
import torch


class BatchBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size: int = buffer_size
        self.batches: Dict[str, List]  = {}

    def add(self, batch: dict):
        # if it is not the first push
        if self.batches != {}:
            for k, v in batch.items():
                self.batches[k].append(v)

        # if it is the first push
        else:
            for k, v in batch.items():
                self.batches[k] = []
                self.batches[k].append(v)

    def build_dataset(self, duplicates=1):
        total_data = {}

        for k, v in self.batches.items():
            data = torch.cat(v)
            total_data[k] = data[-self.buffer_size:]
            # self.batches[k] = [total_data[k],] # handle off-policy samples
        return BufferDataset(total_data, duplicates)

    def __len__(self):
        keys = [k for k in self.batches.keys()]
        if len(keys) == 0:
            return 0
        return len(self.batches[keys[0]])

    def state_dict(self) -> None:
        return {
            "batches": self.batches,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.batches = state_dict["batches"]
    
    def clean(self):
        self.batches = {}


class ChemBatchBuffer(BatchBuffer):
    def build_dataset(self, duplicates=1):
        for k, v in self.batches.items():
            self.batches[k] = v[-self.buffer_size:]
        return BufferDataset(self.batches, duplicates)


def chem_collate(batch):
    if len(batch) == 1:
        return batch[0]
    else:
        keys = batch[0].keys()
        dict_of_lists = {key: [d[key] for d in batch] for key in keys}
        return dict_of_lists


class BufferDataset(torch.utils.data.Dataset):
    def __init__(self, total_data: dict, duplicates: int):
        super().__init__()
        # has more than 1 key(s) & all values have same length
        assert len(total_data.keys()) > 0
        assert 'log_rnd' in total_data.keys()
        assert 'term_cost' in total_data.keys()

        keys = list(total_data.keys())
        self.len = len(total_data[keys[0]])
        for v in total_data.values():
            assert len(v) == self.len

        self.total_data = total_data
        self.duplicates = duplicates # expand factor

    def __getitem__(self, idx):
        return {k: v[idx % self.len] for k, v in self.total_data.items()}

    def __len__(self):
        return self.len * self.duplicates

