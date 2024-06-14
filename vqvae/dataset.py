from torch_geometric.data import Dataset

class NASBenchDataset(Dataset):
    def __init__(self, nasbench_data):
        if not nasbench_data:
            raise Exception("nasbench_data cannot be None")
        super(NASBenchDataset, self).__init__()
        self.data = nasbench_data

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]