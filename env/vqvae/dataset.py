from torch_geometric.data import Dataset      # TODO: Rename and create a file dataset.py for the CIFAR-10 dataset
                                              # TODO: Save the processed dataset as .pt files
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