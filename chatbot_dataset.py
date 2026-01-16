from torch.utils.data import Dataset


class ChatbotDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.x_data = X_data
        self.y_data = y_data


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    

    def __len__(self):
        return len(self.y_data)
    

if __name__ == '__main__':
    pass
