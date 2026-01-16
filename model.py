import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input):
        return self.layers(input)
    

if __name__ == '__main__':
    pass
