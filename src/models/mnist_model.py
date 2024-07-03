import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

class MNISTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)

def get_mnist_data(batch_size, valid_batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)

    collate_fn = lambda x: {'inputs': x[0][0].unsqueeze(0), 'targets': torch.tensor(x[0][1]).unsqueeze(0)}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(test_dataset, batch_size=valid_batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, valid_loader