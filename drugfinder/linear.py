import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class pharos(Dataset):
    def __init__(self, x, y):
        super(pharos, self).__init__()
        self.data = torch.from_numpy(x).float()
        self.labels = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.data

if os.path.exists('/content'):
    if os.path.exists('/content/drive/MyDrive'):
        os.chdir('/content/drive/MyDrive')
        df = pd.read_csv('esm2_320_dimensions_with_labels.csv') 
    else:
        os.chdir('/home/musong/Desktop')
        df = pd.read_csv('/home/musong/Desktop/paper/data/drugfinder/esm2_320_dimensions_with_labels.csv') 
else:
    os.chdir('/home/musong/Desktop')
    df = pd.read_csv('/home/musong/Desktop/paper/data/drugfinder/esm2_320_dimensions_with_labels.csv') 

# df = df.iloc[500:600]
X = df.drop(['label', 'UniProt_id'], axis=1)
y = df['label'].apply(lambda x: 0 if x != 1 else x)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

batch_size = 64
learning_rate = 0.0005
num_epochs = 200
train_set = pharos(np.array(X_train), np.array(y_train))
test_set = pharos(np.array(X_train), np.array(y_test))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
# Define hyperparameters


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define a simple feedforward neural network model
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(320, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.fc4 = nn.Linear(10, 2)  # 2 output classes (0 and 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Create the model and move it to the GPU
model = SimpleClassifier().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_losses = []
# Assuming you have already loaded your data into train_loader and test_loader

# Training loop
for epoch in range(num_epochs):
    model.train()
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())
    
    print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item()}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)  # Move data to GPU
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test set: {100 * correct / total}%')

plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()