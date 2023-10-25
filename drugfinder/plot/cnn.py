from torch.nn import Module, Conv1d, Linear, Dropout, MaxPool1d, functional as F, BatchNorm1d, LazyLinear
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, f1_score, recall_score, accuracy_score
import csv

class Cnn(Module):
    """
    CNN model
    """
    def __init__(self, kernel_size=3, output_dim=1, input_dim=320, drop_out=0, stride=2, padding=1):
        super(Cnn, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.drop_out = drop_out
        self.padding = padding

        self.kernel_1 = kernel_size
        self.channel_1 = 32

        self.conv_1 = Conv1d(kernel_size=self.kernel_1, out_channels=self.channel_1, in_channels=1, stride=1, padding=self.padding)
        self.normalizer_1 = BatchNorm1d(self.channel_1)
        self.pooling_1 = MaxPool1d(kernel_size=self.kernel_1, stride=stride)

        self.dropout = Dropout(p=drop_out)
        self.fc1 = LazyLinear(64)
        self.normalizer_2 = BatchNorm1d(64)
        self.fc2 = Linear(64, 2)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # (batch, embedding_dim) -> (batch, 1, embedding_dim)
        c_1 = self.pooling_1(F.relu(self.normalizer_1(self.conv_1(x))))

        c_2 = torch.flatten(c_1, start_dim=1)
        c_2 = self.dropout(c_2)
        out = F.relu(self.normalizer_2(self.fc1(c_2)))
        out = self.fc2(out)
        out = torch.softmax(out, dim=-1)
        return out

class CustomDataset(Dataset):
    def __init__(self, x, y):
        super(CustomDataset, self).__init__()
        self.data = torch.from_numpy(x).float()
        self.labels = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.data


def get_th_dataset(x, y):
    """
    assemble a dataset with the given data and labels
    :param x:
    :param y:
    :return:
    """
    _dataset = CustomDataset(x, y)
    return _dataset

df = pd.read_csv('data/drugfinder/esm2_320_dimensions_with_labels.csv') 
y = df['label'].apply(lambda x: 0 if x != 1 else x).to_numpy().astype(np.int64)
X = df.drop(['label', 'UniProt_id'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)
test_set = get_th_dataset(X_test, y_test)


model_type = [
    'drugfinder/plot/res/kernel=3,epochs=100,cnn.pt',
    'drugfinder/plot/res/kernel=3,epochs=300,cnn.pt',
    'drugfinder/plot/res/kernel=3,epochs=500,cnn.pt',
    'drugfinder/plot/res/kernel=5,epochs=100,cnn.pt',
    'drugfinder/plot/res/kernel=5,epochs=300,cnn.pt',
    'drugfinder/plot/res/kernel=5,epochs=500,cnn.pt',
    'drugfinder/plot/res/kernel=7,epochs=100,cnn.pt',
    'drugfinder/plot/res/kernel=7,epochs=300,cnn.pt',
    'drugfinder/plot/res/kernel=7,epochs=500,cnn.pt',
    'drugfinder/plot/res/kernel=9,epochs=100,cnn.pt',
    'drugfinder/plot/res/kernel=9,epochs=300,cnn.pt',
    'drugfinder/plot/res/kernel=9,epochs=500,cnn.pt',
    'drugfinder/plot/res/kernel=11,epochs=100,cnn.pt',
    'drugfinder/plot/res/kernel=11,epochs=300,cnn.pt',
    'drugfinder/plot/res/kernel=11,epochs=500,cnn.pt',
]

def write_header():
    header = pd.DataFrame({
    'model_type': [],
    'accuracy': [],
    'mcc': [],
    'f1 score': [],
    'recall': [],
    'fpr': [],
    'tpr': []
})

# Save the header to a CSV file
    header.to_csv('drugfinder/plot/res/plot.csv', index=False)

def model_for_reference(model, model_type):
    model.load_state_dict(torch.load(model_type)['model_state_dict'])
    model.eval()

    with torch.no_grad():
        y_score = model(test_set.get_data())
    y_predict = []
    for i in range(len(y_score)):
        temp = y_score[i]
        if(temp[0] >= 0.5):
            temp_ = 1 - temp[0]
        else:
            temp_ = temp[1]
        y_predict.append(temp_.item())
    y_predict = np.array(y_predict)
    y_test = test_set.get_labels().cpu().numpy()

    fpr, tpr, thresholds = roc_curve(y_test, y_predict)
    roc_auc = auc(fpr, tpr)

    accuracy = accuracy_score(y_test, y_predict > 0.5)
    mcc = matthews_corrcoef(y_test, y_predict > 0.5)
    f1 = f1_score(y_test, y_predict > 0.5)
    recall = recall_score(y_test, y_predict > 0.5)

    data = [model_type,accuracy, mcc,f1,recall]
    with open('drugfinder/plot/res/plot.csv', 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

    print(model_type)
    print("accuracy:", accuracy)
    print("MCC:", mcc)
    print("F1 Score:", f1)
    print("Recall:", recall)
    return fpr, tpr

write_header()
model = Cnn(kernel_size=3, output_dim=1, input_dim=320, drop_out=0, stride=2, padding=1)
model_for_reference(model, model_type[0])
model_for_reference(model, model_type[1])
model_for_reference(model, model_type[2])

model = Cnn(kernel_size=5, output_dim=1, input_dim=320, drop_out=0, stride=2, padding=2)
model_for_reference(model, model_type[3])
model_for_reference(model, model_type[4])
model_for_reference(model, model_type[5])

model = Cnn(kernel_size=7, output_dim=1, input_dim=320, drop_out=0, stride=2, padding=3)
model_for_reference(model, model_type[6])
model_for_reference(model, model_type[7])
model_for_reference(model, model_type[8])

model = Cnn(kernel_size=9, output_dim=1, input_dim=320, drop_out=0, stride=2, padding=4)
model_for_reference(model, model_type[9])
model_for_reference(model, model_type[10])
model_for_reference(model, model_type[11])

model = Cnn(kernel_size=11, output_dim=1, input_dim=320, drop_out=0, stride=2, padding=5)
model_for_reference(model, model_type[12])
model_for_reference(model, model_type[13])
model_for_reference(model, model_type[14])