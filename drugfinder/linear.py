import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, f1_score, recall_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 

df = pd.read_csv('data/drugfinder/esm2_320_dimensions_with_labels.csv') 
X = df.drop(['label', 'UniProt_id'], axis=1).values
y = df['label'].apply(lambda x: 0 if x != 1 else x).values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(320, 180)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(180, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 30)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
    
model = Deep()
model.load_state_dict(torch.load('drugfinder/linear.pt'))
with torch.no_grad():
    y = model(X_test).reshape(-1)
    y_predict = y.numpy()
y_test = y_test.reshape(-1).numpy()

fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)

accuracy = accuracy_score(y_test, y_predict > 0.5)
mcc = matthews_corrcoef(y_test, y_predict > 0.5)
f1 = f1_score(y_test, y_predict > 0.5)
recall = recall_score(y_test, y_predict > 0.5)

print("accuracy:", accuracy)
print("MCC:", mcc)
print("F1 Score:", f1)
print("Recall:", recall)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()