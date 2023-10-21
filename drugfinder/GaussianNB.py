import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, f1_score, recall_score
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/drugfinder/esm2_320_dimensions_with_labels.csv') 

X = df.drop(['label', 'UniProt_id'], axis=1)
y = df['label'].apply(lambda x: 0 if x != 1 else x).to_numpy().astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

y_score = nb_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

mcc = matthews_corrcoef(y_test, y_score > 0.5)
f1 = f1_score(y_test, y_score > 0.5)
recall = recall_score(y_test, y_score > 0.5)

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
plt.title('NB')
plt.legend(loc="lower right")
plt.savefig('debug/GaussianNB.png', dpi=500)