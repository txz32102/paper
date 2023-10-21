import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, f1_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/drugfinder/esm2_320_dimensions_with_labels.csv')

# Prepare the data
X = df.drop(['label', 'UniProt_id'], axis=1)
y = df['label'].apply(lambda x: 0 if x != 1 else x).to_numpy().astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

# Train Gaussian Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_y_predict = nb_classifier.predict_proba(X_test)[:, 1]
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_y_predict)
nb_roc_auc = auc(nb_fpr, nb_tpr)
mcc_nb = matthews_corrcoef(y_test, nb_y_predict > 0.5)
f1_nb = f1_score(y_test, nb_y_predict > 0.5)
recall_nb = recall_score(y_test, nb_y_predict > 0.5)

print('GaussianNB')
print("MCC:", mcc_nb)
print("F1 Score:", f1_nb)
print("Recall:", recall_nb)

# Train Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_y_predict = rf_classifier.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_predict)
rf_roc_auc = auc(rf_fpr, rf_tpr)
mcc_rf = matthews_corrcoef(y_test, rf_y_predict > 0.5)
f1_rf = f1_score(y_test, rf_y_predict > 0.5)
recall_rf = recall_score(y_test, rf_y_predict > 0.5)

print('Random Forest')
print("MCC:", mcc_rf)
print("F1 Score:", f1_rf)
print("Recall:", recall_rf)

# Train Support Vector Machine (SVM)
svm_classifier = SVC(probability=True, random_state=42)
svm_classifier.fit(X_train, y_train)
svm_y_predict = svm_classifier.predict_proba(X_test)[:, 1]
svm_fpr, svm_tpr, thresholds = roc_curve(y_test, svm_y_predict)
svm_roc_auc = auc(svm_fpr, svm_tpr)
mcc = matthews_corrcoef(y_test, svm_y_predict > 0.5)
f1 = f1_score(y_test, svm_y_predict > 0.5)
recall = recall_score(y_test, svm_y_predict > 0.5)

print("SVM")
print("MCC:", mcc)
print("F1 Score:", f1)
print("Recall:", recall)

# Create a new figure for both ROC curves
plt.figure(figsize=(8, 6))

# Plot Gaussian Naive Bayes ROC curve
plt.plot(nb_fpr, nb_tpr, color='darkorange', lw=2, label=f'Gaussian Naive Bayes (AUC = {nb_roc_auc:.2f})')

# Plot Random Forest ROC curve
plt.plot(rf_fpr, rf_tpr, color='blue', lw=2, label=f'Random Forest (AUC = {rf_roc_auc:.2f})')

# Plot SVM ROC curve
plt.plot(svm_fpr, svm_tpr, color='pink', lw=2, label=f'SVM (AUC = {svm_roc_auc:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

# Configure the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")

# Save the figure
plt.savefig('debug/ML.png', dpi=500)

plt.show()  # Optionally display the plot interactively
