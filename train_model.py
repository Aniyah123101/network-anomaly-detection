import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt

# Simulated dataset loading from JSON
with open('protocol_data.json', 'r') as f:
    data = json.load(f)

X = np.array([[v] for v in data.values()])
y = np.array([1 if k in ['TCP', 'HTTPS'] else 0 for k in data.keys()])  # Assume TCP/HTTPS = benign

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy and Loss
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

train_loss = log_loss(y_train, model.predict_proba(X_train))
test_loss = log_loss(y_test, model.predict_proba(X_test))

# Save metrics for visualization
metrics = {
    'train_accuracy': train_acc,
    'test_accuracy': test_acc,
    'train_loss': train_loss,
    'test_loss': test_loss
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Model trained and metrics saved.")
