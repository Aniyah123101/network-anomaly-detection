import json
import matplotlib.pyplot as plt

with open('metrics.json', 'r') as f:
    metrics = json.load(f)

labels = ['Training', 'Validation']

# Accuracy plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(labels, [metrics['train_accuracy'], metrics['test_accuracy']], marker='o', color='green')
plt.title('Accuracy')
plt.ylim(0, 1)
plt.ylabel('Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(labels, [metrics['train_loss'], metrics['test_loss']], marker='o', color='red')
plt.title('Loss')
plt.ylabel('Log Loss')

plt.tight_layout()
plt.show()
