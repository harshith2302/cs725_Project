import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to logs
log_paths = {
    'Baseline (SGD, Tanh, 0.01)': 'logs/baseline_paper/training_log.csv',
    'SGD High LR (0.1)': 'logs/sgd_high_lr/training_log.csv',
    'Modern (Adam, ReLU, 0.001)': 'logs/modern_adam_relu/training_log.csv'
}

# Loading data
dfs = {}
for name, path in log_paths.items():
    if os.path.exists(path):
        dfs[name] = pd.read_csv(path)
    else:
        print(f"Warning: {path} not found.")

# Plot Training Loss
plt.figure(figsize=(10, 6))
for name, df in dfs.items():
    plt.plot(df['epoch'], df['train_loss'], label=name, marker='o')

plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('docs/figures/training_loss.png')
print("Saved docs/figures/training_loss.png")

# Plot Test Accuracy
plt.figure(figsize=(10, 6))
for name, df in dfs.items():
    plt.plot(df['epoch'], df['test_accuracy'], label=name, marker='o')

plt.title('Test Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('docs/figures/test_accuracy.png')
print("Saved docs/figures/test_accuracy.png")
