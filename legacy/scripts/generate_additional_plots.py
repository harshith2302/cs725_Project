import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths to logs
log_paths = {
    'Baseline (LeNet)': 'logs/baseline_paper/training_log.csv',
    'MLP': 'logs/mlp_baseline/training_log.csv',
    'LeNet + Augmentation': 'logs/lenet_augmented/training_log.csv',
    'LeNet (10% Data)': 'logs/lenet_10pct/training_log.csv',
    'LeNet (50% Data)': 'logs/lenet_50pct/training_log.csv'
}

# Loading data
dfs = {}
for name, path in log_paths.items():
    if os.path.exists(path):
        dfs[name] = pd.read_csv(path)
    else:
        print(f"Warning: {path} not found.")

# 1. Model Comparison (Bar Chart)
plt.figure(figsize=(8, 6))
models = ['Baseline (LeNet)', 'MLP']
accuracies = [dfs[m]['test_accuracy'].max() for m in models]
plt.bar(models, accuracies, color=['blue', 'orange'])
plt.title('Model Comparison: LeNet vs MLP')
plt.ylabel('Best Test Accuracy (%)')
plt.ylim(95, 100)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.1, f"{v:.2f}%", ha='center')
plt.savefig('docs/figures/model_comparison.png')
print("Saved docs/figures/model_comparison.png")

# 2. Augmentation Comparison (Bar Chart)
plt.figure(figsize=(8, 6))
aug_models = ['Baseline (LeNet)', 'LeNet + Augmentation']
aug_accs = [dfs[m]['test_accuracy'].max() for m in aug_models]
plt.bar(aug_models, aug_accs, color=['blue', 'green'])
plt.title('Effect of Data Augmentation')
plt.ylabel('Best Test Accuracy (%)')
plt.ylim(98, 100)
for i, v in enumerate(aug_accs):
    plt.text(i, v + 0.05, f"{v:.2f}%", ha='center')
plt.savefig('docs/figures/augmentation_comparison.png')
print("Saved docs/figures/augmentation_comparison.png")

# 3. Training Set Size (Line Chart)
plt.figure(figsize=(10, 6))
sizes = [10, 50, 100]
size_accs = [
    dfs['LeNet (10% Data)']['test_accuracy'].max(),
    dfs['LeNet (50% Data)']['test_accuracy'].max(),
    dfs['Baseline (LeNet)']['test_accuracy'].max()
]
plt.plot(sizes, size_accs, marker='o', linestyle='-', color='purple')
plt.title('Effect of Training Set Size on Accuracy')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Best Test Accuracy (%)')
plt.xticks(sizes)
plt.grid(True)
for i, txt in enumerate(size_accs):
    plt.annotate(f"{txt:.2f}%", (sizes[i], size_accs[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.savefig('docs/figures/training_size_effect.png')
print("Saved docs/figures/training_size_effect.png")
