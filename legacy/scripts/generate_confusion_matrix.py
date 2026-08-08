import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.models.lenet import LeNet5
from src.data.loader import get_data_loaders
import os
import numpy as np

def generate_confusion_matrix():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    _, test_loader = get_data_loaders()
    
    # Load Model
    model = LeNet5().to(device)
    model_path = 'logs/baseline_paper/best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1).cpu().numpy()
            targets = target.cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    # Compute Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Baseline Model)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('docs/figures/confusion_matrix.png')
    print("Saved docs/figures/confusion_matrix.png")

if __name__ == "__main__":
    generate_confusion_matrix()
