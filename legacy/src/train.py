import torch
import torch.nn as nn
import torch.optim as optim
from src.models.lenet import LeNet5
from src.models.mlp import MLP
from src.data.loader import get_data_loaders
import os
import csv
import argparse

def train(epochs=10, lr=0.01, momentum=0.9, log_dir='logs/baseline_paper', activation='tanh', optimizer_name='sgd', model_type='lenet', augment=False, train_pct=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Configuration: Model={model_type}, Augment={augment}, TrainPct={train_pct}, Activation={activation}, Optimizer={optimizer_name}, LR={lr}, LogDir={log_dir}")

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Data Loaders
    train_loader, test_loader = get_data_loaders(augment=augment, train_pct=train_pct)

    # Model
    if model_type.lower() == 'lenet':
        model = LeNet5(activation=activation).to(device)
    elif model_type.lower() == 'mlp':
        model = MLP().to(device)
    else:
        raise ValueError(f"Unsupported model: {model_type}")

    # Optimizer
    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    criterion = nn.CrossEntropyLoss()

    # Logging
    log_file = os.path.join(log_dir, 'training_log.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'test_loss', 'test_accuracy'])

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)

        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)

        print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Log results
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, avg_test_loss, accuracy])

        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))

    print(f"Training complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LeNet-5 on MNIST')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--log_dir', type=str, default='logs/baseline_paper', help='Directory for logs')
    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'relu'], help='Activation function')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer')
    parser.add_argument('--model', type=str, default='lenet', choices=['lenet', 'mlp'], help='Model type')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--train_pct', type=float, default=1.0, help='Percentage of training data to use')

    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr, momentum=args.momentum, log_dir=args.log_dir, activation=args.activation, optimizer_name=args.optimizer, model_type=args.model, augment=args.augment, train_pct=args.train_pct)
