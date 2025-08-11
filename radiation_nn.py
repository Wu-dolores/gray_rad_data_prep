
# Import necessary libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tqdm import tqdm
import os
from termcolor import cprint




# Define the CNN model for radiation prediction
class RadiationCNN(nn.Module):
    def __init__(self):
        super(RadiationCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        # Second convolutional layer
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        # Third convolutional layer
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        # Fully connected layer to output upward and downward fluxes
        self.fc = nn.Linear(64, 2)  # Predict two fluxes per layer: upward and downward

    def forward(self, x):
        # Change input shape for Conv1d: (batch, features, layers)
        x = x.permute(0, 2, 1)  # (batch, 3, 50)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        # Restore shape for fully connected layer
        x = x.permute(0, 2, 1)  # (batch, 50, 64)
        x = self.fc(x)           # (batch, 50, 2)
        return x


# Visualize model predictions vs. ground truth for the test set
def vis_prediction(args, model, X_test_tensor, Y_test_tensor, scaler_Y, criterion, vis_num):
    model.eval()
    with torch.no_grad():
        # Get model predictions for test set
        Y_pred = model(X_test_tensor)
        test_loss = criterion(Y_pred, Y_test_tensor)
        print(f"Test Loss: {test_loss.item():.4f}")

    # Convert tensors to numpy arrays
    Y_pred_np = Y_pred.cpu().numpy()
    Y_test_np = Y_test_tensor.cpu().numpy()

    # Inverse transform to original scale
    Y_pred_inv = scaler_Y.inverse_transform(Y_pred_np.reshape(-1, 2)).reshape(Y_pred_np.shape)
    Y_true_inv = scaler_Y.inverse_transform(Y_test_np.reshape(-1, 2)).reshape(Y_test_np.shape)

    # Visualize the first sample's fluxes
    for sample_idx in range(vis_num):
        plt.figure(figsize=(8, 5))
        plt.plot(Y_true_inv[sample_idx, :, 0], label="True Upward Flux")
        plt.plot(Y_pred_inv[sample_idx, :, 0], label="Pred Upward Flux")
        plt.plot(Y_true_inv[sample_idx, :, 1], label="True Downward Flux")
        plt.plot(Y_pred_inv[sample_idx, :, 1], label="Pred Downward Flux")
        plt.xlabel("Layer Index")
        plt.ylabel("Flux")
        plt.legend()
        plt.tight_layout()
        save_dir = os.path.join(args.output_path, f"prediction_{sample_idx:06d}.png")
        plt.savefig(save_dir)

# Visualize training and validation loss curves
def vis_loss(args, train_losses, val_losses, filename="loss.png"):
    num_epochs = args.epochs
    save_dir = os.path.join(args.output_path, filename)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir)

# Prepare and preprocess the dataset
def prepare_data(args):
    # Load dataset from CSV file
    df = pd.read_csv(args.data_path)

    # Separate input features and output targets
    X = df[['p', 'T','density']].values
    Y = df[['up_flux', 'down_flux']].values
    # print(X.shape,Y.shape)

    # Normalize input features
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    # print(X_scaled.shape)
    # Normalize output targets
    scaler_Y = MinMaxScaler()
    Y_scaled = scaler_Y.fit_transform(Y)
    # print(Y_scaled.shape)

    # Reshape to (num_cases, num_layers, num_features)
    X = X_scaled.reshape(100000, 50, 3)  # 100 cases, 50 layers per case, 3 features per layer
    Y = Y_scaled.reshape(100000, 50, 2)  # 100 cases, 50 layers per case, 2 outputs per layer

    # Split into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=args.device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32, device=args.device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=args.device)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32, device=args.device)

    # Print shapes for confirmation
    cprint(f"Train X shape: {X_train_tensor.shape}, Y shape: {Y_train_tensor.shape}", "green")
    cprint(f"Test X shape: {X_test_tensor.shape}, Y shape: {Y_test_tensor.shape}", "green")

    # Create DataLoader for training and test sets
    train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, X_test_tensor, Y_test_tensor, scaler_Y


# Train the model and record losses
def train(args, model, criterion, optimizer, train_loader, val_loader):
    model.train()
    train_losses = []
    val_losses = []

    num_epochs = args.epochs
    # Create tqdm progress bar
    pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch", total=num_epochs)
    for epoch in pbar:
        model.train()  # Set model to training mode
        epoch_train_losses = []
        for X_train_tensor, Y_train_tensor in train_loader:
            X_train_tensor = X_train_tensor.to(args.device)
            Y_train_tensor = Y_train_tensor.to(args.device)
            optimizer.zero_grad()  # Clear gradients
            output_train = model(X_train_tensor)  # Forward pass

            # Flatten outputs and targets for loss calculation
            output_train_flat = output_train.reshape(-1, 2)
            Y_train_flat = Y_train_tensor.reshape(-1, 2)
            
            # Compute training loss
            train_loss = criterion(output_train_flat, Y_train_flat)
            
            train_loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            
            # Record training loss for this batch
            epoch_train_losses.append(train_loss.item())

        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)

        # Compute validation loss
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            epoch_val_losses = []
            for X_val_tensor, Y_val_tensor in val_loader:
                X_val_tensor = X_val_tensor.to(args.device)
                Y_val_tensor = Y_val_tensor.to(args.device)
                output_val = model(X_val_tensor)
                output_val_flat = output_val.reshape(-1, 2)
                Y_val_flat = Y_val_tensor.reshape(-1, 2)
                val_loss = criterion(output_val_flat, Y_val_flat)
                epoch_val_losses.append(val_loss.item())

            avg_val_loss = np.mean(epoch_val_losses)
            val_losses.append(avg_val_loss)

        # Update tqdm description with current losses
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    return train_losses, val_losses


# Main function to parse arguments and run training and evaluation
def main():
    parser = argparse.ArgumentParser(description="Train a CNN model for radiation data")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--data_path", type=str, default="atmospheric_radiation_dataset.csv", help="Path to the dataset")
    parser.add_argument("--output_path", type=str, default="outputs", help="Path to save the model output")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train the model on")
    parser.add_argument("--vis_num", type=int, default=50, help="Number of samples to visualize")
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    cprint("="*60, "cyan")
    cprint("Training Configuration", "green", attrs=["bold"])
    cprint(f"Device: {args.device}", "green")
    cprint(f"Learning Rate: {args.lr}", "green")
    cprint(f"Batch Size: {args.batch_size}", "green")
    cprint(f"Epochs: {args.epochs}", "green")
    cprint("-"*60, "cyan")
    # Initialize model and move to device
    model = RadiationCNN()
    model = model.to(args.device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Prepare data loaders and test tensors
    cprint("Preparing Data Loaders", "green", attrs=["bold"])
    train_loader, test_loader, X_test_tensor, Y_test_tensor, scaler_Y = prepare_data(args)

    # Train the model
    cprint("Starting Training", "green", attrs=["bold"])
    train_losses, val_losses = train(args, model, criterion, optimizer, train_loader, test_loader)

    cprint("Training Complete, visualizing results...", "green", attrs=["bold"])
    # Visualize loss curves
    vis_loss(args, train_losses, val_losses, "loss.png")
    # Visualize predictions
    vis_prediction(args, model, X_test_tensor, Y_test_tensor, scaler_Y, criterion, vis_num=args.vis_num)

# Entry point
if __name__ == "__main__":
    main()