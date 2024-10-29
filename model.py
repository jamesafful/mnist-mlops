import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb
from sklearn.metrics import precision_score, recall_score

# Define the Logistic Regression model with hidden layers
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters to experiment with
config = {
    "learning_rate": 0.01,
    "epochs": 10,
    "batch_size": 64,
}

# Initialize wandb
wandb.init(project="mnist-mlops", config=config)

# Watch the model
model = LogisticRegression(input_dim=28*28, output_dim=10)
wandb.watch(model, log="all")

# Load and preprocess data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Initialize loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Function to log predictions
def log_predictions(model, data, target, num_samples=10):
    model.eval()
    with torch.no_grad():
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        
        # Log predictions
        for i in range(num_samples):
            wandb.log({
                "sample_prediction": wandb.Image(data[i], caption=f'True: {target[i].item()}, Predicted: {predicted[i].item()}')
            })

# Training loop
for epoch in range(config["epochs"]):
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradients
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # Log loss and learning rate
        wandb.log({"loss": loss.item(), "learning_rate": config["learning_rate"]})

        # Log gradient norms
        grad_norm = torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]))
        wandb.log({"gradient_norm": grad_norm.item()})

    # Validation
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)  # Forward pass
            _, predicted = torch.max(output.data, 1)  # Get the predicted class
            total += target.size(0)  # Total number of samples
            correct += (predicted == target).sum().item()  # Count correct predictions
            
            all_targets.extend(target.numpy())  # Collect true labels
            all_predictions.extend(predicted.numpy())  # Collect predicted labels

    accuracy = 100 * correct / total  # Calculate accuracy
    wandb.log({"accuracy": accuracy})  # Log accuracy with W&B

    # Calculate and log precision and recall
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    wandb.log({"precision": precision, "recall": recall})

    # Log sample predictions
    log_predictions(model, next(iter(test_loader))[0][:10], next(iter(test_loader))[1][:10])
    
    print(f'Epoch {epoch + 1}/{config["epochs"]}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}')
# After the training loop
torch.save(model.state_dict(), "model_weights.pth")  # Save the model weights

wandb.finish()  # Finish W&B session

