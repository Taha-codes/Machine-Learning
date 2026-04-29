import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

# [B] = BOILERPLATE  — identical in every PyTorch project, memorise it
# [P] = PROJECT      — you decide this based on your specific task

# ─────────────────────────────────────────────
# 0. DEVICE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # [B]


# ─────────────────────────────────────────────
# 1. HYPERPARAMETERS
# ─────────────────────────────────────────────
BATCH_SIZE    = 64      # [P] — try 32, 64, 128
LEARNING_RATE = 0.001   # [P] — Adam default; tune if loss doesn't decrease
NUM_EPOCHS    = 10      # [P] — stop when val loss stops improving
NUM_CLASSES   = 10      # [P] — depends on your dataset (CIFAR-10 → 10)


# ─────────────────────────────────────────────
# 2. DATA
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),                                   # [B]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [P] — mean/std of YOUR dataset
])

train_data   = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)  # [P]
test_data    = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)  # [P]

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)   # [B]
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)  # [B]


# ─────────────────────────────────────────────
# 3. MODEL DEFINITION
# ─────────────────────────────────────────────
class CNN(nn.Module):                    # [B] — always subclass nn.Module

    def __init__(self):
        super().__init__()               # [B] — always call this first

        self.features = nn.Sequential(
            # Block 1: (batch, 3, 32, 32) → (batch, 32, 16, 16)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [P] — in_ch, out_ch, kernel
            nn.ReLU(),                                   # [P] — activation choice
            nn.MaxPool2d(2),                             # [P] — pooling strategy

            # Block 2: (batch, 32, 16, 16) → (batch, 64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [P]
            nn.ReLU(),                                   # [P]
            nn.MaxPool2d(2),                             # [P]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                        # [B]
            nn.Linear(64 * 8 * 8, 512),          # [P] — in_features must match flattened size
            nn.ReLU(),                           # [P]
            nn.Linear(512, NUM_CLASSES),         # [P] — out_features = number of classes
        )

    def forward(self, x):        # [B] — signature never changes
        x = self.features(x)    # [B] — call your layers on x
        x = self.classifier(x)  # [B]
        return x                 # [B] — return raw logits


# ─────────────────────────────────────────────
# 4. LOSS & OPTIMIZER
# ─────────────────────────────────────────────
model     = CNN().to(device)                                    # [B]
criterion = nn.CrossEntropyLoss()                               # [P] — loss depends on task
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)    # [P] — optimizer + lr


# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────
def train(model, loader, criterion, optimizer):
    model.train()                                       # [B]
    total_loss, correct, total = 0, 0, 0                # [B]

    for images, labels in loader:                       # [B]
        images, labels = images.to(device), labels.to(device)  # [B]

        optimizer.zero_grad()                           # [B] ← step 1: clear gradients
        outputs = model(images)                         # [B] ← step 2: forward pass
        loss    = criterion(outputs, labels)            # [B] ← step 3: compute loss
        loss.backward()                                 # [B] ← step 4: backprop
        optimizer.step()                                # [B] ← step 5: update weights

        total_loss += loss.item()                       # [B]
        _, predicted = outputs.max(1)                   # [P] — accuracy metric (classification)
        correct += predicted.eq(labels).sum().item()    # [P]
        total   += labels.size(0)                       # [B]

    return total_loss / len(loader), correct / total    # [P] — what you return/log


# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()                            # [B] — always switch to eval mode
    total_loss, correct, total = 0, 0, 0    # [B]

    with torch.no_grad():                   # [B] — always wrap eval in no_grad
        for images, labels in loader:                       # [B]
            images, labels = images.to(device), labels.to(device)  # [B]

            outputs = model(images)                         # [B]
            loss    = criterion(outputs, labels)            # [B]

            total_loss += loss.item()                       # [B]
            _, predicted = outputs.max(1)                   # [P]
            correct += predicted.eq(labels).sum().item()    # [P]
            total   += labels.size(0)                       # [B]

    return total_loss / len(loader), correct / total        # [P]


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
for epoch in range(1, NUM_EPOCHS + 1):                          # [B]
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)    # [B]
    val_loss,   val_acc   = evaluate(model, test_loader, criterion)             # [B]

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Train  loss: {train_loss:.3f}  acc: {train_acc*100:.1f}% | "
        f"Val    loss: {val_loss:.3f}  acc: {val_acc*100:.1f}%"
    )
