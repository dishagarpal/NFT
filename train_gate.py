import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "gate_training.json"

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

MODEL_OUTPUT = "gating_model.pth"


# ============================================================
# LOAD DATA
# ============================================================

with open(DATA_FILE, "r") as f:
    data = json.load(f)

print("Total examples:", len(data))


# ============================================================
# CONVERT DATA
# ============================================================

X = []
y = []

for example in data:

    features = example["features"]

    # Class-specific confidence + presence
    X.append([
        features["a_conf"],
        features["b_conf"],
        features["c_conf"],

        features["a_present"],
        features["b_present"],
        features["c_present"]
    ])

    target = example["target"]

    target_id = {
        "A": 0,
        "B": 1,
        "C": 2
    }[target]

    y.append(target_id)


X = torch.tensor(
    X,
    dtype=torch.float32
)

y = torch.tensor(
    y,
    dtype=torch.long
)

print("Feature size:", X.shape[1])


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training examples:", len(X_train))
print("Validation examples:", len(X_val))


# ============================================================
# DATASET
# ============================================================

class GateDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (
            self.X[index],
            self.y[index]
        )


train_dataset = GateDataset(
    X_train,
    y_train
)

val_dataset = GateDataset(
    X_val,
    y_val
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)


# ============================================================
# GATING NETWORK
# ============================================================

class GatingNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(6, 32),

            nn.ReLU(),

            nn.Linear(32, 16),

            nn.ReLU(),

            nn.Linear(16, 3)
        )

    def forward(self, x):

        return self.network(x)


model = GatingNetwork()


# ============================================================
# HANDLE CLASS IMBALANCE
# ============================================================

class_counts = torch.bincount(y_train)

weights = (
    len(y_train)
    /
    (
        len(class_counts)
        * class_counts.float()
    )
)

print("\nClass weights:")

print(
    "YOLO-A:",
    float(weights[0])
)

print(
    "YOLO-B:",
    float(weights[1])
)

print(
    "YOLO-C:",
    float(weights[2])
)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n==============================")
print("TRAINING CLASS-AWARE GATING MODEL")
print("==============================")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == batch_y
        ).sum().item()

        total += len(batch_y)


    train_accuracy = correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            outputs = model(batch_X)

            predictions = outputs.argmax(
                dim=1
            )

            val_correct += (
                predictions == batch_y
            ).sum().item()

            val_total += len(batch_y)


    val_accuracy = val_correct / val_total


    print(
        f"Epoch "
        f"{epoch + 1:02d}/{EPOCHS} | "
        f"Loss: "
        f"{total_loss / len(train_loader):.4f} | "
        f"Train Acc: "
        f"{train_accuracy:.3f} | "
        f"Val Acc: "
        f"{val_accuracy:.3f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_OUTPUT
)

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")

print(
    "Saved gating model:",
    MODEL_OUTPUT
)