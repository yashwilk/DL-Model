import torch
import torch.nn as nn

class StockDirection(nn.Module):
    def __init__(self, input_size):
        super(StockDirection, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# Sample data: 200 samples with 5 features, binary labels
features = ["open", "high", "low", "close", "volume"]
X_train = torch.randn(200, len(features))
y_train = torch.randint(0, 2, (200, 1)).float()

model = StockDirection(input_size=len(features))

criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


model.eval()