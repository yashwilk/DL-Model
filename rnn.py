import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.RNN(input_size=1,hidden_size=64,num_layers=2,batch_first=True)
        self.fc=nn.Linear(64,1)
    def forward(self,x):
        out,hidden=self.rnn(x)
        out=out[:,-1,:]
        out=self.fc(out)
        return out

# Sample data: 100 sequences of length 20, each with 1 feature
X_train = torch.randn(100, 20, 1)
y_train = torch.randn(100, 1)

model=RNNModel()
criterion=nn.MSELoss()
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.01)
epochs=10
for epoch in range(epochs):
    optimizer.zero_grad()
    prediction=model(X_train)
    loss=criterion(prediction,y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

