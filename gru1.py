import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, use_aux_targets):
        super().__init__()
        self.use_aux_targets = use_aux_targets

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, 1)
        if use_aux_targets:
            self.aux_head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]

        target_pred = self.fc(out).squeeze(1)

        if self.use_aux_targets:
            aux_pred = self.aux_head(out)
            return target_pred, aux_pred

        return target_pred, None
 

 def train_one_epoch(model, loader, optimizer, cfg):

    model.train()

    criterion = nn.MSELoss()
    total_loss = 0
    for x, y, aux in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        aux = aux.to(cfg.device)

        optimizer.zero_grad()
        target_pred, aux_pred = model(x)
        loss = criterion(target_pred, y)
        if cfg.use_aux_targets:
            aux_loss = criterion(aux_pred, aux)
            loss = loss + 0.3 * aux_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)