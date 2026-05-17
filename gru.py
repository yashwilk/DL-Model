import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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


def run_fold(df, cfg, fold_name, valid_start, valid_end):
    feature_cols = [c for c in df.columns if c.startswith("feature_")]

    train_df = df[df["date_id"] < valid_start].copy()
    valid_df = df[(df["date_id"] >= valid_start) & (df["date_id"] <= valid_end)].copy()

    train_dataset = TimeSeriesDataset(train_df, feature_cols, cfg)
    valid_dataset = TimeSeriesDataset(valid_df, feature_cols, cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    model = GRUModel(
        input_size=len(feature_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        use_aux_targets=cfg.use_aux_targets,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_score = -999

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg)
        score = validate(model, valid_loader, cfg)

        best_score = max(best_score, score)

        print(
            f"{fold_name} | Epoch {epoch+1} | "
            f"Loss: {train_loss:.5f} | Score: {score:.5f}"
        )

        if cfg.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                f"{fold_name}/loss": train_loss,
                f"{fold_name}/score": score,
                "epoch": epoch + 1,
            })

    if cfg.use_online_learning:
        print(f"{fold_name} | Running dummy online learning step...")

        online_df = valid_df.copy()
        online_dataset = TimeSeriesDataset(online_df, feature_cols, cfg)

        online_loader = DataLoader(
            online_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )

        train_one_epoch(model, online_loader, optimizer, cfg)

    return best_score